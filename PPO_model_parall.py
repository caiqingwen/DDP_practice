import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_async, remote
import socket

# 查找可用端口，避免端口冲突
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return str(s.getsockname()[1])

# 1. 模型组件定义
class ModelPart1(nn.Module):
    """模型第一部分：运行在指定GPU上"""
    def __init__(self, input_dim=10, embed_dim=64):
        super().__init__()
        self.device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES会限制可见GPU，因此总是使用cuda:0
        self.embedding = nn.Linear(input_dim, embed_dim).to(self.device)
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        ).to(self.device)
    
    def forward(self, x):
        return self.attention(self.embedding(x.to(self.device)))


class ModelPart2(nn.Module):
    """模型第二部分：运行在指定GPU上"""
    def __init__(self, embed_dim=64, num_classes=2):
        super().__init__()
        self.device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES会限制可见GPU，因此总是使用cuda:0
        self.pooling = nn.AdaptiveAvgPool1d(1).to(self.device)
        self.classifier = nn.Linear(embed_dim, num_classes).to(self.device)
    
    def forward(self, x):
        return self.classifier(self.pooling(x.to(self.device).transpose(1, 2)).squeeze(-1))


# 2. 全局模型（协调两部分）
class DistModel(nn.Module):
    def __init__(self, part1_rref, part2_rref):
        super().__init__()
        self.part1_rref = part1_rref
        self.part2_rref = part2_rref
    
    def forward(self, x):
        # 异步调用第一部分
        fut = rpc_async(
            self.part1_rref.owner(),
            ModelPart1.forward,
            args=(self.part1_rref.local_value(), x)
        )
        x = fut.wait()
        
        # 异步调用第二部分
        fut = rpc_async(
            self.part2_rref.owner(),
            ModelPart2.forward,
            args=(self.part2_rref.local_value(), x)
        )
        return fut.wait()


# 3. 数据集
class SyntheticDataset(nn.Module):
    def __init__(self, num_samples=10000, input_dim=10):
        super().__init__()
        self.data = torch.randn(num_samples, input_dim)
        self.labels = (torch.mean(self.data, dim=1) > 0).long()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 4. 参数服务器进程（协调训练）
def run_parameter_server(rank, world_size, port):
    # 初始化RPC
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    rpc.init_rpc(
        "parameter_server",
        rank=rank,
        world_size=world_size,
        backend=rpc.BackendType.TENSORPIPE
    )
    
    print(f"参数服务器已启动，等待工作节点连接...")
    
    # 创建远程模型引用
    part1_rref = remote("worker0", ModelPart1)
    part2_rref = remote("worker1", ModelPart2)
    
    # 正确收集参数（避免序列化生成器）
    def get_params(module_rref):
        params = []
        for param in module_rref.rpc_sync().parameters():
            params.append(param)
        return params
    
    params = get_params(part1_rref) + get_params(part2_rref)
    optimizer = optim.Adam(params, lr=0.001)
    
    # 训练数据
    dataset = SyntheticDataset(num_samples=10000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    # 训练循环
    model = DistModel(part1_rref, part2_rref)
    model.train()
    
    for epoch in range(5):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            labels = labels.to(outputs.device)  # 确保标签与输出在同一设备
            
            # 计算损失
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}, Acc: {100.*correct/total:.2f}%")
    
    # 保存模型
    torch.save({
        'part1_state_dict': part1_rref.rpc_sync().state_dict(),
        'part2_state_dict': part2_rref.rpc_sync().state_dict()
    }, 'distributed_model.pth')
    
    # 关闭RPC
    rpc.shutdown()


# 5. 工作进程（运行模型部分）
def run_worker(rank, world_size, gpu_id, port):
    # 动态检查可用GPU
    if torch.cuda.device_count() <= gpu_id:
        raise RuntimeError(f"GPU {gpu_id} 不可用！系统只有 {torch.cuda.device_count()} 个GPU")
    
    # 设置当前GPU
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 初始化RPC
    name = f"worker{rank-1}"
    rpc.init_rpc(
        name,
        rank=rank,
        world_size=world_size,
        backend=rpc.BackendType.TENSORPIPE
    )
    
    # 等待参数服务器的RPC调用
    print(f"{name} 使用 GPU {gpu_id} ({torch.cuda.get_device_name(0)}) 已启动，等待RPC调用...")
    
    # 阻塞直到所有RPC完成
    rpc.shutdown()


# 6. 进程入口函数
def run_process(rank, world_size, port):
    if rank == 0:  # 参数服务器
        run_parameter_server(rank, world_size, port)
    else:  # 工作节点
        gpu_id = rank - 1  # worker0 -> cuda:0, worker1 -> cuda:1
        run_worker(rank, world_size, gpu_id, port)


# 7. 主函数
def main():
    # 动态检查可用GPU
    available_gpus = torch.cuda.device_count()
    print(f"发现 {available_gpus} 个可用GPU:")
    for i in range(available_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if available_gpus < 2:
        raise RuntimeError(f"需要至少2个GPU，当前只有 {available_gpus} 个")
    
    world_size = 3  # 1个参数服务器 + 2个工作节点
    
    # 查找可用端口
    port = find_free_port()
    print(f"使用端口 {port} 进行RPC通信")
    
    # 使用spawn启动多进程
    mp.spawn(
        run_process,
        args=(world_size, port),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    # 设置多进程启动方法为spawn
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    
    main()
