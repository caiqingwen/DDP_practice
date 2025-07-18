import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# 1. 定义简单的自注意力模型
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        """
        简单的自注意力模块
        Args:
            embed_dim: 嵌入维度
        """
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, embed_dim]
        Returns:
            注意力输出 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # 计算query, key, value
        q = self.query(x)  # [batch_size, seq_len, embed_dim]
        k = self.key(x)    # [batch_size, seq_len, embed_dim]
        v = self.value(x)  # [batch_size, seq_len, embed_dim]
        
        # 计算注意力分数 (缩放点积注意力)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (embed_dim ** 0.5)
        # scores shape: [batch_size, seq_len, seq_len]
        
        # 应用softmax获取注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 加权求和得到输出
        output = torch.matmul(attn_weights, v)
        # output shape: [batch_size, seq_len, embed_dim]
        
        return output

# 2. 定义完整模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, embed_dim=64, num_classes=2):
        """
        包含自注意力模块的简单模型
        Args:
            input_dim: 输入维度
            embed_dim: 嵌入维度
            num_classes: 分类类别数
        """
        super(SimpleModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.attention = SelfAttention(embed_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 序列维度上的全局平均池化
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
        Returns:
            分类 logits [batch_size, num_classes]
        """
        # 嵌入层
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # 自注意力层
        x = self.attention(x)  # [batch_size, seq_len, embed_dim]
        
        # 全局池化: [batch_size, seq_len, embed_dim] -> [batch_size, embed_dim, 1]
        x = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        
        # 分类器
        logits = self.classifier(x)  # [batch_size, num_classes]
        
        return logits

# 3. 生成模拟数据集
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, input_dim=10):
        """
        合成数据集，生成随机数据和标签
        Args:
            num_samples: 样本数量
            seq_len: 序列长度
            input_dim: 特征维度
        """
        super(SyntheticDataset, self).__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # 生成随机输入数据 (在-1到1之间)
        self.data = torch.rand(num_samples, seq_len, input_dim) * 2 - 1
        
        # 生成标签: 如果序列的平均值 > 0，则为类别1，否则为类别0
        self.labels = (torch.mean(self.data, dim=(1, 2)) > 0).long()
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 4. 初始化分布式环境
def setup(rank, world_size):
    """
    初始化分布式训练环境
    Args:
        rank: 当前进程的全局排名
        world_size: 总进程数
    """
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',  # 使用NCCL后端进行GPU通信
        init_method='env://',  # 通过环境变量初始化
        world_size=world_size,
        rank=rank
    )
    
    # 设置当前GPU
    torch.cuda.set_device(rank)

# 5. 清理分布式环境
def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

# 6. 训练函数 (每个进程都会执行)
def train(rank, world_size):
    """
    每个GPU上的训练函数
    Args:
        rank: 当前GPU的排名
        world_size: 总GPU数
    """
    # 初始化分布式环境
    setup(rank, world_size)
    
    # 直接设置参数（替代argparse）
    input_dim = 10       # 输入维度
    embed_dim = 64       # 嵌入维度
    num_classes = 2      # 分类类别数
    num_samples = 10000  # 样本数量
    seq_len = 10         # 序列长度
    batch_size = 32      # 每个GPU的批次大小
    num_workers = 4      # 数据加载工作线程数
    epochs = 5           # 训练轮数
    lr = 0.001           # 学习率
    
    # 打印当前进程使用的GPU信息
    gpu_name = torch.cuda.get_device_name(rank)
    gpu_mem = torch.cuda.get_device_properties(rank).total_memory / 1024**3  # GB
    print(f"进程 {rank} 使用 GPU: {gpu_name} (显存: {gpu_mem:.2f} GB)")
    
    # 创建模型并移动到当前GPU
    model = SimpleModel(input_dim=input_dim, 
                        embed_dim=embed_dim, 
                        num_classes=num_classes).to(rank)
    
    # 包装模型为DDP
    model = DDP(model, device_ids=[rank])
    
    # 创建合成数据集
    dataset = SyntheticDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        input_dim=input_dim
    )
    
    # 创建分布式采样器
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True  # 每个epoch打乱数据
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True  # 加速CPU到GPU的数据传输
    )
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        # 设置采样器的epoch，确保每个epoch的shuffle不同
        dataloader.sampler.set_epoch(epoch)
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # 将数据移到当前GPU
            inputs = inputs.to(rank)
            labels = labels.to(rank)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 每10个batch打印一次信息 (只在主进程打印)
            if rank == 0 and (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(dataloader)}] '
                      f'Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')
        
        # 每个epoch结束后，只在主进程打印汇总信息
        if rank == 0:
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = 100. * correct / total
            print(f'Epoch [{epoch+1}/{epochs}] Summary: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    
    # 保存模型 (只在主进程保存)
    if rank == 0:
        torch.save(model.module.state_dict(), 'model_ddp.pth')
        print("Model saved to model_ddp.pth")
    
    # 清理分布式环境
    cleanup()

# 7. 主函数
def main():
    # 检查可用GPU数量
    if not torch.cuda.is_available():
        print("错误: 没有可用的GPU!")
        return
    
    total_gpus = torch.cuda.device_count()
    print(f"发现 {total_gpus} 个可用GPU:")
    for i in range(total_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        print(f"  GPU {i}: {gpu_name} (显存: {gpu_mem:.2f} GB)")
    
    # 设置世界大小为GPU数量或手动指定
    world_size = min(total_gpus, 2)  # 使用全部GPU或最多2个
    print(f"将使用 {world_size} 个GPU进行训练")
    
    # 使用spawn启动多个进程
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()    
