import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 1. 自注意力模块（与原代码相同）
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (embed_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


# 2. 模型并行改造：拆分到不同GPU
class ModelParallelModel(nn.Module):
    def __init__(self, input_dim=10, embed_dim=64, num_classes=2):
        super(ModelParallelModel, self).__init__()
        # 定义两个GPU设备（需确保至少有2个GPU）
        self.gpu0 = torch.device("cuda:0")
        self.gpu1 = torch.device("cuda:1")
        
        # 第一部分：嵌入层+注意力层 → 分配到GPU0
        self.embedding = nn.Linear(input_dim, embed_dim).to(self.gpu0)
        self.attention = SelfAttention(embed_dim).to(self.gpu0)
        
        # 第二部分：池化层+分类器 → 分配到GPU1
        self.global_pool = nn.AdaptiveAvgPool1d(1).to(self.gpu1)
        self.classifier = nn.Linear(embed_dim, num_classes).to(self.gpu1)
        
    def forward(self, x):
        # 步骤1：输入先传到GPU0（第一部分所在设备）
        x = x.to(self.gpu0)  # 输入移到GPU0
        
        # 步骤2：GPU0上计算嵌入层和注意力层
        x = self.embedding(x)  # 结果在GPU0
        x = self.attention(x)  # 结果在GPU0
        
        # 步骤3：中间结果传到GPU1（第二部分所在设备）
        x = x.to(self.gpu1)  # 移到GPU1
        
        # 步骤4：GPU1上计算池化和分类
        x = self.global_pool(x.transpose(1, 2)).squeeze(-1)  # 结果在GPU1
        logits = self.classifier(x)  # 最终输出在GPU1
        
        return logits


# 3. 数据集保持不变（与原代码相同）
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, input_dim=10):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.data = torch.rand(num_samples, seq_len, input_dim) * 2 - 1  # 随机数据
        self.labels = (torch.mean(self.data, dim=(1, 2)) > 0).long()  # 标签
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 4. 训练函数（模型并行版本）
def train():
    # 超参数
    input_dim = 10
    embed_dim = 64
    num_classes = 2
    num_samples = 10000
    seq_len = 10
    batch_size = 32
    epochs = 5
    lr = 0.001
    
    # 检查GPU数量（至少需要2个）
    if torch.cuda.device_count() < 2:
        raise ValueError("模型并行需要至少2个GPU！")
    print(f"使用GPU0: {torch.cuda.get_device_name(0)}, GPU1: {torch.cuda.get_device_name(1)}")
    
    # 初始化模型（已拆分到两个GPU）
    model = ModelParallelModel(input_dim=input_dim, embed_dim=embed_dim, num_classes=num_classes)
    
    # 数据加载
    dataset = SyntheticDataset(num_samples=num_samples, seq_len=seq_len, input_dim=input_dim)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True  # 加速CPU到GPU的数据传输
    )
    
    # 损失函数和优化器（自动处理跨GPU参数）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 所有GPU的参数都会被优化
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # 输入移到GPU0（模型起始层所在设备）
            inputs = inputs.to(model.gpu0, non_blocking=True)
            # 标签移到GPU1（输出层所在设备，需与logits同设备）
            labels = labels.to(model.gpu1, non_blocking=True)
            
            # 前向传播（自动跨GPU计算）
            outputs = model(inputs)  # outputs在GPU1
            
            # 计算损失（在GPU1上）
            loss = criterion(outputs, labels)
            
            # 反向传播（梯度自动跨GPU传递）
            optimizer.zero_grad()
            loss.backward()  # PyTorch会自动处理跨设备梯度
            optimizer.step()
            
            # 统计指标
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 打印中间结果
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(dataloader)}] '
                      f'Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')
        
        # 每个epoch总结
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/{epochs}] Summary: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    
    # 保存模型
    torch.save(model.state_dict(), 'model_parallel.pth')
    print("模型已保存到 model_parallel.pth")


# 主函数
def main():
    if not torch.cuda.is_available():
        print("错误：无可用GPU！")
        return
    if torch.cuda.device_count() < 2:
        print("错误：至少需要2个GPU进行模型并行！")
        return
    train()


if __name__ == "__main__":
    main()
