import wandb
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 初始化W&B
wandb.init(project="mnist-classification", entity="my-team")

# 定义超参数
wandb.config.learning_rate = 0.001
wandb.config.batch_size = 32
wandb.config.num_epochs = 10

# 加载MNIST数据集
train_dataset = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
test_loader= torch.utils.data.DataLoader(dataset=test_dataset, batch_size=wandb.config.batch_size, shuffle=False)

# 定义模型
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

# 训练模型
total_steps = len(train_loader)
for epoch in range(wandb.config.num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 前向传播和计算损失
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录训练指标和损失
        if (i+1) % 100 == 0:
            wandb.log({"loss": loss.item(), "epoch": epoch+1, "step": i+1})

    # 在测试集上评估模型
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        wandb.log({"accuracy": accuracy, "epoch": epoch+1})

# 结束实验
wandb.finish()