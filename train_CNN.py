from model.CNN import CNN
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

def load_train():
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 将图像大小调整为256x256
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 创建数据集
    # dataset = datasets.ImageFolder(root='girls-new\\train', transform=transform)

    # 创建 DataLoader
    train_dataset = datasets.ImageFolder(root='DiffusionForensics\\images\\train\\lsun_bedroom', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    val_dataset = datasets.ImageFolder(root='DiffusionForensics\\images\\val\\lsun_bedroom', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    return train_loader, val_loader

if __name__ == "__main__":

    cnn = CNN()
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss() # 二元交叉熵损失
    optimizer = optim.Adam(cnn.parameters(), lr=0.0005) # Adam优化器

    # 模拟一次训练步骤
    
    train_loader, val_loader = load_train()
    # loop = tqdm(train_loader, leave=True)
    accuracy_list = []
    for epoch in range(50):
        cnn.train()
        running_loss = 0.0

        for images, labels in train_loader:
            # 清空梯度
            labels = F.one_hot(labels, num_classes=2).float()
            optimizer.zero_grad()
            
            # 前向传播
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        cnn.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                # labels = F.one_hot(labels, num_classes=2).float()
                outputs = cnn(images)
                _, predicted = torch.max(outputs.data, 1)
                # print(predicted, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Epoch {epoch+1}, Accuracy on the validation set: {100 * correct / total}%')
        accuracy = correct / total
        accuracy_list.append(accuracy)
        # loop.set_description(f'Epoch [{epoch+1}/{50}]')
        # loop.set_postfix(loss=running_loss / len(train_loader), accuracy=f'{accuracy:.2f}%')
    torch.save(cnn.state_dict(), 'cnn_bedroom.pth')