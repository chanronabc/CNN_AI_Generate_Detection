import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一层卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二层卷积层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 第三层卷积层
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # dropout层
        self.dropout = nn.Dropout(p=0.5)
        # 全连接层
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)  # 二分类

    def forward(self, x):
        # 通过卷积层和池化层
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # 展平
        x = x.view(-1, 64 * 32 * 32)
        # 通过全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    