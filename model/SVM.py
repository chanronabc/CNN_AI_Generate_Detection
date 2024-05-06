import torch
import torch.nn as nn
import torch.optim as optim

class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        # 图像大小为 256x256，展平后是 65536
        self.fc = nn.Linear(256*256 * 3, 2)  # 假设是二分类任务

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平图像
        x = self.fc(x)
        return x