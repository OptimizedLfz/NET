import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchsummary import summary
import numpy as np
import tqdm


class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()#调用nn.module的初始化函数
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.shortcut = None

        #通道不一致，短路无法连接，需要进行一次conv2d将通道数一致
        #同理stride决定了特征图的大小，特征图大小不一致无法连接
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.shortcut is not None:
            identity = self.shortcut(x)
            print(identity.shape)
        
        out += identity
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, nums=10):
        super(ResNet, self).__init__()
        #分类数
        self.nums = nums
        #手写数字识别，单张图片规格28*28*1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3),#28*28*64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#14*14*64
        )

        self.conv2 = nn.Sequential(
            ResBlock(in_channel=64, out_channel=64),
            ResBlock(in_channel=64, out_channel=64)#14*14*64
        )

        self.conv3 = nn.Sequential(
            ResBlock(in_channel=64, out_channel=128),
            ResBlock(in_channel=128, out_channel=128)#14*14*128
        )

        self.conv4 = nn.Sequential(
            ResBlock(in_channel=128, out_channel=256),
            ResBlock(in_channel=256, out_channel=256)#14*14*256
        )
        
        self.conv5 = nn.Sequential(
            ResBlock(in_channel=256, out_channel=512, stride=2),#7*7*512
            ResBlock(in_channel=512, out_channel=512)#7*7*512
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))#1*1*512

        self.fc = nn.Linear(512, self.nums)#self.nums


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)#n*512*1*1
        x = torch.flatten(x,1)#n*512
        x = self.fc(x)#n*self.nums
        return x

#将label转换为0-1矩阵，用于计算loss
def labels_to_one_hot(labels, num_classes):
    # 使用 PyTorch 的 zeros 创建一个全零矩阵
    one_hot_matrix = torch.zeros((len(labels), num_classes), dtype=torch.float32)
    
    # 将每个标签对应的位置设为 1
    one_hot_matrix[torch.arange(len(labels)), labels] = 1
    
    return one_hot_matrix

#构造数据集，为小批量学习做准备
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

#初始化数据 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.tensor(np.loadtxt(open("train.csv","rb"),delimiter=",",skiprows=1), dtype=torch.float32)
test_data = torch.tensor(np.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=1), dtype=torch.float32)
labels = data[:,0].reshape(42000).long().to(device)
test_labels = test_data[:,0].reshape(42000).long().to(device)
images = data[:,1:785].reshape(42000,1,28,28).to(device)
test_images = test_data[:,1:785].reshape(42000,1,28,28).to(device)
dataset = CustomDataset(images, labels)
test_dataset = CustomDataset(test_images, test_labels)
#训练次数，学习率
num_epochs = 10
lr=0.001
#初始化网络
net = ResNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr)
loss_fn = nn.CrossEntropyLoss()
#抽取小批量
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle = False)
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    num_batches = 0
    net.train()#设置模型为训练模式
    for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
        optimizer.zero_grad()
        y_hat = net(X)
        l = loss_fn(y_hat, y)
        l.backward()
        optimizer.step()
        train_loss += l.item()
        train_acc += (y_hat.argmax(dim=1) == y).sum().item()
        num_batches += 1 

    train_loss /= num_batches
    train_acc /= len(train_loader.dataset)

    test_loss = 0.0
    test_acc = 0.0
    num_batches = 0
    net.eval()
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            y_hat = net(X)
            l = loss_fn(y_hat, y)

            test_loss += l.item()
            test_acc += (y_hat.argmax(dim=1) == y).sum().item()
            num_batches += 1
        
        test_loss /= num_batches
        test_acc /= len(test_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"train_loss={train_loss:.3f}, train_acc={train_acc:.3f},"
              f"test_loss={test_loss:.3f}, test_acc={test_acc:.3f}")