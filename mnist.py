import numpy

import torchvision
import torch
from PIL import Image
from torch.functional import F
import torch.utils.data as Data
import torch.nn as nn
from torchvision.transforms import ToPILImage


# 获取手写数字训练集
from util import to_string

train_data = torchvision.datasets.MNIST(
    # 目录
    root="./mnist",
    # 值为True表示是训练集
    train=True,
    # 图片转换成tensor把一个取值范围时[0,255]的PIL.Image对象转变为[0,1.0]的tensor对象
    transform=torchvision.transforms.ToTensor(),
    # 是否在线下载，如果本地root指定目录不存在情况下download设置为True，反之通过root目录去获取数据集
    download=False
)

# 获取测试集
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
# 加载训练集 batch_size=一批次训练多少， shuffle 是否在下次训练时打乱顺序，False表示不打乱， num_workers 线程数
train_loader = Data.DataLoader(dataset=train_data, batch_size=50, shuffle=True)
# 只选择前2000条进行训练 数据集数据是[0,255]的所以做除法保证数据在[0,1]之间
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets[:2000]


# 创建模型
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 手写图尺寸为 28*28*1 单通道
        self.conv1 = nn.Sequential(
            # 输出结果 [16, 24, 24]
            nn.Conv2d(1, 16, 5),
            # 为了解决在用到激励函数后，梯度丢失或梯度爆炸问题，可以做批标准化的操作
            nn.BatchNorm2d(16),
            # 在训练时会出现过拟合现象，就是当神经元(输出通道)比要训练的数据大的时候
            # 计算机为了更好的包容到每一个点而导致过拟合现象的发生
            # 这时可以利用dropout来进行每一次屏蔽部分神经元的操作，0.2表示20%
            # 随机删除20%的神经元
            nn.Dropout(0.2),
            nn.ReLU(),
            # 最大池化层 kernel_size为2 输出结果 [16, 12, 12]
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            # 输出结果 [32, 8, 8]
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 输出结果 [32, 4, 4]
            nn.MaxPool2d(2)
        )
        # 展平操作
        self.out = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)   # 结果(batch, 32, 4, 4)
        # 展平为二维
        # x.size(0)是获取x中的batch, -1 表示的是 32 * 4 * 4的值
        # 最终成为 (batch, 32 * 4 * 4)
        x = x.view(x.size(0), -1)
        # 通过shape可以看到结果是[50, 512]
        # 32 * 4 * 4 的结果也是512 所以在计算时一定要计算对，否则会报错
        # print(x.shape)
        out = self.out(x)
        return out


cnn = CNN()
# 利用Adam优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.02)
loss_func = nn.CrossEntropyLoss()

# 训练两次
for e in range(2):
    for step, (x, y) in enumerate(train_loader):
        # CPU 计算
        output = cnn(x)
        # GPU计算
        # output = cnn(x).cuda()
        # 计算误差
        loss = loss_func(output, y)
        # 由于每次循环会记录上此次的值，所以需要清零
        optimizer.zero_grad()
        # 反向传递
        loss.backward()
        optimizer.step()
        # 每50次查看一下训练进度
        if step % 50 == 0:
            # 预测模式
            cnn.eval()
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print(f"Epoch: {e} | train los: {loss.data.numpy()}, | test accuracy: {accuracy}")
            # 训练模式
            cnn.train()

# 训练完的数据保存下来
# torch.save(cnn, 'net.pkl')  # 保存整个网络
# 只保存网络中的参数 (速度快, 占内存少)
# torch.save(cnn.state_dict(), "sxsz.pkl")

# 读取训练完成的数据
# 这种方式将会提取整个神经网络, 网络大的时候可能会比较慢.
# cnn = torch.load('sxsz.pkl')
# 这种方式将会提取所有的参数, 然后再放到你的新建网络中.
cnn.load_state_dict(torch.load("sxsz.pkl"))
#
img = Image.open("img/sx.png")
data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
x = torch.unsqueeze(data_transforms(img), 1)/255.
test_output = cnn(x)
img.show()
# img = ToPILImage()(test_x[0])
# img.show()
# test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
to_string("手写图片训练", pred_y=pred_y, test_y=test_y[:10].numpy())
# img.show()
