import torch
import torch.utils.data as Data
import torchvision
import torch.nn as nn
from torch.functional import F
from PIL import Image
from torch.autograd import Variable
import sys
import os
import numpy
from torchvision.transforms import ToPILImage

captcha_len = 4
alphabet = []
for i in range(48, 59):
    alphabet.append(chr(i))
for i in range(97, 123):
    alphabet.append(chr(i))


def char_to_index(c):
    """
    对带有数字和字母的字符使用独热码 先保存字母加数字到list中，后续取list中的下标
    例: 1.验证码 3Qcq 转全小写 3qcq
        2.对0-9数字以及a-z字母保存在list [0,1,2,3,..9,a,b,c...q..z]
        3.获取在list中下标 [3, 26, 12, 26] 对应的就是3qcq
        4.生成一个 36(数字+字母总和) * 4(验证码大小)的全是0 的列表
        5.验证码每一个字符分到 36倍数中 并标记此处值为1
        6.[0,0,0,1,0,...1....]
        7.order 记录下标 之后恢复原值用
    对于分类变量，建模时要进行转换，通常直接转换为数字;
    转换后的值会影响同一特征在样本中的权重。比如转换为1000和转换为1对模型影响明显不同。
    one-hot编码的定义是用N位状态寄存器来对N个状态进行编码。
    比如上面的例子 3qcq -> [3, 26, 12, 26]，有4个分类值，因此N为4;
    对应的one-hot编码可以表示为100,100110,10010, 100110;
    此处我选择用4个36为长度的list 来填入对应下标的值
    :param c:
    :return:
    """
    targets = [0] * (len(alphabet) * captcha_len)
    order = []
    c_list = list(c.lower())
    for ind, i in enumerate(c_list):
        index = ind * len(alphabet) + alphabet.index(i)
        targets[index] = 1
        order.append(index)
    return targets, order


def index_to_str(index):
    s = []
    for i in index:
        if i:
            char_index = i % len(alphabet)
            s.append(alphabet[char_index])
        else:
            s.append(i)
    return s


class MyDataSet(Data.Dataset):

    def __init__(self, root, file_paths, transform=None):
        super(MyDataSet, self).__init__()
        with open(os.path.join(root, file_paths)) as f:
            label_path_list = f.read().split("\n")
        self.data = []
        self.targets = []
        self.orders = []
        self.transform = transform
        for path_label in label_path_list:
            if not path_label:
                continue
            p, l = path_label.split(" ")
            img = MyDataSet.get_img(p)
            self.data.append(img)
            # 独热码
            targets, orders = char_to_index(l)
            self.targets.append(targets)
            self.orders.append(orders)
        self.data = torch.Tensor(self.data)
        self.targets = torch.Tensor(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, targets, orders = self.data[index], self.targets[index], self.orders[index]
        img = Image.fromarray(img.numpy())
        if self.transform:
            img = self.transform(img)
        return img, targets, orders

    @staticmethod
    def get_img(path):
        img = Image.open(path).convert("L")
        return numpy.array(img)


data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_data = MyDataSet("codeDemo", "train.txt", data_transform)
train_load = Data.DataLoader(dataset=train_data, batch_size=20, shuffle=True)
test_data = MyDataSet("codeDemo", "test.txt", data_transform)
test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)
# 测试500张验证码
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:500]/255.
test_y = test_data.targets[:500]


# 创建神经网络
class CNN(nn.Module):

    # 三层卷积 两层全连接
    def __init__(self):
        super(CNN, self).__init__()
        # 图片大小为 100 * 60 * 1 单通道
        self.conv1 = nn.Sequential(
            # 100 60
            nn.Conv2d(1, 16, 3, padding=1),
            # 防止梯度消失或梯度爆炸
            nn.BatchNorm2d(16),
            # 防止过拟合 每次随机去除20%的神经元
            nn.Dropout(0.2),
            nn.ReLU(),
            # 50,30
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            # 30, 30
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            # 防止过拟合 每次随机去除30%的点
            nn.Dropout(0.4),
            nn.ReLU(),
            # 25, 15
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            # 25, 15
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            # 防止过拟合 每次随机去除30%的点
            nn.Dropout(0.4),
            nn.ReLU(),
            # 12, 7
            nn.MaxPool2d(2)
        )

        # 此处除8 可以当成是通过 2*2*2 算出，因为卷积核和padding计算完的尺寸会相互抵消
        self.fc1 = nn.Linear(64 * (100 // 8) * (60 // 8), 1024)
        # 输出层 输出数量因设为想要的长度 此处是因为把验证码进行独热编码
        self.fc2 = nn.Linear(1024, len(alphabet) * captcha_len)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out


cnn = CNN()
loss_func = nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)


def train():
    # 开始训练
    for i in range(5):
        for ind, (x, y, order) in enumerate(train_load):
            x, y = Variable(x), Variable(y)
            out_x = cnn(x)
            loss = loss_func(out_x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ind % 50 == 0:
                print(f"Epoch: {i} | train los: {loss.data.numpy()}")

    torch.save(cnn.state_dict(), "yzm.pth")


def ts():
    cnn.load_state_dict(torch.load("yzm.pth"))
    # success = []
    # for ind, (x, y, orders) in enumerate(test_loader):
    #     test_output = cnn(x)
    #     # ToPILImage()(x[0]).show()
    #     captcha = index_to_str(numpy.array(orders))
    #     train_captcha = []
    #     for i in range(captcha_len):
    #         train_captcha.append(torch.argmax(test_output[0, i * len(alphabet): (i + 1) * len(alphabet)], 0))
    #     train_captcha_str = index_to_str(numpy.array(train_captcha))
    #     try:
    #         if captcha == train_captcha_str:
    #             success.append([captcha, train_captcha_str])
    #         print(ind, "原值:", "".join(captcha), "预测值:", "".join(train_captcha_str), "命中:", len(success))
    #     except:
    #         pass
    # print("命中率:", len(success) / len(test_loader))

    img = Image.open("codeDemo/test/0ai4.png")
    img = img.convert("L")
    # 图片转tensor 并对行添加维度
    x = data_transform(img).unsqueeze(1)
    test_output = cnn(x)
    train_captcha = []
    # 遍历验证码的长度分批次获取 (0-36) (36-36*2) (36*2 - 36*3) (36*3 - 36*4) 的下标位
    for i in range(captcha_len):
        # argmax 获取以dim定义的列最大值下标或行最大值下标 此处取列最大值下标
        train_captcha.append(torch.argmax(test_output[0, i * len(alphabet): (i + 1) * len(alphabet)], 0))
        train_captcha_str = index_to_str(numpy.array(train_captcha))
    print("验证码:", train_captcha_str)
    # img.show()

ts()
print("结束")




