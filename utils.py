import math
import numpy as np
import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
from d2l import torch as d2l
from IPython import display
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        # d2l.plt.pause(0.1)
        
        
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def synthetic_data(w, b, num_examples):
    """
        生成数据，y = Xw + b + 噪声
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    # y = torch.matmul(X, w) + b
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
    

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def accuracy(y_hat, y):
    """准确度"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数，预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）

    Defined in :numref:`sec_softmax_scratch`"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（定义见第3章）

    Defined in :numref:`sec_softmax_scratch`"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.1, 1.0],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        # print(train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    # print(train_loss, train_acc)
    # assert train_loss < 0.5, train_loss
    # assert train_acc <= 1 and train_acc > 0.7, train_acc
    # assert test_acc <= 1 and test_acc > 0.7, test_acc


def sgd(params, lr, batch_size):
    """
        小批量随机梯度下降
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            

def linreg(X, w, b):
    """
        线性回归模型
    """
    return torch.matmul(X, w) + b
            

def squared_loss(y_hat, y):
    """
        均方误差
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
            
            
def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def get_dataloader_workers():
    """
        使用n个进程来读取数据
    """
    return 0


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中

    Defined in :numref:`sec_fashion_mnist`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]

    Defined in :numref:`sec_use_gpu`"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def time_s2dhms(s):
    """时间秒转日时分秒"""
    s = int(s)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return f'{d}:{h:02d}:{m:02d}:{s:02d}'

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(net.parameters()).device
    # 正确预测的数量，总预测的数量
    y_hat_true_count = 0
    y_hat_num_count = 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            
            y_hat_true_count += accuracy(net(X), y)
            y_hat_num_count += y.numel()
    return y_hat_true_count / y_hat_num_count


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    time_0 = time.time()
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print(f'training on: [{device}], [{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]')
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    
    # 时间用时列表
    time_list = list()
    # 总训练过的数据数量
    all_train_num_count = 0
    # 初始化参数：训练集损失，训练集准确率，测试集准确率
    train_l = train_acc = test_acc = None
    # 训练损失之和，训练准确率之和，样本数
    train_loss = train_hat_true_count = train_num_count = 0
    
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确数之和，样本数
        train_loss = 0
        train_hat_true_count = 0
        train_num_count = 0
        # 训练模式
        net.train()
        for i, (X, y) in enumerate(train_iter):
            start_time = time.time()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss += l * X.shape[0]
                train_hat_true_count += accuracy(y_hat, y)
                train_num_count += X.shape[0]

            time_list.append(time.time() - start_time)
            
            train_l = train_loss / train_num_count
            train_acc = train_hat_true_count / train_num_count
            
        all_train_num_count += train_num_count
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        
        print(f'epoch: {epoch+1}/{num_epochs}, loss {train_l:.3f}, '
              f'train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    
    print(f'*** {all_train_num_count / sum(time_list):.1f} examples/sec '
          f'on {str(device)} - [{time_s2dhms(sum(time_list))}], '
          f'all: [{time_s2dhms(time.time() - time_0)}] ***')
    
    
def train_gpus(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=try_all_gpus()):
    """用多GPU进行模型训练"""
    time_0 = time.time()
    # 时间用时列表
    time_list = list()
    # 总训练过的数据数量
    all_train_num_count = 0
    # 初始化参数：训练集损失，训练集准确率，测试集准确率
    train_l = train_acc = test_acc = None
    # 训练损失之和，训练准确率之和，样本数
    train_loss = train_hat_true_count = train_num_count = 0
    
    num_batches = len(train_iter)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # 4个维度：训练损失之和，训练准确数之和，样本数，标签特点数
        train_loss = 0
        train_hat_true_count = 0
        train_num_count = 0
        labels_num_element_count = 0
        # 训练模式
        net.train()
        for i, (X, y) in enumerate(train_iter):
            start_time = time.time()
            
            if isinstance(X, list):
                # 微调BERT中所需
                X = [x.to(devices[0]) for x in X]
            else:
                X = X.to(devices[0])
            y = y.to(devices[0])
            
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.sum().backward()
            trainer.step()
            with torch.no_grad():
                train_loss += l.sum()
                train_hat_true_count += accuracy(y_hat, y)
                train_num_count += X.shape[0]
            
            time_list.append(time.time() - start_time)
            
            train_l = train_loss / train_num_count
            train_acc = train_hat_true_count / train_num_count
            
        all_train_num_count += train_num_count
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        
        print(f'epoch: {epoch+1}/{num_epochs}, loss {train_l:.3f}, '
              f'train acc {train_acc:.3f}, test acc {test_acc:.3f}')
        
    print(f'*** {all_train_num_count / sum(time_list):.1f} examples/sec '
          f'on {str(devices)} - [{time_s2dhms(sum(time_list))}], '
          f'all: [{time_s2dhms(time.time() - time_0)}] ***')


    
class Residual(nn.Module):
    """
        残差模块
    """
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = lambda x: x
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        """
            主路与短路保持同样的shape
        """
        # 主路
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        # 短路
        X = self.conv3(X)
        # 主路与短路复合
        Y += X
        return F.relu(Y)
    
    
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """
        绘制图像列表
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def resnet18(num_classes, in_channels=1):
    """
        稍加修改的ResNet-18模型
    """
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 该模型使用了更小的卷积核、步长和填充，而且删除了最大汇聚层
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
    return net