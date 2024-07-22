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
from pathlib import Path
import pandas as pd
from torchvision.io import image


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
    train_time_list = list()
    # 总训练过的数据数量
    all_train_num_count = 0
    # 初始化参数：训练集损失，训练集准确率，测试集准确率
    train_l = train_acc = test_acc = None
    # 训练损失之和，训练准确率之和，样本数
    train_loss = train_hat_true_count = train_num_count = 0
    
    num_batches = len(train_iter)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    print(f'training on: {devices}, [{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]')
    for epoch in range(num_epochs):
        epoch_time_0 = time.time()
        # 4个维度：训练损失之和，训练准确数之和，样本数，标签特点数
        train_loss = 0
        train_hat_true_count = 0
        train_num_count = 0
        labels_num_element_count = 0
        # 训练模式
        net.train()
        for i, (X, y) in enumerate(train_iter):
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
                train_num_count += X.shape[0]
                train_hat_true_count += accuracy(y_hat, y)
                labels_num_element_count += y.numel()
            
        train_l = train_loss / train_num_count
        train_acc = train_hat_true_count / labels_num_element_count
            
        train_time_list.append(time.time() - epoch_time_0)    # 训练阶段耗时记录
        
        all_train_num_count += train_num_count
        test_acc = evaluate_accuracy_gpu(net, test_iter) if test_iter is not None else 0
        
        epoch_time = time.time() - epoch_time_0    # 当前epoch耗时
        print(f'epoch: {epoch+1}/{num_epochs}, loss: {train_l:.3f}, '
              f'train_acc: {train_acc:.3f}, test_acc: {test_acc:.3f}, '
              f'epoch_time: [{time_s2dhms(epoch_time)}]')
        
    print(f'*** training speed: {all_train_num_count / sum(train_time_list):.1f} examples/sec '
          f'on {str(devices)}, all_time: [{time_s2dhms(time.time() - time_0)}] ***')


    
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


def box_corner_to_center(boxes):
    """从（左上x，左上y，右下x，右下y）转换到（中间x，中间y，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=1)
    return boxes


def box_center_to_corner(boxes):
    """从（中间x，中间y，宽度，高度）转换到（左上x，左上y，右下x，右下y）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=1)
    return boxes


def bbox_to_rect(bbox, color):
    # 对角边框（左上x，左上y，右下x，右下y）
    # 绘制bbox
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)


def multibox_prior(data, area_sizes, w_h_ratios, w_h_ratios_normalization=False):
    """
        生成以每个像素为中心具有不同形状的锚框
        data图片数据，area_sizes锚框占图片面积率，w_h_ratios锚框宽高比
    """
    img_h, img_w = data.shape[-2:]
    device, num_area_sizes, num_w_h_ratios = data.device, len(area_sizes), len(w_h_ratios)
    # 每个像素点需要生成的锚框数
    num_boxes_per_pixel = num_area_sizes + num_w_h_ratios - 1
    # area_sizes，w_h_ratios 转tensor
    area_sizes_tensor = torch.tensor(area_sizes, device=device)
    w_h_ratios_tensor = torch.tensor(w_h_ratios, device=device)

    #每个像素1*1，坐标（x,y），将像素点中心定义为该像素生成锚框的中心，即锚框中心点坐标（x+0.5, y+0.5）
    offset_h, offset_w = 0.5, 0.5

    # 生成锚框的所有中心点
    center_h = torch.arange(img_h, device=device) + offset_h
    center_w = torch.arange(img_w, device=device) + offset_w
    # 生成img所有坐标点的锚框中心坐标，每个点的y值center_y_anchor，每个点的x值center_x_anchor
    center_y_anchor, center_x_anchor = torch.meshgrid(center_h, center_w)
    # flatten
    center_y_anchor, center_x_anchor = center_y_anchor.reshape(-1), center_x_anchor.reshape(-1)
    
    # 生成锚框宽高，集合area_sizes和w_h_ratios都只取第一个元素和另一个集合的所有组合
    # 生成“num_boxes_per_pixel”个锚框宽和高
    if w_h_ratios_normalization is False:
        # 1、宽高未归一化，area_size为面积比，w_h_ratio为最终锚框的宽高比
        # 由下两式联立：
        # w_h_ratio = w_anchor / h_anchor
        # area_size = (w_anchor * h_anchor) / (img_w * img_h) 
        # 推出锚框宽高计算公式：
        # w_anchor = sqrt(img_w * img_h * area_size * w_h_ratio)
        # h_anchor = sqrt(img_w * img_h * area_size / w_h_ratio)
        area_img = img_w * img_h
        w_anchor = torch.cat(
            tensors=[torch.sqrt(area_img * area_sizes_tensor * w_h_ratios_tensor[0]), 
                     torch.sqrt(area_img * area_sizes_tensor[0] * w_h_ratios_tensor[1:])],
            dim=0
        )
        h_anchor = torch.cat(
            tensors=[torch.sqrt(area_img * area_sizes_tensor / w_h_ratios_tensor[0]), 
                     torch.sqrt(area_img * area_sizes_tensor[0] / w_h_ratios_tensor[1:])],
            dim=0
        )
    else:
        # 2、宽高归一化，area_size为面积比，w_h_ratio表示的宽高比是归一化后的
        # 由下两式联立：
        # w_h_ratio = (w_anchor/img_w) / (h_anchor/img_h)
        # area_size = (w_anchor * h_anchor) / (img_w * img_h) 
        # 推出锚框宽高计算公式：
        # w_anchor = img_w * sqrt(area_size * w_h_ratio)
        # h_anchor = img_h * sqrt(area_size / w_h_ratio)
        w_anchor = torch.cat(
            tensors=[torch.sqrt(area_sizes_tensor * w_h_ratios_tensor[0]), 
                     torch.sqrt(area_sizes_tensor[0] * w_h_ratios_tensor[1:])],
            dim=0
        ) * img_w
        h_anchor = torch.cat(
            tensors=[torch.sqrt(area_sizes_tensor / w_h_ratios_tensor[0]), 
                     torch.sqrt(area_sizes_tensor[0] / w_h_ratios_tensor[1:])],
            dim=0
        ) * img_h
        
    # 使用小数表示（锚框与图片宽高占比）
    w_anchor /= img_w
    h_anchor /= img_h
    center_x_anchor /= img_w
    center_y_anchor /= img_h
    # 即每个像素都要生成num_boxes_per_pixel个锚框，每个像素的锚框宽高尺寸为w_anchor，h_anchor
    # 下面要根据每个像素的具体位置为每个像素都生成出所有锚框具体位置
    
    # 相对于center的偏移(-w_anchor/2, -h_anchor/2, w_anchor/2, h_anchor/2)
    # 创建单个像素的锚框偏移张量，张量shape为 (num_boxes_per_pixel, 4)，锚框为对角坐标表示
    # 第一个维度重复img_h * img_w，第二个维度一次，得到每个像素的锚框偏移张量，
    # repeat后张量shape为 (num_boxes_per_pixel * img_h * img_w, 4)
    boxes_all_pixel_tensor = torch.stack(
        [-w_anchor, -h_anchor, w_anchor, h_anchor], 
        dim=1
    ).repeat(img_h * img_w, 1) / 2
    
    # 锚框中心坐标(center_x_anchor, center_y_anchor, center_x_anchor, center_y_anchor)
    # 每个中心点都将有“num_boxes_per_pixel”个锚框，
    # 锚框中心坐标张量shape为(img_h * img_w, 4)，所以生成含所有锚框中心的网格，
    # 需要张量内部重复，即每行元素（每个像素）重复了“num_boxes_per_pixel”次
    # repeat_interleave后张量shape为 (num_boxes_per_pixel * img_h * img_w, 4)
    boxes_center_all_pixel_tensor = torch.stack(
        [center_x_anchor, center_y_anchor, center_x_anchor, center_y_anchor], 
        dim=1
    ).repeat_interleave(num_boxes_per_pixel, dim=0)
    
    # 相加得锚框完整坐标，shape为 (num_boxes_per_pixel * img_h * img_w, 4)
    # (center_x_anchor-w_anchor/2, center_y_anchor-h_anchor/2, center_x_anchor+w_anchor/2, center_y_anchor+h_anchor/2)
    boxes_all_pixel_tensor = boxes_center_all_pixel_tensor + boxes_all_pixel_tensor
    
    # 增加一个第0个维度，shape为 (1, num_boxes_per_pixel * img_h * img_w, 4)
    return boxes_all_pixel_tensor.unsqueeze(0)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        """将对象做成list"""
        obj = obj or default_values
        return obj if isinstance(obj, (list, tuple)) else [obj]

    labels = _make_list(labels)
    colors = _make_list(colors, default_values=['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center', 
                      fontsize=9, color=text_color, bbox=dict(facecolor=color, lw=0))
            
            
def box_iou(boxes1, boxes2):
    """
        计算两组框列表中两两成对的交并比
        （boxes1, boxes2 均为对角标记坐标）
    """
    box_area = lambda boxes: (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # 该操作连续触发两次广播，第一次广播相当于矩阵1升维后复制扩充该维度【复制len(矩阵2)个】，再使矩阵2触发一次广播
    # inter_upperlefts = torch.max(boxes1[:, None, :2].repeat_interleave(len(boxes2), dim=1), boxes2[:, :2])
    # inter_lowerrights = torch.min(boxes1[:, None, 2:].repeat_interleave(len(boxes2), dim=1), boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areas和union_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU, shape为(num_anchors, num_gt_boxes)
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量(锚框到真实边界框映射表)
    anchors_bbox_map = torch.full(size=(num_anchors,), fill_value=-1, dtype=torch.long, device=device)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)    # iou满足阈值的锚框索引
    box_j = indices[max_ious >= iou_threshold]    # iou满足阈值的类别索引
    anchors_bbox_map[anc_i] = box_j
    # 此刻已将满足阈值的锚框标记好真实类别
    
    # 极端情况下，有可能大多数锚框均对某一真实类iou最大，此刻可以适当压制这一类，使其余真实类可以最少被一个锚框匹配出
    # 标记为“无效”行列的填充为-1
    col_discard = torch.full(size=(num_anchors,), fill_value=-1)
    row_discard = torch.full(size=(num_gt_boxes,), fill_value=-1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)    # 不写维度dim，返回flattened索引
        box_idx = (max_idx % num_gt_boxes).long()    # 类别索引
        anc_idx = (max_idx / num_gt_boxes).long()    # 锚框索引
        anchors_bbox_map[anc_idx] = box_idx    # 此真实类别找到iou表现最好的锚框，标记此锚框的类别
        # 将该行列值置为-1，即压制该类别在其余锚框的iou表现，使其余真实类可以最少被一个锚框匹配出
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """
        计算锚框与真实边框的偏移量
        anchors, assigned_bb 均为对角表示坐标
    """
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], dim=1).to(torch.float16)
    return offset


def multibox_target(anchors, labels):
    """
        使用真实边界框标记锚框
        标记锚框的类别和偏移量。 将背景类别的索引设置为零，然后将新类别的整数索引递增1
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(dim=0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        # 每一个label元素： (类别，左上x，左上y，右下x，右下y)
        label = labels[i, :, :]
        # 给每一个锚框分配类别
        anchors_bbox_map = assign_anchor_to_bbox(ground_truth=label[:, 1:], anchors=anchors, device=device)
        # bbox_mask的shape为 (num_anchors, 4)，有匹配类别值是1，无匹配类别值是0
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        # 将真实类标签和分配的边界框坐标初始化为零（包含背景类别）
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        has_class_anchors_ids = torch.nonzero(anchors_bbox_map >= 0)    # 有正类别的锚框索引
        class_ids = anchors_bbox_map[has_class_anchors_ids]    # 有正类别的锚框对应的类别索引
         # 将有正类别的锚框的类别索引+1，其余为默认值背景负类0
        class_labels[has_class_anchors_ids] = label[class_ids, 0].long() + 1   
        # 将有正类别的锚框的类别所在边框坐标收集在assigned_bb中
        assigned_bb[has_class_anchors_ids] = label[class_ids, 1:]    
        # 计算偏移量
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask    # 仅计算有正类标记的锚框与label的offset
        
        # 收集当前批的偏移量、锚框的正类蒙板、正类
        # offset和bbox_mask的shape都是(num_anchors, 4); class_labels的shape是(num_anchors,)
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))   
        batch_class_labels.append(class_labels)
        
    # bbox_offset和bbox_mask的shape都是(batch_size, num_anchors * 4)
    # class_labels的shape是(batch_size, num_anchors)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] / 10 * anc[:, 2:]) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序，非极大值抑制置信度高且iou小的框"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的索引
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: 
            break
        # iou的shape为(len(B[1:]),)
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        # 将iou低于阈值的 B[1:]索引留存下来，继续下一回合
        ids = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[ids + 1]    # ids是从当前B的第二个元素开始，所以索引+1
    return torch.tensor(keep, device=boxes.device)


def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        # 第i批数据
        # cls_prob的shape为(num_cls, num_anchors), offset_pred的shape为(num_anchors, 4)
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        # 每个锚框的最大正类置信度和类别索引（索引从第一个正类开始，不包含负类背景0）
        conf, class_id = torch.max(cls_prob[1:], dim=0)    # conf和class_id的shape都为(num_anchors,)，返回max的value和index
        predicted_bb = offset_inverse(anchors, offset_pred)    # 根据锚框和偏移量转换为预测边界框（新锚框）
        keep = nms(predicted_bb, conf, nms_threshold)    # 非极大值抑制后，保留下来的边界框索引，是锚框/新锚框的索引向量

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))    # 保留的锚框索引和所有的锚框索引合并
        uniques, counts = combined.unique(return_counts=True)    # 返回不重复的值（锚框索引值）的向量，和出现次数的向量
        non_keep = uniques[counts == 1]    # 只出现一次的值（锚框索引值）的向量
        all_id_sorted = torch.cat((keep, non_keep))    # 保留下来的锚框索引和认为是背景的锚框索引拼接，得到有序的锚框全索引
        class_id[non_keep] = -1    # 将包含nms没通过的不保留的类别预测均重置为背景类别
        class_id = class_id[all_id_sorted]    # 按照all_id_sorted索引顺序重置类别顺序
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]    # 重置置信度和新锚框顺序
        # pos_threshold是一个用于非背景预测的阈值
        below_pos_threshold_idx = (conf < pos_threshold)
        class_id[below_pos_threshold_idx] = -1    # 将pos_threshold置信度不到阈值的预测都认定为背景类别
        conf[below_pos_threshold_idx] = 1 - conf[below_pos_threshold_idx]    # 将低于阈值的正类置信度都重置为负类背景的置信度（1-conf）
        # 拼接 类别、置信度、新锚框 ，pred_info的shape为(num_anchors, 1+1+4)
        pred_info = torch.cat((class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
    # 多批数据合并
    return torch.stack(out)


def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir_name = 'banana-detection'
    data_path = Path(rf'../data/{data_dir_name}')
    
    train_or_valid_dir = data_path / 'bananas_train' if is_train else data_path / 'bananas_val'
    csv_fname = train_or_valid_dir / 'label.csv'
    csv_data = pd.read_csv(csv_fname.as_posix())
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        _img = image.read_image((train_or_valid_dir / 'images' / img_name).as_posix())
        images.append(_img)
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256    # 将坐标缩放到[0, 1]，类别为0，无影响


class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print(f'read {str(len(self.features))} {"training" if is_train else "validation"} examples')

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
    
    
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True), batch_size, shuffle=True)
    valid_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False), batch_size)
    return train_iter, valid_iter


def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_dir = voc_dir / 'ImageSets' / 'Segmentation'
    txt_fpath = txt_dir / 'train.txt' if is_train else txt_dir / 'val.txt'
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fpath.as_posix(), 'r') as f:
        image_name_list = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(image_name_list):
        features.append(torchvision.io.read_image((voc_dir / 'JPEGImages' / f'{fname}.jpg').as_posix()))
        labels.append(torchvision.io.read_image((voc_dir / 'SegmentationClass' / f'{fname}.png').as_posix(), mode))
    return features, labels

# 常量，RGB颜色值和类名
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def voc_colormap2label():
    """
        构建从RGB到VOC类别索引的映射
        即：可以给定一个rgb颜色索引即可对应出类别
    """
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)    # rgb三通道空间大小为256 * 256 *256，将rgb以256进制计算为一个数字
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """
        将VOC标签中的RGB值映射到它们的类别索引
    """
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = (colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2]    # 通过不同权重叠加整合，三通道合并为一通道
    return colormap2label[idx]

def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label


class VOCSegDataset(torch.utils.data.Dataset):
    """
        一个用于加载VOC数据集的自定义数据集
    """
    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        self.colormap2label = voc_colormap2label()
        txt_dir = voc_dir / 'ImageSets' / 'Segmentation'
        txt_fpath = txt_dir / 'train.txt' if is_train else txt_dir / 'val.txt'
        self.mode = torchvision.io.image.ImageReadMode.RGB
        with open(txt_fpath.as_posix(), 'r') as f:
            image_name_list = f.read().split()
        self._img_path_list = [voc_dir / 'JPEGImages' / f'{fname}.jpg' for fname in image_name_list]
        self._label_path_list = [voc_dir / 'SegmentationClass' / f'{fname}.png' for fname in image_name_list]
        
        self.img_path_list = list()
        self.label_path_list = list()
        for img_path in self._img_path_list:
            img = torchvision.io.read_image(img_path.as_posix())
            if self.crop_check(img):
                self.img_path_list.append(img_path)
        for label_path in self._label_path_list:
            label = torchvision.io.read_image(label_path.as_posix(), self.mode)
            if self.crop_check(label):
                self.label_path_list.append(label_path) 
                
        self._img_path_list = self._label_path_list = None
        print('read ' + str(len(self.img_path_list)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def crop_check(self, img):
        return img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1]
    
    def filter(self, imgs):
        return [img for img in imgs if (img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        _img = self.normalize_image(torchvision.io.read_image(self.img_path_list[idx].as_posix()))
        _label = torchvision.io.read_image(self.label_path_list[idx].as_posix(), self.mode)
        img, label = voc_rand_crop(_img, _label, *self.crop_size)
        return (img, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.img_path_list)
    
    
def load_data_voc(batch_size, crop_size, voc_dir):
    """加载VOC语义分割数据集"""
    num_workers = get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(VOCSegDataset(True, crop_size, voc_dir), batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(VOCSegDataset(False, crop_size, voc_dir), batch_size, drop_last=True, num_workers=num_workers)
    return train_iter, test_iter