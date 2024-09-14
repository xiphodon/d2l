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
import collections
import re
import random


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
                # train_num_count += X.shape[0]
                train_num_count += len(X)
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


santi_txt_path = Path(r'../data/ebook/三体1.txt')
sanguo_txt_path = Path(r'../data/ebook/ThreeKingdoms.txt')


def read_ebook_txt(txt_path: Path):
    """
        将txt电子书加载到文本行类别中
    """
    with open(txt_path.as_posix(), 'r', encoding='utf8') as fp:
        lines = fp.readlines()
        
    new_lines = list()
    for line in lines:
        new_line = re.sub('^\S+', ' ', line).strip().lower()
        if len(new_line) > 0:
            new_lines.append(new_line)
    return new_lines


def tokenize(lines, token='char'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)
        
        
class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序降序排列
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def load_corpus_txt(txt_path: Path, max_tokens=-1):
    """返回数据集的词元索引列表和词表"""
    lines = read_ebook_txt(txt_path)
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
        
        
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
        
        
class SeqDataLoader:  
    """加载序列数据的迭代器"""
    def __init__(self, txt_Path: Path, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_txt(txt_Path, max_tokens=max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
    

def load_data_txt(txt_path: Path, batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """返回数据集的迭代器和词表"""
    data_iter = SeqDataLoader(txt_path, batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device, get_params_fn, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params_fn(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
        self.device = device

    def __call__(self, X, state):
        X = X.to(self.device)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, self.device)
    
    
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        # 只更新state（隐状态），不记录输出
        _, state = net(get_input(), state)
        outputs.append(vocab[y])    # 预热时，output只收集实际值索引
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))    # 推理时，output收集预测值索引
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    time_0 = time.time()
    state = None
    # 训练损失之和,词元数量
    loss_sum = 0
    token_sum = 0
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        loss_sum += l * y.numel()
        token_sum += y.numel()
    return math.exp(loss_sum / token_sum), token_sum / (time.time() - time_0)


def train_ch8(net, train_iter, vocab, lr, num_epochs, device, predict_prefix, use_random_iter=False, predict_len=200):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, predict_len, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 50 == 0:
            print(predict(predict_prefix[0]))
            print(f'epoch: {epoch+1}/{num_epochs}, ppl: {ppl}')
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict(predict_prefix[0]))
    print(predict(predict_prefix[1]))
    
    
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device))
        

def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_path = Path(r'../data/fra-eng/fra.txt')
    with open(data_path.as_posix(), 'r', encoding='utf-8') as f:
        return f.read()
    

def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '
    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    plt.hist([[len(l) for l in xlist], [len(l) for l in ylist]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.show()


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]    # 将每一行转成词表索引
    lines = [l + [vocab['<eos>']] for l in lines]    # 将每一行的词表索引后加入<eos>索引
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])    # 将每一行词表索引截断或填充
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)    # 计算维度1上即每一行词表索引的有效长度
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
        
        
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
        

class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
    
    
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds', shrink=0.6):
    """
        显示四维矩阵热力图
    """
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=shrink);
    
    
# class Seq2SeqEncoder(Encoder):
#     """用于序列到序列学习的循环神经网络编码器"""
#     def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
#         super(Seq2SeqEncoder, self).__init__(**kwargs)
#         # 嵌入层
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

#     def forward(self, X, *args):
#         # 输出'X'的形状：(batch_size,num_steps,embed_size)
#         X = self.embedding(X)
#         # 在循环神经网络模型中，第一个轴对应于时间步
#         X = X.permute(1, 0, 2)
#         # 如果未提及状态，则默认为0
#         output, state = self.rnn(X)
#         # output的形状:(num_steps,batch_size,num_hiddens)
#         # state的形状:(num_layers,batch_size,num_hiddens)
#         return output, state
    

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项, 用value值填充"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()

    for epoch in range(num_epochs):
        time_0 = time.time()
        # 训练损失总和，词元数量
        loss_sum = 0 
        token_sum = 0
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            batch_bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([batch_bos, Y[:, :-1]], 1)  # 强制教学，将bos加到这批Y的首位，作为解码器输入
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                loss_sum += l.sum()
                token_sum += num_tokens
        if (epoch + 1) % 50 == 0:
            print(f'epoch: {epoch + 1}/{num_epochs}, loss: {loss_sum / token_sum:.3f}')
    print(f'loss {loss_sum / token_sum:.3f}, {token_sum / (time.time() - time_0):.1f} tokens/sec on {str(device)}')
    
    
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k): 
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作, 带蒙版的softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
    
    
    
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # print(f'[params] queries shape: {queries.shape}, keys shape: {keys.shape}, values shape: {values.shape}')
        # forward其中查询、键和值的形状为（批量大小，步数或词元序列长度，特征大小）
        queries, keys = self.W_q(queries), self.W_k(keys)
        # print(f'[params * W] queries shape: {queries.shape}, keys shape: {keys.shape}, values shape: {values.shape}')
        # queries形状：(batch_size，queries步数或词元序列长度_查询的个数，num_hidden)
        # keys：(batch_size，keys步数或词元序列长度_“键－值”对的个数，num_hidden)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和后，形状：(batch_size，查询的个数，“键－值”对的个数，num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # print(f'[q k dim cat] features shape: {features.shape}')
        # tanh 激活放缩数值，不影响形状
        features = torch.tanh(features)
        # print(f'[tanh] features shape: {features.shape}')
        # scores操作逻辑理解意义，scores操作前，batch_size单独看，每批是 (查询的个数, “键-值”对的个数, num_hiddens), 
        # 即每个 查询和键 由num_hiddens个元素表示，w_v形状为(num_hiddens, 1), 
        # scores操作 features * w_v 相当于将每个 查询和键 由num_hiddens个元素变为1个元素表示，
        # 所以scores后的形状：(batch_size，查询的个数，“键-值”对的个数, 1)，self.w_v后仅有一个输出，因此从形状中移除最后那个维度。
        # scores形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        # print(f'[features * W] scores shape: {scores.shape}')
        # masked_softmax 只改数值，将有效长度外的权重降到0，不影响形状
        self.attention_weights = masked_softmax(scores, valid_lens)
        # print(f'[masked_softmax scores] attention_weights shape: {self.attention_weights.shape}')
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        # 输出形状（批量大小，查询的步数，值的维度）
        result = torch.bmm(self.dropout(self.attention_weights), values)
        # print(f'[bmm attention_weights values] result shape: {result.shape}')
        # print('--------')
        return result
    
    
class DotProductAttention(nn.Module):
    """
        缩放点积注意力
        使用点积可以得到计算效率更高的评分函数， 但是点积操作要求查询和键具有相同的长度
    """
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        # scores 形状：(batch_size，查询的个数，“键－值”对的个数)
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
    
    
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状: (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens的形状: (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values的形状: (batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
    
    
class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
    
    
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        # 输入X的形状（批量大小，时间步数或序列长度，隐单元数或特征维度）
        # 输出形状为（批量大小，时间步数，ffn_num_outputs）
        return self.dense2(self.relu(self.dense1(X)))
    
    
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

    
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, 
                 num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(Encoder):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, 
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f'block{i}', EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, 
                                                           ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，因此嵌入值乘以嵌入维度的平方根进行缩放，然后再与位置编码相加。
        # 使位置信息不要过大，从而可保持原信息重要程度
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X