# Pytorch_study
For Pytorch study.
Task_1:Pytorch的基本概念.

1.什么是Pytorch，为什么选择Pytroch？
>> Pytorch是一个基于 Python 的科学计算包，作为NumPy 的替代品，可以利用 GPU 的性能进行计算；是深度学习研究平台，拥有足够的灵活性和速度。
>> 因为PyTorch是当前难得的简洁优雅且高效快速的框架，具有简洁、快速、易用、活跃的社区的特点。

2.Pytroch的安装。
>> 我的是windows10并且没有GPU，则在官网https://pytorch.org/ 上选择相应版本然后通过Anaconda安装PyTorch，在命令行窗口输入命令：conda install pytorch-cpu torchvision-cpu -c pytorch 即可。

3.配置Python环境。
>> 安装Anaconda，在环境变量中Path中添加Python目录。

4.准备Python管理器。
>> 安装Anaconda。

5.通过命令行安装PyTorch。
>> 按Win+R键进入windows运行，输入cmd进入命令行窗口，在命令行窗口输入命令：conda install pytorch-cpu torchvision-cpu -c pytorch 即可进行PyTorch的安装。

6.PyTorch基础概念。
>> 不太了解这个问题在问什么以及该回答什么..就了解了一下PyTorch入门要学习Tensors(张量)。Tensors类似于NumPy的ndarrays，同时Tensors可以使用GPU进行计算。对张量也有一定的了解了，可对其进行加减等运算。

7.通用代码实现流程(实现一个深度学习的代码流程)。
>> 我对深度学习不了解，只能复制代码来进行体验了解。我在网上找了定义CNN网络的代码来体验。代码如下：


import torch
import torch.nn as nn
import torch.nn.functional as F


'''
CNN计算

(H - k +2 * P) / S + 1
(W - k +2 * P) / S + 1

LetNet-5 
input: 32*32*3

out_conv1 = (32-5)+1 = 28 
max_pool1 = 28 / 2 = 14
out_conv2 = (14 - 5) + 1 = 10
max_pool2 = 10 / 2 = 5
'''

'''

定义一个神经网络

https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #  conv1层，输入的灰度图，所以 in_channels=1, out_channels=6 说明使用了6个滤波器/卷积核，
        # kernel_size=5卷积核大小5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # conv2层， 输入通道in_channels 要等于上一层的 out_channels
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # an affine operarion: y = Wx + b
        # 全连接层fc1,因为32x32图像输入到fc1层时候，feature map为： 5x5x16
        # 因此，全连接层的输入特征维度为16*5*5，  因为上一层conv2的out_channels=16
        # out_features=84,输出维度为84，代表该层为84个神经元
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 特征图转换为一个１维的向量
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]     # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
--------------------- 
版权声明：本文为CSDN博主「黑桃5200」的原创文章，遵循CC 4.0 by-sa版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/Heitao5200/article/details/90114432


个人总结：我认为此次任务基本上完成了对接下来学习所要用到的基础，但是对于小白来说还是有点懵懵懂懂的，不是很了解只有初步认识。最后的运行深度学习的代码也不是很理解里面的意思。总之，还要学习的还有很多，疑问也还有很多。
