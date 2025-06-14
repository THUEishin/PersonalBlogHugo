---
title: '基于YOLO LITE的人脸口罩检测'
summary: 模式识别课程的期末大作业，对图片内的人脸进行检测定位，并判断是否佩戴口罩。在个人电脑上对YOLO-LITE模型从随机初始化开始训练，最终得到的模型性能结果为mAP@.5约为15%，CPU推断速度为29FPS。
date: '2020-06-29T17:09:35+08:00'
draft: false
author: ["Ruichen Ni"]
tags: ["Pattern Recognition"]
math: true
---

## 项目简介
&emsp;&emsp;这是我这学期模式识别课程的大作业。简单写个博客纪念一下。网上也有很多关于目标检测的YOLO算法的介绍文章，GitHub上也有很多大牛提供的YOLO算法的代码。但是大部分的代码都是提供一个已经训练好的model以及相应的inference代码。很少有人给出整个训练过程的代码。

&emsp;&emsp;经过多方学习，我写了一个特别特别简单的YOLO-LITE代码。卷积层的层数特别少，并且从老师提供的dataset里面截取了一个mini-dataset，可以在14个小时内从随机初始化开始训练得到一个YOLO-LITE模型。可以预料到过拟合现象会非常严重，最终的模型性能结果为mAP@.5约为15%，CPU推断速度为29FPS。

&emsp;&emsp;这篇博客的目的是为了给跟我一样刚入坑的小白对于YOLO-LITE算法的一个简单的指导，大牛还请绕道[手动狗头]。

## 代码地址
该项目的所有代码都被保存在我的GitHub仓库中
```bash
$ git clone https://github.com/THUEishin/Masked-Face-Detection.git
```
具体代码如何使用可以参见该仓库下的`README.md`文件，详细地介绍了如何使用已训练好的model去做inference以及如何从随机初始化开始训练一个YOLO-LITE模型。

## YOLO介绍
### 任务定义及主流方法
人脸口罩检测属于目标检测任务，而目标检测任务主要分为两个子任务：1、确定目标所在的位置；2、判断目标所属的类别。按照执行这两个子任务的顺序可以将现有的目标检测模型分为两阶段（two-stage）模型和一阶段（one-stage）模型。
1. 在两阶段模型中，需要首先大致确定目标所在的位置，即给出建议区域（Region Proposal）；然后在这些建议区域上进行目标类别的分类。目前主流的两阶段模型有RCNN、SPPNet、Fast RCNN、Faster RCNN和Pyramid Networks。
2. 在一阶段模型中，确定位置和判断类别的两个子任务将同时完成，不存在先后顺序。主流的一阶段模型有YOLO、SSD和Retina-Net。

我这里针对本文要实现的YOLO算法进行简单的发展介绍。YOLO算法的核心是通过**卷积**以及**最大值池化**操作对图像进行降采样的同时增大通道数，形成最后的==特征图==（feature map）。

特征图的每一个像素点代表了原始图像上的一块区域，因此可以将每个像素点作为原始图像网格划分的依据。例如YOLOv2中将 \\(416\times416\times3\\) 的图像最终转为 \\(13\times13\times255\\) 的特征图，相当于将原图划分为 \\(13\times13\\) 的网格，并且每个网格代表了原图中 \\(32\times32\\) 的一块区域。而每个像素点上的通道特征向量则是相应的预测值。

YOLO到目前为止存在三个版本，下面简单介绍一下三个版本之间的差异：

1. YOLOv1是针对RCNN的region proposal过程耗时太长提出的一种一阶段模型，可以同时预测边框和类别。在YOLOv1中加入了全连接层，导致测试图片的尺度必须与训练尺度一致；另外最终的特征图的大小确定，导致对于不同尺寸物体的检测精度存在较大的差异。

2. YOLOv2在YOLOv1的基础上进行了相当多的改进：在卷积层后添加了Batch Normalization，帮助正则化模型；增加了Anchor Box的先验框，并且每个Anchor Box可以独立地进行类别的预测，提高了正确率；增加路由层（passthrough layer），将不同尺度的特征图一起进行预测，提高了对细粒度特征的检测；去掉了全连接层，使得模型可以对不同尺寸的输入图像进行预测。

3. YOLOv3相对于YOLOv2的改进较小，主要是增加了卷积层的深度，YOLOv2使用的backbone网络是Darknet-19，即19个卷积层，而YOLOv3使用的backbone网络是Darknet-53，即53个卷积层；并且借鉴ResNet处理深层网络梯度消失的方法，在不同的卷积层之间设置了快捷链路（shortcut）。

### 模型轻量化的意义
随着计算机视觉的蓬勃发展，越来越多的成果被应用到生活实际中，例如支付宝付款时的刷脸支付、快递集散中心的分拣堆垛以及自动驾驶等等。最影响用户体验的往往不是精度，而是速度。

然而我们手机的芯片考虑到价格、体积等多方面因素，不可能像我们的电脑CPU或者GPU那么强大。因此近些年来越来越多的学者开始关注如何在保证预测精度的情况下提升预测的速度，即模型的轻量化。

Rachel Huang和Jonathan Pedoeem在YOLOv2-Tiny的基础上提出了YOLO-LITE[<sup>[1]</sup>](#Rachel-YOLO)：1、减少卷积层的通道数；2、减少卷积层数；3、对于浅层模型作出了无需Batch Normalization的实验验证。在尽可能少损失精度的情况下（YOLOv2-Tiny在VOC07上的mAP=40.48\%，YOLO-LITE在VOC07上的mAP=33.57\%）减少模型的参数以提高模型预测的效率，YOLOv2-Tiny的模型预测效率为2.4FPS，YOLO-LITE的预测效率为21FPS。

### 性能指标的定义
#### IoU
**图像的交并比**（Intersection to Union，**IoU**）定义如下：

$$IoU(A,B)=\frac{A\cap B}{A\cup B}$$

假设长方形图像A的左下角的节点坐标为\\(x_{min}^{1}\\)和\\(y_{min}^{1}\\)，右上角的节点坐标为\\(x_{max}^{1}\\)和\\(y_{max}^{1}\\)；长方形图像B的左下角的节点坐标为\\(x_{min}^{2}\\)和\\(y_{min}^{2}\\)，右上角的节点坐标为\\(x_{max}^{2}\\)和\\(y_{max}^{2}\\)。则相交部分的左下角节点坐标为：

$$x_{min}=\max\left(x_{min}^{1},x_{min}^{2}\right)$$

$$y_{min}=\max\left(y_{min}^{1},y_{min}^{2}\right)$$ 

右上角节点坐标为：

$$x_{max}=\min\left(x_{max}^{1},x_{max}^{2}\right)$$

$$y_{max}=\min\left(y_{max}^{1},y_{max}^{2}\right)$$

则相交部分的面积可以用如下公式进行计算：

$$A\cap B=(x_{max}-x_{min})\cdot(y_{max}-y_{min})$$

而并集的面积可以用容斥定理给出：

$$A\cup B=A+B-A\cap B$$

#### mAP
采用PASCAL VOC风格的11点插值法来进行mAP的计算。首先我们来定义正确率和召回率

$$precision=\frac{TP}{TP+FP}$$

$$recall=\frac{TP}{TP+FN}$$

其中TP表示True Positive，FP表示False Positive，FN表示False Negative。

对于某一个类别的预测结果，我们给定一个置信度的阈值，如果预测的置信度大于该阈值则认为是正样本，否则认为是负样本。在正样本中，给定一个IoU的阈值，如果预测框和真实框之间的IoU值大于该阈值，则认为这是一个真阳性样本，否则这是一个假阳性样本，真阳性样本的总数即为TP，假阳性样本的总数为FP。测试集中所有的该类别的正样本数量减去真阳性样本数量就得到了假阴性样本的数量FN。

通过上述公式，在给定的置信度阈值下我们得到了一组precision和recall，改变置信度阈值的大小，我们可以得到给定IoU阈值下的precision-recall曲线。对于该曲线采取如下PASCAL VOC风格的11点插值公式就可以计算得到该类别在给定IoU阈值下的平均正确率AP：

$$AP_{IoU}=\frac{1}{11}\sum_{i=0}^{10}\max_{recall>0.1i}\left(precision\right)$$

对戴口罩和不戴口罩两个类别的AP取算术平均就可以得到$mAP@IoU$：

$$mAP@IoU=\sum_{class=1}^{2}AP_{IoU}^{class}$$

#### 电脑配置
我个人笔记本电脑的配置为i7-4600U双核CPU@2.1GHz，超频可以达到2.69GHz，内存8G（系统常年占用3~4G，因此可用的内存空间只有4G），无GPU。

即使是YOLO-LITE，作者采用的仍然是GPU进行模型的训练，最终的训练时间仍然达到了7个小时，显然是我个人笔记本电脑所无法完成的。因此，我在YOLO-LITE的基础上进一步简化，使得YOLO模型能够在个人笔记本上完成从随机初始化开始的训练。

## 数据整理
### 数据来源
AIZOO作者开源的7959张人脸标注图片，数据集如下表所示来自于WIDER Face和MAFA数据集，并重新修改了标注和校验。

|  数据集   | WIDER | MAFA  | 合计  |
| :-------: | :---: | :---: | :---: |
| 训练集/张 | 3114  | 3006  | 6120  |
| 验证集/张 |  780  | 1059  | 1839  |

由于磁盘读写非常的耗费时间，因此我从训练集中选取了1024张图片作为mini-训练集，考虑到WIDER中的人脸基本都没有戴口罩，MAFA中的人脸基本都是带着口罩的，我从两个数据集各选取了512张（我取batch size为64，512张图片方便划分batch）。并且，从验证集中选取了256张图片作为mini-验证集，从验证集中另外选取了128张图片作为测试集。数据来源如下表所示：

|  数据集   | WIDER | MAFA  | 合计  |
| :-------: | :---: | :---: | :---: |
| 训练集/张 |  512  |  512  | 1024  |
| 验证集/张 |  128  |  128  |  256  |
| 测试集/张 |  64   |  64   |  128  |

在我的YOLO模型中，图像的输入大小为\\(208\times208\times3\\)（16倍降采样后变成\\(13\times13\times3\\) ），因此为了进一步缩短训练时图片读取的时间，我提前对训练集和验证集的图像和相应的标注尺寸进行了缩放后保存，而测试集的数据则保持不变。最终模型使用的数据集可以到清华云盘[这里](https://cloud.tsinghua.edu.cn/f/29f2223b1f6b490ab2a2/)下载。

### 数据可视化
我从mini-数据集中选了两张图片进行可视化展示如下：

原始图像

![训练集原始图片](/images/PatternRecognition/YOLO-LITE/origin-train.jpeg "340x240") &emsp;![验证集原始图像](/images/PatternRecognition/YOLO-LITE/origin-test.jpeg "240x240")

在保证原图像宽高比不变的条件下，我对训练集和测试集图片进行了缩放，即缩放后的图像宽或者高的大小为208。
然后将缩放后的图像放置到大小为\\(208\times208\times3\\)的灰色画布中。在图像缩放的同时，对边界框的标签也同等程度地进行缩放和平移。最终得到的结果如下图所示：

![训练集缩放后的图像](/images/PatternRecognition/YOLO-LITE/resize-train.jpeg "normal") &emsp; ![验证集缩放后的图像](/images/PatternRecognition/YOLO-LITE/resize-test.jpeg "normal")

## 模型设计
### Anchor框的大小确定
YOLO算法中使用C聚类的方法来确定Anchor框的大小，使用一般的欧式距离会导致大尺寸的Anchor框之间的距离要远大于小尺寸的Anchor框之间的距离，导致C聚类算法的不稳定。因此YOLOv2的作者在文章中提出使用IoU来构建两个Anchor框之间的距离：

$$distance=1-IoU(Anchor1,Anchor2)$$

这样构建的距离就与Anchor框的尺寸无关，而只与Anchor框之间的相似性有关。
C聚类算法如果采用随机初始化类别会造成收敛慢和不稳定的问题，因此我采用了最远批次的方法初始化C聚类算法的初始类别划分。我设置聚类数$k=6$可以得到如下的结果：

| Anchor框编号 |   1   |   2   |   3   |   4   |   5   |   6   |
| :----------: | :---: | :---: | :---: | :---: | :---: | :---: |
|      宽      | 0.87  |  1.4  |   1   | 6.08  | 15.05 | 22.45 | 52.93 |
|      高      | 1.42  | 2.00  | 8.14  | 19.62 | 30.32 | 70.13 |
|    宽高比    |  0.6  |   1   | 0.71  | 0.75  | 0.77  | 0.74  | 0.75  |

由于我采用的YOLO最后的特征图是16倍降采样后的结果（\\(13\times13\times filters\\)），因此尺寸特别小的目标是无法识别的，也即不使用第1个和第2个Anchor框。排除小尺寸的Anchor框后，我们可以发现所有Anchor框的宽高比基本都在0.75左右，这与一般人的人脸宽高比基本一致。而AIZOO的作者在他的公众号推送中说有很多种不同的宽高比，查看他的开源代码后发现，原因在于他的图像缩放没有保证原图像的宽高比。

综上所述，我最终程序中使用的先验Anchor框为上表中编号为\\(3\sim 6\\)的Anchor框大小。
### YOLO-LITE模型结构
Rachel Huang和Jonathan Pedoeem文章中的训练集是PASCAL VOC2007和2012，里面有20个类别，因此每个网格上每个Anchor框的维度是\\(4(loc)+1(conf)+20(cls)=25\\)，最终YOLO-LITE每个网格使用了5个Anchor框，所以他们YOLO预测层前最后一层卷积层的通道数为\\(25\times5=125\\)。

在人脸口罩识别的任务中，只有戴口罩与不戴口罩两个类别，我最后使用的Anchor数量为4，因此我模型中YOLO预测层前最后一层卷积层的通道数为\\(7\times4=28\\)。综上所述，我给出每一层的参数如下：

1. 16通道的\\(3\times3\\)卷积核，padding=1，stride=1\\(\rightarrow\\)leakyReLU激活函数\\(\rightarrow 2\times2\\)的max pooling，stride=2；（共432个参数）

2. 32通道的\\(3\times3\\)卷积核，padding=1，stride=1\\(\rightarrow\\)leakyReLU激活函数\\(\rightarrow 2\times2\\)的max pooling，stride=2；（共4608个参数）

3. 64通道的\\(3\times3\\)卷积核，padding=1，stride=1\\(\rightarrow\\)leakyReLU激活函数\\(\rightarrow 2\times2\\)的max pooling，stride=2；（共18432个参数）

4. 128通道的\\(3\times3\\)卷积核，padding=1，stride=1\\(\rightarrow\\)leakyReLU激活函数\\(\rightarrow 2\times2\\)的max pooling，stride=2；（共73728个参数）

5. 128通道的\\(3\times3\\)卷积核，padding=1，stride=1\\(\rightarrow\\)leakyReLU激活函数；（共147456个参数）

6. 256通道的\\(3\times3\\)卷积核，padding=1，stride=1\\(\rightarrow\\)leakyReLU激活函数；（共294912个参数）

7. 28通道的\\(1\times1\\)卷积核，padding=0，stride=1；（共7168个参数）

8. YOLO预测层

所有层的参数都将从随机初始化开始训练，统计可知模型待训练的参数总数为546304个参数。
### 数据类型的确定
卷积核函数、leakyReLU激活函数和最大值池化函数我都是直接调用的pytorch的nn模块下相应的函数。由于pytorch在训练过程中并不会自动地进行数据类型转换，一旦输入数据的类型和模型函数使用的数据类型不符时就会报错

（pytorch的torch.tensor()函数默认是构建double类型的张量，而nn模块下的函数其待训练的参数默认全是float类型的变量，一开始我以为是没有对输入数据做Variable包装，debug了半天才发现是pytorch不会做自动数据类型转换）

由于运行过程存在中间数据，通过监视任务管理器中的内存占用量变化可以发现，使用float类型的参数时，整个训练过程最大将占用4G左右的内存，算上系统长期占用的3G左右的内存，最高的内存使用率将达到90\%左右。

由于double类型的字长是float类型的两倍，可以预见到如果使用double类型的参数，我电脑的内存（8G）将远远不足。因此最终使用的是float类型的参数。

## 从随机初始化开始的训练过程
### 误差的定义
我沿用了YOLO作者在YOLOv2中给出的坐标变换公式：
![损失函数定义](/images/PatternRecognition/YOLO-LITE/loss-function.png)

在此基础上，定义损失函数如下：

$$
loss = \lambda_{coord}\sum_{i=0}^{S^{2}}\sum_{j=0}^{B}\mathcal{I}_{ij}^{obj}MSE\left( \left(b_x, b_y\right) _{A_i}, (b_x,b_y) _{L_i},reduction\right)
$$

$$
\quad\quad+\lambda_{coord}\sum_{i=0}^{S^{2}}\sum_{j=0}^{B}\mathcal{I}_{ij}^{obj}MSE\left((b_w, b_h) _{A_i}, (b_w, b_h) _{L_i},reduction\right)
$$

$$
\quad\quad+\sum_{i=0}^{S^{2}}\sum_{j=0}^{B}\mathcal{I}_{ij}^{obj}BinaryCrossEntropy\left(C _{A_i}, C _{L_i},reduction\right)
$$

$$
\quad\quad+\lambda_{no\_obj}\sum_{i=0}^{S^{2}}\sum_{j=0}^{B}\mathcal{I}_{ij}^{no\_obj}BinaryCrossEntropy\left(C _{A_i},C _{L_i},reduction\right)
$$

$$
\quad\quad+\sum_{i=0}^{S^{2}} \sum_{c \in classes} {\mathcal{I}_{ij}^{obj} CrossEntropy \left((p_c) _{A_i}, (p_c) _{L_i}, reduction\right)}
$$

其中，当相应的Anchor框中存在object时，不存在object时则相反

\\(\mathcal{I} _{ij}^{obj}=1, \mathcal{I} _{ij}^{no-obj}=0\\)

\\(\lambda_{coord}, \lambda_{no-obj}\\)是误差缩放系数；
\\(A_{i}\\)表示第i个网格内用于预测的那个Anchor框；
\\(L_{i}\\)表示第i个网格内的标签；
\\(C\\)表示置信度系数；
\\(p_{c}\\)表示该Anchor框对于类别的预测。

reduction我们沿用YOLOv1论文中的sum求和方式。BinaryCrossEntropy是一个多类分类器的误差函数，比较适合用来统计一张图片中存在多个标签的置信度误差，我直接使用的是nn模块下的BCELoss函数。
### 从随机初始化开始的模型训练过程
由于对边界框的宽和高相对于先验Anchor框做了对数归一化，使得\\(b_{w}\\)和\\(b_{h}\\)的取值范围为\\( (-\infty,+\infty)\\)，前期如果梯度太大就有可能造成训练的发散。而前期训练的学习速率太小又会导致整个训练过程收敛较慢。

另外可以发现，每张图片最终特征图上的Anchor数量为\\(13\times13\times4=676\\)，而一张图片的标签数往往是个位数的量级，会造成正负样本的比例失调，这也是造成收敛发散的主要原因。

因此我设置\\(\lambda_{coord}=5, \lambda_{no-obj}=0.001\\)以保证模型前期在坐标维度上的收敛速度，同时可以很好地避免训练发散。随后，我逐步增大\\(\lambda_{no\_obj}\\)以增加模型在置信度误差上的收敛速度。以下是每组具体的训练参数、学习速率和训练的Epoch数：

1. \\(\lambda_{coord}=5,\lambda_{no-obj}=0.001\\)；起始学习速率为1e-4，每隔5个epoch学习速率降低20\%；共训练100 Epoch（寻找最为合适的学习速率）

2. \\(\lambda_{coord}=5,\lambda_{no-obj}=0.001\\)；学习速率设置为1e-5；共训练100 Epoch

3. \\(\lambda_{coord}=5,\lambda_{no-obj}=0.01\\)；学习速率设置为1e-5；共训练100 Epoch

4. \\(\lambda_{coord}=5,\lambda_{no-obj}=0.1\\)；起始学习速率为1e-5，每隔10个epoch学习速率降低5\%；共训练100 Epoch

## 模型的训练结果与讨论
### 模型性能
由于最后一个100Epoch的训练过程中过拟合现象非常严重，因此我分别统计了320 Epoch模型和400 Epoch模型在测试集上的mAP性能如下表所示：
1.训练了320 Epoch的模型

|  IoU阈值   |  0.5   |  0.7  |  0.9  | 0.5:0.95 |
| :--------: | :----: | :---: | :---: | :------: |
|  戴口罩AP  | 20.06% | 6.87% | 0.70% |  7.59%   |
| 不戴口罩AP | 9.08%  | 1.05% |  0%   |  2.72%   |
|    mAP     | 14.57% | 3.96% | 0.35% |  5.16%   |

2.训练了400 Epoch的模型

|  IoU阈值   |  0.5   |  0.7   |  0.9  | 0.5:0.95 |
| :--------: | :----: | :----: | :---: | :------: |
|  戴口罩AP  | 18.52% | 10.11% |  0%   |  7.43%   |
| 不戴口罩AP | 9.33%  | 0.73%  |  0%   |  2.38%   |
|    mAP     | 13.92% | 5.42%  |  0%   |  4.91%   |

### 模型改进展望
本模型的一个最大的优势就是能够在个人CPU笔记本电脑上完成从随机初始化开始的模型训练过程，并且最终的模型性能也还不错：\\(mAP@.5\approx15\%\\)，CPU检测速度\\(29.7 FPS\\)。

但是模型仍然有可以改进的地方。由于大作业的时间有限，没有时间再进一步训练优化模型了，这里我指出模型可以进一步优化的方向。
#### 先验Anchor框的选取与多尺度目标检测
由于C聚类的聚类结果和聚类个数以及初始化密切相关，我在做C聚类的时候简单地取了聚类数为6，这里应该使用谱聚类来分析一下具体聚类数应该取多大比较好，这样就不会漏掉\\(\left(34,50\right)\\)这个先验Anchor框的大小。我后续看测试集的Inference输出时发现，限制正确率的主要原因还是很多中等规模的目标都没有被检测出来。

另外，还有一个多尺度目标检测的问题。YOLOv3通过引入路由层将不同大小的特征图结合起来一起进行推断预测，很好地解决了不同尺寸目标感受野的问题。如果像我的模型一样只使用一种尺寸的特征图，就有很大概率发生不同类别的预测框嵌套的情况。
#### 过拟合问题

由于训练时间有限，我只选取了mini数据集。而WIDER数据集不同的场景类别多，而且差别迥异，每一类只选取少量的样本作为训练集无疑增加了训练难度和过拟合的概率。由于个人CPU电脑的内存往往很小（通常是8G，刨去系统占用就只剩下4G空间了），一口气把所有图片都读取进来显然不太现实，在这种情况下想要增加训练数据量有两种方法：

1. 分批次读入训练数据。比如我现在的训练batch_size是64，那可以选择读512张图片在内存中，这512张图片训练10个Epoch后换下一批512张图片。但是磁盘读写速度是很慢的，这样会导致训练时间变得非常长；

2. 数据增强。观察测试集Inference结果后，我发现我的模型对于游泳的那些图片预测很差，主要是训练集的图片中人脸基本都是端正竖直的，而游泳的人脸基本都是横着的，因此可以通过数据增强的方式来增加训练集数据量：每一个Epoch对训练数据图片做一个随机小角度的旋转。
#### 高层次特征的学习

我的模型对口罩和人脸的特征描述层次很低，特别是对于口罩的特征简单的认为是一块颜色较深的纯色区域。

我个人认为这个问题是YOLO-LITE这种浅层模型所无法解决的问题，这个就真的只能够使用像YOLOv3那样的深层Darknet网络（Darknet-53）。
## 代码参考
由于对pytorch中的矩阵变换不是太熟悉，我YOLOmodel.py文件中的prediction函数借鉴了[ayooshkathuria仓库](https://github.com/ayooshkathuria/pytorch-yolo-v3/blob/master/util.py)中的predict_transform函数。

<div id="Rachel-YOLO"></div>

- [1] Rachel Huang, Jonathan Pedoeem, and Cuixian Chen. *Yolo-lite*: a real-time object detection algorithm optimized for non-gpu computers. In 2018 IEEE International Conference on Big Data (Big Data), pages 2503–2510. IEEE, 2018.