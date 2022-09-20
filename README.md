# MVBench

引用测试：这是一条引用<sup><a href="#ref1">[1]</a></sup>



## 1. Introduction

xxx



## 2. Get started

### 2.1 Installation

`MVBench` is published as a Python package and you can install it by:

```bash
pip install mvbench
```

All dependencies (`torch==1.7.0, torchvision==x.x.x, ...`) will be resolved automatically if they are not prepared in your environment.



### 2.2 Build scenario

#### 2.2.1 By UI (recommended)

You can launch the UI interface by:

```bash
python -m mvbench.ui
```

Then a web app will be launched in your browser. You can follow the tutorial in it and view the whole building process. Finally, a code will be generated and you should copy it into your project.

#### 2.2.2 By code

We provide an universal API to builid scenario:

```python
build_scenario_manually(
    source_datasets_name=['SVHN', 'CIFAR10'], 
    target_datasets_order=['MNIST', 'STL10', 'USPS', 'MNIST', 'STL10', 'USPS'], 
    da_mode='da',
    num_samples_in_each_target_domain=100
)
```

Arguments:

- `source_datasets_name`：源域数据集名称

- `target_datasets_order`：目标域数据集到来顺序

- `da_mode`：DA设定，值范围为`'da' | 'partial_da' | 'open_set_da' | 'universal_da'`。

  以上述代码为例，源域中`CIFAR10`数据集和目标域中`STL10`数据集都有10个类，但只有9个类是一样的，`CIFAR10`和`STL10`各有一个私有类。四种`da_mode`对输出数据集的影响如下：

  - `'da'`：将`CIFAR10`和`STL10`中的私有类去除，保证`CIFAR10`和`STL10`的标签空间完全一致
  - `'partial_da'`：将`STL10`中的私有类去除，保证`CIFAR10`的标签空间大于`STL10`的标签空间
  - `'open_set_da'`：将`CIFAR10`中的私有类去除，保证`CIFAR10`的标签空间小于`STL10`的标签空间
  - `'universal_da'`：不对数据集标签空间做处理

- `num_samples_in_each_target_domain`：每个目标域中可用的样本数



### 2.3 Implement your algorithm

#### 2.3.1 Under unified interface (recommended)

Like a popular DG benchmark [DomainBed](https://github.com/facebookresearch/DomainBed) (paper<sup><a href="#ref1">[1]</a></sup>), each algorithms are implemented as a class which is inherited from an unified abstract class. By this, all algorithms can be evaluated in a uniform manner conveniently. 

(xxx).

#### 2.3.2 Based on your existed code

Sometimes you may want integrate your existed algorithm implementation with `MVBench` quickly while it's time-consuming to re-schedule your code to match our unified algorithm interface. The popular distribution shift benchmark [WILDS](https://ai.stanford.edu/blog/wilds/) (paper<sup><a href="#ref1">[2]</a></sup>) just provides APIs of building datasets and don't care about algorithm implementation and unified evaluation.

(xxx).



### 2.4 Evaluation

Run your algorithm by xxx.

Metrics (accuracy, time usage, memory footprint, etc.) will be recorded automatically in the DA process. After that, a summarized evaluation report will be generated.



## 3. Scenario

我们聚焦于移动端计算机视觉中在线持续的领域自适应场景，其可定义为：

> 离线阶段：服务器上，模型在源域数据上进行训练后，部署在移动设备上进行推理。此时无法接触到任何目标域数据。
>
> 在线阶段：移动设备所在环境**持续变化**，使得模型输入数据特征空间（可能以及标签空间）相对于源域持续发生变化（即目标域）。模型需要利用每个目标域上的**少量无标签**数据，**持续**进行**快速调整**才能不断弥补下降的精度。

在DA的框架下，该场景有以下特点：

1. 有一个源域，其可能由一个或多个数据集组成
2. 每个目标域由一个数据集组成
3. 模型在源域上预训练完成后，每遇到一个新目标域，都需要利用其中的少量无标签样本对模型进行调优

> 示例场景：
>
> 源域：将CIFAR-10, SVHN合并为一个数据集
>
> 目标域：STL-10 -> MNIST -> USPS -> STL-10 -> USPS -> ...

根据源域/目标域的特征空间、标签空间、对应关系，可对该场景进行更具体的分类，如下。**与之对应的数据集、算法和模型将在后续章节进行更详细的汇总展示。**

#### 3.1.1 根据特征空间 (X) 进行分类

域迁移体现在输入数据的分布变化上，如环境影响导致的图像光照、模糊变化，图像预处理阶段中产生的噪点、图像压缩等。下表根据CVPR'22和NeuriPS'20对Distribution Shifts进行了详尽的分类。~~但需要注意的是，Adversarial Shifts通常不纳入领域自适应的研究范围，本benchmark暂不对其进行研究。~~

| Shift                           | 描述                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| Background Shifts               | 图片背景产生变化、但物体主体完全一致                         |
| Corruption Shifts               | 图像中存在一些来自于环境影响（如光照、天气）或者图像预处理阶段（如噪点、JPEG压缩）的干扰 |
| Consistency Shifts              | 从视频中取出来的一段连续帧，对人来说很相似、但是能让模型输出不同预测 |
| Adversarially Filtered Shifts   | 将一个普通预训练模型预测错的图片单独组成一个数据集           |
| Geometric Transformation Shifts | 对图像施加几何变换（旋转、翻转、拉伸、视角变换等）           |
| Texture Shifts                  | 图像中物体形状保持完全一致，但纹理发生变化（如使用风格迁移生成的图片） |
| ~~Adversarial Shifts~~          | ~~使用对抗攻击生成的图片~~                                   |
| Dataset Shifts                  | 类别空间有交叉的两个数据集之间的shift                        |

#### 3.1.2 根据标签空间 (Y) 进行分类

源域和目标域在特征空间不同的同时，标签空间可能也不同。根据源域和目标域标签空间的交叉程度可分为以下四类：

|              | 源域/目标域标签空间的关系                |
| ------------ | ---------------------------------------- |
| DA           | 源域标签空间等于目标域标签空间           |
| Partial DA   | 源域标签空间包含目标域标签空间           |
| Open Set DA  | 目标域标签空间包含源域标签空间           |
| Universal DA | 源域标签空间和目标域标签空间只有部分交集 |

#### 3.1.3 根据源域数量进行分类

移动设备在同一时间只可能位于一个目标域中，因此不存在multi-target DA问题。

但针对一个目标域，源域中可能有一个或者多个数据集与之相关，由此衍生出两种DA场景：

1. single source and single target DA：对于每个目标域，源域中有且仅有一个数据集与之相关；
2. single source and multi target DA：对于每个目标域，源域中可能有多个数据集与之相关；此时，multi-target DA算法可能能更好的利用多个源域数据集的信息。



## 4. Workload

### 4.1 Dataset

以下章节根据2.1.1 标签空间 (Y) 进行分类对数据集进行汇总和分类。

#### 4.1.1 图像分类

| 数据集    | 物体大类 | Domain Shift类型                     |
| --------- | -------- | ------------------------------------ |
| MNIST     | 数字     | 与其它数据集构成Dataset Shifts       |
| EMNIST    | 数字字母 | 与其它数据集构成Dataset Shifts       |
| CIFAR10   | 通用物体 | 与其它数据集构成Dataset Shifts       |
| CIFAR10-C | 通用物体 | 与CIFAR10构成Image Corruption Shifts |
| ...       | ...      | ...                                  |

#### 4.1.2 目标检测

| 数据集 | 物体大类 | Domain Shift类型 |
| ------ | -------- | ---------------- |
| ...    | ...      | ...              |



### 4.2 Method

对于连续领域自适应场景，有一类针对性的算法Continual DA，但其仅适用于源域/目标域标签空间完全相同的单源域自适应，并不能覆盖2.1中场景的不同分类。因此，我们对传统UDA算法也进行了统计与实现，在Continual DA算法无法使用的场景中（如Open Set DA）我们默认使用传统UDA算法。

#### 4.2.1 根据特征空间 (X) 进行分类

大部分DA算法都没有明确说明适用于哪些domain shift，但至少可以从其实验所用数据集来判断其支持的domain shift。下表对Continual DA算法进行统计：

| 算法 | 实验数据集                                                   | Shift Type                           |
| ---- | ------------------------------------------------------------ | ------------------------------------ |
| ONDA | [KTH Handtool](https://www.nada.kth.se/cas/data/handtool/)   | Background Shifts, Corruption Shifts |
| Tent | ImageNet-C, CIFAR-10-C, CIFAR-100-C<br>SVHN/MNIST/USPS<br>GTA/CityScapes | Corruption Shifts, Dataset Shifts    |
| SHOT | Office, Office-Home, VisDA-C<br>SVHN/MNIST/USPS              | Dataset Shifts                       |
| ...  | ...                                                          | ...                                  |

#### 4.2.2 根据标签空间 (Y) 进行分类

#### 4.2.3 根据源域数量进行分类



### 4.3 Model

#### 4.3.1 图像分类

#### 4.3.2 目标检测



## 参考文献

- [1] <span id="ref1">(ICLR'21) In Search of Lost Domain Generalization</span>
- [2] <span id="ref2">(ICML'21) WILDS: A Benchmark of in-the-Wild Distribution Shifts</span>


