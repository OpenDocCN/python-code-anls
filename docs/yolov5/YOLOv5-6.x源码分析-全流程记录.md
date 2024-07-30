<!--yml
category: 游戏
date: 2023-09-17 14:43:35
-->

# YOLOv5-6.x源码分析 全流程记录

> 来源：[https://blog.csdn.net/weixin_51322383/article/details/130353834](https://blog.csdn.net/weixin_51322383/article/details/130353834)

### 文章目录

*   [前言](#_1)
*   [导航](#_19)

# 前言

**这个系列的博客是YOLOv5-6.x的源码解析讲解，接下来一段时间我会将YOLOv5的每一个脚本文件梳理一遍，帮助自己，也帮助后面想要搞懂YOLOv5内部运作的同学们。**

**博主本人从去年开始接触深度学习，研究得最多的就是YOLO家族，但我总觉得不能完全通透地理解整个网络运作的流程。YOLOv5如今已经更新到了v7，可以说是工程界用得最多的项目，也是YOLO系列最稳定的。所以通读他的源码，对我们的帮助很大。**

**之前打算一星期写完这个专栏，但三天下来后，我发现我想多了。。。且不说去抠每一个细节，有很多重要的函数仍然难以理解，后续我也会再对写过的博客进行修改，加入自己的理解。鉴于本人的水平有限，如有错误，还请博主们指教。同时，感谢已经写过YOLOv5源码分析的各大博主们，我的灵感来源大多于你们！！！**

**重新立一个flag，两周内，完成该专栏！！！**

* * *

**2023-05-01 15:40**

**有时候我经常在想，有必要看那么细吗？这对我有什么帮助吗？包括我也问了我的老师和学长，有必要去挖YOLOv5的源码吗，老师也说没有必要。但是对于我来说，我是在这方面喜欢刨根问底的人，尽管我用了这么久的YOLOv5，但我总对他的很多执行过程云里雾里，让我去挖一遍源码，真的能解决我很多的疑惑。比如我之前一直没搞清bbox损失和置信度损失之间的关系，经过这次的学习后，让我大彻大悟。**

**这段时间看了很多博主的源码剖析，不得不说，他们做的都非常的好，讲的也非常的通透，而我许多地方看不懂仍然还是写了上去，算是为了给这个博客专栏做得更加完善吧，所以如果你看到我的很多地方讲的不透彻，你可以再看看我最下方References，或者去搜搜别的博主，肯定能解决你的疑惑。当然，我也非常欢迎你能和我进行讨论。如果你也能像我一样去写一个属于你自己的专栏解析，那我觉得这件事情。。。泰裤辣！！！**

* * *

# 导航

**YOLOv5**
[YOLOv5-6.x源码分析（一）---- detect.py](https://blog.csdn.net/weixin_51322383/article/details/130306871)
[YOLOv5-6.x源码分析（二）---- val.py](https://blog.csdn.net/weixin_51322383/article/details/130317934)
[YOLOv5-6.x源码分析（三）---- train.py](https://blog.csdn.net/weixin_51322383/article/details/130336035)
[YOLOv5-6.x源码分析（四）---- 模型搭建之yolo.py](https://blog.csdn.net/weixin_51322383/article/details/130353750)
[YOLOv5-6.x源码分析（五）---- 模型搭建之model.py](https://blog.csdn.net/weixin_51322383/article/details/130379982)
[YOLOv5-6.x源码分析（六）---- 数据集创建之datasets.py](https://blog.csdn.net/weixin_51322383/article/details/130387945)
[YOLOv5-6.x源码分析（七）---- 数据增强之augmentations.py](https://blog.csdn.net/weixin_51322383/article/details/130409656)
[YOLOv5-6.x源码分析（八）---- loss.py](https://blog.csdn.net/weixin_51322383/article/details/130426149)
[YOLOv5-6.x源码分析（九）---- general.py](https://blog.csdn.net/weixin_51322383/article/details/130447757)
[YOLOv5-6.x源码分析（十）---- metrics.py](https://blog.csdn.net/weixin_51322383/article/details/130454335)

* * *

**Reference**

1.  CSDN 路人贾： [YOLOv5源码逐行超详细注释与解读（1）——项目目录结构解析](https://blog.csdn.net/weixin_43334693/article/details/129356033?spm=1001.2014.3001.5501)
2.  CSDN 满船清梦压星河HK：[【YOLOV5-5.x 源码讲解】整体项目文件导航](https://hukai.blog.csdn.net/article/details/119043919?spm=1001.2014.3001.5502)