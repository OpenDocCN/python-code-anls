<!--yml
category: 游戏
date: 2023-09-17 14:45:23
-->

# YOLOv5-6.x源码分析（一）---- detect.py

> 来源：[https://blog.csdn.net/weixin_51322383/article/details/130306871](https://blog.csdn.net/weixin_51322383/article/details/130306871)

### 文章目录

*   [前引](#_1)
*   [🚀YOLOv5-6.x源码分析（一）---- detect.py](#YOLOv56x_detectpy_17)
*   *   [1\. 导入需要的包](#1__19)
    *   [2\. 执行main函数](#2_main_70)
    *   [3\. 设置opt参数](#3_opt_86)
    *   [4\. 执行run函数](#4_run_124)
    *   *   [4.1 初始化一些配置](#41__125)
        *   [4.2 载入模型](#42__142)
        *   [4.3 加载数据](#43__156)
        *   [4.4 推理部分](#44__172)
        *   *   [4.4.1 热身部分](#441__174)
            *   [4.4.2 对每张图片/视频进行前向推理](#442__203)
            *   [4.4.3 NMS后处理除去多余的框](#443_NMS_216)
            *   [4.4.4 预测过程](#444__225)
            *   [4.4.5 打印目标检测结果](#445__277)

# 前引

这算是我的第一个正式博客文章吧，在准备动手写内容的时候，都有点无从下手的感觉。anyway，以后应该会写的越来越娴熟的。

YOLO系列我已经用了接近一年了吧，从去年暑假开始学习，打算入坑深度学习，其中跑过demo，自己用Flask搭配YOLOv5写过网页端实时检测，还看过源码，可以说已经把YOLO系列玩得已经比较6了。

YOLO系列日新月异，如今已经更新到了第8代，但用得最多的还是第五代，而第五代也已经更新到了v7.0，因为更新多，所以也相对更加稳定，使用的人也更多。

我开始学习深度学习其实到现在也没有一年，我这种半路出家的，如果不好好走每一步，真的很容易出岔子。像上面提到，我用YOLO也已经用得比较多了，项目里面三个有两个都是用的YOLO，所以在到时候面试的时候肯定也是重点询问项目，这样我就更得把YOLO的每一个part熟悉了。

所以正式因为这样，我才会写下这篇博客，并由此作为起点来记录，到最后把每一部分都理解通透。写博客真的很占用时间，但为了不让碎片化信息绑架我，我一定可以坚持写完的！

再定个小目标，这一周之内把YOLOv5的源码解析写完。

Let’s begin!🚀🚀🚀

**导航：**[YOLOv5-6.x源码分析 全流程记录](https://blog.csdn.net/weixin_51322383/article/details/130353834)

# 🚀YOLOv5-6.x源码分析（一）---- detect.py

这个函数是推理脚本，可以输入图片、视频、streams等进行检测。执行的结果会保存在runs/detect/xxx下。

## 1\. 导入需要的包

```py
import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn 
```

首先是导入的常用python库：

*   `argparse：`它是一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息
*   os： 它提供了多种操作系统的接口。通过os模块提供的操作系统接口，我们可以对操作系统里文件、终端、进程等进行操作
*   `sys`： 它是与python解释器交互的一个接口，该模块提供对解释器使用或维护的一些变量的访问和获取，它提供了许多函数和变量来处理 Python 运行时环境的不同部分
*   `pathlib`： 这个库提供了一种面向对象的方式来与文件系统交互，可以让代码更简洁、更易读
*   `torch`： 这是主要的Pytorch库。它提供了构建、训练和评估神经网络的工具
*   `torch.backends. cudnn`： 它提供了一个接口，用于使用cuDNN库，在NVIDIA GPU上高效地进行深度学习。cudnn模块是一个Pytorch库的扩展

```py
FILE = Path(__file__).resolve() # 得到绝对路径 ./yolov5/detect.py
ROOT = FILE.parents[0]  # YOLOv5 root directory 父目录 ./yolov5
if str(ROOT) not in sys.path:   # sys.path 模块的查询路径列表,确保ROOT存在sys.path中
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative，绝对路径转换为相对路径 
```

接着定义了一些文件路径。
这一部分的主要作用有两个：

*   将当前项目添加到系统路径上，以使得项目中的模块可以调用。
*   将当前项目的相对路径保存在ROOT中，便于寻找项目中的文件。

```py
# ----------------- 导入自定义的其他包 -------------------
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync 
```

最后则是一些自定义模块，其中主要包括了：

*   `models/common.py：`定义了一些通用的类模块，比如各种卷积模块。
*   `utils.dataloaders.py：`这个文件定义了两个类，LoadImages和LoadStreams，它们可以加载图像或视频帧，并对它们进行一些预处理，以便进行物体检测或识别。
*   `utils.general.py：`定义一些工具函数，比如日志、坐标转换等。
*   `utils.plot.py：`画图，标框。
*   `utils.torch_utils.py:`定义了一些与pytorch相关的工具函数，比如设备选择等。

通过导入这些模块，可以减少代码的复杂度、耦合性、冗余程度。

## 2\. 执行main函数

```py
def main(opt):
    check_requirements(exclude=('tensorboard', 'thop')) # 检测各种包有没有成功安装;打印参数
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt) 
```

主函数主要就是调用了`run()`函数，将命令行参数opt作为字典参数传递给`run()`函数。

> `if name == main：`的作用：
> 一个python文件通常有两种使用方法，第一是作为脚本直接执行，第二是 import 到其他的 python 脚本中被调用（模块重用）执行。因此 if name == ‘main’:的作用就是控制这两种情况执行代码的过程，在 if name == ‘main’: 下的代码只有在第一种情况下（即文件作为脚本直接执行）才会被执行，而 import 到其他脚本中是不会被执行的。

## 3\. 设置opt参数

```py
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT/'runs/train/strawberry/weights/best.pt', help='model path(s)')    # 权重文件
    # parser.add_argument('--source', type=str, default='http://admin:admin@192.168.43.1:8081', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str, default=ROOT / "data/strawberry", help='file/dir/URL/glob, 0 for webcam') # 测试数据
    parser.add_argument('--data', type=str, default=ROOT/'data/strawberry.yaml', help='(optional) dataset.yaml path')    # 参数文件
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w') # 高、宽
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')              # 置信度阈值
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')                  # 非极大抑制的iou阈值
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')           # 每张图片最大的目标个数
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')               # GPU加速
    parser.add_argument('--view-img', action='store_true', help='show results')                             # 是否展示预测后的图片/视频，默认false
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')                    # 是否将预测的框坐标以txt文件格式保存 默认True 会在runs/detect/expn/labels下生成每张图片预测的txt文件
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')   # 是否保存预测每个目标置信度到预测tx文件中
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')           # 是否需要将预测到的目标从原图中扣出来 剪切好 并保存 会在runs/detect/expn下生成crops文件，将剪切的图片保存在里面  默认False
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')                  # 是否不要保存预测后的图片  默认False 就是默认要保存预测后的图片
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')  # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')                   # 进行nms是否也除去不同类别之间的框 默认False
    parser.add_argument('--augment', action='store_true', help='augmented inference')                       # 预测是否也要采用数据加强
    parser.add_argument('--visualize', action='store_true', help='visualize features')                      # 是否将optimizer从ckpt中删除  更新模型  默认False
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')     # 保存路径
    parser.add_argument('--name', default='exp', help='save results to project/name')                       # 保存的文件名字
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')   # 如果存在文件夹，是否覆盖
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')        # 检测框的线条宽度
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')                # 是否隐藏label
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')             # 是否隐藏置信度
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand [640,640]
    print_args(vars(opt))   # 打印所有参数信息
    return opt 
```

这部分代码主要是设置了一些列参数，这些参数在`run()`中以字典形式传递。

## 4\. 执行run函数

### 4.1 初始化一些配置

```py
# ===================================== 1、初始化一些配置 =====================================
    # 是否保存预测后的图片 默认nosave=False 所以只要传入的文件地址不是以.txt结尾 就都是要保存预测后的图片的
    save_img = not nosave and not source.endswith('.txt')  # save inference images  是否以.txt结尾;为true
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)    # 是否是文件地址 suffix:后缀(1:从j开头)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))   # 是否是网络流地址 false
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)  # 是否是数值(摄像头)、.txt、网络流且不是文件地址
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories 保存路径
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run 增量路径（检测保存路径下的数字到几了）
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir 
```

这段代码包括了一些保存路径之类的定义。

### 4.2 载入模型

```py
# ===================================== 2、载入模型 =====================================
    # Load model 模型加载
    device = select_device(device)  # 设备选择
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)   # 权重、设备、false、.yaml、半精度推理过程
    stride, names, pt = model.stride, model.names, model.pt # 步长、类别名、是否为pytorch
    imgsz = check_img_size(imgsz, s=stride)  # check image size 640是32的倍数 
```

前面两行代码都是在自己定义的包中，在后面再具体讲解吧，这里大致只需要了解到他是选择设备（cpu还是cuda）、载入模型。
接着下面获取了模型的**stride、name、pt**等参数。
最后调用`check_img_size`检查图片是否符合要求，不符合则需要调整。

### 4.3 加载数据

```py
# ===================================== 3、加载数据 =====================================
    # Dataloader
    if webcam:  # false
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt) # 加载图片文件
        bs = 1  # batch_size    每次输入一张图片
    vid_path, vid_writer = [None] * bs, [None] * bs 
```

这里的主要函数是`LoadImages()`，载入数据。

### 4.4 推理部分

这个part是整个算法的核心部分，通过for循环对加载的数据进行遍历，如果是视频流则一帧一帧地推理，然后进行NMS，最后画框，预测类别。

#### 4.4.1 热身部分

```py
# Run inference 模型推理过程
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup(热身初始化)
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]  # dt：寸尺时间
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        # 1、处理每一张图片/视频的格式
        im = torch.from_numpy(im).to(device)    #从numpy转成tensor格式，放到device上
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32  判断有没有用半精度
        im /= 255  # 0 - 255 to 0.0 - 1.0   归一化
        if len(im.shape) == 3:  # 是否是3通道
            im = im[None]  # expand for batch dim [1,3,640,480]
        t2 = time_sync()
        dt[0] += t2 - t1 
```

**热身**操作，即对模型进行一些预处理以加速后续的推理过程。

> 作用：来自ChitGPT的答案：**深度学习模型训练热身的作用是为了使初始权重更好地适应数据分布，提高最终模型的收敛速度和泛化能力。通过热身训练，可以有效减少梯度下降的震荡，加速收敛速度，并降低局部极小值的影响。**

说简单点就是在模型训练初期给他一个较大的学习率，**因为较大的学习率就不那么容易会使模型学偏**，然后在训练的后期再减小学习率，使其收敛。
具体可看[深度学习之“训练热身”（warm up）–学习率的设置](https://blog.csdn.net/weixin_40051325/article/details/107465843)

在这个阶段，还定义了一些变量，包括`seen`、`windows`和`dt`，分别表示已处理的图片数量、窗口列表和时间消耗列表。遍历dataset，整理图片信息。

接着是对数据集的图片进行预处理：

*   将图片转化为tensor格式，放到device上，并转换为FP16/32。
*   将像素值0 ~ 255归一化，变为0 ~ 1，并为批处理增加一维度（batch）。
*   记录时间消耗并更新dt

#### 4.4.2 对每张图片/视频进行前向推理

```py
# Inference     默认False
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        # 2、对每张图片/视频进行前向推理
        pred = model(im, augment=augment, visualize=visualize)  # augment：数据增强 pred:得到检测框
        t3 = time_sync()
        dt[1] += t3 - t2 
```

**这里对每张图片进行前向推理。**
第二行代码，使用`model`对图像进行预测，`augment`和`visualize`参数是用于指示是否在预测时使用数据增强和可视化。
后面的代码记录了当前时间，并计算**从上一个时间点到这个时间点的时间差**，然后将这个时间差**加到一个名为dt的时间差列表中的第二个元素上**。

#### 4.4.3 NMS后处理除去多余的框

```py
# NMS 去除多余的框
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) # 1,5（5个检测框）,6(前四个为坐标，置信度，类别）
        dt[2] += time_sync() - t3 
```

这段是YOLO的经典代码：**非极大值抑制（NMS）**，用于筛选预测结果。
再次更新计时器，记录NMS所耗费的时间。

#### 4.4.4 预测过程

```py
# Process predictions  后续保存或者打印预测信息
        # 对每张图片进行处理  将pred(相对img_size 640)映射回原图img0 size
        for i, det in enumerate(pred):  # per image
            seen += 1   # 计数
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0) # 如果有frame则为0 
```

对筛选后的结果进行for循环遍历，这一段主要是判断是否采用网络摄像头。

*   **如果使用的是网络摄像头**，则代码会遍历每个图像并复制一份备份到变量`im0`中，同时将当前图像的路径和计数器记录到变量`p`和`frame`中。最后，将当前处理的物体索引和相关信息记录到字符串变量`s`中。

*   **如果没有使用网络摄像头**，则会直接使用`im0`变量中的图像，将图像路径和计数器记录到变量`p`和`frame`中。同时，还会检查数据集中是否有"frame"属性，如果有，则将其值记录到变量`frame`中。
    `det`是pred的每一张图片内容，det就是一张图片的东西，在后续的代码中会用到，这里先按下不表。

```py
p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string 打印图片信息
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh   获得宽和高的大小
            imc = im0.copy() if save_crop else im0  # for save_crop 是否把检测框裁剪下来保存成一张图片
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))	# 以便于在图像上绘制检测结果 
```

这一部分主要是路径转换，`save_crop`来选择是否把检测框裁剪下来保存成一张图片。
最后创建了一个`annotator`对象，以便于在图像上绘制检测结果。

```py
 if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():   # 统计所有框的类别
                    # 若为这几个类别才进行结果打印
                    # if names[int(c)] == 'person' or names[int(c)] == 'bicycle' or names[int(c)] == 'car' or names[int(c)] == 'motorcycle' \
                    # or names[int(c)] == 'bus' or names[int(c)] == 'truck':
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n}  {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # else:   # 如果不是，则continue
                    #     continue 
```

这一部分是判断有没有框，如果有物体，则会执行操作。

我打注释的代码可以只检测部分物体。

首先，`scale_coords`会将检测结果中的物体坐标从缩放的图片大小变回去。
然后遍历`det`的内容，前面说了`det`就是一张图片的信息，其实`det`里面包含了每一个物体的信息，将其**类别和数量**
添加到`s`字符串中。方便后面打印。

#### 4.4.5 打印目标检测结果

```py
# Write results 保存结果
                for *xyxy, conf, cls in reversed(det):  # 保存.txt    # reversed：逆序
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image  画到原图上
                        c = int(cls)  # integer class

                        # # # 只检测几个类别:person、bicycle、car、motorcycle、bus、truck
                        # if names[int(cls)] == 'person' or names[int(cls)] == 'bicycle' or names[int(cls)] == 'car' or names[int(cls)] == 'motorcycle' \
                        #         or names[int(cls)] == 'bus' or names[int(cls)] == 'truck':
                        #
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}  {conf:.2f}')  # hide_labels隐藏标签   hide_conf隐藏置信度
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # else:
                        #     continue
                    if save_crop:   # 是否截取目标框为图片
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True) 
```

如果存在目标检测结果，就会执行下一步操作。

这里将det中每一个物体的坐标信息进行转换，采用的函数时`xyxy2xywh()`；
并将label画到图片`annotator`上，调用的函数是`box_label`。

* * *

剩下的就是保存图片的功能和打印功能的实现了，不是很难，就不想写了。主要是我写这篇博客的时候，电脑太卡了，我都重启网页了好多次了。这电脑真得换了，散热是真的不行，打游戏不行就算了，现在连最基本的办公也难以实现。下半年就换！

* * *

`detect.py`算是大概写完了吧。

进行一个总结：

**这一段代码就是目标检测算法中的inference推理阶段，大致流程就是1\. 载入数据；2\. 推理；3\. NMS后处理；4\. 画框；5\. 保存结果、打印信息。**

现在是20：01，正好到了下班时间，健身去了~
明天继续肝 ~