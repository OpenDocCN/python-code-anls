# Yolov5DNF源码解析 1

# yolov5-DNF
使用yolov5检测DNF游戏画面，通过设计相应的算法来控制角色自动打怪。

详细教学请移步b站，有很详细的讲解：https://www.bilibili.com/video/BV18r4y1A7BF/

对于代码的使用，有几点要注意：

1. 代码中涉及到使用opencv对小地图和技能栏进行模板匹配和二值化等操作，注意，不同游戏分辨率和电脑显示器分辨率是不一致的，代码中给出的（0,0,1280,800）是本人游戏中的分辨率，而small_recgonize.py和skill_recgonize.py中的img[45:65, 1107:1270]， img[733: 793, 538:750, 2]是根据不同显示器的分辨率决定的，使用时需要自己调整。

2. 本人训练的yolov5模型还有待提高，我的训练集只有294张图片，因此效果一般。


# `skill_recgnize.py`

这段代码的主要目的是对一张图片进行评分。评分是通过检测图片中像素是否大于127来进行的。大于127的像素将会被计入到一个变量中，该变量代表整个图片中大于127的像素数量占总像素数量的比例。最终，该代码将计算出评分并将结果输出。

此外，代码还提供了一个图片显示函数，该函数使用OpenCV库中的imshow函数来显示图片。如果调用该函数，将会显示一个窗口，用户可以在窗口中移动鼠标并进行其他操作。


```py
import cv2 as cv
import numpy as np

def score(img):
    counter = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > 127:
                counter += 1
    return counter/(img.shape[0] * img.shape[1])

def img_show(img):
    cv.imshow("win", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

```

这段代码定义了一个函数 `skill_rec()`，用于检查给定的技能名称（通过参数 `skill_name`）的图像是否为有价值的图像。`skill_rec()`函数接收两个参数：技能名称（字符串类型）和图像（图像数据类型）。

首先，函数计算两个值：

1. 793 - 733 = 60，然后将此值除以2得到 30。这是技能高度。
2. 750 - 538 = 212，然后将此值除以7并取整得到 32。这是技能宽度。

接下来，函数定义了一个字典 `dict`，其中包含各个技能的图像数据。每个键（例如 `"A"`）对应一个元组，包含技能名称、技能高度和技能宽度。

然后，函数定义了一个函数 `score()`，用于比较图像的分数。这个函数接收一个图像数据类型（例如 `image`）作为参数，并返回一个浮点数（表示图像的质量）。如果得分大于0.1，说明图像是有价值的，否则不是。

最后，函数使用 `skill_rec()` 函数来获取技能名称和对应的图像，并比较它们之间的匹配程度。如果技能名称和图像匹配，函数返回 `True`；否则返回 `False`。


```py
skill_height = int((793-733)/2)
skill_width = int((750-538)/7)

dict = {"A": (733+skill_height, 538), "S": (733+skill_height, 538+skill_width), "D": (733+skill_height, 538+2*skill_width),
        "F": (733+skill_height, 538+3*skill_width), "G": (733+skill_height, 538+4*skill_width),
        "H": (733+skill_height, 538+5*skill_width), "Q": (733, 538), "W": (733, 538+skill_width), "E": (733, 538+2*skill_width),
        "R": (733, 538+3*skill_width), "T": (733, 538+4*skill_width), "Y": (733, 538+5*skill_width)}


def skill_rec(skill_name, img):
    if skill_name == "X":
        return True
    skill_img = img[dict[skill_name][0]: dict[skill_name][0]+skill_height,
                dict[skill_name][1]: dict[skill_name][1]+skill_width, 2]
    if score(skill_img) > 0.1:
        return True
    else:
        return False

```

It looks like this code is a script for a game or a simulation. It uses the Pygame library to display images and calculate the game's difficulty level based on the player's performance.

The script loads an image and displays it using the `imgdisplay` function. This function takes two arguments: the image data and the parameters for how the image should be displayed. The example code provided uses the following parameters:

* `[img]`: The image data. This is a two-dimensional array that represents the image, with the last dimension representing the color channels.
* `[D]`: The difficulty level of the game. A lower number means the game is easier, while a higher number means it is harder.
* `[F]`: The font for the game's difficulty level display.
* `[W]`: The font for the game's score display.
* `[R]`: The font for the game's settings.
* `[imgQ]`: The quality of the image data.
* `[imgS]`: The speed of the image data.
* `[imgD]`: The direction of the image data.
* `[imgH]`: The size of the image.
* `[imgG]`: The type of image data.
* `[imgE]`: The engagement level of the game.
* `[imgQ]`: The quality of the image data.
* `[imgS]`: The speed of the image data.
* `[imgD]`: The direction of the image data.
* `[imgH]`: The size of the image.
* `[imgG]`: The type of image data.
* `[imgF]`: The font for the game's difficulty level display.
* `[imgW]`: The font for the game's score display.
* `[imgR]`: The font for the game's settings.
* `[imgY]`: The font for the game's engagement level display.
* `[imgH]`: The size of the image.
* `[imgW]`: The type of image data.
* `[imgQ]`: The quality of the image data.
* `[imgS]`: The speed of the image data.
* `[imgD]`: The direction of the image data.
* `[imgH]`: The size of the image.
* `[imgG]`: The type of image data.
* `[imgF]`: The font for the game's difficulty level display.
* `[imgW]`: The font for the game's score display.
* `[imgR]`: The font for the game's settings.
* `[imgH]`: The size of the image.
* `[imgW]`: The type of image data.
* `[imgQ]`: The quality of the image data.
* `[imgS]`: The speed of the image data.
* `[imgD]`: The direction of the image data.
* `[imgH]`: The size of the image.
* `[imgG]`: The type of image data.
* `[imgF]`: The font for the game's difficulty level display.
* `[imgW]`: The font for the game's score display.
* `[imgR]`: The font for the game's settings.
* `[imgH]`: The size of the image.
* `[imgW]`: The type of image data.
* `[imgQ]`: The quality of the image data.
* `[imgS]`: The speed of the image data.
* `[imgD]`: The direction of the image data.
* `[imgH]`: The size of the image.
* `[imgG]`: The type of image data.
* `[imgF]`: The font for the game's difficulty level display.
* `[imgW]`: The font for the game's score display.
* `[imgR]`: The font for the game's settings.
* `[imgH]`: The size of the image.
* `[imgW]`: The type of image data.
* `[imgQ]`: The quality of the image data.
* `[imgS]`: The speed of the image data.
* `[imgD]`: The direction of the image data.
* `[imgH]`: The size of the image.
* `[imgG]`: The type of image data.
* `[imgF]`: The font for the game's difficulty level display.

The `imgdisplay` function


```py
if __name__ == "__main__":
    img_path = "datasets/guiqi/test/20_93.jpg"
    img = cv.imread(img_path)
    print(skill_height, skill_width)
    print(img.shape)
    skill_img = img[733: 793, 538:750, 2]
    img_show(skill_img)


    skill_imgA = img[dict["A"][0]: dict["A"][0]+skill_height, dict["A"][1]: dict["A"][1]+skill_width, 2]
    skill_imgH= img[dict["H"][0]: dict["H"][0]+skill_height, dict["H"][1]: dict["H"][1]+skill_width, 2]
    skill_imgG= img[dict["G"][0]: dict["G"][0]+skill_height, dict["G"][1]: dict["G"][1]+skill_width, 2]
    skill_imgE= img[dict["E"][0]: dict["E"][0]+skill_height, dict["E"][1]: dict["E"][1]+skill_width, 2]
    skill_imgQ= img[dict["Q"][0]: dict["Q"][0]+skill_height, dict["Q"][1]: dict["Q"][1]+skill_width, 2]
    skill_imgS= img[dict["S"][0]: dict["S"][0]+skill_height, dict["S"][1]: dict["S"][1]+skill_width, 2]
    skill_imgY= img[dict["Y"][0]: dict["Y"][0]+skill_height, dict["Y"][1]: dict["Y"][1]+skill_width, 2]
    skill_imgD = img[dict["D"][0]: dict["D"][0]+skill_height, dict["D"][1]: dict["D"][1]+skill_width, 2]
    skill_imgF = img[dict["F"][0]: dict["F"][0]+skill_height, dict["F"][1]: dict["F"][1]+skill_width, 2]
    skill_imgW = img[dict["W"][0]: dict["W"][0]+skill_height, dict["W"][1]: dict["W"][1]+skill_width, 2]
    skill_imgR = img[dict["R"][0]: dict["R"][0]+skill_height, dict["R"][1]: dict["R"][1]+skill_width, 2]

    # print("A", np.mean(skill_imgA))
    # print("H", np.mean(skill_imgH))
    # print("G", np.mean(skill_imgG))
    # print("E", np.mean(skill_imgE))
    # print("Q", np.mean(skill_imgQ))
    # print("S", np.mean(skill_imgS))
    # print("Y", np.mean(skill_imgY))

    print("A", score(skill_imgA))
    print("Q", score(skill_imgQ))
    print("S", score(skill_imgS))
    print("D", score(skill_imgD))
    print("F", score(skill_imgF))
    print("W", score(skill_imgW))
    print("R", score(skill_imgR))
    print("Y", score(skill_imgY))
    print("H", score(skill_imgH))
    print("G", score(skill_imgG))
    print("E", score(skill_imgE))

    print(skill_rec("W", img))


```

# `small_recgonize.py`

这段代码的主要作用是读取一张图片，并对这张图片中的目标进行定位。

首先，通过使用 cv2.imread() 函数从图片文件中读取图片数据，然后使用 cv.imshow() 函数来显示图片。

然后，定义了一个名为 img_show() 的函数，用于显示图片。

接着，定义了一个名为 current_door() 的函数，用于检测图片中目标的位置。函数中使用了 numpy 库中的 argmax() 函数来定位图片中的目标，并使用步长参数 stride 来对目标进行下采样，以获得每个房间的编号。最后，返回目标所在的房间数。


```py
import cv2 as cv
import numpy as np


img_path = "datasets/guiqi/test/61_93.jpg"
img = cv.imread(img_path)

def img_show(img):
    cv.imshow("win", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def current_door(img, stride = 17):
    crop = img[45:65, 1107:1270, 0]
    # img_show(crop)
    index = np.unravel_index(crop.argmax(), crop.shape)
    i = int((index[1] // stride) + 1)
    return i  # 返回的是在第几个房间

```

这段代码定义了一个名为 `next_door` 的函数，它接受一个图像（img）作为参数。函数的主要作用是查找图像中的目标区域（目标）并返回一个编号，该编号对应于目标区域在原始图像中的位置。

具体来说，以下是函数的主要步骤：

1. 从 `np.load()` 函数中加载一个名为 "问号模板.npy" 的模型。这个模型可能是一个预定义的结构方程模型（PEM）或机器学习模型，用于对目标区域进行定位和匹配。

2. 从输入图像（img）中提取目标区域（目标）并将其保存到一个新的变量（target）。

3. 使用 `cv.matchTemplate()` 函数查找目标区域（目标）在原始图像（img）中的位置。这个函数返回一个与目标区域匹配的分数分布（result），其中分数从0到`cv.NO_IVEGRET_VALUE` 范围。

4. 使用 `cv.normalize()` 函数将分数分布（result）归一化到 [0, 1] 范围内，消除最小值低于某个阈值（通常是 1e-10）的情况。

5. 使用 `cv.minMaxLoc()` 函数找到分数分布（result）中的最小值（min_val）和最大值（max_val），以及最小值（min_loc）和最大值（max_loc）在原始图像中的位置。

6. 如果最小值（min_val）低于预设的阈值（通常是 1e-10），函数将打印出最小值（min_val）并显示一个矩形框（theight, twidth）在目标区域（target）上。

7. 函数返回目标区域（target）中匹配结果（next_door_id）的值。

需要注意的是，这个函数的具体实现可能会因使用的模型和阈值不同而有所差异。


```py
def next_door(img):
    img_temp = np.load("问号模板.npy")
    # img_show(img_temp)
    target = img[45:65, 1107:1270]
    result = cv.matchTemplate(target, img_temp, cv.TM_SQDIFF_NORMED)
    cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    next_door_id = 0
    if min_val < 1e-10:
        # print(min_val, max_val, min_loc, max_loc)
        strmin_val = str(min_val)
        theight, twidth = img_temp.shape[:2]
        # cv.rectangle(target, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (225, 0, 0), 2)
        # cv.imshow("MatchResult----MatchingValue=" + strmin_val, target)
        # cv.waitKey()
        # cv.destroyAllWindows()
        next_door_id = int(((min_loc[0] + 0.5 * twidth) // 18.11) + 1)
    return next_door_id

```

这段代码的作用是：

1. 判断当前门的状态（通过调用函数door_status()来实现的），并输出当前门的图像。
2. 判断下一扇门的位置（通过调用函数door_position()来实现的），并输出下一扇门的图像。
3. 如果当前门和下一扇门的图像都已经准备好，那么调用函数show_image()来显示它们。
4. 如果当前门的状态为"open"，那么调用函数 template_file() 来将当前门的图像保存为模板文件。


```py
if __name__ == "__main__":
    print(current_door(img))
    print(next_door(img))
    # img_show(img[45:65, 1144:1162])
    # np.save("问号模板", img[45:65, 1144:1162])



```

# `yolo5_detect.py`

这段代码的作用是实现一个物体检测算法。它主要做了以下几件事情：

1. 导入必要的库：numpy, grabscreen, cv2, torch, torch.autograd, directkeys, utils.torch_utils, utils.general。

2. 从grabscreen中抓取屏幕截图，并转换为numpy数组。

3. 从屏幕截图中提取像素数据，并导入cv2库进行处理。

4. 从numpy数组中导入cv2处理后的像素数据，并使用cv2库的函数实现非最大抑制（non-maximum suppression, NMS）。

5. 将处理过的图像输入到torch的torchVision库中，并将图像的尺寸转换为torch.Size类型。

6. 在PyTorch中使用Variable和Torch.autograd实现物体检测算法的训练和优化。

7. 使用DirectKeys库实现对物体检测算法的实时控制，包括按下键盘上的键来激活检测。

8. 实现物体检测算法的训练和测试，并使用现有的物体检测算法进行物体检测。


```py
import numpy as np
from grabscreen import grab_screen
import cv2
import time
import directkeys
import torch
from torch.autograd import Variable
from directkeys import PressKey, ReleaseKey, key_down, key_up
from getkeys import key_check
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from models.experimental import attempt_load
import random

```




```py
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

```

这段代码是一个用于检测物体并在一张图片中绘制出物体的PyTorch实现。它主要分为以下几个部分：

1. 加载权重文件：`weights = r'E:\Computer_vision\yolov5\YOLO5\yolov5-master\runs\exp0\weights\best.pt'`。这个文件包含了训练好的物体检测模型（YOLOv5）的权重信息，是加载模型时需要使用的。

2. 设置设备：`device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")`。这段代码判断当前设备（CPU或GPU）是否支持 CUDA，如果支持，则使用 CUDA，否则使用 CPU。这里的作用是确保在训练模型时能够使用 GPU。

3. 加载预训练的模型：`model = attempt_load(weights, map_location=device)`。这段代码尝试使用 `weights` 中的权重文件加载一个预训练好的模型，并将其存储在 `model` 变量中。如果模型加载成功，则可以进行后续的训练和检测。

4. 设置窗口大小和最大物体数量：`window_size = (0,0,1200,750)`。这段代码定义了窗口的大小（宽度和高度）和能够包含的最大物体数量（20个窗口）。

5. 循环检测每一帧：`for i in list(range(5))[::-1]:`。这段代码循环遍历每个检测框的序号，从第 0 个开始，直到循环到第 4 个（最后一个循环用于绘制最后一个框）。

6. 在每次循环中，先检测是否存在物体：`last_time = time.time()`。这一步用于记录每个检测框的检测时间，以便在训练模型时知道是哪一帧的图片。

7. 如果存在物体，则绘制物体：`img_size = 608`。这一步定义了图片的大小，这段代码中的图片大小为 608x750。

8. 设置暂停和保存txt的变量：`paused = False` 和 `save_txt = False`。这两段代码设置了一个暂停标志（`paused`）和一个保存txt的变量（`save_txt`）。如果暂停标志为True，则说明检测过程暂停，当前图片暂时不绘制，但仍然保存了检测结果。如果检测成功，则可以将图片保存为txt文件。

9. 设置检测阈值：`conf_thres = 0.3` 和 `iou_thres = 0.2`。这两段代码设置了物体检测的阈值，用于判断当前检测到的框是否是真正的物体。具体来说，`conf_thres` 是用于设置物体宽度和高度的阈值（默认值是 0.3），而 `iou_thres` 是用于设置物体与背景的交集阈值（默认值是 0.2）。

10. 在每次循环中，先加载模型：`model = attempt_load(weights, map_location=device)`。这一步加载我们已经设置好的预训练模型，并将其存储在 `model` 变量中。

11. 循环遍历每个检测框：`for i in range(5):`。这一段代码循环遍历每个检测框的序号，从第 0 个开始，直到循环到第 4 个（最后一个循环用于绘制最后一个框）。

12. 如果存在物体，则绘制物体：`img_size = 608`。这一步定义了图片的大小，这段代码中的图片大小为 608x750。

13. 设置循环的步长：`for i in list(range(5))[::-1]:`。这一段代码中的 for循环步长为-1，说明两两成对出现，即每次循环包含一个前一个循环的最后一个元素和下一个循环的前一个元素。


```py
weights = r'E:\Computer_vision\yolov5\YOLO5\yolov5-master\runs\exp0\weights\best.pt'
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = attempt_load(weights, map_location=device)  # load FP32 model
window_size = (0,0,1200,750)
last_time = time.time()
for i in list(range(5))[::-1]:
    print(i + 1)
    time.sleep(1)
img_size = 608
paused = False
half = device.type != 'cpu'
view_img = True
save_txt = False
conf_thres = 0.3
iou_thres = 0.2
```

This code performs inference on a series of imageNet objects, using the Object Detection algorithm, and performs NMS on the detected boxes. The inference and NMS time are printed, and the detected boxes are displayed in the image if the "save\_txt" flag is set to 1. The boxes are also plotted in the image if the "view\_img" flag is set to 1.

The key\_check function is used to check if the user presses the 'q' key to quit the inference and it returns True.

It is important to note that this code snippet runs on a TensorFlow rather than PyTorch environment.


```py
classes = None
agnostic_nms = True
names = ['hero', 'small_map', "monster", 'money', 'material', 'door', 'BOSS', 'box', 'options']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
if half:
    model.half()  # to FP16

while (True):
    if not paused:
        img0 = grab_screen(window_size)
        print('loop took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)
        # Padded resize
        img = letterbox(img0, new_shape=img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device).unsqueeze(0)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        t1 = time_synchronized()
        # print(img.shape)
        pred = model(img, augment=False)[0]

        # Apply NMS
        det = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()
        print("inference and NMS time: ", t2 - t1)
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        det = det[0]
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # if save_txt:  # Write to file
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #     with open(txt_path + '.txt', 'a') as f:
                #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                if view_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=2)

        img0 = cv2.resize(img0, (600, 375))
        # Stream results
        if view_img:
            cv2.imshow('window', img0)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                raise StopIteration

        # Setting pause and unpause
        keys = key_check()
        if 'P' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                time.sleep(1)
```

# `models/common.py`

这段代码是一个PyTorch脚本，定义了一些通用的功能，可以被某些模型共享。具体来说，这段代码包含以下内容：

1. 引入math模块，用于某些数学计算。
2. 引入torch库，用于在PyTorch环境中操作。
3. 导入torch.nn模块，因为该模块是PyTorch中神经网络的标准接口，可以用来创建神经网络模型。
4. 从torch.nn.functional模块中导入非最大抑制（NMS）函数，用于在训练过程中去除模型的僵尸层。
5. 定义了一个autopad函数，用于对一个图形卷积层中的k值进行填充。该函数可以接受一个参数p，用于指定填充的边界。函数首先判断k是否为整数，如果是，则将k除以2并向上取整，否则创建一个包含k//2个x的列表。最后函数返回p。

这段代码的作用是为某些神经网络模型提供一些通用的功能，包括填充函数和NMS函数。


```py
# This file contains modules common to various models
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.general import non_max_suppression


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


```

这段代码定义了一个名为 DWConv 的函数，用于执行深度卷积操作。该函数接受两个输入参数：c1 和 c2，以及一个参数 k，表示卷积核的大小。函数还有一个参数 s，表示步长，用于控制每次卷积操作后步长的变化。最后一个参数 act，表示是否执行激活函数。

函数的实现基于一个名为 Conv 的类，该类继承自 PyTorch 中的 nn.Module 类。在 Conv 类中，有一个 standard_conv 方法，用于执行标准的卷积操作。该方法包含一个构造函数，用于设置卷积参数，以及一个 forward 方法，用于在 forward 方法中进行前向传播。

DWConv 函数的作用是在输入参数 c1 和 c2 上执行深度卷积操作，可以看作是在标准卷积操作的基础上增加了一个步长参数 s，用于控制每次卷积操作后步长的大小。通过传入不同的参数，可以对卷积操作的结果产生不同的影响。


```py
def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


```

This is a PyTorch implementation of the CSP (Convolutional Sub-pixel Evolution) model. CSP is a technique to evolve the feature maps of a given model by training a series of convolutional neural networks.

The CSP model takes a bottleneck network and adds a series of convolutional neural networks, where each block adds a new convolutional neural network with different filter sizes, followed by a ReLU activation function and a bottleneck average. The number of channels and the number of groups are specified by the user.

The CSP model can be used for image classification or other tasks.


```py
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


```

这段代码定义了两个类，SPP和Focus。SPP类是一个用于YOLOv3中的SPP（Spatial Pyramid Pooling）层的神经网络模型类，而Focus类则是一个将输入的目标特征映射到c空间的类。

具体来说，SPP类包含了一个在SPP层中使用的卷积层，以及一个在SPP层中使用的池化层。SPP层的输出是一个包含多个不同尺度的特征图，每个特征图都是通过对输入进行不同大小的池化操作得到的结果。SPP类中的一个重要参数是k，它定义了SPP层中使用的不同尺度的特征图的数量，以及每个特征图的大小和步长。

Focus类则包含一个卷积层，用于将输入的目标特征映射到c空间。Focus类的SPP层中的不同尺度的特征图数与SPP类中的相同，但是Focus类的SPP层中的特征图大小是针对每个特征图通道而不是每个输入样本。此外，Focus类中的SPP层中还包含一个归一化层，用于对特征图进行归一化处理，以便增强模型的稳定性。

SPP类和Focus类中的各个卷积层和池化层的参数都是根据需要来定义的，并且可以通过在__init__函数中进行修改来自定义这些参数。在训练时，这些参数都会在网络中实时更新，以帮助网络更好地捕捉输入数据中的特征信息。


```py
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


```



这段代码定义了两个类，一个是 `Concat`，另一个是 `NMS`。

`Concat` 类继承自 PyTorch 的 `nn.Module` 类，并重写了其 `__init__` 方法。该方法初始化了一个 `dim` 参数，表示要连接的维度数。`__init__` 方法在 `__init__` 函数中执行，因此可以设置 `dim` 参数来控制 concatenate 函数的行为。

`NMS` 类也继承自 `nn.Module` 类，并重写了其 `__init__` 方法。该方法和 `Concat` 类不同，该类包含一个 `conf` 参数和一个 `classes` 参数，用于设置非最大抑制 (NMS) 的参数。`__init__` 方法在 `__init__` 函数中执行，因此可以设置 `conf` 和 `classes` 参数来控制 NMS 函数的行为。

`Concat` 和 `NMS` 类都包含一个 `forward` 方法，用于在网络中前向传递数据。这些方法的实现超出了 `nn.Module` 的基本功能，因此需要手动编写。在这里，我会提供一些基本的指导，但是需要更多的上下文才能回答您的问题。


```py
class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.3  # confidence threshold
    iou = 0.6  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, dimension=1):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


```

这段代码定义了一个名为 "Flatten" 的类，该类继承自 PyTorch 中的nn.Module类。这个类实现了一个将输入张量 x  flatten（扁平化）至只有一维张量的函数，并返回此张量。

接着，定义了一个名为 "Classify" 的类，该类也继承自nn.Module类。这个类包含了一个可变的参数 ch1 和 c2，以及一个可变的参数 k、s、p 和 g。

在 Classify类的__init__方法中，首先调用父类的初始化函数，然后初始化自己的参数。

在 Classify类的forward函数中，首先将输入 x 应用AveragePool2d（平均池化）操作，将结果存储在变量 z 中。接着，将 x 应用Conv2d（卷积层）操作，将输入张量转换为只有一维张量，并添加到变量 z 中。最后，将通过 Flatten类创建的只有一维张量 z 返回。

这段代码的作用是，将输入张量 x 按照给定的 ch1 和 c2 参数进行扁平化操作，并返回扁平后的结果。通过 Classify类，用户可以对不同输入张量应用不同的模型，如 ResNet、VGG 等分类模型。


```py
class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


```

This is a PyTorch implementation of a neural network model called "FastFuse". This model is for image classification and has a catchy name because it is designed to be fast and efficient in training and deployment.

This model takes in an input image x and an image feature map W_features. It uses a two-stage attention mechanism to attend to different parts of the image and features map, as well as a channel attention mechanism to focus on specific channels of the input image.

The output of this model is a binary classification of the input image, with probability outputed as the output class. This model uses the information from both the attention mechanism and the channel attention mechanism to produce a clean and discriminative output.

This implementation assumes that the model has been trained on an image classification dataset and has been fine-tuned for a specific task. This implementation does not include any methods for training or deployment and is intended to be used as is in a production environment.


```py
# ===================
#     RGA Module
# ===================

class RGA_Module(nn.Module):
    def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True, \
                 cha_ratio=8, spa_ratio=8, down_ratio=8):
        super(RGA_Module, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial

        self.use_spatial = use_spatial
        self.use_channel = use_channel

        # print('Use_Spatial_Att: {};\tUse_Channel_Att: {}.'.format(self.use_spatial, self.use_channel))

        self.inter_channel = in_channel // cha_ratio
        self.inter_spatial = in_spatial // spa_ratio

        # Embedding functions for original features
        if self.use_spatial:
            self.gx_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.gx_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

        # Embedding functions for relation features
        if self.use_spatial:
            self.gg_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
        if self.use_channel:
            self.gg_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel * 2, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )

        # Networks for learning attention weights
        if self.use_spatial:
            num_channel_s = 1 + self.inter_spatial
            self.W_spatial = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_s // down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_s // down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )
        if self.use_channel:
            num_channel_c = 1 + self.inter_channel
            self.W_channel = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_c, out_channels=num_channel_c // down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_c // down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_c // down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )

        # Embedding functions for modeling relations
        if self.use_spatial:
            self.theta_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            self.phi_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.theta_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
            self.phi_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

    def forward(self, x):
        b, c, h, w = x.size()

        if self.use_spatial:
            # spatial attention
            theta_xs = self.theta_spatial(x)  # 1 20 32 32
            phi_xs = self.phi_spatial(x)  # 1 20 32 32
            theta_xs = theta_xs.view(b, self.inter_channel, -1)  # 1 20 32*32
            theta_xs = theta_xs.permute(0, 2, 1)  # 1 32*32 20
            phi_xs = phi_xs.view(b, self.inter_channel, -1)  # 1 20 32*32
            Gs = torch.matmul(theta_xs, phi_xs)  # 1 1024 1024
            Gs_in = Gs.permute(0, 2, 1).view(b, h * w, h, w)  # 1 1024 32 32 调换下顺序
            Gs_out = Gs.view(b, h * w, h, w)  # 1 1024 32 32
            Gs_joint = torch.cat((Gs_in, Gs_out), 1)  # 8 4096 64 32
            Gs_joint = self.gg_spatial(Gs_joint)  # 8 256 64 32

            g_xs = self.gx_spatial(x)  # 8 32 64 32
            g_xs = torch.mean(g_xs, dim=1, keepdim=True)  # 8 1 64 32
            ys = torch.cat((g_xs, Gs_joint), 1)  # 8 257 64 32

            W_ys = self.W_spatial(ys)  # 8 1 64 32
            if not self.use_channel:
                out = F.sigmoid(W_ys.expand_as(x)) * x  # 位置特征，不同特征图，位置相同的
                return out
            else:
                x = F.sigmoid(W_ys.expand_as(x)) * x
        if self.use_channel:
            # channel attention
            xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)  # 8 2048 256 1
            theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)  # 8 256 256
            phi_xc = self.phi_channel(xc).squeeze(-1)  # 8 256 256
            Gc = torch.matmul(theta_xc, phi_xc)  # 8 256 256
            Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)  # 8 256 256 1
            Gc_out = Gc.unsqueeze(-1)  # 8 256 256 1
            Gc_joint = torch.cat((Gc_in, Gc_out), 1)  # 8 512 256 1
            Gc_joint = self.gg_channel(Gc_joint)  # 8 32 256 1

            g_xc = self.gx_channel(xc)  # 8 256 256 1
            g_xc = torch.mean(g_xc, dim=1, keepdim=True)  # 8 1 256 1
            yc = torch.cat((g_xc, Gc_joint), 1)  # 8 33 256 1
            W_yc = self.W_channel(yc).transpose(1, 2)  # 8 256 1 1 得到权重分配
            out = F.sigmoid(W_yc) * x

            return out
```

# `models/experimental.py`

这段代码定义了一个名为 "CrossConv" 的 PyTorch 模型类，它用于实现 2D 卷积操作。这个模型类的实现基于两个卷积层，它们之间通过一个短路连接（shortcut）进行连接。短路连接在模型类中被称为 "self.add"，如果它为True，则会在输入 x 前面加入一个 leftshuffle 操作，然后对 x 进行两次卷积操作。如果短路连接为False，则不会对输入 x 进行额外的操作。

CrossConv 模型的输入是两个卷积层的输出，输出也是一个卷积层的输出。在 forward 方法中，第一个卷积层使用 idx 参数指定的卷积核（k=3, s=1）和步幅（expansion=1.0）进行前向传播，第二个卷积层使用 idx 参数指定的卷积核（k=3, s=1）和步幅（expansion=1.0）进行前向传播。然后，如果短路连接为True，则第一个卷积层的输出会被拼接到第二个卷积层的输入中，否则第二个卷积层的输出就是第二个卷积层的输入。


```py
# This file contains experimental modules

import numpy as np
import torch
import torch.nn as nn

from models.common import Conv, DWConv
from utils.google_utils import attempt_download


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


```

这段代码定义了一个名为 "C3" 的类，继承自 PyTorch 中的nn.Module类。

C3类包含了一个 CSP（Convolutional Sub-Pixel Convolutional）层，具有以下参数：

* c1：输入通道，输出通道和扩展参数
* c2：输出通道和扩展参数
* n：输入通道的数量
* shortcut：是否使用快捷通道（在输入通道数量较短时自动使用）
* g：输入通道和输出通道的 groups 数
* e：输入通道和输出通道的扩张比例，用于防止输入通道和输出通道的维度和平衡

在C3类中，还包含了一个名为 "forward" 的方法，用于将输入 x 传递给 CSP 层并返回结果。

该代码的作用是实现了一个具有 CSP 层的神经网络模型，主要应用于图像分割任务中。CSP 层可以在不同通道之间进行特征图的跨通道传递，同时避免了传统卷积层中的过拟合问题，具有较强的泛化能力。通过在不同通道之间应用 CSP 层，可以轻松地构建出具有多种不同特征描述的神经网络，例如，可以实现 RGB 和 depth 信息流的特征融合。


```py
class C3(nn.Module):
    # Cross Convolution CSP
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


```

这段代码定义了一个名为 "Sum" 的类，继承自 PyTorch 中的 nn.Module 类。这个类的实现与一个加权求和的操作，其输入是一个包含 n 个输入的序列 x，输出是 x 的第一个元素。

Sum 类的初始化方法接受一个参数 weight，表示是否应用权重。如果 weight 为 True，则使用一个从 1 到 n 的编号为 n/2 的正权重向量作为加权求和的第一项，权重向量还跨越了整个序列。如果 weight 为 False，则使用一个从 1 到 n 的标量作为加权求和的第一项。

Sum 类包含一个 forward 方法，用于前向传播输入序列 x 到输出。在 forward 方法中，如果没有应用权重，则执行以下操作：对于每个输入元素 i，将其与 x 的下一个元素 j 相乘，然后将其与输入向量中 i 对应的权重向量相加，得到当前输入元素 i 的前向传播结果。如果应用了体重，则将当前的权重向量乘以 sigmoid 函数，然后将其乘以 2。最后，返回输入序列 x 的第一个元素。


```py
class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


```

这段代码定义了一个名为 "GhostConv" 的类，用于实现 ghostnet 架构中的卷积神经网络。该类包含一个 "forward" 方法，用于前向传播数据。

具体来说，该代码首先定义了一个 "GhostConv" 类，该类继承自 PyTorch 中的 nn.Module 类。然后，在类的初始化函数中，定义了模型的参数，包括隐藏通道数、卷积核大小、步长、Groups 控制等。接着，创建了两个卷积层，用于对输入数据进行前向传递。最后，通过 concatenate 方法将两个卷积层的输出结果，作为模型的 forward 方法的前一个输出，用于快速获取输出。

接下来，定义了一个名为 "GhostBottleneck" 的类，用于实现 ghostnet 架构中的瓶颈模块。该类包含一个 "forward" 方法，用于前向传播数据。在该方法中，对输入数据首先通过一个 GhostConv 层进行前向传递，然后使用一个 DWConv（一种用于执行卷积操作的计算模块）来执行卷积操作。接着，使用一个 DBConv（一种特殊的 DWConv，用于执行刘备操作）来执行偏移量计算，并将其结果与输入数据相加。最后，返回前向传递的结果。


```py
class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k, s):
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


```

这段代码定义了一个名为MixConv2d的类，继承自PyTorch中的nn.Module类。这个类在图像处理领域中用于实现深度卷积操作。

MixConv2d类包含了一个初始化函数__init__，这个函数在创建MixConv2d实例时需要传入参数c1、c2和k，其中c1和c2是输入张量的大小，k是卷积核的尺寸，可以是任意实数。此外，参数s表示是否等权，如果为True，则每个卷积层中的通道数相等。

MixConv2d类中还包含了一个__repr__函数，这个函数返回的是MixConv2d实例的哈希表示，可以用于将MixConv2d实例存储在PyTorch中的一个字典中，如：

model = MixConv2d(c1, c2, k=((1, 3), 2), s=True, equal_ch=True)

MixConv2d类中的M混合层是该类实例中的主要卷积层，M混合层包含了一个卷积核函数，该函数使用了一个Mix机制来计算每个卷积层的通道数。在__init__函数中，首先创建了一个M mix层，然后使用这个M mix层来计算每个卷积层的通道数，最后将计算得到的通道数存储在实例变量m中。

在forward函数中，对输入的张量x进行加法操作，并使用MixConv2d实例中的act函数对结果进行非线性激活。最后，使用MixConv2d实例中的bn函数对卷积层中的张量进行归一化，然后将计算得到的张量x + act(bn(torch.cat(m(x), 1)))返回。


```py
class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


```

这段代码定义了一个名为 `Ensemble` 的类，继承自 `nn.ModuleList` 类。这个类的创建和普通 `nn.ModuleList` 类类似，但会在其中添加一个前向传播函数 `forward`。

`Ensemble` 类的 `__init__` 函数接受一个参数 `x`，如果需要，还会传递一个 `augment` 参数，表示是否需要对输入数据进行增强。`__forward__` 函数用于将输入数据 `x` 传递给每个 `Ensemble` 中的模型，并将每个模型的输出存储在一个列表 `y` 中。最后，`Ensemble` 类还提供了前向传播函数 `forward`，用于在训练和推理时输出一个 tensor。


```py
class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        y = torch.cat(y, 1)  # nms ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output


```



这段代码定义了一个函数 `attempt_load`，它接受一个或多个模型权重(即一个或多个模型的权重列表)，并返回一个 ensemble 对象。

函数中首先定义了一个 ensemble 对象 `model`，然后遍历输入的每个权重，对于每个权重，尝试从不同的位置(即 `map_location` 参数)加载该权重，然后将模型的权重(即模型的 FP32 版本的权重)加载到 `model` 中。

接着，如果加载的权重数量为 1，函数将直接返回该权重对应的模型。否则，函数将打印出所有加载的模型的名称，并将模型中的所有名称(即字符串)设置为它们所对应的键，以使函数可以可读性更好。最后，函数返回一个 ensemble 对象。

总的来说，这段代码的作用是加载多个模型的权重，并返回由这些模型组成的 ensemble 对象。


```py
def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble

```

# `models/export.py`

这段代码是一个用于将YOLOv5模型导出为ONNX和TorchScript格式的Python脚本。它使用PyTorch中的`argparse`模块来解析用户输入的参数，并使用`subprocess`模块从命令行中读取这些参数。

具体来说，这段代码首先将当前工作目录（即当前目录）添加到PyTorch中的`sys.path`列表中，以便在运行脚本时可以访问到模型文件和数据集。然后它使用`argparse`模块中的`parser`函数来解析用户输入的参数，其中`weights`参数指定了要导出的模型权重文件的位置，`img`参数指定了图像的大小，`batch`参数指定了批处理大小。

在导出模型之后，它会将模型文件导出为ONNX和TorchScript格式。ONNX是一种用于表示PyTorch模型的接口，而TorchScript则是一种用于将PyTorch模型打包成可执行文件的方法。


```py
"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""
#首先pip install onnx
import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
sys.path.append('../') 
import torch
import torch.nn as nn

```



This is a script that exports a PyTorch model to ONNX, and optionally to CoreML. The script takes as input a trained PyTorch model (as saved by the `torchscript` tool), an image, and optionally provides the path to aweights file.

The script first converts the PyTorch model to ONNX format, by applying a pixel scaling factor to the input image. This is done by replacing the last layer of the model with a custom `ImageType` that specifies the input shape, scale, and scaling factor. The `save` method of the ONNX model is then called to save the converted model to a file.

If an image is provided, it is first converted to the ONNX format, and then saved to a file. Finally, the script exports the model to CoreML. This is done by converting the ONNX model to a `coreml.Model`, and then saving it to a file.

The script also prints the time taken to perform the export operations.

Overall, this script appears to be a useful tool for exporting a PyTorch model to ONNX and CoreML.


```py
import models
from models.experimental import attempt_load
from utils.activations import Hardswish
from utils.general import set_logging, check_img_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv) and isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()  # assign activation
        # if isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run

    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img)
        ts.save(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # CoreML export
    try:
        import coremltools as ct

        print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
        # convert model from torchscript and apply pixel scaling as per detect.py
        model = ct.convert(ts, inputs=[ct.ImageType(name='image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        f = opt.weights.replace('.pt', '.mlmodel')  # filename
        model.save(f)
        print('CoreML export success, saved as %s' % f)
    except Exception as e:
        print('CoreML export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))

```

# `models/yolo.py`

这段代码使用了多个Python库，包括argparse、logging和math，以及来自torch和torchvision的几个模块。它的作用是创建一个用于图像分类任务的神经网络模型。

具体来说，它实现了一个具有多个层的高分辨率图像分类模型。从左到右数，这些层包括：

1. Conv层：通过`argparse`库导入的一个自定义的卷积层，其中包含多个参数。
2. Bottleneck层：通过`argparse`库导入的一个自定义的卷积层，其中包含多个Bottleneck模块。
3. SPP层：通过`argparse`库导入的一个自定义的全连接层，其中包含多个SPP模块。
4. DWConv层：通过`argparse`库导入的一个自定义的卷积层，其中包含多个DWConv模块。
5. Focus层：通过`argparse`库导入的一个自定义的卷积层，其中包含多个Focus模块。
6. BottleneckCSP层：通过`argparse`库导入的一个自定义的卷积层，其中包含多个BottleneckCSP模块。
7. Concat层：通过`argparse`库导入的一个自定义的卷积层，其中包含多个Concat模块。
8. NMS层：通过`argparse`库导入的一个自定义的卷积层，其中包含多个NMS模块。
9. RGA_Module层：通过`argparse`库导入的一个自定义的卷积层，其中包含多个RGA_Module模块。

在训练期间，这些模块被组合在一起作为一个神经网络，并在输出时进行一些额外的处理。


```py
import argparse
import logging
import math
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat, NMS, RGA_Module
from models.experimental import MixConv2d, CrossConv, C3
```

This is a PyTorch implementation of a 2D convolutional neural network (CNN) model. The model has a defined forward method and a forward function.

The model consists of a convolutional neural network and a fully connected (linear) layer. The convolutional neural network has a two-dimensional grid with a specified number of channels (20 in this case), a stride size, and an anchor dimension. The convolutional neural network applies two standard convolutional layers with a ReLU activation function in between. The output of the convolutional neural network is then fed into a fully connected (linear) layer.

The forward method takes a input tensor `x`, and the output is determined by the parameters of the convolutional neural network. The output is a tensor `z` that has the same shape as the input tensor `x`. If the `model.training` flag is `False`, the output tensor `z` will have the same shape as the input tensor `x`; otherwise, it will have shape `(batch_size, shape(batch_size, input_shape(x), shape(batch_size,), shape(batch_size, input_shape(x), shape(batch_size))))`.

The forward function applies the convolutional neural network to the input tensor `x` and returns the output tensor `z`.


```py
from utils.general import check_anchor_order, make_divisible, check_file, set_logging
from utils.torch_utils import (
    time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, select_device)


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


```

This is a PyTorch implementation of a customizable Conv2d model. The model has a top-level node called `Bottleneck`, which is a combination of a Conv2d layer, a BatchNorm2d layer, and a shortcut to the input.

The `Bottleneck` node takes two arguments: `self` and `num_classes`, and returns an instance of the `Bottleneck` class.

The `Bottleneck` node is designed to take in a pre-trained model and apply持久的 buffering to the convolutional feature map, allowing for faster inference.

The `Bottleneck` node also provides an optional method called `fuse()`, which fuses the model layers to save memory and improve inference speed.

The `Bottleneck` node has a dependency called `NMS()`, which is used to perform non-maximum suppression (NMS) on object detection outputs.

The `NMS()` method has a dependency called `f()`, which returns the negative log likelihood of each detected object.

The `NMS()` method also has a dependency called `i` and `w` arguments, which are the current index and weight of the detected objects.

The `NMS()` method has a constructor and an initializer, both of which take no arguments.

The `NMS()` method has a `model.add_module()` method, which adds the module to the model.

The `NMS()` method has a `model.info()` method, which prints information about the model.

The `NMS()` method has a `model.fuse()` method, which fuses the model layers to improve inference speed.


```py
class Model(nn.Module):
    def __init__(self, img_size, cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), img_size, ch=[ch])  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = img_size[0]  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        print('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        i = 1
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))
            x = m(x)  # run
            #print('层数：',i,'特征图大小：',x.shape)
            i+=1
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def add_nms(self):  # fuse model Conv2d() + BatchNorm2d() layers
        if type(self.model[-1]) is not NMS:  # if missing NMS
            print('Adding NMS module... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
        return self

    def info(self, verbose=False):  # print model information
        model_info(self, verbose)


```

This is a Python implementation of a PyTorch custom layer implementation. The layer subclass is implementing the make_divisible and concatenation operations.

The `make_divisible` function takes in a layer, a divisor, and an optional integer `电源参数` (default is 2). It returns a new layer that has all elements of the original layer divided by the specified divisor, while preserving the largest possible remainder.

The `concat` function takes in a list of integers or a tuple of integers, and concatenates them. If the input is a tuple, it takes the first element as a scalar value and the second element as a tuple.

The `nn.Module` is a base class from PyTorch that defines the structure of a custom neural network module. The `nn.Sequential` class is a type of `nn.Module` that can be used to build up custom neural network layers.

The `Detect` class is a subclass of `nn.Module` that implements a custom neural network module for object detection. The `nn.Sequential` class can be used to build up custom neural network layers, and the `m()` method can be used to apply an operation to the parameters of a layer.

The `nn.BatchNorm2d` class is a subclass of `nn.Module` that implements a custom neural network module for batch normalization. The `nn.Conv2d` class is a subclass of `nn.Module` that implements a custom neural network module for convolutional neural networks. The `nn.MaxPool2d` class is a subclass of `nn.Module` that implements a custom neural network module for max pooling.


```py
def parse_model(d, img_size, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is RGA_Module:
            args = [round(gw * args[0]), (img_size[0]//args[1])*(img_size[1]//args[1])]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


```

这段代码是一个PyTorch脚本，其主要作用是定义和训练一个目标检测模型。它使用了Argon2（ONNX）文档生成器和summarymerator库，以及PyTorch中的一些方便函数。以下是该代码的详细解释：

1. 初始化：定义了两个参数文件（.yaml），用于配置目标检测模型的参数。这些参数文件将包含模型架构、损失函数、优化器等。

2. 创建 ArgumentParser 对象：用于创建命令行参数的解析器。

3. 添加参数：在 ArgumentParser 实例中添加了两个参数：--cfg 和 --device。通过调用 ArgumentParser.parse_args() 方法，将用户提供的参数解析并保存到 opt 变量中。

4. 检查配置文件：使用 check_file() 函数检查 .yaml 文件是否正确配置。如果配置文件不正确，程序将崩溃并退出。

5. 设置日志记录：使用 set_logging() 函数设置日志记录的最低级别，以便记录更多的错误信息。

6. 选择 GPU：通过 --device 参数指定使用 GPU 而不是 CPU。

7. 加载预训练模型：使用 Model.from_pretrained() 方法加载预定义的 PyTorch 模型，并将其转换到指定的设备上。

8. 训练模型：使用 Model.train() 方法进入训练模式，以便在运行时创建新的数据张量。

9. 创建数据张量：根据需要从用户那里获取图像数据，并将其转换为张量。

10. 运行 ONNX 导出：根据 --device 参数指定使用 GPU。然后，通过 torch.onnx.export() 函数将模型导出为 ONNX 张量，以便在其他设备上运行。

11. 创建并添加数据到 TensorBoard：使用 SummaryWriter 和 add_image 和 add_graph 方法将数据添加到 TensorBoard。如果 TensorBoard 无法运行，程序将崩溃并退出。


```py
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # ONNX export
    # model.model[-1].export = True
    # torch.onnx.export(model, img, opt.cfg.replace('.yaml', '.onnx'), verbose=True, opset_version=11)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard

```