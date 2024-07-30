# `.\comic-translate\modules\ocr\pororo\pororo\models\brainOCR\recognition.py`

```py
# 从标准库中导入数学模块
import math

# 导入 NumPy 库，并用 np 别名表示
import numpy as np

# 导入 PyTorch 库
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms

# 从 PIL 库中导入 Image 模块
from PIL import Image

# 从当前包中导入 Model 类
from .model import Model

# 从当前包中导入 CTCLabelConverter 函数
from .utils import CTCLabelConverter

# 定义一个函数 contrast_grey，用于计算灰度图像的对比度信息
def contrast_grey(img):
    # 计算图像的高百分位值（90% 分位）和低百分位值（10% 分位）
    high = np.percentile(img, 90)
    low = np.percentile(img, 10)
    # 返回对比度值及其对应的高低百分位值
    return (high - low) / np.maximum(10, high + low), high, low

# 定义一个函数 adjust_contrast_grey，用于调整灰度图像的对比度到目标值
def adjust_contrast_grey(img, target: float = 0.4):
    # 获得图像的对比度及其高低百分位值
    contrast, high, low = contrast_grey(img)
    # 如果图像的对比度低于目标值，则进行对比度调整
    if contrast < target:
        # 将图像转换为整数类型
        img = img.astype(int)
        # 计算调整比例
        ratio = 200.0 / np.maximum(10, high - low)
        # 对图像进行调整
        img = (img - low + 25) * ratio
        img = np.maximum(
            np.full(img.shape, 0),
            np.minimum(
                np.full(img.shape, 255),
                img,
            ),
        ).astype(np.uint8)
    # 返回调整后的图像
    return img

# 定义一个类 NormalizePAD，用于将图像标准化并进行填充处理
class NormalizePAD(object):

    def __init__(self, max_size, PAD_type: str = "right"):
        # 初始化图像转换器为 ToTensor 类型
        self.toTensor = transforms.ToTensor()
        # 设置最大尺寸
        self.max_size = max_size
        # 计算最大宽度的一半
        self.max_width_half = math.floor(max_size[2] / 2)
        # 设置填充类型
        self.PAD_type = PAD_type

    def __call__(self, img):
        # 将图像转换为 Tensor 类型并标准化
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        # 获取图像的通道数、高度和宽度
        c, h, w = img.size()
        # 创建填充后的图像对象
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        # 将图像内容复制到填充后的图像对象中（右填充）
        Pad_img[:, :, :w] = img
        # 如果图像宽度不等于最大宽度，则添加边界填充
        if self.max_size[2] != w:
            Pad_img[:, :, w:] = (img[:, :, w - 1].unsqueeze(2).expand(
                c,
                h,
                self.max_size[2] - w,
            ))
        # 返回填充后的图像对象
        return Pad_img

# 定义一个类 ListDataset，用于创建图像数据集
class ListDataset(torch.utils.data.Dataset):

    def __init__(self, image_list: list):
        # 初始化图像列表和样本数
        self.image_list = image_list
        self.nSamples = len(image_list)

    def __len__(self):
        # 返回数据集的长度
        return self.nSamples

    def __getitem__(self, index):
        # 获取指定索引位置的图像并返回成 PIL Image 格式
        img = self.image_list[index]
        return Image.fromarray(img, "L")

# 定义一个类 AlignCollate，用于对齐和整理图像数据
class AlignCollate(object):

    def __init__(self, imgH: int, imgW: int, adjust_contrast: float):
        # 设置图像的目标高度和宽度
        self.imgH = imgH
        self.imgW = imgW
        # 保持比例并进行填充
        self.keep_ratio_with_pad = True  # Do Not Change
        # 设置图像对比度调整的目标值
        self.adjust_contrast = adjust_contrast
    # 定义一个方法，使对象可以像函数一样被调用，处理输入的批次数据
    def __call__(self, batch):
        # 使用 lambda 函数过滤掉批次中为 None 的元素
        batch = filter(lambda x: x is not None, batch)
        # 将过滤后的批次赋值给 images 变量
        images = batch

        # 设置调整后图像的最大宽度为 imgW
        resized_max_w = self.imgW
        # 输入图像的通道数设为 1
        input_channel = 1
        # 创建图像转换对象，设置图像归一化和填充尺寸
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        # 初始化空列表用于存放调整后的图像
        resized_images = []
        # 遍历批次中的每张图像
        for image in images:
            # 获取当前图像的宽度 w 和高度 h
            w, h = image.size
            # 如果需要调整对比度
            if self.adjust_contrast > 0:
                # 将图像转换为灰度图并转为 NumPy 数组
                image = np.array(image.convert("L"))
                # 调整图像的对比度
                image = adjust_contrast_grey(image, target=self.adjust_contrast)
                # 将 NumPy 数组转换为 PIL 图像
                image = Image.fromarray(image, "L")

            # 计算图像的宽高比
            ratio = w / float(h)
            # 如果按比例调整后的宽度超过了设定的最大宽度 imgW
            if math.ceil(self.imgH * ratio) > self.imgW:
                # 将调整后的宽度设置为 imgW
                resized_w = self.imgW
            else:
                # 否则按比例调整宽度
                resized_w = math.ceil(self.imgH * ratio)

            # 使用双三次插值法调整图像尺寸为 (resized_w, self.imgH)
            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            # 将调整后的图像应用预定义的转换
            resized_images.append(transform(resized_image))

        # 将调整后的图像堆叠成张量，增加一个维度
        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        # 返回处理后的图像张量
        return image_tensors
def get_text(image_list, recognizer, converter, opt2val: dict):
    # 获取图像宽度和高度
    imgW = opt2val["imgW"]
    imgH = opt2val["imgH"]
    # 获取图像对比度调整参数
    adjust_contrast = opt2val["adjust_contrast"]
    # 获取批处理大小和工作线程数
    batch_size = opt2val["batch_size"]
    n_workers = opt2val["n_workers"]
    # 获取对比度阈值
    contrast_ths = opt2val["contrast_ths"]

    # TODO: 弄清楚这段代码的作用
    # batch_max_length = int(imgW / 10)

    # 提取图像列表中的坐标和图像数据
    coord = [item[0] for item in image_list]
    img_list = [item[1] for item in image_list]
    
    # 创建用于对齐和拼合的数据集
    AlignCollate_normal = AlignCollate(imgH, imgW, adjust_contrast)
    # 使用图像列表创建 ListDataset 对象
    test_data = ListDataset(img_list)
    test_loader = torch.utils.data.DataLoader(
        test_data,                       # 使用给定的测试数据创建一个数据加载器
        batch_size=batch_size,           # 指定批处理大小
        shuffle=False,                   # 禁用数据重排，保持顺序
        num_workers=n_workers,           # 指定用于数据加载的工作线程数目
        collate_fn=AlignCollate_normal,  # 使用指定的数据对齐函数对数据进行处理
        pin_memory=True,                 # 将数据加载到 CUDA 固定内存中，提高效率
    )

    # 预测第一轮
    result1 = recognizer_predict(recognizer, converter, test_loader, opt2val)

    # 预测第二轮
    low_confident_idx = [
        i for i, item in enumerate(result1) if (item[1] < contrast_ths)
    ]
    if len(low_confident_idx) > 0:
        # 提取低置信度索引对应的图像列表
        img_list2 = [img_list[i] for i in low_confident_idx]
        # 创建用于对比增强的数据对齐函数
        AlignCollate_contrast = AlignCollate(imgH, imgW, adjust_contrast)
        # 使用新的图像列表创建测试数据集
        test_data = ListDataset(img_list2)
        # 使用新的数据加载器加载测试数据
        test_loader = torch.utils.data.DataLoader(
            test_data,                       # 使用给定的测试数据创建一个数据加载器
            batch_size=batch_size,           # 指定批处理大小
            shuffle=False,                   # 禁用数据重排，保持顺序
            num_workers=n_workers,           # 指定用于数据加载的工作线程数目
            collate_fn=AlignCollate_contrast, # 使用对比增强的数据对齐函数对数据进行处理
            pin_memory=True,                 # 将数据加载到 CUDA 固定内存中，提高效率
        )
        # 对低置信度图像进行第二轮预测
        result2 = recognizer_predict(recognizer, converter, test_loader,
                                     opt2val)

    # 合并两轮预测结果
    result = []
    for i, zipped in enumerate(zip(coord, result1)):
        box, pred1 = zipped
        if i in low_confident_idx:
            pred2 = result2[low_confident_idx.index(i)]
            # 比较两轮预测的置信度并选择更高的作为最终结果
            if pred1[1] > pred2[1]:
                result.append((box, pred1[0], pred1[1]))
            else:
                result.append((box, pred2[0], pred2[1]))
        else:
            result.append((box, pred1[0], pred1[1]))

    return result
```