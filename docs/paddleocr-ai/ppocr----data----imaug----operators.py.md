# `.\PaddleOCR\ppocr\data\imaug\operators.py`

```
"""
# 版权声明
# 2020年版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何担保或条件，无论是明示的还是暗示的。
# 请查看许可证以获取特定语言的权限和限制。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 导入所需的库
import sys
import six
import cv2
import numpy as np
import math
from PIL import Image

# 定义一个 DecodeImage 类，用于解码图像
class DecodeImage(object):
    """ decode image """

    # 初始化方法，设置图像模式、通道顺序和是否忽略方向
    def __init__(self,
                 img_mode='RGB',
                 channel_first=False,
                 ignore_orientation=False,
                 **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first
        self.ignore_orientation = ignore_orientation
    # 定义一个类方法，用于处理输入的数据
    def __call__(self, data):
        # 从输入数据中获取图像数据
        img = data['image']
        # 在 Python 2 中，图像数据应该是字符串类型且长度大于0
        if six.PY2:
            assert type(img) is str and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        # 在 Python 3 中，图像数据应该是字节类型且长度大于0
        else:
            assert type(img) is bytes and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        # 将字节数据转换为 numpy 数组
        img = np.frombuffer(img, dtype='uint8')
        # 如果忽略方向信息，则使用 cv2.IMREAD_IGNORE_ORIENTATION 和 cv2.IMREAD_COLOR 参数解码图像
        if self.ignore_orientation:
            img = cv2.imdecode(img, cv2.IMREAD_IGNORE_ORIENTATION |
                               cv2.IMREAD_COLOR)
        # 否则，使用默认参数解码图像
        else:
            img = cv2.imdecode(img, 1)
        # 如果图像为空，则返回空值
        if img is None:
            return None
        # 如果图像模式为灰度，则将图像转换为灰度图像
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 如果图像模式为 RGB，则确保图像通道数为3，然后将通道顺序反转
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]

        # 如果通道优先标志为真，则将图像通道维度置于第一维
        if self.channel_first:
            img = img.transpose((2, 0, 1))

        # 更新输入数据中的图像数据为处理后的图像数据
        data['image'] = img
        # 返回处理后的数据
        return data
# 定义一个类，用于对图像进行归一化处理，包括减去均值、除以标准差
class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        # 如果 scale 是字符串，则将其转换为对应的值
        if isinstance(scale, str):
            scale = eval(scale)
        # 设置 scale，默认为 1.0 / 255.0
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        # 设置均值，默认为 [0.485, 0.456, 0.406]
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        # 设置标准差，默认为 [0.229, 0.224, 0.225]
        std = std if std is not None else [0.229, 0.224, 0.225]

        # 根据 order 设置 shape
        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        # 将均值转换为 numpy 数组，并设置为 float32 类型
        self.mean = np.array(mean).reshape(shape).astype('float32')
        # 将标准差转换为 numpy 数组，并设置为 float32 类型
        self.std = np.array(std).reshape(shape).astype('float32')

    # 对数据进行归一化处理
    def __call__(self, data):
        img = data['image']
        from PIL import Image
        # 如果图像是 PIL.Image 类型，则转换为 numpy 数组
        if isinstance(img, Image.Image):
            img = np.array(img)
        # 断言图像是 numpy 数组类型
        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
        # 对图像进行归一化处理
        data['image'] = (img.astype('float32') * self.scale - self.mean) / self.std
        return data


# 定义一个类，用于将 hwc 格式的图像转换为 chw 格式
class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    # 对数据进行转换
    def __call__(self, data):
        img = data['image']
        from PIL import Image
        # 如果图像是 PIL.Image 类型，则转换为 numpy 数组
        if isinstance(img, Image.Image):
            img = np.array(img)
        # 将图像转置为 chw 格式
        data['image'] = img.transpose((2, 0, 1))
        return data


# 定义一个类，用于加载 fasttext 模型并进行预测
class Fasttext(object):
    def __init__(self, path="None", **kwargs):
        import fasttext
        # 加载 fasttext 模型
        self.fast_model = fasttext.load_model(path)

    # 对数据进行预测
    def __call__(self, data):
        label = data['label']
        # 使用 fasttext 模型进行预测
        fast_label = self.fast_model[label]
        data['fast_label'] = fast_label
        return data


# 定义一个类，用于保留指定的键对应的值
class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    # 保留指定键对应的值
    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


# 定义一个类，用于填充数据
class Pad(object):
    # 初始化函数，设置图像大小和大小的倍数
    def __init__(self, size=None, size_div=32, **kwargs):
        # 如果指定了图像大小且类型不是整数、列表或元组，则抛出类型错误异常
        if size is not None and not isinstance(size, (int, list, tuple)):
            raise TypeError("Type of target_size is invalid. Now is {}".format(
                type(size)))
        # 如果图像大小是整数，则转换为列表
        if isinstance(size, int):
            size = [size, size]
        # 设置图像大小和大小的倍数
        self.size = size
        self.size_div = size_div

    # 调用函数，对输入的数据进行处理
    def __call__(self, data):

        # 获取输入数据中的图像
        img = data['image']
        img_h, img_w = img.shape[0], img.shape[1]
        # 如果指定了图像大小
        if self.size:
            resize_h2, resize_w2 = self.size
            # 断言目标大小的高度和宽度应大于图像的高度和宽度
            assert (
                img_h < resize_h2 and img_w < resize_w2
            ), '(h, w) of target size should be greater than (img_h, img_w)'
        else:
            # 计算调整后的高度和宽度，使其为大小的倍数
            resize_h2 = max(
                int(math.ceil(img.shape[0] / self.size_div) * self.size_div),
                self.size_div)
            resize_w2 = max(
                int(math.ceil(img.shape[1] / self.size_div) * self.size_div),
                self.size_div)
        # 使用常数值0对图像进行边界填充
        img = cv2.copyMakeBorder(
            img,
            0,
            resize_h2 - img_h,
            0,
            resize_w2 - img_w,
            cv2.BORDER_CONSTANT,
            value=0)
        # 更新数据中的图像
        data['image'] = img
        # 返回处理后的数据
        return data
# 定义一个 Resize 类，用于调整图像大小
class Resize(object):
    # 初始化方法，设置默认大小为 (640, 640)
    def __init__(self, size=(640, 640), **kwargs):
        self.size = size

    # 调整图像大小的方法
    def resize_image(self, img):
        # 获取目标大小
        resize_h, resize_w = self.size
        # 获取原始图像大小
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        # 计算高度和宽度的缩放比例
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        # 调整图像大小
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # 返回调整后的图像和缩放比例
        return img, [ratio_h, ratio_w]

    # 调用对象时执行的方法
    def __call__(self, data):
        # 获取图像数据
        img = data['image']
        # 如果数据中包含多边形信息
        if 'polys' in data:
            text_polys = data['polys']

        # 调整图像大小
        img_resize, [ratio_h, ratio_w] = self.resize_image(img)
        # 如果数据中包含多边形信息
        if 'polys' in data:
            new_boxes = []
            # 遍历每个多边形的顶点
            for box in text_polys:
                new_box = []
                for cord in box:
                    # 根据缩放比例调整顶点坐标
                    new_box.append([cord[0] * ratio_w, cord[1] * ratio_h])
                new_boxes.append(new_box)
            # 更新多边形信息
            data['polys'] = np.array(new_boxes, dtype=np.float32)
        # 更新图像数据
        data['image'] = img_resize
        # 返回更新后的数据
        return data


# 定义一个 DetResizeForTest 类，用于测试时调整检测结果的大小
class DetResizeForTest(object):
    # 初始化方法
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        # 设置默认参数
        self.resize_type = 0
        self.keep_ratio = False
        # 如果传入了图像形状参数
        if 'image_shape' in kwargs:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
            # 如果传入了保持比例参数
            if 'keep_ratio' in kwargs:
                self.keep_ratio = kwargs['keep_ratio']
        # 如果传入了限制边长参数
        elif 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        # 如果传入了调整长边参数
        elif 'resize_long' in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get('resize_long', 960)
        # 否则设置默认参数
        else:
            self.limit_side_len = 736
            self.limit_type = 'min'
    # 定义一个类方法，用于对输入数据进行处理
    def __call__(self, data):
        # 获取输入数据中的图像数据
        img = data['image']
        # 获取图像的高度、宽度和通道数
        src_h, src_w, _ = img.shape
        # 如果图像的高度和宽度之和小于64，则进行填充操作
        if sum([src_h, src_w]) < 64:
            img = self.image_padding(img)

        # 根据不同的resize类型进行图像的resize操作
        if self.resize_type == 0:
            # 调用resize_image_type0方法进行图像resize，并获取resize比例
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            # 调用resize_image_type2方法进行图像resize，并获取resize比例
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            # 调用resize_image_type1方法进行图像resize，并获取resize比例
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        # 更新输入数据中的图像数据
        data['image'] = img
        # 更新输入数据中的形状信息，包括原始高度、宽度以及resize比例
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        # 返回处理后的数据
        return data

    # 定义一个方法，用于对图像进行填充操作
    def image_padding(self, im, value=0):
        # 获取图像的高度、宽度和通道数
        h, w, c = im.shape
        # 创建一个指定大小的全零矩阵，并填充指定值
        im_pad = np.zeros((max(32, h), max(32, w), c), np.uint8) + value
        # 将原始图像数据填充到新矩阵中
        im_pad[:h, :w, :] = im
        # 返回填充后的图像数据
        return im_pad

    # 定义一个方法，用于对图像进行resize操作（类型1）
    def resize_image_type1(self, img):
        # 获取目标resize后的高度和宽度
        resize_h, resize_w = self.image_shape
        # 获取原始图像的高度和宽度
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        # 如果需要保持宽高比，则重新计算resize宽度
        if self.keep_ratio is True:
            resize_w = ori_w * resize_h / ori_h
            N = math.ceil(resize_w / 32)
            resize_w = N * 32
        # 计算resize比例
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        # 对图像进行resize操作
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # 返回resize后的图像数据和resize比例
        return img, [ratio_h, ratio_w]
    # 将图像调整为网络所需的大小，即32的倍数
    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        # 获取图像的高度、宽度和通道数
        limit_side_len = self.limit_side_len
        h, w, c = img.shape

        # 限制最大边
        if self.limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        # 限制最小边
        elif self.limit_type == 'min':
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        # 调整长边
        elif self.limit_type == 'resize_long':
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise Exception('not support limit type, image ')
        # 计算调整后的高度和宽度
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        # 将调整后的高度和宽度调整为32的倍数
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        try:
            # 如果调整后的宽度或高度小于等于0，则返回空值
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            # 调整图像大小
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            # 捕获异常并打印相关信息
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        # 计算高度和宽度的比例
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]
    # 定义一个方法用于调整图像大小
    def resize_image_type2(self, img):
        # 获取图像的高度、宽度和通道数
        h, w, _ = img.shape

        # 初始化调整后的宽度和高度为原始宽度和高度
        resize_w = w
        resize_h = h

        # 根据高度和宽度的比较确定缩放比例
        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        # 根据比例调整高度和宽度
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio

        # 设置最大步长为128，将调整后的高度和宽度调整为128的倍数
        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride

        # 使用OpenCV的resize函数调整图像大小
        img = cv2.resize(img, (int(resize_w), int(resize_h)))

        # 计算高度和宽度的缩放比例
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        # 返回调整后的图像和高度、宽度的缩放比例
        return img, [ratio_h, ratio_w]
# 定义一个用于测试端到端调整大小的类
class E2EResizeForTest(object):
    # 初始化方法，接收关键字参数
    def __init__(self, **kwargs):
        # 调用父类的初始化方法
        super(E2EResizeForTest, self).__init__()
        # 设置最大边长和验证集
        self.max_side_len = kwargs['max_side_len']
        self.valid_set = kwargs['valid_set']

    # 调用对象时执行的方法，接收数据并调整大小
    def __call__(self, data):
        # 获取图像数据
        img = data['image']
        # 获取原始图像的高度和宽度
        src_h, src_w, _ = img.shape
        # 根据验证集类型选择调整大小的方法
        if self.valid_set == 'totaltext':
            # 调整图像大小并返回调整比例
            im_resized, [ratio_h, ratio_w] = self.resize_image_for_totaltext(
                img, max_side_len=self.max_side_len)
        else:
            # 调整图像大小并返回调整比例
            im_resized, (ratio_h, ratio_w) = self.resize_image(
                img, max_side_len=self.max_side_len)
        # 更新数据中的图像和形状信息
        data['image'] = im_resized
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        # 返回更新后的数据
        return data

    # 调整图像大小的方法，针对特定验证集
    def resize_image_for_totaltext(self, im, max_side_len=512):
        # 获取图像的高度、宽度和通道数
        h, w, _ = im.shape
        # 初始化调整后的宽度和高度
        resize_w = w
        resize_h = h
        # 初始化调整比例
        ratio = 1.25
        # 根据比例调整高度和宽度
        if h * ratio > max_side_len:
            ratio = float(max_side_len) / resize_h
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)
        # 设置最大步长
        max_stride = 128
        # 根据步长调整高度和宽度
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        # 调整图像大小并计算调整比例
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        # 返回调整后的图像和比例
        return im, (ratio_h, ratio_w)
    def resize_image(self, im, max_side_len=512):
        """
        resize image to a size multiple of max_stride which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        """
        # 获取原始图像的高度、宽度和通道数
        h, w, _ = im.shape

        # 初始化调整后的宽度和高度为原始图像的宽度和高度
        resize_w = w
        resize_h = h

        # 根据最长边进行调整
        if resize_h > resize_w:
            ratio = float(max_side_len) / resize_h
        else:
            ratio = float(max_side_len) / resize_w

        # 根据比例调整高度和宽度
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        # 设置最大步长为128，将高度和宽度调整为最接近的128的倍数
        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride

        # 使用OpenCV库中的resize函数对图像进行调整
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        
        # 计算高度和宽度的调整比例
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        # 返回调整后的图像和调整比例
        return im, (ratio_h, ratio_w)
# 定义一个名为 KieResize 的类
class KieResize(object):
    # 初始化方法，接收关键字参数 kwargs
    def __init__(self, **kwargs):
        # 调用父类的初始化方法
        super(KieResize, self).__init__()
        # 从 kwargs 中获取 img_scale 的值，分别赋给 max_side 和 min_side
        self.max_side, self.min_side = kwargs['img_scale'][0], kwargs['img_scale'][1]

    # 定义一个调用方法，接收参数 data
    def __call__(self, data):
        # 从 data 中获取 image 和 points
        img = data['image']
        points = data['points']
        # 获取图像的高度、宽度和通道数
        src_h, src_w, _ = img.shape
        # 调用 resize_image 方法对图像进行缩放
        im_resized, scale_factor, [ratio_h, ratio_w], [new_h, new_w] = self.resize_image(img)
        # 调用 resize_boxes 方法对坐标点进行缩放
        resize_points = self.resize_boxes(img, points, scale_factor)
        # 更新 data 中的相关字段
        data['ori_image'] = img
        data['ori_boxes'] = points
        data['points'] = resize_points
        data['image'] = im_resized
        data['shape'] = np.array([new_h, new_w])
        # 返回更新后的 data
        return data

    # 定义一个方法用于对图像进行缩放
    def resize_image(self, img):
        # 创建一个全零数组作为缩放后的图像
        norm_img = np.zeros([1024, 1024, 3], dtype='float32')
        # 设置缩放的目标尺寸
        scale = [512, 1024]
        # 获取原始图像的高度和宽度
        h, w = img.shape[:2]
        # 计算缩放因子
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
        # 计算缩放后的宽度和高度
        resize_w, resize_h = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
        # 设置最大步长
        max_stride = 32
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        # 调用 OpenCV 的 resize 方法对图像进行缩放
        im = cv2.resize(img, (resize_w, resize_h))
        new_h, new_w = im.shape[:2]
        w_scale = new_w / w
        h_scale = new_h / h
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        norm_img[:new_h, :new_w, :] = im
        # 返回缩放后的图像、缩放因子、高度和宽度比例、新的高度和宽度
        return norm_img, scale_factor, [h_scale, w_scale], [new_h, new_w]

    # 定义一个方法用于对坐标点进行缩放
    def resize_boxes(self, im, points, scale_factor):
        # 根据缩放因子对坐标点进行缩放
        points = points * scale_factor
        img_shape = im.shape[:2]
        # 将坐标点限制在图像范围内
        points[:, 0::2] = np.clip(points[:, 0::2], 0, img_shape[1])
        points[:, 1::2] = np.clip(points[:, 1::2], 0, img_shape[0])
        # 返回缩放后的坐标点
        return points
class SRResize(object):
    # 定义一个图像缩放类
    def __init__(self,
                 imgH=32,
                 imgW=128,
                 down_sample_scale=4,
                 keep_ratio=False,
                 min_ratio=1,
                 mask=False,
                 infer_mode=False,
                 **kwargs):
        # 初始化函数，设置图像高度、宽度、缩放比例等参数
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.mask = mask
        self.infer_mode = infer_mode

    def __call__(self, data):
        # 定义调用函数，对输入数据进行处理
        imgH = self.imgH
        imgW = self.imgW
        images_lr = data["image_lr"]
        transform2 = ResizeNormalize(
            (imgW // self.down_sample_scale, imgH // self.down_sample_scale))
        images_lr = transform2(images_lr)
        data["img_lr"] = images_lr
        if self.infer_mode:
            return data

        images_HR = data["image_hr"]
        label_strs = data["label"]
        transform = ResizeNormalize((imgW, imgH))
        images_HR = transform(images_HR)
        data["img_hr"] = images_HR
        return data


class ResizeNormalize(object):
    # 定义一个图像缩放和归一化类
    def __init__(self, size, interpolation=Image.BICUBIC):
        # 初始化函数，设置缩放尺寸和插值方法
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # 定义调用函数，对输入图像进行缩放和归一化处理
        img = img.resize(self.size, self.interpolation)
        img_numpy = np.array(img).astype("float32")
        img_numpy = img_numpy.transpose((2, 0, 1)) / 255
        return img_numpy


class GrayImageChannelFormat(object):
    """
    format gray scale image's channel: (3,h,w) -> (1,h,w)
    Args:
        inverse: inverse gray image 
    """
    # 定义一个灰度图像通道格式化类
    def __init__(self, inverse=False, **kwargs):
        # 初始化函数，设置是否反转灰度图像的参数
        self.inverse = inverse
    # 定义一个类的调用方法，接受一个数据字典作为参数
    def __call__(self, data):
        # 从数据字典中获取图像数据
        img = data['image']
        # 将彩色图像转换为单通道灰度图像
        img_single_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 在第0维度上扩展图像数据
        img_expanded = np.expand_dims(img_single_channel, 0)

        # 如果需要反转图像
        if self.inverse:
            # 对图像数据进行反转处理
            data['image'] = np.abs(img_expanded - 1)
        else:
            # 否则保持原图像数据不变
            data['image'] = img_expanded

        # 将原始图像数据存储在数据字典中
        data['src_image'] = img
        # 返回处理后的数据字典
        return data
```