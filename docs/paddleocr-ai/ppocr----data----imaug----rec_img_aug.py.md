# `.\PaddleOCR\ppocr\data\imaug\rec_img_aug.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入所需的库
import math
import cv2
import numpy as np
import random
import copy
from PIL import Image
# 导入自定义的图像处理函数
from .text_image_aug import tia_perspective, tia_stretch, tia_distort
from .abinet_aug import CVGeometry, CVDeterioration, CVColorJitter, SVTRGeometry, SVTRDeterioration
# 导入 PaddlePaddle 的图像处理函数
from paddle.vision.transforms import Compose

# 定义 RecAug 类
class RecAug(object):
    def __init__(self,
                 tia_prob=0.4,
                 crop_prob=0.4,
                 reverse_prob=0.4,
                 noise_prob=0.4,
                 jitter_prob=0.4,
                 blur_prob=0.4,
                 hsv_aug_prob=0.4,
                 **kwargs):
        # 初始化参数
        self.tia_prob = tia_prob
        # 创建 BaseDataAugmentation 实例
        self.bda = BaseDataAugmentation(crop_prob, reverse_prob, noise_prob,
                                        jitter_prob, blur_prob, hsv_aug_prob)

    # 定义 __call__ 方法
    def __call__(self, data):
        # 获取图像数据
        img = data['image']
        h, w, _ = img.shape

        # 图像增强处理
        # tia
        if random.random() <= self.tia_prob:
            if h >= 20 and w >= 20:
                # 扭曲和拉伸
                img = tia_distort(img, random.randint(3, 6))
                img = tia_stretch(img, random.randint(3, 6))
            # 透视变换
            img = tia_perspective(img)

        # bda
        data['image'] = img
        # 调用 BaseDataAugmentation 实例处理数据
        data = self.bda(data)
        return data

# 定义 BaseDataAugmentation 类
class BaseDataAugmentation(object):
    # 初始化函数，设置各种数据增强的概率参数
    def __init__(self,
                 crop_prob=0.4,
                 reverse_prob=0.4,
                 noise_prob=0.4,
                 jitter_prob=0.4,
                 blur_prob=0.4,
                 hsv_aug_prob=0.4,
                 **kwargs):
        # 设置裁剪的概率参数
        self.crop_prob = crop_prob
        # 设置翻转的概率参数
        self.reverse_prob = reverse_prob
        # 设置添加噪声的概率参数
        self.noise_prob = noise_prob
        # 设置颜色抖动的概率参数
        self.jitter_prob = jitter_prob
        # 设置模糊的概率参数
        self.blur_prob = blur_prob
        # 设置HSV颜色增强的概率参数
        self.hsv_aug_prob = hsv_aug_prob
        # 为高斯模糊准备滤波器
        self.fil = cv2.getGaussianKernel(ksize=5, sigma=1, ktype=cv2.CV_32F)

    # 数据增强的调用函数
    def __call__(self, data):
        # 获取图像数据
        img = data['image']
        # 获取图像的高度和宽度
        h, w, _ = img.shape

        # 如果随机数小于等于裁剪概率，并且图像高度和宽度都大于等于20，则进行裁剪
        if random.random() <= self.crop_prob and h >= 20 and w >= 20:
            img = get_crop(img)

        # 如果随机数小于等于模糊概率，则进行高斯模糊
        if random.random() <= self.blur_prob:
            # 使用高斯滤波器进行模糊处理
            img = cv2.sepFilter2D(img, -1, self.fil, self.fil)

        # 如果随机数小于等于HSV颜色增强概率，则进行HSV颜色增强
        if random.random() <= self.hsv_aug_prob:
            img = hsv_aug(img)

        # 如果随机数小于等于颜色抖动概率，则进行颜色抖动
        if random.random() <= self.jitter_prob:
            img = jitter(img)

        # 如果随机数小于等于添加噪声概率，则添加高斯噪声
        if random.random() <= self.noise_prob:
            img = add_gasuss_noise(img)

        # 如果随机数小于等于翻转概率，则进行图像翻转
        if random.random() <= self.reverse_prob:
            img = 255 - img

        # 更新数据中的图像信息
        data['image'] = img
        # 返回更新后的数据
        return data
# 定义一个类 ABINetRecAug，用于数据增强
class ABINetRecAug(object):
    # 初始化函数，设置几何变换、恶化、颜色调整的概率
    def __init__(self,
                 geometry_p=0.5,
                 deterioration_p=0.25,
                 colorjitter_p=0.25,
                 **kwargs):
        # 创建数据增强的组合
        self.transforms = Compose([
            # 几何变换
            CVGeometry(
                degrees=45,
                translate=(0.0, 0.0),
                scale=(0.5, 2.),
                shear=(45, 15),
                distortion=0.5,
                p=geometry_p), 
            # 恶化
            CVDeterioration(
                var=20, degrees=6, factor=4, p=deterioration_p),
            # 颜色调整
            CVColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1,
                p=colorjitter_p)
        ])

    # 数据增强的调用函数
    def __call__(self, data):
        # 获取图像数据
        img = data['image']
        # 对图像进行数据增强
        img = self.transforms(img)
        # 更新数据中的图像数据
        data['image'] = img
        return data

# 定义一个类 RecConAug，用于合并额外的数据
class RecConAug(object):
    # 初始化函数，设置合并概率、图像形状、最大文本长度、额外数据数量等参数
    def __init__(self,
                 prob=0.5,
                 image_shape=(32, 320, 3),
                 max_text_length=25,
                 ext_data_num=1,
                 **kwargs):
        # 设置额外数据数量
        self.ext_data_num = ext_data_num
        # 设置合并概率
        self.prob = prob
        # 设置最大文本长度
        self.max_text_length = max_text_length
        # 设置图像形状
        self.image_shape = image_shape
        # 计算图像的最大宽高比
        self.max_wh_ratio = self.image_shape[1] / self.image_shape[0]

    # 合并额外数据的函数
    def merge_ext_data(self, data, ext_data):
        # 计算原始图像的宽度
        ori_w = round(data['image'].shape[1] / data['image'].shape[0] * self.image_shape[0])
        # 计算额外数据的宽度
        ext_w = round(ext_data['image'].shape[1] / ext_data['image'].shape[0] * self.image_shape[0])
        # 调整原始图像的大小
        data['image'] = cv2.resize(data['image'], (ori_w, self.image_shape[0]))
        # 调整额外数据的大小
        ext_data['image'] = cv2.resize(ext_data['image'], (ext_w, self.image_shape[0]))
        # 拼接原始图像和额外数据的图像
        data['image'] = np.concatenate([data['image'], ext_data['image']], axis=1)
        # 更新数据中的标签信息
        data["label"] += ext_data["label"]
        return data
    # 定义一个类的方法，用于处理数据
    def __call__(self, data):
        # 生成一个随机数
        rnd_num = random.random()
        # 如果随机数大于给定概率，则直接返回数据
        if rnd_num > self.prob:
            return data
        # 遍历数据中的扩展数据
        for idx, ext_data in enumerate(data["ext_data"]):
            # 如果当前标签长度加上扩展数据的标签长度超过最大文本长度，则跳出循环
            if len(data["label"]) + len(ext_data["label"]) > self.max_text_length:
                break
            # 计算当前图像和扩展数据图像的宽高比之和
            concat_ratio = data['image'].shape[1] / data['image'].shape[0] + ext_data['image'].shape[1] / ext_data['image'].shape[0]
            # 如果宽高比之和超过最大宽高比，则跳出循环
            if concat_ratio > self.max_wh_ratio:
                break
            # 合并当前数据和扩展数据
            data = self.merge_ext_data(data, ext_data)
        # 移除数据中的扩展数据
        data.pop("ext_data")
        # 返回处理后的数据
        return data
# 定义一个类 SVTRRecAug，用于进行数据增强操作
class SVTRRecAug(object):
    # 初始化方法，设置默认参数和数据增强操作
    def __init__(self,
                 aug_type=0,
                 geometry_p=0.5,
                 deterioration_p=0.25,
                 colorjitter_p=0.25,
                 **kwargs):
        # 组合多种数据增强操作
        self.transforms = Compose([
            # 几何变换操作
            SVTRGeometry(
                aug_type=aug_type,
                degrees=45,
                translate=(0.0, 0.0),
                scale=(0.5, 2.),
                shear=(45, 15),
                distortion=0.5,
                p=geometry_p), 
            # 图像劣化操作
            SVTRDeterioration(
                var=20, degrees=6, factor=4, p=deterioration_p),
            # 颜色调整操作
            CVColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1,
                p=colorjitter_p)
        ])

    # 调用方法，对输入数据进行数据增强操作
    def __call__(self, data):
        # 获取图像数据
        img = data['image']
        # 对图像进行数据增强操作
        img = self.transforms(img)
        # 更新数据中的图像数据
        data['image'] = img
        # 返回更新后的数据
        return data

# 定义一个类 ClsResizeImg，用于调整分类任务的图像大小
class ClsResizeImg(object):
    # 初始化方法，设置图像大小参数
    def __init__(self, image_shape, **kwargs):
        self.image_shape = image_shape

    # 调用方法，对输入数据进行图像大小调整操作
    def __call__(self, data):
        # 获取图像数据
        img = data['image']
        # 调用 resize_norm_img 函数对图像进行大小调整和归一化处理
        norm_img, _ = resize_norm_img(img, self.image_shape)
        # 更新数据中的图像数据
        data['image'] = norm_img
        # 返回更新后的数据
        return data

# 定义一个类 RecResizeImg，用于调整文本识别任务的图像大小
class RecResizeImg(object):
    # 初始化方法，设置图像大小和其他参数
    def __init__(self,
                 image_shape,
                 infer_mode=False,
                 eval_mode=False,
                 character_dict_path='./ppocr/utils/ppocr_keys_v1.txt',
                 padding=True,
                 **kwargs):
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.eval_mode = eval_mode
        self.character_dict_path = character_dict_path
        self.padding = padding
    # 定义一个类的方法，用于处理输入数据
    def __call__(self, data):
        # 从输入数据中获取图像数据
        img = data['image']
        # 如果处于评估模式或者推理模式且字符字典路径不为空
        if self.eval_mode or (self.infer_mode and
                              self.character_dict_path is not None):
            # 调用 resize_norm_img_chinese 函数对图像进行归一化和调整大小
            norm_img, valid_ratio = resize_norm_img_chinese(img,
                                                            self.image_shape)
        else:
            # 调用 resize_norm_img 函数对图像进行归一化、调整大小和填充
            norm_img, valid_ratio = resize_norm_img(img, self.image_shape,
                                                    self.padding)
        # 更新数据中的图像信息为处理后的图像
        data['image'] = norm_img
        # 更新数据中的有效比例信息
        data['valid_ratio'] = valid_ratio
        # 返回处理后的数据
        return data
# 定义一个类 VLRecResizeImg，用于处理图像的尺寸调整
class VLRecResizeImg(object):
    # 初始化方法，接受图像形状、推断模式、字符字典路径和填充参数等
    def __init__(self,
                 image_shape,
                 infer_mode=False,
                 character_dict_path='./ppocr/utils/ppocr_keys_v1.txt',
                 padding=True,
                 **kwargs):
        # 初始化类的属性
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.character_dict_path = character_dict_path
        self.padding = padding

    # 定义类的调用方法，用于处理数据
    def __call__(self, data):
        # 从数据中获取图像
        img = data['image']

        # 获取图像的通道数、高度和宽度
        imgC, imgH, imgW = self.image_shape
        # 调整图像大小为指定的形状
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_w = imgW
        # 将调整后的图像转换为 float32 类型
        resized_image = resized_image.astype('float32')
        # 如果图像通道数为 1，则将像素值归一化到 [0, 1] 范围
        if self.image_shape[0] == 1:
            resized_image = resized_image / 255
            norm_img = resized_image[np.newaxis, :]
        else:
            # 否则将图像通道维度置换到第一维，并归一化
            norm_img = resized_image.transpose((2, 0, 1)) / 255
        # 计算有效比例，即调整后图像宽度与原始宽度的比值
        valid_ratio = min(1.0, float(resized_w / imgW))

        # 更新数据中的图像和有效比例字段，并返回处理后的数据
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        return data


# 定义一个类 RFLRecResizeImg，用于处理图像的尺寸调整
class RFLRecResizeImg(object):
    # 初始化方法，接受图像形状、填充参数、插值方法等
    def __init__(self, image_shape, padding=True, interpolation=1, **kwargs):
        # 初始化类的属性
        self.image_shape = image_shape
        self.padding = padding

        # 根据插值方法选择相应的 OpenCV 插值方式
        self.interpolation = interpolation
        if self.interpolation == 0:
            self.interpolation = cv2.INTER_NEAREST
        elif self.interpolation == 1:
            self.interpolation = cv2.INTER_LINEAR
        elif self.interpolation == 2:
            self.interpolation = cv2.INTER_CUBIC
        elif self.interpolation == 3:
            self.interpolation = cv2.INTER_AREA
        else:
            # 如果插值方法不在支持范围内，则抛出异常
            raise Exception("Unsupported interpolation type !!!")
    # 定义一个类的方法，用于处理输入数据
    def __call__(self, data):
        # 从输入数据中获取图像数据
        img = data['image']
        # 将图像从彩色转换为灰度
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 调用 resize_norm_img 函数对图像进行缩放和归一化处理，并返回处理后的图像和有效比例
        norm_img, valid_ratio = resize_norm_img(
            img, self.image_shape, self.padding, self.interpolation)
        # 更新输入数据中的图像数据为处理后的图像
        data['image'] = norm_img
        # 更新输入数据中的有效比例信息
        data['valid_ratio'] = valid_ratio
        # 返回处理后的数据
        return data
# 定义一个类 SRNRecResizeImg，用于处理图片数据的调整和预处理
class SRNRecResizeImg(object):
    # 初始化方法，接受图片形状、头数和最大文本长度等参数
    def __init__(self, image_shape, num_heads, max_text_length, **kwargs):
        self.image_shape = image_shape
        self.num_heads = num_heads
        self.max_text_length = max_text_length

    # 调用方法，对输入的数据进行处理
    def __call__(self, data):
        # 获取输入数据中的图片
        img = data['image']
        # 调用 resize_norm_img_srn 方法对图片进行归一化和调整大小
        norm_img = resize_norm_img_srn(img, self.image_shape)
        data['image'] = norm_img
        # 调用 srn_other_inputs 方法获取其他输入数据
        [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = \
            srn_other_inputs(self.image_shape, self.num_heads, self.max_text_length)

        # 将获取的其他输入数据添加到输入数据中
        data['encoder_word_pos'] = encoder_word_pos
        data['gsrm_word_pos'] = gsrm_word_pos
        data['gsrm_slf_attn_bias1'] = gsrm_slf_attn_bias1
        data['gsrm_slf_attn_bias2'] = gsrm_slf_attn_bias2
        return data

# 定义一个类 SARRecResizeImg，用于处理图片数据的调整和预处理
class SARRecResizeImg(object):
    # 初始化方法，接受图片形状和宽度下采样比例等参数
    def __init__(self, image_shape, width_downsample_ratio=0.25, **kwargs):
        self.image_shape = image_shape
        self.width_downsample_ratio = width_downsample_ratio

    # 调用方法，对输入的数据进行处理
    def __call__(self, data):
        # 获取输入数据中的图片
        img = data['image']
        # 调用 resize_norm_img_sar 方法对图片进行归一化、调整大小和填充
        norm_img, resize_shape, pad_shape, valid_ratio = resize_norm_img_sar(
            img, self.image_shape, self.width_downsample_ratio)
        data['image'] = norm_img
        data['resized_shape'] = resize_shape
        data['pad_shape'] = pad_shape
        data['valid_ratio'] = valid_ratio
        return data

# 定义一个类 PRENResizeImg，用于处理图片数据的调整和预处理
class PRENResizeImg(object):
    # 初始化方法，接受图片形状等参数
    def __init__(self, image_shape, **kwargs):
        """
        根据原始论文的实现，这里是一个硬调整大小的方法。
        因此，你可能需要优化它以更好地适应你的任务。
        """
        self.dst_h, self.dst_w = image_shape
    # 定义一个类的方法，用于对输入数据进行处理
    def __call__(self, data):
        # 从输入数据中获取图像数据
        img = data['image']
        # 调用 OpenCV 库中的 resize 方法，将图像进行缩放处理
        resized_img = cv2.resize(
            img, (self.dst_w, self.dst_h), interpolation=cv2.INTER_LINEAR)
        # 调整图像的维度顺序，将通道维度放在最前面，并进行归一化处理
        resized_img = resized_img.transpose((2, 0, 1)) / 255
        # 对图像进行标准化处理，减去均值并除以标准差
        resized_img -= 0.5
        resized_img /= 0.5
        # 将处理后的图像数据存回输入数据中，并转换数据类型为 np.float32
        data['image'] = resized_img.astype(np.float32)
        # 返回处理后的数据
        return data
# 定义一个类 SPINRecResizeImg，用于处理图像的缩放和灰度处理
class SPINRecResizeImg(object):
    # 初始化函数，设置图像形状、插值方法、均值和标准差等参数
    def __init__(self,
                 image_shape,
                 interpolation=2,
                 mean=(127.5, 127.5, 127.5),
                 std=(127.5, 127.5, 127.5),
                 **kwargs):
        # 保存图像形状
        self.image_shape = image_shape
        # 将均值和标准差转换为 numpy 数组
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        # 保存插值方法
        self.interpolation = interpolation

    # 定义 __call__ 方法，用于处理数据
    def __call__(self, data):
        # 获取图像数据
        img = data['image']
        # 将图像转换为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 根据插值类型设置对应的 OpenCV 插值方法
        if self.interpolation == 0:
            interpolation = cv2.INTER_NEAREST
        elif self.interpolation == 1:
            interpolation = cv2.INTER_LINEAR
        elif self.interpolation == 2:
            interpolation = cv2.INTER_CUBIC
        elif self.interpolation == 3:
            interpolation = cv2.INTER_AREA
        else:
            # 抛出异常，表示不支持的插值类型
            raise Exception("Unsupported interpolation type !!!")
        # 处理图像加载过程中的错误
        if img is None:
            return None
        # 调整图像大小
        img = cv2.resize(img, tuple(self.image_shape), interpolation)
        img = np.array(img, np.float32)
        # 在最后一个维度上增加一个维度
        img = np.expand_dims(img, -1)
        # 调整图像维度顺序
        img = img.transpose((2, 0, 1))
        # 标准化图像
        img = img.copy().astype(np.float32)
        mean = np.float64(self.mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.std.reshape(1, -1))
        img -= mean
        img *= stdinv
        # 更新数据中的图像信息
        data['image'] = img
        return data

# 定义类 GrayRecResizeImg
class GrayRecResizeImg(object):
    # 初始化函数，接受图像形状、调整类型、插值类型、是否缩放、是否填充等参数
    def __init__(self,
                 image_shape,
                 resize_type,
                 inter_type='Image.LANCZOS',
                 scale=True,
                 padding=False,
                 **kwargs):
        # 设置图像形状、调整类型、是否填充、插值类型、是否缩放等属性
        self.image_shape = image_shape
        self.resize_type = resize_type
        self.padding = padding
        self.inter_type = eval(inter_type)
        self.scale = scale

    # 调用函数，对输入的数据进行处理
    def __call__(self, data):
        # 获取输入数据中的图像
        img = data['image']
        # 将图像转换为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 获取图像形状
        image_shape = self.image_shape
        # 如果需要填充
        if self.padding:
            # 获取图像通道数、高度、宽度
            imgC, imgH, imgW = image_shape
            # 计算图像的宽高比
            h = img.shape[0]
            w = img.shape[1]
            ratio = w / float(h)
            # 根据比例调整图像宽度
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            # 调整图像大小
            resized_image = cv2.resize(img, (resized_w, imgH))
            # 对调整后的图像进行归一化处理
            norm_img = np.expand_dims(resized_image, -1)
            norm_img = norm_img.transpose((2, 0, 1))
            resized_image = norm_img.astype(np.float32) / 128. - 1.
            # 创建一个全零填充的图像
            padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
            padding_im[:, :, 0:resized_w] = resized_image
            # 更新数据中的图像
            data['image'] = padding_im
            return data
        # 如果调整类型为 PIL
        if self.resize_type == 'PIL':
            # 将图像转换为 PIL 格式
            image_pil = Image.fromarray(np.uint8(img))
            # 调整图像大小
            img = image_pil.resize(self.image_shape, self.inter_type)
            img = np.array(img)
        # 如果调整类型为 OpenCV
        if self.resize_type == 'OpenCV':
            # 使用 OpenCV 调整图像大小
            img = cv2.resize(img, self.image_shape)
        # 对调整后的图像进行归一化处理
        norm_img = np.expand_dims(img, -1)
        norm_img = norm_img.transpose((2, 0, 1))
        # 如果需要缩放
        if self.scale:
            # 对图像进行归一化处理
            data['image'] = norm_img.astype(np.float32) / 128. - 1.
        else:
            # 对图像进行归一化处理
            data['image'] = norm_img.astype(np.float32) / 255.
        # 返回处理后的数据
        return data
# 定义一个类，用于处理 ABINet 模型的图像数据，包括初始化和调用方法
class ABINetRecResizeImg(object):
    # 初始化方法，接收图像形状参数和其他关键字参数
    def __init__(self, image_shape, **kwargs):
        self.image_shape = image_shape

    # 调用方法，对输入的数据进行处理
    def __call__(self, data):
        # 获取数据中的图像
        img = data['image']
        # 调用 resize_norm_img_abinet 函数对图像进行规范化和缩放
        norm_img, valid_ratio = resize_norm_img_abinet(img, self.image_shape)
        # 更新数据中的图像和有效比例信息
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        return data

# 定义一个类，用于处理 SVTR 模型的图像数据，包括初始化和调用方法
class SVTRRecResizeImg(object):
    # 初始化方法，接收图像形状参数、是否填充等关键字参数
    def __init__(self, image_shape, padding=True, **kwargs):
        self.image_shape = image_shape
        self.padding = padding

    # 调用方法，对输入的数据进行处理
    def __call__(self, data):
        # 获取数据中的图像
        img = data['image']
        # 调用 resize_norm_img 函数对图像进行规范化和缩放
        norm_img, valid_ratio = resize_norm_img(img, self.image_shape, self.padding)
        # 更新数据中的图像和有效比例信息
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        return data

# 定义一个类，用于处理 RobustScanner 模型的图像数据，包括初始化和调用方法
class RobustScannerRecResizeImg(object):
    # 初始化方法，接收图像形状参数、最大文本长度、宽度下采样比例等关键字参数
    def __init__(self, image_shape, max_text_length, width_downsample_ratio=0.25, **kwargs):
        self.image_shape = image_shape
        self.width_downsample_ratio = width_downsample_ratio
        self.max_text_length = max_text_length

    # 调用方法，对输入的数据进行处理
    def __call__(self, data):
        # 获取数据中的图像
        img = data['image']
        # 调用 resize_norm_img_sar 函数对图像进行规范化和缩放
        norm_img, resize_shape, pad_shape, valid_ratio = resize_norm_img_sar(img, self.image_shape, self.width_downsample_ratio)
        # 创建一个包含最大文本长度范围的数组
        word_positons = np.array(range(0, self.max_text_length)).astype('int64')
        # 更新数据中的图像、调整后形状、填充形状、有效比例和文本位置信息
        data['image'] = norm_img
        data['resized_shape'] = resize_shape
        data['pad_shape'] = pad_shape
        data['valid_ratio'] = valid_ratio
        data['word_positons'] = word_positons
        return data

# 定义一个函数，用于对图像进行规范化和缩放
def resize_norm_img_sar(img, image_shape, width_downsample_ratio=0.25):
    # 解析图像形状参数
    imgC, imgH, imgW_min, imgW_max = image_shape
    # 获取图像的高度和宽度
    h = img.shape[0]
    w = img.shape[1]
    valid_ratio = 1.0
    # 确保新宽度是宽度下采样比例的整数倍
    width_divisor = int(1 / width_downsample_ratio)
    # 调整图像大小
    # 计算宽高比
    ratio = w / float(h)
    # 根据宽高比和目标高度计算新的宽度
    resize_w = math.ceil(imgH * ratio)
    # 如果新的宽度不能被width_divisor整除，则调整为最接近的可以被整除的宽度
    if resize_w % width_divisor != 0:
        resize_w = round(resize_w / width_divisor) * width_divisor
    # 如果设置了最小宽度限制，则取最大值
    if imgW_min is not None:
        resize_w = max(imgW_min, resize_w)
    # 如果设置了最大宽度限制，则计算有效比例并取最小值
    if imgW_max is not None:
        valid_ratio = min(1.0, 1.0 * resize_w / imgW_max)
        resize_w = min(imgW_max, resize_w)
    # 调整图像大小
    resized_image = cv2.resize(img, (resize_w, imgH))
    # 转换数据类型为float32
    resized_image = resized_image.astype('float32')
    # 归一化处理
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    resize_shape = resized_image.shape
    # 创建一个填充图像，用-1填充
    padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
    # 将调整大小后的图像放入填充图像中
    padding_im[:, :, 0:resize_w] = resized_image
    pad_shape = padding_im.shape

    return padding_im, resize_shape, pad_shape, valid_ratio
# 调整并标准化图像大小
def resize_norm_img(img,
                    image_shape,
                    padding=True,
                    interpolation=cv2.INTER_LINEAR):
    # 获取图像通道数、高度和宽度
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    # 如果不需要填充
    if not padding:
        # 调整图像大小，保持宽高比，使用指定的插值方法
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=interpolation)
        resized_w = imgW
    else:
        # 计算宽高比
        ratio = w / float(h)
        # 根据宽高比调整宽度
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
    # 转换图像数据类型为 float32
    resized_image = resized_image.astype('float32')
    # 如果图像通道数为 1
    if image_shape[0] == 1:
        # 将图像像素值归一化到 [0, 1] 范围
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        # 调整图像维度顺序，并将像素值归一化到 [0, 1] 范围
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    # 像素值标准化
    resized_image -= 0.5
    resized_image /= 0.5
    # 创建一个全零填充的图像数组
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    # 计算有效宽度比例
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio


# 调整并标准化中文图像大小
def resize_norm_img_chinese(img, image_shape):
    imgC, imgH, imgW = image_shape
    # todo: change to 0 and modified image shape
    # 计算最大宽高比
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(imgH * max_wh_ratio)
    # 根据宽高比调整宽度
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    # 如果图像通道数为 1
    if image_shape[0] == 1:
        # 将图像像素值归一化到 [0, 1] 范围
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        # 调整图像维度顺序，并将像素值归一化到 [0, 1] 范围
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    # 像素值标准化
    resized_image -= 0.5
    resized_image /= 0.5
    # 创建一个全零填充的图像数组
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    # 将调整大小后的图像数据填充到原始图像的左侧
    padding_im[:, :, 0:resized_w] = resized_image
    # 计算有效比例，即调整大小后的宽度与原始宽度的比值，取最小值为1.0
    valid_ratio = min(1.0, float(resized_w / imgW))
    # 返回填充后的图像数据和有效比例
    return padding_im, valid_ratio
# 调整并标准化图像大小，返回调整后的图像数据
def resize_norm_img_srn(img, image_shape):
    # 获取图像通道数、高度和宽度
    imgC, imgH, imgW = image_shape

    # 创建一个全黑的图像数组，大小为指定的高度和宽度
    img_black = np.zeros((imgH, imgW))
    # 获取输入图像的高度和宽度
    im_hei = img.shape[0]
    im_wid = img.shape[1]

    # 根据宽高比例调整图像大小
    if im_wid <= im_hei * 1:
        img_new = cv2.resize(img, (imgH * 1, imgH))
    elif im_wid <= im_hei * 2:
        img_new = cv2.resize(img, (imgH * 2, imgH))
    elif im_wid <= im_hei * 3:
        img_new = cv2.resize(img, (imgH * 3, imgH))
    else:
        img_new = cv2.resize(img, (imgW, imgH))

    # 将调整后的图像转换为灰度图像
    img_np = np.asarray(img_new)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    # 将灰度图像复制到全黑图像的左侧
    img_black[:, 0:img_np.shape[1]] = img_np
    img_black = img_black[:, :, np.newaxis]

    # 获取图像的行数、列数和通道数
    row, col, c = img_black.shape
    c = 1

    # 返回调整后的图像数据，转换为 float32 类型
    return np.reshape(img_black, (c, row, col)).astype(np.float32)


# 调整并标准化图像大小，返回调整后的图像数据和有效比例
def resize_norm_img_abinet(img, image_shape):
    # 获取图像通道数、高度和宽度
    imgC, imgH, imgW = image_shape

    # 调整图像大小并进行标准化处理
    resized_image = cv2.resize(
        img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
    resized_w = imgW
    resized_image = resized_image.astype('float32')
    resized_image = resized_image / 255.

    # 计算均值和标准差，对图像进行标准化处理
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    resized_image = (
        resized_image - mean[None, None, ...]) / std[None, None, ...]
    resized_image = resized_image.transpose((2, 0, 1))
    resized_image = resized_image.astype('float32')

    # 计算有效比例并返回调整后的图像数据和有效比例
    valid_ratio = min(1.0, float(resized_w / imgW))
    return resized_image, valid_ratio


# 生成其他输入数据，返回编码器和解码器的位置信息以及注意力偏置数据
def srn_other_inputs(image_shape, num_heads, max_text_length):
    # 获取图像通道数、高度和宽度
    imgC, imgH, imgW = image_shape
    # 计算特征维度
    feature_dim = int((imgH / 8) * (imgW / 8))

    # 创建编码器和解码器的位置信息数组
    encoder_word_pos = np.array(range(0, feature_dim)).reshape(
        (feature_dim, 1)).astype('int64')
    gsrm_word_pos = np.array(range(0, max_text_length)).reshape(
        (max_text_length, 1)).astype('int64')

    # 创建全为1的注意力偏置数据和自注意力偏置数据
    gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
    gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
        [1, max_text_length, max_text_length])
    # 创建一个与 gsrm_slf_attn_bias1 相同的数组，每个元素都是 gsrm_slf_attn_bias1 的值乘以 -1e9
    gsrm_slf_attn_bias1 = np.tile(gsrm_slf_attn_bias1,
                                  [num_heads, 1, 1]) * [-1e9]
    
    # 创建一个下三角矩阵，将 gsrm_attn_bias_data 中的值填充到下三角区域，然后重新形状为 [1, max_text_length, max_text_length]
    gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
        [1, max_text_length, max_text_length])
    # 创建一个与 gsrm_slf_attn_bias2 相同的数组，每个元素都是 gsrm_slf_attn_bias2 的值乘以 -1e9
    gsrm_slf_attn_bias2 = np.tile(gsrm_slf_attn_bias2,
                                  [num_heads, 1, 1]) * [-1e9]
    
    # 返回一个包含 encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2 的列表
    return [
        encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
        gsrm_slf_attn_bias2
    ]
def flag():
    """
    flag
    """
    # 生成一个随机数，如果大于0.5000001返回1，否则返回-1
    return 1 if random.random() > 0.5000001 else -1


def hsv_aug(img):
    """
    cvtColor
    """
    # 将图像从BGR颜色空间转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 生成一个随机数乘以0.001，再乘以flag()函数的返回值，得到delta值
    delta = 0.001 * random.random() * flag()
    # 调整HSV图像的亮度
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    # 将调整后的HSV图像转换回BGR颜色空间
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def blur(img):
    """
    blur
    """
    h, w, _ = img.shape
    # 如果图像的高度和宽度都大于10，则对图像进行高斯模糊处理
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (5, 5), 1)
    else:
        return img


def jitter(img):
    """
    jitter
    """
    w, h, _ = img.shape
    # 如果图像的高度和宽度都大于10，则对图像进行抖动处理
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        for i in range(s):
            img[i:, i:, :] = src_img[:w - i, :h - i, :]
        return img
    else:
        return img


def add_gasuss_noise(image, mean=0, var=0.1):
    """
    Gasuss noise
    """
    # 生成服从正态分布的噪声
    noise = np.random.normal(mean, var**0.5, image.shape)
    # 将噪声添加到图像上
    out = image + 0.5 * noise
    # 将像素值限制在0到255之间
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


def get_crop(image):
    """
    random crop
    """
    h, w, _ = image.shape
    top_min = 1
    top_max = 8
    # 随机选择裁剪的上边界
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = image.copy()
    ratio = random.randint(0, 1)
    # 根据ratio选择裁剪的方式
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0:h - top_crop, :, :]
    return crop_img


def rad(x):
    """
    rad
    """
    # 将角度转换为弧度
    return x * np.pi / 180


def get_warpR(config):
    """
    get_warpR
    """
    anglex, angley, anglez, fov, w, h, r = \
        config.anglex, config.angley, config.anglez, config.fov, config.w, config.h, config.r
    if w > 69 and w < 112:
        anglex = anglex * 1.5

    z = np.sqrt(w**2 + h**2) / 2 / np.tan(rad(fov / 2))
    # Homogeneous coordinate transformation matrix
    # 生成绕 x 轴旋转的变换矩阵
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0], [
                       0,
                       -np.sin(rad(anglex)),
                       np.cos(rad(anglex)),
                       0,
                   ], [0, 0, 0, 1]], np.float32)
    # 生成绕 y 轴旋转的变换矩阵
    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0], [
                       -np.sin(rad(angley)),
                       0,
                       np.cos(rad(angley)),
                       0,
                   ], [0, 0, 0, 1]], np.float32)
    # 生成绕 z 轴旋转的变换矩阵
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    # 计算综合旋转矩阵
    r = rx.dot(ry).dot(rz)
    # 生成中心点
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)
    # 生成四个点
    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter
    # 应用旋转矩阵到四个点
    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)
    list_dst = np.array([dst1, dst2, dst3, dst4])
    org = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
    dst = np.zeros((4, 2), np.float32)
    # 投影到图像平面
    dst[:, 0] = list_dst[:, 0] * z / (z - list_dst[:, 2]) + pcenter[0]
    dst[:, 1] = list_dst[:, 1] * z / (z - list_dst[:, 2]) + pcenter[1]

    # 获取透视变换矩阵
    warpR = cv2.getPerspectiveTransform(org, dst)

    # 获取变换后的四个点坐标
    dst1, dst2, dst3, dst4 = dst
    # 计算变换后的矩形区域
    r1 = int(min(dst1[1], dst2[1]))
    r2 = int(max(dst3[1], dst4[1]))
    c1 = int(min(dst1[0], dst3[0]))
    c2 = int(max(dst2[0], dst4[0]))

    # 计算缩放比例
    ratio = min(1.0 * h / (r2 - r1), 1.0 * w / (c2 - c1))

    dx = -c1
    dy = -r1
    T1 = np.float32([[1., 0, dx], [0, 1., dy], [0, 0, 1.0 / ratio]])
    # 计算最终的透视变换矩阵
    ret = T1.dot(warpR)
    # 捕获所有异常
    except:
        # 设置比例为1.0
        ratio = 1.0
        # 创建一个3x3的浮点型矩阵T1
        T1 = np.float32([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        # 将T1赋值给ret
        ret = T1
    # 返回ret, (-r1, -c1), ratio, dst
    return ret, (-r1, -c1), ratio, dst
# 定义一个函数，用于生成仿射变换矩阵
def get_warpAffine(config):
    """
    get_warpAffine
    """
    # 从配置中获取旋转角度
    anglez = config.anglez
    # 根据旋转角度计算旋转矩阵
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0]], np.float32)
    # 返回计算得到的旋转矩阵
    return rz
```