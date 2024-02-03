# `.\PaddleOCR\ppocr\data\imaug\abinet_aug.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 代码参考自：
# https://github.com/FangShancheng/ABINet/blob/main/transforms.py
"""
import math
import numbers
import random

import cv2
import numpy as np
from paddle.vision.transforms import Compose, ColorJitter

# 定义一个函数，用于生成服从 beta 分布的随机数，乘以 magnitude
def sample_asym(magnitude, size=None):
    return np.random.beta(1, 4, size) * magnitude

# 定义一个函数，用于生成对称的服从 beta 分布的随机数，乘以 magnitude
def sample_sym(magnitude, size=None):
    return (np.random.beta(4, 4, size=size) - 0.5) * 2 * magnitude

# 定义一个函数，用于生成服从均匀分布的随机数
def sample_uniform(low, high, size=None):
    return np.random.uniform(low, high, size=size

# 根据输入的插值类型，返回相应的插值方法
def get_interpolation(type='random'):
    if type == 'random':
        choice = [
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA
        ]
        interpolation = choice[random.randint(0, len(choice) - 1)]
    elif type == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif type == 'linear':
        interpolation = cv2.INTER_LINEAR
    elif type == 'cubic':
        interpolation = cv2.INTER_CUBIC
    elif type == 'area':
        interpolation = cv2.INTER_AREA
    else:
        raise TypeError(
            'Interpolation types only nearest, linear, cubic, area are supported!'
        )
    return interpolation

# 定义一个类 CVRandomRotation
class CVRandomRotation(object):
    # 初始化函数，设置旋转角度，默认为15度
    def __init__(self, degrees=15):
        # 检查 degrees 是否为数字类型
        assert isinstance(degrees, numbers.Number), "degree should be a single number."
        # 检查 degrees 是否为非负数
        assert degrees >= 0, "degree must be positive."
        # 将 degrees 赋值给对象的 degrees 属性
        self.degrees = degrees

    # 静态方法，获取旋转角度参数
    @staticmethod
    def get_params(degrees):
        # 调用 sample_sym 函数获取旋转角度
        return sample_sym(degrees)

    # 调用对象时执行的函数，对图像进行旋转
    def __call__(self, img):
        # 获取旋转角度
        angle = self.get_params(self.degrees)
        # 获取图像的高度和宽度
        src_h, src_w = img.shape[:2]
        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(
            center=(src_w / 2, src_h / 2), angle=angle, scale=1.0)
        # 计算旋转矩阵的绝对值的余弦和正弦值
        abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
        # 计算旋转后的图像宽度和高度
        dst_w = int(src_h * abs_sin + src_w * abs_cos)
        dst_h = int(src_h * abs_cos + src_w * abs_sin)
        # 调整旋转矩阵的平移参数
        M[0, 2] += (dst_w - src_w) / 2
        M[1, 2] += (dst_h - src_h) / 2

        # 获取插值方法的标志
        flags = get_interpolation()
        # 对图像进行仿射变换
        return cv2.warpAffine(
            img,
            M, (dst_w, dst_h),
            flags=flags,
            borderMode=cv2.BORDER_REPLICATE)
# 定义一个类 CVRandomAffine，用于生成随机仿射变换
class CVRandomAffine(object):
    # 初始化方法，接受旋转角度、平移、缩放和剪切参数
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        # 断言旋转角度为数字类型，并且为正数
        assert isinstance(degrees, numbers.Number), "degree should be a single number."
        assert degrees >= 0, "degree must be positive."
        # 将旋转角度赋值给实例变量
        self.degrees = degrees

        # 如果有平移参数
        if translate is not None:
            # 断言平移参数为列表或元组，并且长度为2
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            # 检查平移参数的取值范围是否在0到1之间
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
            # 将平移参数赋值给实例变量
            self.translate = translate

        # 如果有缩放参数
        if scale is not None:
            # 断言缩放参数为列表或元组，并且长度为2
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            # 检查缩放参数是否为正数
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
            # 将缩放参数赋值给实例变量
            self.scale = scale

        # 如果有剪切参数
        if shear is not None:
            # 如果剪切参数为单个数字
            if isinstance(shear, numbers.Number):
                # 断言剪切参数为正数
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                # 将剪切参数转换为列表形式
                self.shear = [shear]
            else:
                # 断言剪切参数为列表或元组，并且长度为2
                assert isinstance(shear, (tuple, list)) and (len(shear) == 2), \
                    "shear should be a list or tuple and it must be of length 2."
                # 将剪切参数赋值给实例变量
                self.shear = shear
        else:
            # 如果没有剪切参数，则将实例变量设为None
            self.shear = shear
    # 获取仿射矩阵的逆矩阵，包括旋转中心、角度、平移、缩放和剪切参数
    def _get_inverse_affine_matrix(self, center, angle, translate, scale,
                                   shear):
        # 引入 sin、cos、tan 函数
        from numpy import sin, cos, tan

        # 如果剪切参数是单个数字，则转换为包含两个相同值的列表
        if isinstance(shear, numbers.Number):
            shear = [shear, 0]

        # 如果剪切参数不是元组或列表，或者长度不为2，则引发异常
        if not isinstance(shear, (tuple, list)) and len(shear) == 2:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing " +
                "two values. Got {}".format(shear))

        # 将角度转换为弧度
        rot = math.radians(angle)
        sx, sy = [math.radians(s) for s in shear]

        cx, cy = center
        tx, ty = translate

        # 无缩放的旋转矩阵
        a = cos(rot - sy) / cos(sy)
        b = -cos(rot - sy) * tan(sx) / cos(sy) - sin(rot)
        c = sin(rot - sy) / cos(sy)
        d = -sin(rot - sy) * tan(sx) / cos(sy) + cos(rot)

        # 带有缩放和剪切的逆旋转矩阵
        # det([[a, b], [c, d]]) == 1，因为 det(rotation) = 1 且 det(shear) = 1
        M = [d, -b, 0, -c, a, 0]
        M = [x / scale for x in M]

        # 应用平移和中心平移的逆操作：RSS^-1 * C^-1 * T^-1
        M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
        M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)

        # 应用中心平移：C * RSS^-1 * C^-1 * T^-1
        M[2] += cx
        M[5] += cy
        # 返回计算得到的逆仿射矩阵
        return M

    @staticmethod
    # 获取图像增强的参数，包括旋转角度、平移、缩放范围、错切和图像高度
    def get_params(degrees, translate, scale_ranges, shears, height):
        # 从角度范围中随机选择一个角度
        angle = sample_sym(degrees)
        
        # 如果有平移参数，则计算最大平移距离
        if translate is not None:
            max_dx = translate[0] * height
            max_dy = translate[1] * height
            # 随机选择平移距离
            translations = (np.round(sample_sym(max_dx)),
                            np.round(sample_sym(max_dy)))
        else:
            translations = (0, 0)

        # 如果有缩放范围参数，则随机选择一个缩放比例
        if scale_ranges is not None:
            scale = sample_uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        # 如果有错切参数，则随机选择错切值
        if shears is not None:
            if len(shears) == 1:
                shear = [sample_sym(shears[0]), 0.]
            elif len(shears) == 2:
                shear = [sample_sym(shears[0]), sample_sym(shears[1])]
        else:
            shear = 0.0

        # 返回计算得到的参数：旋转角度、平移距离、缩放比例、错切值
        return angle, translations, scale, shear
    # 定义一个函数，用于对输入的图像进行仿射变换
    def __call__(self, img):
        # 获取输入图像的高度和宽度
        src_h, src_w = img.shape[:2]
        # 获取旋转角度、平移、缩放和剪切参数
        angle, translate, scale, shear = self.get_params(
            self.degrees, self.translate, self.scale, self.shear, src_h)

        # 获取仿射变换矩阵
        M = self._get_inverse_affine_matrix((src_w / 2, src_h / 2), angle,
                                            (0, 0), scale, shear)
        M = np.array(M).reshape(2, 3)

        # 定义起始点坐标
        startpoints = [(0, 0), (src_w - 1, 0), (src_w - 1, src_h - 1),
                       (0, src_h - 1)]
        # 定义投影函数
        project = lambda x, y, a, b, c: int(a * x + b * y + c)
        # 计算结束点坐标
        endpoints = [(project(x, y, *M[0]), project(x, y, *M[1]))
                     for x, y in startpoints]

        # 获取最小外接矩形
        rect = cv2.minAreaRect(np.array(endpoints))
        # 获取外接矩形的顶点坐标
        bbox = cv2.boxPoints(rect).astype(dtype=np.int32)
        # 计算外接矩形的最大和最小坐标
        max_x, max_y = bbox[:, 0].max(), bbox[:, 1].max()
        min_x, min_y = bbox[:, 0].min(), bbox[:, 1].min()

        # 计算目标图像的宽度和高度
        dst_w = int(max_x - min_x)
        dst_h = int(max_y - min_y)
        # 调整仿射变换矩阵的平移参数
        M[0, 2] += (dst_w - src_w) / 2
        M[1, 2] += (dst_h - src_h) / 2

        # 添加平移
        dst_w += int(abs(translate[0]))
        dst_h += int(abs(translate[1]))
        if translate[0] < 0: M[0, 2] += abs(translate[0])
        if translate[1] < 0: M[1, 2] += abs(translate[1])

        # 获取插值标志
        flags = get_interpolation()
        # 返回仿射变换后的图像
        return cv2.warpAffine(
            img,
            M, (dst_w, dst_h),
            flags=flags,
            borderMode=cv2.BORDER_REPLICATE)
# 定义一个类 CVRandomPerspective，用于执行随机透视变换
class CVRandomPerspective(object):
    # 初始化方法，设置扭曲程度参数
    def __init__(self, distortion=0.5):
        self.distortion = distortion

    # 获取透视变换的参数
    def get_params(self, width, height, distortion):
        # 生成高度方向和宽度方向的偏移量
        offset_h = sample_asym(
            distortion * height / 2, size=4).astype(dtype=np.int32)
        offset_w = sample_asym(
            distortion * width / 2, size=4).astype(dtype=np.int32)
        # 计算四个顶点的坐标
        topleft = (offset_w[0], offset_h[0])
        topright = (width - 1 - offset_w[1], offset_h[1])
        botright = (width - 1 - offset_w[2], height - 1 - offset_h[2])
        botleft = (offset_w[3], height - 1 - offset_h[3])

        # 定义起始点和结束点
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1),
                       (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return np.array(
            startpoints, dtype=np.float32), np.array(
                endpoints, dtype=np.float32)

    # 实现对象的调用方法，对图像进行透视变换
    def __call__(self, img):
        # 获取图像的高度和宽度
        height, width = img.shape[:2]
        # 获取透视变换的起始点和结束点
        startpoints, endpoints = self.get_params(width, height, self.distortion)
        # 获取透视变换矩阵
        M = cv2.getPerspectiveTransform(startpoints, endpoints)

        # TODO: more robust way to crop image
        # 获取最小外接矩形
        rect = cv2.minAreaRect(endpoints)
        # 获取外接矩形的四个顶点坐标
        bbox = cv2.boxPoints(rect).astype(dtype=np.int32)
        max_x, max_y = bbox[:, 0].max(), bbox[:, 1].max()
        min_x, min_y = bbox[:, 0].min(), bbox[:, 1].min()
        min_x, min_y = max(min_x, 0), max(min_y, 0)

        # 获取插值方式的标志
        flags = get_interpolation()
        # 对图像进行透视变换
        img = cv2.warpPerspective(
            img,
            M, (max_x, max_y),
            flags=flags,
            borderMode=cv2.BORDER_REPLICATE)
        # 裁剪图像
        img = img[min_y:, min_x:]
        return img


# 定义一个类 CVRescale
class CVRescale(object):
    # 初始化函数，定义图像尺度使用高斯金字塔，并将图像重新调整到目标尺度
    def __init__(self, factor=4, base_size=(128, 512)):
        """ Define image scales using gaussian pyramid and rescale image to target scale.
        
        Args:
            factor: the decayed factor from base size, factor=4 keeps target scale by default.
            base_size: base size the build the bottom layer of pyramid
        """
        # 如果 factor 是数字类型，则将其四舍五入取整
        if isinstance(factor, numbers.Number):
            self.factor = round(sample_uniform(0, factor))
        # 如果 factor 是元组或列表类型且长度为2，则在指定范围内随机选择一个值
        elif isinstance(factor, (tuple, list)) and len(factor) == 2:
            self.factor = round(sample_uniform(factor[0], factor[1]))
        else:
            # 抛出异常，factor 必须是数字或长度为2的列表
            raise Exception('factor must be number or list with length 2')
        # 确保 factor 是有效的
        self.base_h, self.base_w = base_size[:2]

    # 调用函数，根据 factor 对图像进行缩放
    def __call__(self, img):
        # 如果 factor 为 0，则直接返回原图像
        if self.factor == 0: return img
        # 获取原图像的高度和宽度
        src_h, src_w = img.shape[:2]
        # 设置当前宽度和高度为基础宽度和高度
        cur_w, cur_h = self.base_w, self.base_h
        # 将图像缩放到当前宽度和高度
        scale_img = cv2.resize(
            img, (cur_w, cur_h), interpolation=get_interpolation())
        # 根据 factor 进行多次高斯金字塔降采样
        for _ in range(self.factor):
            scale_img = cv2.pyrDown(scale_img)
        # 将缩放后的图像重新调整到原始图像的尺寸
        scale_img = cv2.resize(
            scale_img, (src_w, src_h), interpolation=get_interpolation())
        # 返回缩放后的图像
        return scale_img
# 定义一个添加高斯噪声的类
class CVGaussianNoise(object):
    # 初始化函数，设置均值和方差
    def __init__(self, mean=0, var=20):
        self.mean = mean
        # 判断方差的类型，如果是数字则取绝对值，如果是元组或列表则取均匀分布的随机值
        if isinstance(var, numbers.Number):
            self.var = max(int(sample_asym(var)), 1)
        elif isinstance(var, (tuple, list)) and len(var) == 2:
            self.var = int(sample_uniform(var[0], var[1]))
        else:
            raise Exception('degree must be number or list with length 2')

    # 添加高斯噪声的调用函数
    def __call__(self, img):
        # 生成符合正态分布的噪声
        noise = np.random.normal(self.mean, self.var**0.5, img.shape)
        # 将噪声添加到图像上，并将结果限制在0到255之间
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return img

# 定义一个添加运动模糊的类
class CVMotionBlur(object):
    # 初始化函数，设置模糊程度和角度
    def __init__(self, degrees=12, angle=90):
        # 判断模糊程度的类型，如果是数字则取绝对值，如果是元组或列表则取均匀分布的随机值
        if isinstance(degrees, numbers.Number):
            self.degree = max(int(sample_asym(degrees)), 1)
        elif isinstance(degrees, (tuple, list)) and len(degrees) == 2:
            self.degree = int(sample_uniform(degrees[0], degrees[1]))
        else:
            raise Exception('degree must be number or list with length 2')
        # 随机生成一个角度
        self.angle = sample_uniform(-angle, angle)

    # 添加运动模糊的调用函数
    def __call__(self, img):
        # 生成旋转矩阵
        M = cv2.getRotationMatrix2D((self.degree // 2, self.degree // 2), self.angle, 1)
        # 创建运动模糊核
        motion_blur_kernel = np.zeros((self.degree, self.degree))
        motion_blur_kernel[self.degree // 2, :] = 1
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (self.degree, self.degree))
        motion_blur_kernel = motion_blur_kernel / self.degree
        # 对图像进行滤波处理
        img = cv2.filter2D(img, -1, motion_blur_kernel)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

# 定义一个几何变换类
class CVGeometry(object):
    # 初始化函数，设置默认参数值
    def __init__(self,
                 degrees=15,  # 旋转角度范围
                 translate=(0.3, 0.3),  # 平移范围
                 scale=(0.5, 2.),  # 缩放范围
                 shear=(45, 15),  # 剪切范围
                 distortion=0.5,  # 扭曲程度
                 p=0.5):  # 概率
        self.p = p  # 将概率值赋给实例变量
        type_p = random.random()  # 生成一个随机数
        # 根据随机数的大小选择不同的数据增强方式
        if type_p < 0.33:
            self.transforms = CVRandomRotation(degrees=degrees)  # 随机旋转
        elif type_p < 0.66:
            self.transforms = CVRandomAffine(
                degrees=degrees, translate=translate, scale=scale, shear=shear)  # 随机仿射变换
        else:
            self.transforms = CVRandomPerspective(distortion=distortion)  # 随机透视变换
    
    # 调用函数，对输入的图像进行数据增强
    def __call__(self, img):
        # 根据概率决定是否进行数据增强
        if random.random() < self.p:
            return self.transforms(img)  # 对图像进行数据增强
        else:
            return img  # 返回原始图像
# 定义一个类 CVDeterioration，用于图像数据的退化处理
class CVDeterioration(object):
    # 初始化方法，接受变量 var、degrees、factor 和概率 p
    def __init__(self, var, degrees, factor, p=0.5):
        # 将概率 p 存储到实例变量中
        self.p = p
        # 初始化一个空列表 transforms，用于存储图像变换操作
        transforms = []
        # 如果 var 不为 None，则添加高斯噪声变换操作到 transforms 列表中
        if var is not None:
            transforms.append(CVGaussianNoise(var=var))
        # 如果 degrees 不为 None，则添加运动模糊变换操作到 transforms 列表中
        if degrees is not None:
            transforms.append(CVMotionBlur(degrees=degrees))
        # 如果 factor 不为 None，则添加缩放变换操作到 transforms 列表中
        if factor is not None:
            transforms.append(CVRescale(factor=factor))

        # 随机打乱 transforms 列表中的顺序
        random.shuffle(transforms)
        # 将 transforms 列表中的变换操作组合成一个整体
        transforms = Compose(transforms)
        # 将组合后的变换操作存储到实例变量中
        self.transforms = transforms

    # 定义 __call__ 方法，用于对图像进行变换操作
    def __call__(self, img):
        # 如果生成的随机数小于概率 p
        if random.random() < self.p:
            # 对图像应用之前定义的变换操作
            return self.transforms(img)
        else:
            # 否则返回原始图像
            return img

# 定义一个类 CVColorJitter，用于图像颜色调整
class CVColorJitter(object):
    # 初始化方法，接受亮度、对比度、饱和度、色调和概率 p
    def __init__(self,
                 brightness=0.5,
                 contrast=0.5,
                 saturation=0.5,
                 hue=0.1,
                 p=0.5):
        # 将概率 p 存储到实例变量中
        self.p = p
        # 初始化一个颜色调整操作对象，包括亮度、对比度、饱和度和色调
        self.transforms = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue)

    # 定义 __call__ 方法，用于对图像进行颜色调整操作
    def __call__(self, img):
        # 如果生成的随机数小于概率 p，则应用颜色调整操作
        if random.random() < self.p: return self.transforms(img)
        else: return img

# 定义一个类 SVTRDeterioration，用于视频数据的退化处理
class SVTRDeterioration(object):
    # 初始化方法，接受变量 var、degrees、factor 和概率 p
    def __init__(self, var, degrees, factor, p=0.5):
        # 将概率 p 存储到实例变量中
        self.p = p
        # 初始化一个空列表 transforms，用于存储视频变换操作
        transforms = []
        # 如果 var 不为 None，则添加高斯噪声变换操作到 transforms 列表中
        if var is not None:
            transforms.append(CVGaussianNoise(var=var))
        # 如果 degrees 不为 None，则添加运动模糊变换操作到 transforms 列表中
        if degrees is not None:
            transforms.append(CVMotionBlur(degrees=degrees))
        # 如果 factor 不为 None，则添加缩放变换操作到 transforms 列表中
        if factor is not None:
            transforms.append(CVRescale(factor=factor))
        # 将视频变换操作存储到实例变量中
        self.transforms = transforms

    # 定义 __call__ 方法，用于对视频进行变换操作
    def __call__(self, img):
        # 如果生成的随机数小于概率 p
        if random.random() < self.p:
            # 随机打乱视频变换操作的顺序
            random.shuffle(self.transforms)
            # 将打乱后的视频变换操作组合成一个整体
            transforms = Compose(self.transforms)
            # 对视频应用组合后的变换操作
            return transforms(img)
        else:
            # 否则返回原始视频
            return img

# 定义一个类 SVTRGeometry，用于视频几何变换
    # 初始化函数，设置数据增强的参数
    def __init__(self,
                 aug_type=0,  # 数据增强类型，默认为0
                 degrees=15,  # 旋转角度，默认为15
                 translate=(0.3, 0.3),  # 平移范围，默认为(0.3, 0.3)
                 scale=(0.5, 2.),  # 缩放范围，默认为(0.5, 2.)
                 shear=(45, 15),  # 剪切范围，默认为(45, 15)
                 distortion=0.5,  # 扭曲程度，默认为0.5
                 p=0.5):  # 数据增强的概率，默认为0.5
        self.aug_type = aug_type  # 存储数据增强类型
        self.p = p  # 存储数据增强的概率
        self.transforms = []  # 初始化数据增强的变换列表
        self.transforms.append(CVRandomRotation(degrees=degrees))  # 添加随机旋转变换
        self.transforms.append(
            CVRandomAffine(
                degrees=degrees, translate=translate, scale=scale, shear=shear))  # 添加随机仿射变换
        self.transforms.append(CVRandomPerspective(distortion=distortion))  # 添加随机透视变换
    
    # 调用函数，对输入的图像进行数据增强
    def __call__(self, img):
        if random.random() < self.p:  # 根据数据增强的概率进行判断
            if self.aug_type:  # 如果有指定数据增强类型
                random.shuffle(self.transforms)  # 随机打乱变换列表
                transforms = Compose(self.transforms[:random.randint(1, 3)])  # 随机选择1到3个变换组成组合
                img = transforms(img)  # 对图像应用组合的变换
            else:
                img = self.transforms[random.randint(0, 2)](img)  # 随机选择一个变换应用到图像
            return img  # 返回增强后的图像
        else:
            return img  # 不进行数据增强，直接返回原图像
```