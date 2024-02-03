# `.\PaddleOCR\test_tipc\supplementary\data.py`

```
# 导入 numpy 和 paddle 模块
import numpy as np
import paddle
# 导入 os、cv2 和 glob 模块
import os
import cv2
import glob

# 定义一个函数 transform，用于对数据进行转换
def transform(data, ops=None):
    """ transform """
    # 如果 ops 为空，则初始化为空列表
    if ops is None:
        ops = []
    # 遍历 ops 列表中的操作，对数据进行相应的转换
    for op in ops:
        data = op(data)
        # 如果转换后的数据为空，则返回 None
        if data is None:
            return None
    # 返回转换后的数据
    return data

# 创建操作符函数，根据配置参数创建操作符
def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config
    Args:
        params(list): a dict list, used to create some operators
    """
    # 断言 op_param_list 是一个列表
    assert isinstance(op_param_list, list), ('operator config should be a list')
    # 初始化 ops 列表
    ops = []
    # 遍历操作符参数列表
    for operator in op_param_list:
        # 断言 operator 是一个字典且长度为 1
        assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
        # 获取操作符名称和参数
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        # 如果全局配置不为空，则更新参数
        if global_config is not None:
            param.update(global_config)
        # 根据操作符名称和参数创建操作符对象
        op = eval(op_name)(**param)
        # 将操作符对象添加到 ops 列表中
        ops.append(op)
    # 返回操作符列表
    return ops

# 定义一个 DecodeImage 类，用于解码图像
class DecodeImage(object):
    """ decode image """

    # 初始化函数，设置图像模式和通道顺序
    def __init__(self, img_mode='RGB', channel_first=False, **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first
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
        # 将图像数据转换为 numpy 数组
        img = np.frombuffer(img, dtype='uint8')
        # 使用 OpenCV 解码图像数据
        img = cv2.imdecode(img, 1)
        # 如果解码失败，则返回 None
        if img is None:
            return None
        # 如果图像模式为灰度，则将图像转换为 BGR 模式
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 如果图像模式为 RGB，则确保图像通道数为3，然后将通道顺序反转
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]

        # 如果需要将通道放在第一维，则进行转置操作
        if self.channel_first:
            img = img.transpose((2, 0, 1))

        # 更新输入数据中的图像数据和原始图像数据
        data['image'] = img
        data['src_image'] = img
        # 返回处理后的数据
        return data
class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        # 如果 scale 是字符串，则将其转换为相应的值
        if isinstance(scale, str):
            scale = eval(scale)
        # 设置 scale，默认为 1.0 / 255.0
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        # 设置 mean，默认为 [0.485, 0.456, 0.406]
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        # 设置 std，默认为 [0.229, 0.224, 0.225]
        std = std if std is not None else [0.229, 0.224, 0.225]

        # 根据 order 设置 shape
        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        # 将 mean 转换为 numpy 数组，并设置为 float32 类型
        self.mean = np.array(mean).reshape(shape).astype('float32')
        # 将 std 转换为 numpy 数组，并设置为 float32 类型
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        # 获取图像数据
        img = data['image']
        from PIL import Image
        # 如果图像是 PIL.Image 类型，则转换为 numpy 数组
        if isinstance(img, Image.Image):
            img = np.array(img)
        # 断言图像是 numpy 数组类型
        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        # 对图像进行归一化处理
        data['image'] = (
            img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        # 获取图像数据
        img = data['image']
        from PIL import Image
        # 如果图像是 PIL.Image 类型，则转换为 numpy 数组
        if isinstance(img, Image.Image):
            img = np.array(img)
        # 将图像转置为 chw 格式
        data['image'] = img.transpose((2, 0, 1))

        # 获取原始图像数据
        src_img = data['src_image']
        from PIL import Image
        # 如果图像是 PIL.Image 类型，则转换为 numpy 数组
        if isinstance(img, Image.Image):
            src_img = np.array(src_img)
        # 将原始图像转置为 chw 格式
        data['src_image'] = img.transpose((2, 0, 1))

        return data


class SimpleDataset(nn.Dataset):
    def __init__(self, config, mode, logger, seed=None):
        # 设置 logger 和 mode
        self.logger = logger
        self.mode = mode.lower()

        # 获取数据目录
        data_dir = config['Train']['data_dir']

        # 获取图像列表
        imgs_list = self.get_image_list(data_dir)

        # 创建数据处理操作
        self.ops = create_operators(cfg['transforms'], None)
    # 获取指定目录下所有的 PNG 图像文件路径列表
    def get_image_list(self, img_dir):
        # 使用 glob 模块匹配指定目录下所有的 PNG 图像文件路径
        imgs = glob.glob(os.path.join(img_dir, "*.png"))
        # 如果未找到任何图像文件，则抛出数值错误异常
        if len(imgs) == 0:
            raise ValueError(f"not any images founded in {img_dir}")
        # 返回图像文件路径列表
        return imgs
    
    # 重载索引操作符，返回指定索引位置的元素
    def __getitem__(self, idx):
        # 返回空值，需要根据具体需求进行实现
        return None
```