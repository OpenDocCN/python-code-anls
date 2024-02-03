# `.\PaddleOCR\StyleText\engine\predictors.py`

```py
# 版权声明
# 2020年PaddlePaddle作者保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。
import numpy as np
import cv2
import math
import paddle

# 导入自定义模块
from arch import style_text_rec
from utils.sys_funcs import check_gpu
from utils.logging import get_logger

# 定义StyleTextRecPredictor类
class StyleTextRecPredictor(object):
    def __init__(self, config):
        # 从配置中获取算法名称
        algorithm = config['Predictor']['algorithm']
        # 确保算法名称为"StyleTextRec"
        assert algorithm in ["StyleTextRec"], "Generator {} not supported.".format(algorithm)
        # 从配置中获取是否使用GPU
        use_gpu = config["Global"]['use_gpu']
        # 检查GPU是否可用
        check_gpu(use_gpu)
        # 设置PaddlePaddle设备为GPU或CPU
        paddle.set_device('gpu' if use_gpu else 'cpu')
        # 获取日志记录器
        self.logger = get_logger()
        # 根据算法名称获取生成器对象
        self.generator = getattr(style_text_rec, algorithm)(config)
        # 从配置中获取图像高度和宽度
        self.height = config["Global"]["image_height"]
        self.width = config["Global"]["image_width"]
        # 从配置中获取缩放比例、均值和标准差
        self.scale = config["Predictor"]["scale"]
        self.mean = config["Predictor"]["mean"]
        self.std = config["Predictor"]["std"]
        # 从配置中获取是否扩展结果
        self.expand_result = config["Predictor"]["expand_result"]

    # 将图像列表调整为相同高度
    def reshape_to_same_height(self, img_list):
        # 获取第一个图像的高度
        h = img_list[0].shape[0]
        # 遍历图像列表
        for idx in range(1, len(img_list)):
            # 计算新的宽度以保持高度一致
            new_w = round(1.0 * img_list[idx].shape[1] / img_list[idx].shape[0] * h)
            # 调整图像大小
            img_list[idx] = cv2.resize(img_list[idx], (new_w, h))
        return img_list
    # 预测单个图像的合成结果
    def predict_single_image(self, style_input, text_input):
        # 将样式输入和文本输入进行处理
        style_input = self.rep_style_input(style_input, text_input)
        # 对样式输入进行预处理
        tensor_style_input = self.preprocess(style_input)
        # 对文本输入进行预处理
        tensor_text_input = self.preprocess(text_input)
        # 使用生成器模型进行前向传播，得到样式文本合成结果
        style_text_result = self.generator.forward(tensor_style_input, tensor_text_input)
        # 对合成结果进行后处理，得到合成的图像
        fake_fusion = self.postprocess(style_text_result["fake_fusion"])
        fake_text = self.postprocess(style_text_result["fake_text"])
        fake_sk = self.postprocess(style_text_result["fake_sk"])
        fake_bg = self.postprocess(style_text_result["fake_bg"])
        
        # 获取文本边界框
        bbox = self.get_text_boundary(fake_text)
        # 如果存在文本边界框
        if bbox:
            left, right, top, bottom = bbox
            # 对合成结果进行裁剪，保留文本区域
            fake_fusion = fake_fusion[top:bottom, left:right, :]
            fake_text = fake_text[top:bottom, left:right, :]
            fake_sk = fake_sk[top:bottom, left:right, :]
            fake_bg = fake_bg[top:bottom, left:right, :]

        # 返回合成结果字典
        return {
            "fake_fusion": fake_fusion,
            "fake_text": fake_text,
            "fake_sk": fake_sk,
            "fake_bg": fake_bg,
        }

    # 预测多个文本输入的合成结果
    def predict(self, style_input, text_input_list):
        # 如果文本输入是单个文本，直接调用预测单个图像的方法
        if not isinstance(text_input_list, (tuple, list)):
            return self.predict_single_image(style_input, text_input_list)

        # 存储合成结果的列表
        synth_result_list = []
        # 遍历文本输入列表
        for text_input in text_input_list:
            # 对每个文本输入进行预测
            synth_result = self.predict_single_image(style_input, text_input)
            synth_result_list.append(synth_result)

        # 对合成结果进行处理，使其高度相同
        for key in synth_result:
            res = [r[key] for r in synth_result_list]
            res = self.reshape_to_same_height(res)
            synth_result[key] = np.concatenate(res, axis=1)
        
        # 返回合成结果
        return synth_result
    # 对输入图像进行预处理，包括归一化、尺寸调整等操作
    def preprocess(self, img):
        # 将图像转换为 float32 类型，并进行归一化处理
        img = (img.astype('float32') * self.scale - self.mean) / self.std
        # 获取图像的高度、宽度和通道数
        img_height, img_width, channel = img.shape
        # 断言通道数为 3，即 RGB 图像
        assert channel == 3, "Please use an rgb image."
        # 计算图像宽高比
        ratio = img_width / float(img_height)
        # 根据比例调整图像宽度
        if math.ceil(self.height * ratio) > self.width:
            resized_w = self.width
        else:
            resized_w = int(math.ceil(self.height * ratio))
        # 调整图像尺寸
        img = cv2.resize(img, (resized_w, self.height))

        # 创建新的图像数组，尺寸为指定的高度和宽度
        new_img = np.zeros([self.height, self.width, 3]).astype('float32')
        # 将调整后的图像复制到新图像数组中
        new_img[:, 0:resized_w, :] = img
        # 调整图像维度顺序
        img = new_img.transpose((2, 0, 1))
        # 增加一个维度，用于模型输入
        img = img[np.newaxis, :, :, :]
        return paddle.to_tensor(img)

    # 对模型输出的张量进行后处理，包括反归一化、转换数据类型等操作
    def postprocess(self, tensor):
        # 将张量转换为 numpy 数组
        img = tensor.numpy()[0]
        # 调整图像维度顺序
        img = img.transpose((1, 2, 0))
        # 反归一化处理
        img = (img * self.std + self.mean) / self.scale
        # 将像素值限制在 0 到 255 之间
        img = np.maximum(img, 0.0)
        img = np.minimum(img, 255.0)
        # 转换数据类型为 uint8
        img = img.astype('uint8')
        return img

    # 复制风格输入以适应文本输入的尺寸
    def rep_style_input(self, style_input, text_input):
        # 计算复制次数，以适应文本输入的尺寸
        rep_num = int(1.2 * (text_input.shape[1] / text_input.shape[0]) /
                      (style_input.shape[1] / style_input.shape[0])) + 1
        # 在宽度方向上复制风格输入
        style_input = np.tile(style_input, reps=[1, rep_num, 1])
        # 计算最大宽度，以适应指定的宽高比
        max_width = int(self.width / self.height * style_input.shape[0])
        # 裁剪风格输入的宽度
        style_input = style_input[:, :max_width, :]
        return style_input
    # 获取文本图像的边界框
    def get_text_boundary(self, text_img):
        # 获取文本图像的高度和宽度
        img_height = text_img.shape[0]
        img_width = text_img.shape[1]
        # 设置边界值
        bounder = 3
        # 使用 Canny 边缘检测算法获取文本图像的边缘图像
        text_canny_img = cv2.Canny(text_img, 10, 20)
        # 沿水平方向统计边缘像素点数量
        edge_num_h = text_canny_img.sum(axis=0)
        # 获取水平方向上非零像素点的索引
        no_zero_list_h = np.where(edge_num_h > 0)[0]
        # 沿垂直方向统计边缘像素点数量
        edge_num_w = text_canny_img.sum(axis=1)
        # 获取垂直方向上非零像素点的索引
        no_zero_list_w = np.where(edge_num_w > 0)[0]
        # 如果水平或垂直方向上没有非零像素点，则返回空
        if len(no_zero_list_h) == 0 or len(no_zero_list_w) == 0:
            return None
        # 计算边界框的左、右、上、下边界
        left = max(no_zero_list_h[0] - bounder, 0)
        right = min(no_zero_list_h[-1] + bounder, img_width)
        top = max(no_zero_list_w[0] - bounder, 0)
        bottom = min(no_zero_list_w[-1] + bounder, img_height)
        # 返回边界框的左、右、上、下边界坐标
        return [left, right, top, bottom]
```