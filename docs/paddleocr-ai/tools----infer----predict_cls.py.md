# `.\PaddleOCR\tools\infer\predict_cls.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本使用此文件；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”分发的，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。
import os
import sys

# 获取当前文件所在目录的绝对路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录添加到系统路径中
sys.path.append(__dir__)
# 将当前文件所在目录的上级目录添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

# 导入必要的库
import cv2
import copy
import numpy as np
import math
import time
import traceback

# 导入自定义工具模块
import tools.infer.utility as utility
# 导入后处理模块
from ppocr.postprocess import build_post_process
# 导入日志记录器
from ppocr.utils.logging import get_logger
# 导入工具函数
from ppocr.utils.utility import get_image_file_list, check_and_read

# 获取日志记录器
logger = get_logger()

# 定义文本分类器类
class TextClassifier(object):
    def __init__(self, args):
        # 解析参数中的分类器图像形状
        self.cls_image_shape = [int(v) for v in args.cls_image_shape.split(",")]
        # 解析参数中的分类器批次数
        self.cls_batch_num = args.cls_batch_num
        # 解析参数中的分类器阈值
        self.cls_thresh = args.cls_thresh
        # 构建后处理参数
        postprocess_params = {
            'name': 'ClsPostProcess',
            "label_list": args.label_list,
        }
        # 构建后处理操作
        self.postprocess_op = build_post_process(postprocess_params)
        # 创建预测器、输入张量、输出张量和日志记录器
        self.predictor, self.input_tensor, self.output_tensors, _ = \
            utility.create_predictor(args, 'cls', logger)
        # 检查是否使用 ONNX 模型
        self.use_onnx = args.use_onnx
    # 重新调整并标准化图像大小
    def resize_norm_img(self, img):
        # 获取图像通道数、高度和宽度
        imgC, imgH, imgW = self.cls_image_shape
        # 获取原始图像的高度和宽度
        h = img.shape[0]
        w = img.shape[1]
        # 计算宽高比
        ratio = w / float(h)
        # 根据比例调整宽度
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        # 调整图像大小
        resized_image = cv2.resize(img, (resized_w, imgH))
        # 将图像数据类型转换为 float32
        resized_image = resized_image.astype('float32')
        # 如果图像通道数为1
        if self.cls_image_shape[0] == 1:
            # 对图像进行归一化处理
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            # 调整图像维度顺序，并进行归一化处理
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        # 标准化图像数据
        resized_image -= 0.5
        resized_image /= 0.5
        # 创建一个全零填充的数组，用于存放调整后的图像数据
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        # 将调整后的图像数据填充到全零数组中
        padding_im[:, :, 0:resized_w] = resized_image
        # 返回填充后的图像数据
        return padding_im
# 主函数，接受参数并执行主要逻辑
def main(args):
    # 获取图像文件列表
    image_file_list = get_image_file_list(args.image_dir)
    # 创建文本分类器对象
    text_classifier = TextClassifier(args)
    # 有效图像文件列表
    valid_image_file_list = []
    # 图像列表
    img_list = []
    # 遍历图像文件列表
    for image_file in image_file_list:
        # 检查并读取图像文件
        img, flag, _ = check_and_read(image_file)
        # 如果读取失败，则使用OpenCV重新读取
        if not flag:
            img = cv2.imread(image_file)
        # 如果图像为空
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        # 将有效图像文件添加到列表
        valid_image_file_list.append(image_file)
        # 将图像添加到图像列表
        img_list.append(img)
    try:
        # 调用文本分类器进行分类
        img_list, cls_res, predict_time = text_classifier(img_list)
    except Exception as E:
        # 捕获异常并记录日志
        logger.info(traceback.format_exc())
        logger.info(E)
        # 退出程序
        exit()
    # 遍历图像列表
    for ino in range(len(img_list)):
        # 记录预测结果
        logger.info("Predicts of {}:{}".format(valid_image_file_list[ino],
                                               cls_res[ino]))


if __name__ == "__main__":
    # 解析参数并执行主函数
    main(utility.parse_args())
```