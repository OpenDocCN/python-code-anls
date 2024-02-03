# `.\PaddleOCR\tools\infer\predict_rec.py`

```py
# 版权声明
# 2020年PaddlePaddle作者保留所有权利
#
# 根据Apache许可证2.0版（“许可证”）获得许可
# 您可能不得使用此文件，除非符合许可证的规定
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证分发，基于“按原样”分发，
# 没有任何形式的担保或条件，无论是明示或暗示的。
# 有关特定语言的权限和限制，请参阅许可证
# 限制
import os
import sys
从PIL库中导入Image模块
from PIL import Image
__dir__ = os.path.dirname(os.path.abspath(__file__))
将当前文件所在目录添加到系统路径中
sys.path.append(__dir__)
将当前文件所在目录的上一级目录添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

设置环境变量FLAGS_allocator_strategy为'auto_growth'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

导入cv2（OpenCV）、numpy、math、time、traceback、paddle等库
import cv2
import numpy as np
import math
import time
import traceback
import paddle

从tools.infer.utility模块中导入utility函数
import tools.infer.utility as utility
从ppocr.postprocess模块中导入build_post_process函数
from ppocr.postprocess import build_post_process
从ppocr.utils.logging模块中导入get_logger函数
from ppocr.utils.logging import get_logger
从ppocr.utils.utility模块中导入get_image_file_list、check_and_read函数
from ppocr.utils.utility import get_image_file_list, check_and_read

获取logger对象
logger = get_logger()

定义TextRecognizer类
class TextRecognizer(object):
    定义resize_norm_img_vl方法，接受img和image_shape两个参数
    def resize_norm_img_vl(self, img, image_shape):

        从image_shape中获取图像的通道数、高度和宽度
        imgC, imgH, imgW = image_shape
        将图像从BGR格式转换为RGB格式
        img = img[:, :, ::-1]  # bgr2rgb
        将图像调整大小为指定的imgW和imgH，使用线性插值
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        将调整大小后的图像转换为float32类型
        resized_image = resized_image.astype('float32')
        将图像通道顺序转换为CHW
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        返回调整大小和归一化后的图像
        return resized_image
    # 重新调整并标准化图像大小，返回调整后的图像
    def resize_norm_img_srn(self, img, image_shape):
        # 获取图像通道数、高度和宽度
        imgC, imgH, imgW = image_shape

        # 创建一个全黑的图像数组
        img_black = np.zeros((imgH, imgW))
        # 获取输入图像的高度和宽度
        im_hei = img.shape[0]
        im_wid = img.shape[1]

        # 根据宽高比例进行图像调整
        if im_wid <= im_hei * 1:
            img_new = cv2.resize(img, (imgH * 1, imgH))
        elif im_wid <= im_hei * 2:
            img_new = cv2.resize(img, (imgH * 2, imgH))
        elif im_wid <= im_hei * 3:
            img_new = cv2.resize(img, (imgH * 3, imgH))
        else:
            img_new = cv2.resize(img, (imgW, imgH))

        # 将调整后的图像转换为 NumPy 数组
        img_np = np.asarray(img_new)
        # 将图像转换为灰度图像
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        # 将灰度图像复制到黑色图像的左侧
        img_black[:, 0:img_np.shape[1]] = img_np
        # 增加一个维度，使图像变为三维
        img_black = img_black[:, :, np.newaxis]

        # 获取图像的行数、列数和通道数
        row, col, c = img_black.shape
        # 将通道数设置为1
        c = 1

        # 返回调整后的图像，将图像形状重新调整为(c, row, col)并转换为浮点数类型
        return np.reshape(img_black, (c, row, col)).astype(np.float32)
    # 定义一个方法，用于生成其他输入数据，包括编码器词位置、GSRM词位置、GSRM自注意力偏置1和2
    def srn_other_inputs(self, image_shape, num_heads, max_text_length):

        # 获取图像的通道数、高度和宽度
        imgC, imgH, imgW = image_shape
        # 计算特征维度
        feature_dim = int((imgH / 8) * (imgW / 8))

        # 生成编码器词位置数组
        encoder_word_pos = np.array(range(0, feature_dim)).reshape(
            (feature_dim, 1)).astype('int64')
        # 生成GSRM词位置数组
        gsrm_word_pos = np.array(range(0, max_text_length)).reshape(
            (max_text_length, 1)).astype('int64')

        # 生成全为1的GSRM自注意力偏置数据
        gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
        # 生成上三角矩阵的GSRM自注意力偏置1
        gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias1 = np.tile(
            gsrm_slf_attn_bias1,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        # 生成下三角矩阵的GSRM自注意力偏置2
        gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias2 = np.tile(
            gsrm_slf_attn_bias2,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        # 将编码器词位置和GSRM词位置转为二维数组
        encoder_word_pos = encoder_word_pos[np.newaxis, :]
        gsrm_word_pos = gsrm_word_pos[np.newaxis, :]

        # 返回生成的输入数据
        return [
            encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2
        ]

    # 定义一个方法，用于处理SRN的图像数据
    def process_image_srn(self, img, image_shape, num_heads, max_text_length):
        # 对图像进行归一化和缩放
        norm_img = self.resize_norm_img_srn(img, image_shape)
        norm_img = norm_img[np.newaxis, :]

        # 调用srn_other_inputs方法生成其他输入数据
        [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = \
            self.srn_other_inputs(image_shape, num_heads, max_text_length)

        # 将数据类型转换为float32和int64
        gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1.astype(np.float32)
        gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2.astype(np.float32)
        encoder_word_pos = encoder_word_pos.astype(np.int64)
        gsrm_word_pos = gsrm_word_pos.astype(np.int64)

        # 返回处理后的图像数据和其他输入数据
        return (norm_img, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
                gsrm_slf_attn_bias2)
    # 根据指定的图像形状和宽度下采样比例对图像进行调整和归一化处理
    def resize_norm_img_sar(self, img, image_shape,
                            width_downsample_ratio=0.25):
        # 解析图像形状参数
        imgC, imgH, imgW_min, imgW_max = image_shape
        # 获取图像的高度和宽度
        h = img.shape[0]
        w = img.shape[1]
        valid_ratio = 1.0
        # 确保新的宽度是宽度除数的整数倍
        width_divisor = int(1 / width_downsample_ratio)
        # 计算图像宽高比
        ratio = w / float(h)
        resize_w = math.ceil(imgH * ratio)
        # 调整图像宽度，使其成为宽度除数的整数倍
        if resize_w % width_divisor != 0:
            resize_w = round(resize_w / width_divisor) * width_divisor
        # 如果指定了最小宽度，确保调整后的宽度不小于最小宽度
        if imgW_min is not None:
            resize_w = max(imgW_min, resize_w)
        # 如果指定了最大宽度，计算有效比例并确保调整后的宽度不超过最大宽度
        if imgW_max is not None:
            valid_ratio = min(1.0, 1.0 * resize_w / imgW_max)
            resize_w = min(imgW_max, resize_w)
        # 调整图像大小
        resized_image = cv2.resize(img, (resize_w, imgH))
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
        # 创建一个填充图像，用-1填充，确保宽度为最大宽度
        padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
        padding_im[:, :, 0:resize_w] = resized_image
        pad_shape = padding_im.shape

        return padding_im, resize_shape, pad_shape, valid_ratio
    def resize_norm_img_spin(self, img):
        # 将彩色图像转换为灰度图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 将图像大小调整为指定大小
        img = cv2.resize(img, tuple([100, 32]), cv2.INTER_CUBIC)
        # 将图像转换为 float32 类型的数组
        img = np.array(img, np.float32)
        # 在最后一个维度上添加一个维度
        img = np.expand_dims(img, -1)
        # 调整数组的维度顺序
        img = img.transpose((2, 0, 1))
        # 定义均值和标准差
        mean = [127.5]
        std = [127.5]
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        mean = np.float32(mean.reshape(1, -1))
        stdinv = 1 / np.float32(std.reshape(1, -1))
        # 对图像进行均值和标准差归一化
        img -= mean
        img *= stdinv
        return img

    def resize_norm_img_svtr(self, img, image_shape):
        # 获取图像的通道数、高度和宽度
        imgC, imgH, imgW = image_shape
        # 调整图像大小为指定大小
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        # 将图像转换为 float32 类型
        resized_image = resized_image.astype('float32')
        # 调整数组的维度顺序并归一化
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        return resized_image

    def resize_norm_img_abinet(self, img, image_shape):
        # 获取图像的通道数、高度和宽度
        imgC, imgH, imgW = image_shape
        # 调整图像大小为指定大小
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        # 将图像转换为 float32 类型并归一化
        resized_image = resized_image.astype('float32') / 255.
        # 定义均值和标准差
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        # 对图像进行均值和标准差归一化
        resized_image = (
            resized_image - mean[None, None, ...]) / std[None, None, ...]
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = resized_image.astype('float32')
        return resized_image
    # 标准化图像并将其转换为模型可接受的格式
    def norm_img_can(self, img, image_shape):

        # 将图像转换为灰度图像，因为 CAN 模型只能预测灰度图像
        img = cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY)

        # 如果需要反转图像像素值
        if self.inverse:
            img = 255 - img

        # 如果模型期望输入的图像形状的第一个维度为1
        if self.rec_image_shape[0] == 1:
            h, w = img.shape
            _, imgH, imgW = self.rec_image_shape
            # 如果图像的高度或宽度小于模型期望的高度或宽度
            if h < imgH or w < imgW:
                padding_h = max(imgH - h, 0)
                padding_w = max(imgW - w, 0)
                # 对图像进行填充以匹配模型期望的形状
                img_padded = np.pad(img, ((0, padding_h), (0, padding_w)),
                                    'constant',
                                    constant_values=(255))
                img = img_padded

        # 将图像扩展一个维度，并进行归一化处理
        img = np.expand_dims(img, 0) / 255.0
        img = img.astype('float32')

        # 返回处理后的图像
        return img
# 主函数，接受参数并执行主要逻辑
def main(args):
    # 获取图像文件列表
    image_file_list = get_image_file_list(args.image_dir)
    # 创建文本识别器对象
    text_recognizer = TextRecognizer(args)
    # 有效图像文件列表
    valid_image_file_list = []
    # 图像列表
    img_list = []

    # 输出提示信息
    logger.info(
        "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320', "
        "if you are using recognition model with PP-OCRv2 or an older version, please set --rec_image_shape='3,32,320"
    )
    # 进行两次预热
    if args.warmup:
        # 生成随机图像数据
        img = np.random.uniform(0, 255, [48, 320, 3]).astype(np.uint8)
        for i in range(2):
            # 对随机图像数据进行文本识别
            res = text_recognizer([img] * int(args.rec_batch_num))

    # 遍历图像文件列表
    for image_file in image_file_list:
        # 检查并读取图像文件
        img, flag, _ = check_and_read(image_file)
        if not flag:
            # 如果读取失败，则使用OpenCV重新读取图像文件
            img = cv2.imread(image_file)
        if img is None:
            # 输出错误信息
            logger.info("error in loading image:{}".format(image_file))
            continue
        # 将有效的图像文件添加到列表中
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        # 对图像列表进行文本识别
        rec_res, _ = text_recognizer(img_list)

    except Exception as E:
        # 输出异常信息
        logger.info(traceback.format_exc())
        logger.info(E)
        # 退出程序
        exit()
    # 遍历图像列表，输出预测结果
    for ino in range(len(img_list)):
        logger.info("Predicts of {}:{}".format(valid_image_file_list[ino],
                                               rec_res[ino]))
    # 如果需要进行基准测试
    if args.benchmark:
        # 输出基准测试结果
        text_recognizer.autolog.report()


if __name__ == "__main__":
    # 解析参数并执行主函数
    main(utility.parse_args())
```