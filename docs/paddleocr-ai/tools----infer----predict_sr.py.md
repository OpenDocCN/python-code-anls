# `.\PaddleOCR\tools\infer\predict_sr.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证以“原样”分发，
# 没有任何明示或暗示的保证或条件
# 请查看特定语言的许可证
# 限制和条件
import os
import sys
# 从 PIL 库导入 Image 模块
from PIL import Image
# 获取当前文件所在目录
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到系统路径中
sys.path.insert(0, __dir__)
# 将当前目录的上级目录添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

# 导入 OpenCV 库
import cv2
# 导入 numpy 库
import numpy as np
# 导入 math 模块
import math
# 导入 time 模块
import time
# 导入 traceback 模块
import traceback
# 导入 paddle 库
import paddle

# 导入自定义模块 utility
import tools.infer.utility as utility
# 导入后处理模块 build_post_process
from ppocr.postprocess import build_post_process
# 导入日志记录器模块 get_logger
from ppocr.utils.logging import get_logger
# 导入工具模块 get_image_file_list, check_and_read
from ppocr.utils.utility import get_image_file_list, check_and_read

# 获取日志记录器
logger = get_logger()

# 定义 TextSR 类
class TextSR(object):
    # 初始化函数，根据参数设置超分辨率图像形状和批次数
    def __init__(self, args):
        # 将超分辨率图像形状的字符串转换为整数列表
        self.sr_image_shape = [int(v) for v in args.sr_image_shape.split(",")]
        # 设置超分辨率批次数
        self.sr_batch_num = args.sr_batch_num

        # 创建预测器、输入张量、输出张量和配置信息
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'sr', logger)
        # 是否进行基准测试
        self.benchmark = args.benchmark
        # 如果进行基准测试
        if args.benchmark:
            # 导入自动日志模块
            import auto_log
            # 获取当前进程 ID
            pid = os.getpid()
            # 获取推理 GPU ID
            gpu_id = utility.get_infer_gpuid()
            # 创建自动日志记录器
            self.autolog = auto_log.AutoLogger(
                model_name="sr",
                model_precision=args.precision,
                batch_size=args.sr_batch_num,
                data_shape="dynamic",
                save_path=None,  #args.save_log_path,
                inference_config=self.config,
                pids=pid,
                process_name=None,
                gpu_ids=gpu_id if args.use_gpu else None,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logger)

    # 调整和归一化图像大小
    def resize_norm_img(self, img):
        # 获取图像的通道数、高度和宽度
        imgC, imgH, imgW = self.sr_image_shape
        # 使用双三次插值对图像进行调整大小
        img = img.resize((imgW // 2, imgH // 2), Image.BICUBIC)
        # 将图像转换为浮点型的 NumPy 数组
        img_numpy = np.array(img).astype("float32")
        # 调整数组维度顺序，并进行归一化处理
        img_numpy = img_numpy.transpose((2, 0, 1)) / 255
        # 返回调整大小和归一化后的图像数组
        return img_numpy
    # 定义一个类的方法，用于对输入的图像列表进行处理
    def __call__(self, img_list):
        # 获取输入图像列表的长度
        img_num = len(img_list)
        # 获取每个批次的图像数量
        batch_num = self.sr_batch_num
        # 记录开始时间
        st = time.time()
        # 记录开始时间
        st = time.time()
        # 初始化结果列表
        all_result = [] * img_num
        # 如果开启了基准测试
        if self.benchmark:
            # 记录自动日志的开始时间
            self.autolog.times.start()
        # 遍历图像列表，按批次处理图像
        for beg_img_no in range(0, img_num, batch_num):
            # 计算当前批次的结束图像编号
            end_img_no = min(img_num, beg_img_no + batch_num)
            # 初始化归一化图像批次列表
            norm_img_batch = []
            # 获取图像的通道数、高度和宽度
            imgC, imgH, imgW = self.sr_image_shape
            # 遍历当前批次的图像，对图像进行归一化处理
            for ino in range(beg_img_no, end_img_no):
                # 调用 resize_norm_img 方法对图像进行处理
                norm_img = self.resize_norm_img(img_list[ino])
                # 在第一个维度上增加一个维度
                norm_img = norm_img[np.newaxis, :]
                # 将处理后的图像添加到归一化图像批次列表中
                norm_img_batch.append(norm_img)

            # 将归一化图像批次列表转换为 numpy 数组
            norm_img_batch = np.concatenate(norm_img_batch)
            # 复制归一化图像批次
            norm_img_batch = norm_img_batch.copy()
            # 如果开启了基准测试
            if self.benchmark:
                # 记录自动日志的时间戳
                self.autolog.times.stamp()
            # 将归一化图像批次复制到输入张量
            self.input_tensor.copy_from_cpu(norm_img_batch)
            # 运行预测器
            self.predictor.run()
            # 初始化输出列表
            outputs = []
            # 遍历输出张量列表，将输出复制到 CPU
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)
            # 如果输出列表长度不为1，则将所有输出作为预测结果
            if len(outputs) != 1:
                preds = outputs
            else:
                preds = outputs[0]
            # 将预测结果添加到结果列表中
            all_result.append(outputs)
        # 如果开启了基准测试
        if self.benchmark:
            # 记录自动日志的结束时间
            self.autolog.times.end(stamp=True)
        # 返回所有结果和处理时间
        return all_result, time.time() - st
# 主函数，接受参数并执行主要逻辑
def main(args):
    # 获取图像文件列表
    image_file_list = get_image_file_list(args.image_dir)
    # 创建文本识别器对象
    text_recognizer = TextSR(args)
    # 有效图像文件列表
    valid_image_file_list = []
    # 图像列表
    img_list = []

    # 预热两次
    if args.warmup:
        # 生成随机图像
        img = np.random.uniform(0, 255, [16, 64, 3]).astype(np.uint8)
        for i in range(2):
            # 对图像进行文本识别
            res = text_recognizer([img] * int(args.sr_batch_num))

    # 遍历图像文件列表
    for image_file in image_file_list:
        # 检查并读取图像
        img, flag, _ = check_and_read(image_file)
        if not flag:
            # 如果读取失败，使用PIL库打开图像并转换为RGB格式
            img = Image.open(image_file).convert("RGB")
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        # 将有效图像文件添加到列表
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        # 对图像列表进行文本识别
        preds, _ = text_recognizer(img_list)
        # 遍历预测结果
        for beg_no in range(len(preds)):
            sr_img = preds[beg_no][1]
            lr_img = preds[beg_no][0]
            for i in (range(sr_img.shape[0])):
                # 转换图像格式并保存
                fm_sr = (sr_img[i] * 255).transpose(1, 2, 0).astype(np.uint8)
                fm_lr = (lr_img[i] * 255).transpose(1, 2, 0).astype(np.uint8)
                img_name_pure = os.path.split(valid_image_file_list[
                    beg_no * args.sr_batch_num + i])[-1]
                cv2.imwrite("infer_result/sr_{}".format(img_name_pure),
                            fm_sr[:, :, ::-1])
                logger.info("The visualized image saved in infer_result/sr_{}".
                            format(img_name_pure))

    except Exception as E:
        # 捕获异常并记录日志
        logger.info(traceback.format_exc())
        logger.info(E)
        exit()
    # 如果需要进行基准测试，则生成报告
    if args.benchmark:
        text_recognizer.autolog.report()


if __name__ == "__main__":
    # 解析参数并执行主函数
    main(utility.parse_args())
```