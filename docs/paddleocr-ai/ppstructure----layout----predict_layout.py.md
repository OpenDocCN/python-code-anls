# `.\PaddleOCR\ppstructure\layout\predict_layout.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本使用此文件；
# 除非符合许可证的规定，否则不得使用此文件；
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，不附带任何担保或条件，
# 无论是明示的还是暗示的。
# 请查看特定语言的许可证以获取权限和限制。
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

# 导入 OpenCV 库
import cv2
# 导入 NumPy 库
import numpy as np
# 导入时间模块
import time

# 导入自定义模块 utility 中的函数
import tools.infer.utility as utility
# 导入 ppocr.data 模块中的函数
from ppocr.data import create_operators, transform
# 导入 ppocr.postprocess 模块中的函数
from ppocr.postprocess import build_post_process
# 导入 ppocr.utils.logging 模块中的函数
from ppocr.utils.logging import get_logger
# 导入 ppocr.utils.utility 模块中的函数
from ppocr.utils.utility import get_image_file_list, check_and_read
# 导入 ppstructure.utility 模块中的函数
from ppstructure.utility import parse_args
# 导入 picodet_postprocess 模块中的 PicoDetPostProcess 类
from picodet_postprocess import PicoDetPostProcess

# 获取日志记录器
logger = get_logger()

# 定义 LayoutPredictor 类
class LayoutPredictor(object):
    # 初始化函数，接受参数 args
    def __init__(self, args):
        # 预处理操作列表，包含 Resize、NormalizeImage、ToCHWImage 和 KeepKeys 操作
        pre_process_list = [{
            'Resize': {
                'size': [800, 608]
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image']
            }
        }]
        # 后处理参数，包含 PicoDetPostProcess 的名称、布局字典路径、得分阈值和非极大值抑制阈值
        postprocess_params = {
            'name': 'PicoDetPostProcess',
            "layout_dict_path": args.layout_dict_path,
            "score_threshold": args.layout_score_threshold,
            "nms_threshold": args.layout_nms_threshold,
        }

        # 创建预处理操作
        self.preprocess_op = create_operators(pre_process_list)
        # 构建后处理操作
        self.postprocess_op = build_post_process(postprocess_params)
        # 创建预测器、输入张量、输出张量和配置信息
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'layout', logger)
    # 定义一个类的方法，用于对输入的图像进行处理并返回预测结果和推理时间
    def __call__(self, img):
        # 复制输入图像，保留原始图像
        ori_im = img.copy()
        # 创建一个包含图像的字典
        data = {'image': img}
        # 使用预处理操作对数据进行转换
        data = transform(data, self.preprocess_op)
        # 从转换后的数据中获取处理后的图像
        img = data[0]

        # 如果图像为空，则返回空和推理时间为0
        if img is None:
            return None, 0

        # 在图像数组的第一个维度上增加一个维度
        img = np.expand_dims(img, axis=0)
        # 复制图像数据
        img = img.copy()

        # 初始化预测结果和推理时间
        preds, elapse = 0, 1
        # 记录开始时间
        starttime = time.time()

        # 将图像数据复制到输入张量
        self.input_tensor.copy_from_cpu(img)
        # 运行预测器
        self.predictor.run()

        # 初始化用于存储分数和框的列表
        np_score_list, np_boxes_list = [], []
        # 获取输出张量的名称
        output_names = self.predictor.get_output_names()
        # 计算输出张量的数量
        num_outs = int(len(output_names) / 2)
        # 遍历输出张量
        for out_idx in range(num_outs):
            # 将分数数据复制到 CPU
            np_score_list.append(
                self.predictor.get_output_handle(output_names[out_idx])
                .copy_to_cpu())
            # 将框数据复制到 CPU
            np_boxes_list.append(
                self.predictor.get_output_handle(output_names[
                    out_idx + num_outs]).copy_to_cpu())
        # 将分数和框数据存储为字典
        preds = dict(boxes=np_score_list, boxes_num=np_boxes_list)

        # 对预测结果进行后处理
        post_preds = self.postprocess_op(ori_im, img, preds)
        # 计算推理时间
        elapse = time.time() - starttime
        # 返回后处理后的预测结果和推理时间
        return post_preds, elapse
# 主函数，接受参数并执行主要逻辑
def main(args):
    # 获取图像文件列表
    image_file_list = get_image_file_list(args.image_dir)
    # 创建布局预测器对象
    layout_predictor = LayoutPredictor(args)
    # 初始化计数器和总时间
    count = 0
    total_time = 0

    # 重复次数
    repeats = 50
    # 遍历图像文件列表
    for image_file in image_file_list:
        # 检查并读取图像文件
        img, flag, _ = check_and_read(image_file)
        # 如果读取失败，则使用OpenCV重新读取
        if not flag:
            img = cv2.imread(image_file)
        # 如果图像为空，则记录错误信息并继续下一个文件
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue

        # 进行布局预测，记录预测结果和耗时
        layout_res, elapse = layout_predictor(img)

        logger.info("result: {}".format(layout_res))

        # 如果不是第一次预测，则累加总时间
        if count > 0:
            total_time += elapse
        count += 1
        logger.info("Predict time of {}: {}".format(image_file, elapse))


# 如果当前脚本作为主程序运行，则调用主函数并传入解析后的参数
if __name__ == "__main__":
    main(parse_args())
```