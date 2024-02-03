# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\tools\infer.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
import os
import sys
import pathlib

# 获取当前文件的绝对路径
__dir__ = pathlib.Path(os.path.abspath(__file__))

# 将当前文件所在目录添加到系统路径中
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

# 导入 OpenCV 库
import cv2
# 导入 PaddlePaddle 库
import paddle
from paddle import inference
# 导入 NumPy 库
import numpy as np
# 导入 PIL 库中的 Image 模块
from PIL import Image

# 导入 PaddleVision 中的 transforms 模块
from paddle.vision import transforms
# 导入 tools.predict 模块中的 resize_image 函数
from tools.predict import resize_image
# 导入 post_processing 模块中的 get_post_processing 函数
from post_processing import get_post_processing
# 导入 utils.util 模块中的 draw_bbox 和 save_result 函数

# 创建 InferenceEngine 类
class InferenceEngine(object):
    """InferenceEngine
    
    Inference engina class which contains preprocess, run, postprocess
    """
    # 初始化类的实例
    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
        # 调用父类的初始化方法
        super().__init__()
        # 将参数保存到实例变量中
        self.args = args

        # 初始化推理引擎
        self.predictor, self.config, self.input_tensor, self.output_tensor = self.load_predictor(
            os.path.join(args.model_dir, "inference.pdmodel"),
            os.path.join(args.model_dir, "inference.pdiparams"))

        # 构建数据转换
        self.transforms = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 预热
        if self.args.warmup > 0:
            for idx in range(args.warmup):
                print(idx)
                x = np.random.rand(1, 3, self.args.crop_size,
                                   self.args.crop_size).astype("float32")
                self.input_tensor.copy_from_cpu(x)
                self.predictor.run()
                self.output_tensor.copy_to_cpu()

        # 设置后处理方法
        self.post_process = get_post_processing({
            'type': 'SegDetectorRepresenter',
            'args': {
                'thresh': 0.3,
                'box_thresh': 0.7,
                'max_candidates': 1000,
                'unclip_ratio': 1.5
            }
        })

    # 预处理方法
    def preprocess(self, img_path, short_size):
        """preprocess
        Preprocess to the input.
        Args:
            img_path: Image path.
        Returns: Input data after preprocess.
        """
        # 读取图像
        img = cv2.imread(img_path, 1)
        # 将图像转换为RGB格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        # 调整图像大小
        img = resize_image(img, short_size)
        # 对图像进行数据转换
        img = self.transforms(img)
        img = np.expand_dims(img, axis=0)
        shape_info = {'shape': [(h, w)]}
        return img, shape_info
    # 后处理函数，对推理引擎的输出进行后处理
    def postprocess(self, x, shape_info, is_output_polygon):
        """postprocess
        Postprocess to the inference engine output.
        Args:
            x: Inference engine output.
        Returns: Output data after argmax.
        """
        # 调用后处理函数，获取边界框列表和得分列表
        box_list, score_list = self.post_process(
            shape_info, x, is_output_polygon=is_output_polygon)
        # 取出第一个元素，即边界框列表和得分列表
        box_list, score_list = box_list[0], score_list[0]
        # 如果边界框列表不为空
        if len(box_list) > 0:
            # 如果输出为多边形
            if is_output_polygon:
                # 找出边界框列表中不全为0的索引
                idx = [x.sum() > 0 for x in box_list]
                # 根据索引筛选出非全为0的边界框和得分
                box_list = [box_list[i] for i, v in enumerate(idx) if v]
                score_list = [score_list[i] for i, v in enumerate(idx) if v]
            else:
                # 去掉全为0的边界框
                idx = box_list.reshape(box_list.shape[0], -1).sum(
                    axis=1) > 0  # 去掉全为0的框
                box_list, score_list = box_list[idx], score_list[idx]
        else:
            # 如果边界框列表为空，则置为空列表
            box_list, score_list = [], []
        # 返回处理后的边界框列表和得分列表
        return box_list, score_list

    # 运行函数，使用推理引擎进行推理
    def run(self, x):
        """run
        Inference process using inference engine.
        Args:
            x: Input data after preprocess.
        Returns: Inference engine output
        """
        # 将输入数据从 CPU 复制到输入张量
        self.input_tensor.copy_from_cpu(x)
        # 运行推理
        self.predictor.run()
        # 将输出数据从输出张量复制到 CPU
        output = self.output_tensor.copy_to_cpu()
        # 返回推理引擎的输出
        return output
def get_args(add_help=True):
    """
    parse args
    """
    # 导入 argparse 模块，用于解析命令行参数
    import argparse

    # 定义一个函数，用于将字符串转换为布尔值
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # 创建 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Classification Training", add_help=add_help)

    # 添加命令行参数
    parser.add_argument("--model_dir", default=None, help="inference model dir")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--short_size", default=1024, type=int, help="short size")
    parser.add_argument("--img_path", default="./images/demo.jpg")

    parser.add_argument(
        "--benchmark", default=False, type=str2bool, help="benchmark")
    parser.add_argument("--warmup", default=0, type=int, help="warmup iter")
    parser.add_argument(
        '--polygon', action='store_true', help='output polygon or box')

    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=10)

    # 解析命令行参数并返回结果
    args = parser.parse_args()
    return args


def main(args):
    """
    Main inference function.
    Args:
        args: Parameters generated using argparser.
    Returns:
        class_id: Class index of the input.
        prob: : Probability of the input.
    """
    # 创建 InferenceEngine 对象，传入参数 args
    inference_engine = InferenceEngine(args)

    # 初始化基准测试
    if args.benchmark:
        # 导入 auto_log 模块
        import auto_log
        # 创建 AutoLogger 对象，用于记录基准测试信息
        autolog = auto_log.AutoLogger(
            model_name="db",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    # 启用基准测试
    if args.benchmark:
        autolog.times.start()

    # 预处理
    # 对输入图片进行预处理，获取预处理后的图片和形状信息
    img, shape_info = inference_engine.preprocess(args.img_path,
                                                  args.short_size)

    # 如果需要进行基准测试，则记录时间戳
    if args.benchmark:
        autolog.times.stamp()

    # 运行推理引擎，获取输出结果
    output = inference_engine.run(img)

    # 如果需要进行基准测试，则记录时间戳
    if args.benchmark:
        autolog.times.stamp()

    # 后处理
    # 对输出结果进行后处理，获取边界框列表和置信度列表
    box_list, score_list = inference_engine.postprocess(output, shape_info,
                                                        args.polygon)

    # 如果需要进行基准测试，则记录时间戳，并结束计时，生成报告
    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    # 在原始图片上绘制边界框
    img = draw_bbox(cv2.imread(args.img_path)[:, :, ::-1], box_list)
    # 创建输出文件夹
    os.makedirs('output', exist_ok=True)
    # 获取输入图片的路径和文件名，拼接生成输出图片的路径
    img_path = pathlib.Path(args.img_path)
    output_path = os.path.join('output', img_path.stem + '_infer_result.jpg')
    # 将绘制了边界框的图片保存到输出路径
    cv2.imwrite(output_path, img[:, :, ::-1])
    # 保存结果到文本文件
    save_result(
        output_path.replace('_infer_result.jpg', '.txt'), box_list, score_list,
        args.polygon)
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    # 调用主函数，传入参数
    main(args)
```