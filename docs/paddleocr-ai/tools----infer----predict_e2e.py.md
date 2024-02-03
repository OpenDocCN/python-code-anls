# `.\PaddleOCR\tools\infer\predict_e2e.py`

```
# 版权声明和许可信息
# 该代码版权归 PaddlePaddle 作者所有，遵循 Apache License, Version 2.0
# 只有在遵守许可证的情况下才能使用该文件
# 可以在上述链接获取许可证的副本
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入必要的库
import os
import sys

# 获取当前文件所在目录的绝对路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到系统路径中
sys.path.append(__dir__)
# 将上级目录添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

# 导入必要的库
import cv2
import numpy as np
import time
import sys

# 导入自定义工具模块
import tools.infer.utility as utility
# 导入日志记录模块
from ppocr.utils.logging import get_logger
# 导入图像文件列表获取和检查读取函数
from ppocr.utils.utility import get_image_file_list, check_and_read
# 导入数据操作符创建和转换函数
from ppocr.data import create_operators, transform
# 导入后处理模块构建函数
from ppocr.postprocess import build_post_process

# 获取日志记录器
logger = get_logger()

# 定义 TextE2E 类
class TextE2E(object):
    # 初始化函数，接受参数并设置对象属性
    def __init__(self, args):
        # 将参数保存为对象属性
        self.args = args
        # 设置端到端算法
        self.e2e_algorithm = args.e2e_algorithm
        # 是否使用 ONNX
        self.use_onnx = args.use_onnx
        # 预处理操作列表
        pre_process_list = [{
            'E2EResizeForTest': {}
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
                'keep_keys': ['image', 'shape']
            }
        }]
        # 后处理参数
        postprocess_params = {}
        # 如果端到端算法是 "PGNet"
        if self.e2e_algorithm == "PGNet":
            # 更新预处理操作列表中的第一个操作
            pre_process_list[0] = {
                'E2EResizeForTest': {
                    'max_side_len': args.e2e_limit_side_len,
                    'valid_set': 'totaltext'
                }
            }
            # 设置后处理参数
            postprocess_params['name'] = 'PGPostProcess'
            postprocess_params["score_thresh"] = args.e2e_pgnet_score_thresh
            postprocess_params["character_dict_path"] = args.e2e_char_dict_path
            postprocess_params["valid_set"] = args.e2e_pgnet_valid_set
            postprocess_params["mode"] = args.e2e_pgnet_mode
        else:
            # 如果端到端算法不是 "PGNet"，记录日志并退出程序
            logger.info("unknown e2e_algorithm:{}".format(self.e2e_algorithm))
            sys.exit(0)

        # 创建预处理操作
        self.preprocess_op = create_operators(pre_process_list)
        # 构建后处理操作
        self.postprocess_op = build_post_process(postprocess_params)
        # 创建预测器、输入张量、输出张量
        self.predictor, self.input_tensor, self.output_tensors, _ = utility.create_predictor(
            args, 'e2e', logger)  # paddle.jit.load(args.det_model_dir)
        # self.predictor.eval()

    # 裁剪检测结果，确保点在图像范围内
    def clip_det_res(self, points, img_height, img_width):
        # 遍历所有点
        for pno in range(points.shape[0]):
            # 裁剪 x 坐标
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            # 裁剪 y 坐标
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        # 返回裁剪后的点
        return points
    # 过滤检测结果，仅保留在图像范围内的文本框
    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        # 获取图像的高度和宽度
        img_height, img_width = image_shape[0:2]
        # 创建一个新的文本框列表
        dt_boxes_new = []
        # 遍历原始文本框列表
        for box in dt_boxes:
            # 裁剪文本框，确保在图像范围内
            box = self.clip_det_res(box, img_height, img_width)
            # 将裁剪后的文本框添加到新的列表中
            dt_boxes_new.append(box)
        # 将新的文本框列表转换为 NumPy 数组
        dt_boxes = np.array(dt_boxes_new)
        # 返回处理后的文本框列表
        return dt_boxes

    # 对象的调用方法，用于处理图像
    def __call__(self, img):
        # 复制原始图像
        ori_im = img.copy()
        # 创建包含图像的字典
        data = {'image': img}
        # 对图像进行预处理操作
        data = transform(data, self.preprocess_op)
        # 获取处理后的图像和形状列表
        img, shape_list = data
        # 如果图像为空，则返回空值和 0
        if img is None:
            return None, 0
        # 在图像数组的第一个维度上添加一个维度
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        # 复制图像数据
        img = img.copy()
        # 记录开始时间
        starttime = time.time()

        # 如果使用 ONNX 模型
        if self.use_onnx:
            # 创建输入字典
            input_dict = {}
            input_dict[self.input_tensor.name] = img
            # 运行预测器，获取输出
            outputs = self.predictor.run(self.output_tensors, input_dict)
            preds = {}
            # 将输出分配给不同的预测结果
            preds['f_border'] = outputs[0]
            preds['f_char'] = outputs[1]
            preds['f_direction'] = outputs[2]
            preds['f_score'] = outputs[3]
        else:
            # 将图像数据复制到输入张量
            self.input_tensor.copy_from_cpu(img)
            # 运行预测器
            self.predictor.run()
            outputs = []
            # 将输出张量的数据复制到 CPU
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)

            preds = {}
            # 根据端到端算法不同，分配不同的输出
            if self.e2e_algorithm == 'PGNet':
                preds['f_border'] = outputs[0]
                preds['f_char'] = outputs[1]
                preds['f_direction'] = outputs[2]
                preds['f_score'] = outputs[3]
            else:
                # 抛出未实现错误
                raise NotImplementedError
        # 对预测结果进行后处理操作
        post_result = self.postprocess_op(preds, shape_list)
        # 获取文本框和文本字符串
        points, strs = post_result['points'], post_result['texts']
        # 过滤文本框，仅保留在原始图像范围内的文本框
        dt_boxes = self.filter_tag_det_res_only_clip(points, ori_im.shape)
        # 计算处理时间
        elapse = time.time() - starttime
        # 返回处理后的文本框、文本字符串和处理时间
        return dt_boxes, strs, elapse
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 解析命令行参数
    args = utility.parse_args()
    # 获取图像文件列表
    image_file_list = get_image_file_list(args.image_dir)
    # 创建文本检测器对象
    text_detector = TextE2E(args)
    # 初始化计数器和总时间
    count = 0
    total_time = 0
    # 设置绘制图像保存路径
    draw_img_save = "./inference_results"
    # 如果保存路径不存在，则创建
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    # 遍历图像文件列表
    for image_file in image_file_list:
        # 检查并读取图像文件
        img, flag, _ = check_and_read(image_file)
        # 如果读取失败，则使用OpenCV重新读取
        if not flag:
            img = cv2.imread(image_file)
        # 如果图像为空，则记录错误信息并继续下一张图像
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        # 进行文本检测，获取文本框坐标、文本内容和推理时间
        points, strs, elapse = text_detector(img)
        # 如果不是第一张图像，则累加总时间
        if count > 0:
            total_time += elapse
        # 增加计数器
        count += 1
        # 记录当前图像的推理时间
        logger.info("Predict time of {}: {}".format(image_file, elapse))
        # 绘制文本检测结果并保存
        src_im = utility.draw_e2e_res(points, strs, image_file)
        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(draw_img_save,
                                "e2e_res_{}".format(img_name_pure))
        cv2.imwrite(img_path, src_im)
        logger.info("The visualized image saved in {}".format(img_path))
    # 如果处理多张图像，则计算平均推理时间并记录
    if count > 1:
        logger.info("Avg Time: {}".format(total_time / (count - 1)))
```