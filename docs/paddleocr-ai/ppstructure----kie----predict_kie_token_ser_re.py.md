# `.\PaddleOCR\ppstructure\kie\predict_kie_token_ser_re.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本使用此文件
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
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
import json
import numpy as np
import time

# 导入自定义工具模块
import tools.infer.utility as utility
from tools.infer_kie_token_ser_re import make_input
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.visual import draw_ser_results, draw_re_results
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppstructure.utility import parse_args
from ppstructure.kie.predict_kie_token_ser import SerPredictor

# 获取日志记录器
logger = get_logger()

# 定义 SerRePredictor 类
class SerRePredictor(object):
    def __init__(self, args):
        # 初始化是否使用视觉骨干网络的标志
        self.use_visual_backbone = args.use_visual_backbone
        # 初始化 SER 模型预测器
        self.ser_engine = SerPredictor(args)
        # 如果存在 RE 模型目录
        if args.re_model_dir is not None:
            # 构建后处理参数
            postprocess_params = {'name': 'VQAReTokenLayoutLMPostProcess'}
            # 构建后处理操作
            self.postprocess_op = build_post_process(postprocess_params)
            # 创建 RE 模型预测器、输入张量、输出张量和配置
            self.predictor, self.input_tensor, self.output_tensors, self.config = \
                utility.create_predictor(args, 're', logger)
        else:
            # 如果不存在 RE 模型目录，则预测器为空
            self.predictor = None
    # 定义一个类的调用方法，接受一个图像作为输入
    def __call__(self, img):
        # 记录开始时间
        starttime = time.time()
        # 使用序列识别引擎处理图像，获取识别结果、输入和耗时
        ser_results, ser_inputs, ser_elapse = self.ser_engine(img)
        # 如果预测器为空，则直接返回序列识别结果和耗时
        if self.predictor is None:
            return ser_results, ser_elapse

        # 将序列识别结果和输入转换为模型输入，同时获取实体索引字典
        re_input, entity_idx_dict_batch = make_input(ser_inputs, ser_results)
        # 如果不使用视觉骨干网络，则移除第五个元素
        if self.use_visual_backbone == False:
            re_input.pop(4)
        # 将转换后的输入数据复制到模型的输入张量中
        for idx in range(len(self.input_tensor)):
            self.input_tensor[idx].copy_from_cpu(re_input[idx])

        # 运行模型
        self.predictor.run()
        # 将模型输出复制到CPU，并存储在outputs列表中
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        # 将模型输出整理为字典形式
        preds = dict(
            loss=outputs[1],
            pred_relations=outputs[2],
            hidden_states=outputs[0], )

        # 对模型输出进行后处理，得到最终结果
        post_result = self.postprocess_op(
            preds,
            ser_results=ser_results,
            entity_idx_dict_batch=entity_idx_dict_batch)

        # 计算总耗时
        elapse = time.time() - starttime
        # 返回后处理结果和总耗时
        return post_result, elapse
# 主函数，接收参数并执行主要逻辑
def main(args):
    # 获取图像文件列表
    image_file_list = get_image_file_list(args.image_dir)
    # 创建一个 SerRePredictor 对象
    ser_re_predictor = SerRePredictor(args)
    # 初始化计数器和总时间
    count = 0
    total_time = 0

    # 创建输出目录（如果不存在）
    os.makedirs(args.output, exist_ok=True)
    # 打开一个文件用于写入推理结果
    with open(
            os.path.join(args.output, 'infer.txt'), mode='w',
            encoding='utf-8') as f_w:
        # 遍历图像文件列表
        for image_file in image_file_list:
            # 检查并读取图像文件
            img, flag, _ = check_and_read(image_file)
            # 如果读取失败，尝试使用 OpenCV 重新读取
            if not flag:
                img = cv2.imread(image_file)
                img = img[:, :, ::-1]
            # 如果图像为空，记录错误信息并继续下一个图像
            if img is None:
                logger.info("error in loading image:{}".format(image_file))
                continue
            # 进行序列识别预测，记录预测结果和耗时
            re_res, elapse = ser_re_predictor(img)
            re_res = re_res[0]

            # 构建结果字符串，包含图像文件名和序列识别结果
            res_str = '{}\t{}\n'.format(
                image_file,
                json.dumps(
                    {
                        "ocr_info": re_res,
                    }, ensure_ascii=False))
            # 将结果字符串写入文件
            f_w.write(res_str)
            # 根据预测器是否为空，选择绘制结果的方式和保存路径
            if ser_re_predictor.predictor is not None:
                img_res = draw_re_results(
                    image_file, re_res, font_path=args.vis_font_path)
                img_save_path = os.path.join(
                    args.output,
                    os.path.splitext(os.path.basename(image_file))[0] +
                    "_ser_re.jpg")
            else:
                img_res = draw_ser_results(
                    image_file, re_res, font_path=args.vis_font_path)
                img_save_path = os.path.join(
                    args.output,
                    os.path.splitext(os.path.basename(image_file))[0] +
                    "_ser.jpg")

            # 保存绘制结果的图像
            cv2.imwrite(img_save_path, img_res)
            logger.info("save vis result to {}".format(img_save_path))
            # 计算总耗时
            if count > 0:
                total_time += elapse
            count += 1
            logger.info("Predict time of {}: {}".format(image_file, elapse))


if __name__ == "__main__":
    # 解析命令行参数并调用主函数
    main(parse_args())
```