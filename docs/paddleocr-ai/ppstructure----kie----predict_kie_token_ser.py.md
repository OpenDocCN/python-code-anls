# `.\PaddleOCR\ppstructure\kie\predict_kie_token_ser.py`

```
# 版权声明和许可证信息
# 该代码版权归 PaddlePaddle 作者所有，保留所有权利
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
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

# 导入自定义的 utility 模块中的函数
import tools.infer.utility as utility
# 导入 ppocr.data 模块中的函数
from ppocr.data import create_operators, transform
# 导入 ppocr.postprocess 模块中的函数
from ppocr.postprocess import build_post_process
# 导入 ppocr.utils.logging 模块中的函数
from ppocr.utils.logging import get_logger
# 导入 ppocr.utils.visual 模块中的函数
from ppocr.utils.visual import draw_ser_results
# 导入 ppocr.utils.utility 模块中的函数
from ppocr.utils.utility import get_image_file_list, check_and_read
# 导入 ppstructure.utility 模块中的函数
from ppstructure.utility import parse_args

# 导入 PaddleOCR 模块
from paddleocr import PaddleOCR

# 获取日志记录器
logger = get_logger()

# 定义 SerPredictor 类
class SerPredictor(object):
    # 初始化函数，接受参数并设置 OCR 引擎
    def __init__(self, args):
        # 使用 PaddleOCR 创建 OCR 引擎，设置参数并关闭日志输出
        self.ocr_engine = PaddleOCR(
            use_angle_cls=args.use_angle_cls,
            det_model_dir=args.det_model_dir,
            rec_model_dir=args.rec_model_dir,
            show_log=False,
            use_gpu=args.use_gpu)

        # 预处理操作列表，包括 VQATokenLabelEncode、VQATokenPad、VQASerTokenChunk、Resize、NormalizeImage、ToCHWImage、KeepKeys
        pre_process_list = [{
            'VQATokenLabelEncode': {
                'algorithm': args.kie_algorithm,
                'class_path': args.ser_dict_path,
                'contains_re': False,
                'ocr_engine': self.ocr_engine,
                'order_method': args.ocr_order_method,
            }
        }, {
            'VQATokenPad': {
                'max_seq_len': 512,
                'return_attention_mask': True
            }
        }, {
            'VQASerTokenChunk': {
                'max_seq_len': 512,
                'return_attention_mask': True
            }
        }, {
            'Resize': {
                'size': [224, 224]
            }
        }, {
            'NormalizeImage': {
                'std': [58.395, 57.12, 57.375],
                'mean': [123.675, 116.28, 103.53],
                'scale': '1',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': [
                    'input_ids', 'bbox', 'attention_mask', 'token_type_ids',
                    'image', 'labels', 'segment_offset_id', 'ocr_info',
                    'entities'
                ]
            }
        }]
        
        # 后处理参数，包括 VQASerTokenLayoutLMPostProcess
        postprocess_params = {
            'name': 'VQASerTokenLayoutLMPostProcess',
            "class_path": args.ser_dict_path,
        }

        # 创建预处理操作和后处理操作
        self.preprocess_op = create_operators(pre_process_list,
                                              {'infer_mode': True})
        self.postprocess_op = build_post_process(postprocess_params)
        
        # 创建预测器、输入张量、输出张量和配置
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'ser', logger)
    # 定义一个类的调用方法，接受一个图像作为输入
    def __call__(self, img):
        # 复制原始图像
        ori_im = img.copy()
        # 创建一个包含图像的字典
        data = {'image': img}
        # 对图像进行预处理操作
        data = transform(data, self.preprocess_op)
        # 如果预处理后的数据为空，则返回空和0
        if data[0] is None:
            return None, 0
        # 记录开始时间
        starttime = time.time()

        # 遍历数据，如果是 numpy 数组，则在第0维度上扩展为1
        for idx in range(len(data)):
            if isinstance(data[idx], np.ndarray):
                data[idx] = np.expand_dims(data[idx], axis=0)
            else:
                data[idx] = [data[idx]]

        # 将数据从 CPU 复制到输入张量中
        for idx in range(len(self.input_tensor)):
            self.input_tensor[idx].copy_from_cpu(data[idx])

        # 运行预测器
        self.predictor.run()

        # 将输出从输出张量中复制到 CPU
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        # 获取预测结果
        preds = outputs[0]

        # 对预测结果进行后处理操作
        post_result = self.postprocess_op(
            preds, segment_offset_ids=data[6], ocr_infos=data[7])
        # 计算运行时间
        elapse = time.time() - starttime
        # 返回后处理结果、数据和运行时间
        return post_result, data, elapse
# 主函数，接收参数并执行主要逻辑
def main(args):
    # 获取图像文件列表
    image_file_list = get_image_file_list(args.image_dir)
    # 创建序列识别器
    ser_predictor = SerPredictor(args)
    # 初始化计数器和总时间
    count = 0
    total_time = 0

    # 创建输出目录（如果不存在）
    os.makedirs(args.output, exist_ok=True)
    # 打开输出文件，准备写入推理结果
    with open(
            os.path.join(args.output, 'infer.txt'), mode='w',
            encoding='utf-8') as f_w:
        # 遍历图像文件列表
        for image_file in image_file_list:
            # 检查并读取图像文件
            img, flag, _ = check_and_read(image_file)
            # 如果读取失败，尝试使用opencv读取
            if not flag:
                img = cv2.imread(image_file)
                img = img[:, :, ::-1]
            # 如果图像为空，记录错误信息并继续下一个图像
            if img is None:
                logger.info("error in loading image:{}".format(image_file))
                continue
            # 进行序列识别，获取识别结果和推理时间
            ser_res, _, elapse = ser_predictor(img)
            ser_res = ser_res[0]

            # 构建结果字符串，包含图像文件名和识别结果
            res_str = '{}\t{}\n'.format(
                image_file,
                json.dumps(
                    {
                        "ocr_info": ser_res,
                    }, ensure_ascii=False))
            f_w.write(res_str)

            # 绘制序列识别结果并保存图像
            img_res = draw_ser_results(
                image_file,
                ser_res,
                font_path=args.vis_font_path, )

            img_save_path = os.path.join(args.output,
                                         os.path.basename(image_file))
            cv2.imwrite(img_save_path, img_res)
            logger.info("save vis result to {}".format(img_save_path))
            # 累加推理时间
            if count > 0:
                total_time += elapse
            count += 1
            logger.info("Predict time of {}: {}".format(image_file, elapse))


if __name__ == "__main__":
    # 解析命令行参数并调用主函数
    main(parse_args())
```