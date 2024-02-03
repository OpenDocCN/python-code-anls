# `.\PaddleOCR\tools\infer_kie_token_ser.py`

```
# 版权声明
# 本代码版权归 PaddlePaddle 作者所有
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的
# 没有任何明示或暗示的保证或条件，包括但不限于特定用途的适用性保证
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys

# 获取当前文件所在目录
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到系统路径中
sys.path.append(__dir__)
# 将当前目录的上一级目录添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
# 导入 OpenCV 库
import cv2
# 导入 JSON 库
import json
# 导入 PaddlePaddle 深度学习框架
import paddle

# 导入数据处理相关的模块
from ppocr.data import create_operators, transform
# 导入构建模型的模块
from ppocr.modeling.architectures import build_model
# 导入后处理相关的模块
from ppocr.postprocess import build_post_process
# 导入模型加载和保存相关的函数
from ppocr.utils.save_load import load_model
# 导入可视化相关的函数
from ppocr.utils.visual import draw_ser_results
# 导入工具程序
from ppocr.utils.utility import get_image_file_list, load_vqa_bio_label_maps
import tools.program as program

# 定义一个函数，将数据转换为张量
def to_tensor(data):
    import numbers
    from collections import defaultdict
    # 创建一个默认值为列表的字典
    data_dict = defaultdict(list)
    # 存储需要转换为张量的数据索引
    to_tensor_idxs = []

    # 遍历数据
    for idx, v in enumerate(data):
        # 如果数据是 numpy 数组、Paddle 张量或数字类型
        if isinstance(v, (np.ndarray, paddle.Tensor, numbers.Number)):
            # 如果索引不在需要转换为张量的索引列表中，则添加到列表中
            if idx not in to_tensor_idxs:
                to_tensor_idxs.append(idx)
        # 将数据添加到字典中对应索引的值列表中
        data_dict[idx].append(v)
    # 将需要转换为张量的数据转换为张量
    for idx in to_tensor_idxs:
        data_dict[idx] = paddle.to_tensor(data_dict[idx])
    # 返回转换后的数据列表
    return list(data_dict.values())

# 定义一个类 SerPredictor
class SerPredictor(object):
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 获取全局配置信息
        global_config = config['Global']
        # 获取算法名称
        self.algorithm = config['Architecture']["algorithm"]

        # 构建后处理过程
        self.post_process_class = build_post_process(config['PostProcess'],
                                                     global_config)

        # 构建模型
        self.model = build_model(config['Architecture'])

        # 加载模型
        load_model(
            config, self.model, model_type=config['Architecture']["model_type"])

        # 导入 PaddleOCR 模块
        from paddleocr import PaddleOCR

        # 初始化 OCR 引擎
        self.ocr_engine = PaddleOCR(
            use_angle_cls=False,
            show_log=False,
            rec_model_dir=global_config.get("kie_rec_model_dir", None),
            det_model_dir=global_config.get("kie_det_model_dir", None),
            use_gpu=global_config['use_gpu'])

        # 创建数据操作
        transforms = []
        # 遍历数据集的转换操作
        for op in config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            # 如果操作名称中包含 'Label'，则将 OCR 引擎传递给该操作
            if 'Label' in op_name:
                op[op_name]['ocr_engine'] = self.ocr_engine
            # 如果操作名称为 'KeepKeys'，则设置保留的键列表
            elif op_name == 'KeepKeys':
                op[op_name]['keep_keys'] = [
                    'input_ids', 'bbox', 'attention_mask', 'token_type_ids',
                    'image', 'labels', 'segment_offset_id', 'ocr_info',
                    'entities'
                ]

            # 将操作添加到 transforms 列表中
            transforms.append(op)
        
        # 如果全局配置中没有推理模式，则设置为 True
        if config["Global"].get("infer_mode", None) is None:
            global_config['infer_mode'] = True
        
        # 创建操作符
        self.ops = create_operators(config['Eval']['dataset']['transforms'],
                                    global_config)
        # 将模型设置为评估模式
        self.model.eval()
    # 定义一个类的方法，用于处理输入数据
    def __call__(self, data):
        # 打开并读取指定路径的图像文件
        with open(data["img_path"], 'rb') as f:
            img = f.read()
        # 将读取的图像数据存储在数据字典中的"image"键下
        data["image"] = img
        # 对数据进行一系列变换操作
        batch = transform(data, self.ops)
        # 将处理后的数据转换为张量
        batch = to_tensor(batch)
        # 使用模型对处理后的数据进行预测
        preds = self.model(batch)

        # 对模型预测结果进行后处理，传入预测结果、分段偏移ID和OCR信息
        post_result = self.post_process_class(
            preds, segment_offset_ids=batch[6], ocr_infos=batch[7])
        # 返回后处理结果和处理后的数据
        return post_result, batch
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 进行预处理，获取配置、设备、日志器和可视化写入器
    config, device, logger, vdl_writer = program.preprocess()
    # 创建保存结果路径
    os.makedirs(config['Global']['save_res_path'], exist_ok=True)

    # 创建 SER 预测器对象
    ser_engine = SerPredictor(config)

    # 根据配置选择不同的推理模式
    if config["Global"].get("infer_mode", None) is False:
        # 获取数据集目录和推理图片列表
        data_dir = config['Eval']['dataset']['data_dir']
        with open(config['Global']['infer_img'], "rb") as f:
            infer_imgs = f.readlines()
    else:
        # 获取推理图片列表
        infer_imgs = get_image_file_list(config['Global']['infer_img'])

    # 打开保存推理结果的文件
    with open(
            os.path.join(config['Global']['save_res_path'],
                         "infer_results.txt"),
            "w",
            encoding='utf-8') as fout:
        # 遍历推理图片列表
        for idx, info in enumerate(infer_imgs):
            # 根据推理模式解析数据信息
            if config["Global"].get("infer_mode", None) is False:
                data_line = info.decode('utf-8')
                substr = data_line.strip("\n").split("\t")
                img_path = os.path.join(data_dir, substr[0])
                data = {'img_path': img_path, 'label': substr[1]}
            else:
                img_path = info
                data = {'img_path': img_path}

            # 构建保存结果图片路径
            save_img_path = os.path.join(
                config['Global']['save_res_path'],
                os.path.splitext(os.path.basename(img_path))[0] + "_ser.jpg")

            # 进行 SER 模型推理
            result, _ = ser_engine(data)
            result = result[0]
            # 将推理结果写入文件
            fout.write(img_path + "\t" + json.dumps(
                {
                    "ocr_info": result,
                }, ensure_ascii=False) + "\n")
            # 绘制 SER 结果并保存图片
            img_res = draw_ser_results(img_path, result)
            cv2.imwrite(save_img_path, img_res)

            # 记录处理进度
            logger.info("process: [{}/{}], save result to {}".format(
                idx, len(infer_imgs), save_img_path))
```