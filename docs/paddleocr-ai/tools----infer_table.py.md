# `.\PaddleOCR\tools\infer_table.py`

```py
# 版权声明和许可证信息
# 导入必要的库
import numpy as np

# 导入操作系统、系统和 JSON 库
import os
import sys
import json

# 获取当前文件所在目录路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到系统路径中
sys.path.append(__dir__)
# 将当前目录的上一级目录添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

# 导入 PaddlePaddle 相关库
import paddle
from paddle.jit import to_static

# 导入数据处理相关的库
from ppocr.data import create_operators, transform
# 导入模型构建相关的库
from ppocr.modeling.architectures import build_model
# 导入后处理相关的库
from ppocr.postprocess import build_post_process
# 导入模型加载和保存相关的库
from ppocr.utils.save_load import load_model
# 导入图像文件列表获取相关的库
from ppocr.utils.utility import get_image_file_list
# 导入绘制矩形框相关的库
from ppocr.utils.visual import draw_rectangle
# 导入绘制框相关的库
from tools.infer.utility import draw_boxes
# 导入程序相关的库
import tools.program as program
# 导入 OpenCV 库
import cv2

# 禁用梯度计算
@paddle.no_grad()
def main(config, device, logger, vdl_writer):
    global_config = config['Global']

    # 构建后处理对象
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # 构建模型
    if hasattr(post_process_class, 'character'):
        # 如果后处理对象有 'character' 属性，则设置模型输出通道数为字符集的长度
        config['Architecture']["Head"]['out_channels'] = len(
            getattr(post_process_class, 'character'))

    model = build_model(config['Architecture'])
    algorithm = config['Architecture']['algorithm']

    # 加载模型参数
    load_model(config, model)

    # 创建数据操作
    # 初始化一个空列表用于存储数据增强操作
    transforms = []
    # 遍历配置文件中数据集的数据增强操作
    for op in config['Eval']['dataset']['transforms']:
        # 获取数据增强操作的名称
        op_name = list(op)[0]
        # 如果操作名称中包含'Encode'，则跳过当前循环
        if 'Encode' in op_name:
            continue
        # 如果操作名称为'KeepKeys'，则设置其保留的键为'image'和'shape'
        if op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        # 将操作添加到数据增强操作列表中
        transforms.append(op)

    # 设置全局配置中的'infer_mode'为True
    global_config['infer_mode'] = True
    # 根据数据增强操作列表和全局配置创建操作符
    ops = create_operators(transforms, global_config)

    # 获取全局配置中的结果保存路径
    save_res_path = config['Global']['save_res_path']
    # 如果结果保存路径不存在，则创建该路径
    os.makedirs(save_res_path, exist_ok=True)

    # 将模型设置为评估模式
    model.eval()
    # 打开文件 'infer.txt' 以写入模式，编码为 utf-8，保存在指定路径下
    with open(
            os.path.join(save_res_path, 'infer.txt'), mode='w',
            encoding='utf-8') as f_w:
        # 遍历获取配置文件中指定路径下的所有图片文件
        for file in get_image_file_list(config['Global']['infer_img']):
            # 记录日志，输出当前处理的图片文件名
            logger.info("infer_img: {}".format(file))
            # 打开当前图片文件，以二进制模式读取图片数据
            with open(file, 'rb') as f:
                img = f.read()
                # 将图片数据封装成字典
                data = {'image': img}
            # 对图片数据进行转换操作，得到处理后的批量数据
            batch = transform(data, ops)
            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
    
            # 将处理后的图片数据转换为 PaddlePaddle 的 Tensor 格式
            images = paddle.to_tensor(images)
            # 使用模型进行推理，得到预测结果
            preds = model(images)
            # 对预测结果进行后处理，得到分类后的结果
            post_result = post_process_class(preds, [shape_list])
    
            # 获取结构化字符串列表和边界框列表
            structure_str_list = post_result['structure_batch_list'][0]
            bbox_list = post_result['bbox_batch_list'][0]
            structure_str_list = structure_str_list[0]
            # 将结构化字符串列表转换为 HTML 格式
            structure_str_list = [
                '<html>', '<body>', '<table>'
            ] + structure_str_list + ['</table>', '</body>', '</html>']
            bbox_list_str = json.dumps(bbox_list.tolist())
    
            # 记录日志，输出处理后的结果
            logger.info("result: {}, {}".format(structure_str_list,
                                                bbox_list_str))
            # 将处理后的结果写入文件 'infer.txt'
            f_w.write("result: {}, {}\n".format(structure_str_list,
                                                bbox_list_str))
    
            # 如果边界框列表不为空且第一个边界框有四个坐标值
            if len(bbox_list) > 0 and len(bbox_list[0]) == 4:
                # 在图片上绘制矩形框
                img = draw_rectangle(file, bbox_list)
            else:
                # 在图片上绘制边界框
                img = draw_boxes(cv2.imread(file), bbox_list)
            # 将处理后的图片保存到指定路径下
            cv2.imwrite(
                os.path.join(save_res_path, os.path.basename(file)), img)
            # 记录日志，输出保存结果的路径
            logger.info('save result to {}'.format(save_res_path))
        # 记录日志，输出处理成功信息
        logger.info("success!")
# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 调用 program 模块的 preprocess 函数，获取配置、设备、日志器和可视化写入器
    config, device, logger, vdl_writer = program.preprocess()
    # 调用 main 函数，传入配置、设备、日志器和可视化写入器作为参数
    main(config, device, logger, vdl_writer)
```