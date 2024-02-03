# `.\PaddleOCR\tools\infer_cls.py`

```py
# 版权声明和许可证信息
# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 numpy 库
import numpy as np

# 导入 os 和 sys 库
import os
import sys

# 获取当前文件所在目录路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录路径添加到系统路径中
sys.path.append(__dir__)
# 将当前文件所在目录的上级目录路径添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

# 设置环境变量 FLAGS_allocator_strategy 为 'auto_growth'
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

# 导入 paddle 库
import paddle

# 导入数据处理相关的模块
from ppocr.data import create_operators, transform
# 导入模型构建相关的模块
from ppocr.modeling.architectures import build_model
# 导入后处理相关的模块
from ppocr.postprocess import build_post_process
# 导入模型保存和加载相关的模块
from ppocr.utils.save_load import load_model
# 导入工具函数
from ppocr.utils.utility import get_image_file_list
# 导入程序相关的模块
import tools.program as program

# 主函数
def main():
    # 获取全局配置信息
    global_config = config['Global']

    # 构建后处理对象
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # 构建模型
    model = build_model(config['Architecture'])

    # 加载模型参数
    load_model(config, model)

    # 创建数据操作
    transforms = []
    # 遍历数据集的变换操作
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        # 如果操作名中包含 'Label'，则跳过
        if 'Label' in op_name:
            continue
        # 如果操作名为 'KeepKeys'，则设置保留的键为 ['image']
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image']
        # 如果操作名为 'SSLRotateResize'，则设置模式为 'test'
        elif op_name == "SSLRotateResize":
            op[op_name]["mode"] = "test"
        # 将操作添加到 transforms 列表中
        transforms.append(op)
    # 设置全局配置中的 infer_mode 为 True
    global_config['infer_mode'] = True
    # 根据给定的转换和全局配置创建操作符
    ops = create_operators(transforms, global_config)

    # 将模型设置为评估模式
    model.eval()
    
    # 遍历获取图像文件列表
    for file in get_image_file_list(config['Global']['infer_img']):
        # 打印当前处理的图像文件名
        logger.info("infer_img: {}".format(file))
        
        # 以二进制只读方式打开文件
        with open(file, 'rb') as f:
            # 读取文件内容到img变量
            img = f.read()
            # 将图像数据存储在字典中
            data = {'image': img}
        
        # 对数据进行转换操作
        batch = transform(data, ops)

        # 在第0维度上扩展batch[0]，并转换为张量
        images = np.expand_dims(batch[0], axis=0)
        images = paddle.to_tensor(images)
        
        # 使用模型进行推理
        preds = model(images)
        
        # 对预测结果进行后处理
        post_result = post_process_class(preds)
        
        # 遍历后处理结果并打印
        for rec_result in post_result:
            logger.info('\t result: {}'.format(rec_result))
    
    # 打印成功信息
    logger.info("success!")
# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 调用 program 模块的 preprocess 函数，获取 config, device, logger, vdl_writer 四个变量
    config, device, logger, vdl_writer = program.preprocess()
    # 调用主函数 main()
    main()
```