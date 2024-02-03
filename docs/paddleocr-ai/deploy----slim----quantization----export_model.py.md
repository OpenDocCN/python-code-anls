# `.\PaddleOCR\deploy\slim\quantization\export_model.py`

```
# 导入必要的库
import os
import sys

# 获取当前文件所在目录的绝对路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录添加到系统路径中
sys.path.append(__dir__)
# 将上级目录的绝对路径添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..', '..', '..')))
# 将上上级目录的 tools 目录的绝对路径添加到系统路径中
sys.path.insert(
    0, os.path.abspath(os.path.join(__dir__, '..', '..', '..', 'tools')))

# 导入 argparse 库
import argparse

# 导入 paddle 库
import paddle
# 从 paddle.jit 模块中导入 to_static 函数
from paddle.jit import to_static

# 从 ppocr.modeling.architectures 模块中导入 build_model 函数
from ppocr.modeling.architectures import build_model
# 从 ppocr.postprocess 模块中导入 build_post_process 函数
from ppocr.postprocess import build_post_process
# 从 ppocr.utils.save_load 模块中导入 load_model 函数
from ppocr.utils.save_load import load_model
# 从 ppocr.utils.logging 模块中导入 get_logger 函数
from ppocr.utils.logging import get_logger
# 从 tools.program 模块中导入 load_config、merge_config、ArgsParser 函数
from tools.program import load_config, merge_config, ArgsParser
# 从 ppocr.metrics 模块中导入 build_metric 函数
from ppocr.metrics import build_metric
# 导入 tools.program 模块并重命名为 program
import tools.program as program
# 从 paddleslim.dygraph.quant 模块中导入 QAT 类
from paddleslim.dygraph.quant import QAT
# 从 ppocr.data 模块中导入 build_dataloader、set_signal_handlers 函数
from ppocr.data import build_dataloader, set_signal_handlers
# 从 tools.export_model 模块中导入 export_single_model 函数

# 定义主函数
def main():
    ############################################################################################################
    # 1. 量化配置
    ############################################################################################################
    # 定义量化配置字典，包括权重和激活的预处理类型、量化类型、位数、数据类型等
    quant_config = {
        'weight_preprocess_type': None,  # 权重预处理类型，默认为None，不进行预处理
        'activation_preprocess_type': None,  # 激活预处理类型，默认为None，不进行预处理
        'weight_quantize_type': 'channel_wise_abs_max',  # 权重量化类型，默认为'channel_wise_abs_max'
        'activation_quantize_type': 'moving_average_abs_max',  # 激活量化类型，默认为'moving_average_abs_max'
        'weight_bits': 8,  # 权重量化位数，默认为8
        'activation_bits': 8,  # 激活量化位数，默认为8
        'dtype': 'int8',  # 量化后的数据类型，如'uint8'、'int8'等，默认为'int8'
        'window_size': 10000,  # 'range_abs_max'量化的窗口大小，默认为10000
        'moving_rate': 0.9,  # 移动平均的衰减系数，默认为0.9
        'quantizable_layer_type': ['Conv2D', 'Linear'],  # 可量化层的类型列表，默认为['Conv2D', 'Linear']
    }
    # 解析命令行参数
    FLAGS = ArgsParser().parse_args()
    # 加载配置文件
    config = load_config(FLAGS.config)
    # 合并配置文件和命令行参数
    config = merge_config(config, FLAGS.opt)
    # 获取日志记录器
    logger = get_logger()
    
    # 构建后处理类
    post_process_class = build_post_process(config['PostProcess'],
                                            config['Global'])
    
    # 构建模型
    model = build_model(config['Architecture'])
    
    # 获取QAT模型
    quanter = QAT(config=quant_config)
    quanter.quantize(model)
    
    # 加载模型
    load_model(config, model)
    
    # 构建评估指标
    eval_class = build_metric(config['Metric'])
    
    # 构建数据加载器
    set_signal_handlers()
    valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    
    # 判断是否使用SRN算法
    use_srn = config['Architecture']['algorithm'] == "SRN"
    model_type = config['Architecture'].get('model_type', None)
    # 开始评估
    # 使用程序评估模型在验证数据集上的性能，并返回评估指标
    metric = program.eval(model, valid_dataloader, post_process_class,
                          eval_class, model_type, use_srn)
    # 将模型设置为评估模式
    model.eval()

    # 记录评估指标信息
    logger.info('metric eval ***************')
    # 遍历评估指标字典，输出每个指标的键值对
    for k, v in metric.items():
        logger.info('{}:{}'.format(k, v))

    # 获取保存推理结果的路径
    save_path = config["Global"]["save_inference_dir"]

    # 获取模型架构配置信息
    arch_config = config["Architecture"]

    # 如果算法是 SVTR 且头部不是 MultiHead
    if arch_config["algorithm"] == "SVTR" and arch_config["Head"][
            "name"] != 'MultiHead':
        # 获取输入图像形状
        input_shape = config["Eval"]["dataset"]["transforms"][-2][
            'SVTRRecResizeImg']['image_shape']
    else:
        input_shape = None

    # 如果算法是 Distillation，即蒸馏模型
    if arch_config["algorithm"] in ["Distillation", ]:
        # 获取所有模型架构
        archs = list(arch_config["Models"].values())
        # 遍历模型名称列表
        for idx, name in enumerate(model.model_name_list):
            # 设置子模型保存路径
            sub_model_save_path = os.path.join(save_path, name, "inference")
            # 导出单个模型
            export_single_model(model.model_list[idx], archs[idx],
                                sub_model_save_path, logger, input_shape,
                                quanter)
    else:
        # 设置保存路径为推理结果路径下的 inference 文件夹
        save_path = os.path.join(save_path, "inference")
        # 导出单个模型
        export_single_model(model, arch_config, save_path, logger, input_shape,
                            quanter)
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 调用 program 模块的 preprocess 函数，获取 config, device, logger, vdl_writer 四个变量
    config, device, logger, vdl_writer = program.preprocess()
    # 调用 main 函数
    main()
```