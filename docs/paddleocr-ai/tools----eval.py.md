# `.\PaddleOCR\tools\eval.py`

```
# 版权声明和许可证信息
# 该代码版权归 PaddlePaddle 作者所有，遵循 Apache License, Version 2.0 许可证
# 可以在遵守许可证的前提下使用该文件
# 许可证详情请参考 http://www.apache.org/licenses/LICENSE-2.0

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

# 获取当前文件所在目录路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前目录路径添加到系统路径中
sys.path.insert(0, __dir__)
# 将当前目录的上一级目录路径添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

# 导入 paddle 库
import paddle
# 导入数据处理相关的模块
from ppocr.data import build_dataloader, set_signal_handlers
# 导入模型构建相关的模块
from ppocr.modeling.architectures import build_model
# 导入后处理相关的模块
from ppocr.postprocess import build_post_process
# 导入评估指标相关的模块
from ppocr.metrics import build_metric
# 导入模型保存和加载相关的模块
from ppocr.utils.save_load import load_model
# 导入程序相关的模块
import tools.program as program

# 主函数
def main():
    # 获取全局配置信息
    global_config = config['Global']
    
    # 构建数据加载器
    set_signal_handlers()
    valid_dataloader = build_dataloader(config, 'Eval', device, logger)

    # 构建后处理模块
    post_process_class = build_post_process(config['PostProcess'], global_config)

    # 构建模型
    # 对于文本识别算法
    model = build_model(config['Architecture'])
    # 额外输入模型列表
    extra_input_models = [
        "SRN", "NRTR", "SAR", "SEED", "SVTR", "SVTR_LCNet", "VisionLAN",
        "RobustScanner", "SVTR_HGNet"
    ]
    # 是否有额外输入
    extra_input = False
    # 如果配置中的算法是Distillation，则遍历模型字典中的每个模型，检查是否需要额外输入
    if config['Architecture']['algorithm'] == 'Distillation':
        for key in config['Architecture']["Models"]:
            # 如果extra_input为False且当前模型的算法在额外输入模型列表中，则将extra_input设置为True
            extra_input = extra_input or config['Architecture']['Models'][key]['algorithm'] in extra_input_models
    # 如果配置中的算法不是Distillation，则检查当前算法是否在额外输入模型列表中
    else:
        extra_input = config['Architecture']['algorithm'] in extra_input_models
    # 如果配置中包含"model_type"键
    if "model_type" in config['Architecture'].keys():
        # 如果算法是CAN，则将model_type设置为'can'，否则使用配置中的model_type
        if config['Architecture']['algorithm'] == 'CAN':
            model_type = 'can'
        else:
            model_type = config['Architecture']['model_type']
    else:
        # 如果配置中不包含"model_type"键，则将model_type设置为None
        model_type = None

    # 构建评估指标
    eval_class = build_metric(config['Metric'])
    # 检查是否使用自动混合精度
    use_amp = config["Global"].get("use_amp", False)
    # 获取自动混合精度级别
    amp_level = config["Global"].get("amp_level", 'O2')
    # 获取自定义的自动混合精度黑名单
    amp_custom_black_list = config['Global'].get('amp_custom_black_list', [])
    if use_amp:
        # 设置与自动混合精度相关的标志
        AMP_RELATED_FLAGS_SETTING = {
            'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
            'FLAGS_max_inplace_grad_add': 8,
        }
        paddle.set_flags(AMP_RELATED_FLAGS_SETTING)
        # 获取损失缩放比例
        scale_loss = config["Global"].get("scale_loss", 1.0)
        # 获取是否使用动态损失缩放
        use_dynamic_loss_scaling = config["Global"].get("use_dynamic_loss_scaling", False)
        # 创建GradScaler对象
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling)
        # 如果自动混合精度级别为"O2"，则使用paddle.amp.decorate装饰模型
        if amp_level == "O2":
            model = paddle.amp.decorate(
                models=model, level=amp_level, master_weight=True)
    else:
        scaler = None

    # 加载最佳模型
    best_model_dict = load_model(
        config, model, model_type=config['Architecture']["model_type"])
    # 如果最佳模型字典不为空，则打印日志信息
    if len(best_model_dict):
        logger.info('metric in ckpt ***************')
        for k, v in best_model_dict.items():
            logger.info('{}:{}'.format(k, v))

    # 开始评估
    # 使用程序、模型、验证数据加载器、后处理类、评估类、模型类型、额外输入、缩放器、混合精度级别、混合精度自定义黑名单计算指标
    metric = program.eval(model, valid_dataloader, post_process_class,
                          eval_class, model_type, extra_input, scaler,
                          amp_level, amp_custom_black_list)
    # 记录日志，提示评估指标开始
    logger.info('metric eval ***************')
    # 遍历评估指标字典，记录每个指标的键值对
    for k, v in metric.items():
        logger.info('{}:{}'.format(k, v))
# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 调用 program 模块的 preprocess 函数，获取 config, device, logger, vdl_writer 四个变量
    config, device, logger, vdl_writer = program.preprocess()
    # 调用主函数 main()
    main()
```