# `.\PaddleOCR\deploy\slim\prune\export_prune_model.py`

```py
# 版权声明
# 从未来模块导入绝对路径、除法和打印函数
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入操作系统和系统模块
import os
import sys

# 获取当前文件所在目录
__dir__ = os.path.dirname(__file__)
# 将当前目录添加到系统路径中
sys.path.append(__dir__)
# 将当前目录的上级目录添加到系统路径中
sys.path.append(os.path.join(__dir__, '..', '..', '..'))
# 将当前目录的上级目录的上级目录添加到系统路径中
sys.path.append(os.path.join(__dir__, '..', '..', '..', 'tools'))

# 导入paddle模块
import paddle
# 从ppocr.data模块中导入构建数据加载器和设置信号处理程序
from ppocr.data import build_dataloader, set_signal_handlers
# 从ppocr.modeling.architectures模块中导入构建模型
from ppocr.modeling.architectures import build_model
# 从ppocr.postprocess模块中导入构建后处理
from ppocr.postprocess import build_post_process
# 从ppocr.metrics模块中导入构建度量
from ppocr.metrics import build_metric
# 从ppocr.utils.save_load模块中导入加载模型
from ppocr.utils.save_load import load_model
# 导入tools.program模块
import tools.program as program

# 主函数
def main(config, device, logger, vdl_writer):

    global_config = config['Global']

    # 构建数据加载器
    set_signal_handlers()
    valid_dataloader = build_dataloader(config, 'Eval', device, logger)

    # 构建后处理
    post_process_class = build_post_process(config['PostProcess'], global_config)

    # 构建模型
    # 对于rec算法
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        # 设置输出通道数为字符数
        config['Architecture']["Head"]['out_channels'] = char_num
    model = build_model(config['Architecture'])

    if config['Architecture']['model_type'] == 'det':
        input_shape = [1, 3, 640, 640]
    # 如果模型类型是'rec'，则设置输入形状为[1, 3, 32, 320]
    elif config['Architecture']['model_type'] == 'rec':
        input_shape = [1, 3, 32, 320]

    # 计算模型的 FLOPs（浮点运算数）并输出
    flops = paddle.flops(model, input_shape)
    logger.info("FLOPs before pruning: {}".format(flops))

    # 导入 FPGMFilterPruner 类
    from paddleslim.dygraph import FPGMFilterPruner
    # 将模型设置为训练模式
    model.train()
    # 创建 FPGMFilterPruner 对象
    pruner = FPGMFilterPruner(model, input_shape)

    # 构建评估指标
    eval_class = build_metric(config['Metric'])

    # 定义评估函数
    def eval_fn():
        # 调用 program 模块的 eval 函数进行评估
        metric = program.eval(model, valid_dataloader, post_process_class, eval_class)
        # 根据模型类型选择主要指标
        if config['Architecture']['model_type'] == 'det':
            main_indicator = 'hmean'
        else:
            main_indicator = 'acc'
        logger.info("metric[{}]: {}".format(main_indicator, metric[main_indicator]))
        return metric[main_indicator]

    # 获取参数的敏感度分析结果
    params_sensitive = pruner.sensitive(
        eval_func=eval_fn,
        sen_file="./sen.pickle",
        skip_vars=[
            "conv2d_57.w_0", "conv2d_transpose_2.w_0", "conv2d_transpose_3.w_0"
        ])

    # 输出模型参数的敏感度分析结果保存在 sen.pickle 文件中
    logger.info("The sensitivity analysis results of model parameters saved in sen.pickle")

    # 计算被剪枝参数的比例
    params_sensitive = pruner._get_ratios_by_loss(params_sensitive, loss=0.02)
    for key in params_sensitive.keys():
        logger.info("{}, {}".format(key, params_sensitive[key])

    # 执行剪枝操作
    plan = pruner.prune_vars(params_sensitive, [0])

    # 重新计算模型的 FLOPs 并输出
    flops = paddle.flops(model, input_shape)
    logger.info("FLOPs after pruning: {}".format(flops))

    # 加载预训练模型
    load_model(config, model)
    # 评估模型
    metric = program.eval(model, valid_dataloader, post_process_class, eval_class)
    # 根据模型类型选择主要指标
    if config['Architecture']['model_type'] == 'det':
        main_indicator = 'hmean'
    else:
        main_indicator = 'acc'
    logger.info("metric['']: {}".format(main_indicator, metric[main_indicator])

    # 开始导出模型
    from paddle.jit import to_static

    # 设置推理形状为[3, -1, -1]
    infer_shape = [3, -1, -1]
    # 如果模型类型是rec，则推断形状为[3, 32, -1]，其中H必须为32
    if config['Architecture']['model_type'] == "rec":
        infer_shape = [3, 32, -1]

        # 如果网络中有TPS变换，并且不支持可变长度输入，则输入大小需要与训练期间相同
        if 'Transform' in config['Architecture'] and config['Architecture'][
                'Transform'] is not None and config['Architecture'][
                    'Transform']['name'] == 'TPS':
            logger.info(
                'When there is tps in the network, variable length input is not supported, and the input size needs to be the same as during training'
            )
            infer_shape[-1] = 100
    # 将模型转换为静态图模型，指定输入规格为[None] + 推断形状，数据类型为float32
    model = to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None] + infer_shape, dtype='float32')
        ])

    # 保存推断模型到指定路径
    save_path = '{}/inference'.format(config['Global']['save_inference_dir'])
    paddle.jit.save(model, save_path)
    logger.info('inference model is saved to {}'.format(save_path))
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 调用 program 模块的 preprocess 函数，传入 is_train=True 参数，获取 config, device, logger, vdl_writer 四个变量
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    # 调用 main 函数，传入 config, device, logger, vdl_writer 四个变量作为参数
    main(config, device, logger, vdl_writer)
```