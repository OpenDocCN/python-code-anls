# `.\PaddleOCR\deploy\slim\prune\sensitivity_anal.py`

```
# 版权声明和许可证信息
# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入操作系统和系统库
import os
import sys

# 获取当前文件所在目录
__dir__ = os.path.dirname(__file__)
# 将当前目录添加到系统路径中
sys.path.append(__dir__)
# 将当前目录的上级目录添加到系统路径中
sys.path.append(os.path.join(__dir__, '..', '..', '..'))
# 将当前目录的上级目录的tools目录添加到系统路径中
sys.path.append(os.path.join(__dir__, '..', '..', '..', 'tools'))

# 导入paddle相关库
import paddle
import paddle.distributed as dist
# 导入构建数据加载器的函数和设置信号处理程序的函数
from ppocr.data import build_dataloader, set_signal_handlers
# 导入构建模型的函数
from ppocr.modeling.architectures import build_model
# 导入构建损失函数的函数
from ppocr.losses import build_loss
# 导入构建优化器的函数
from ppocr.optimizer import build_optimizer
# 导入构建后处理过程的函数
from ppocr.postprocess import build_post_process
# 导入构建度量指标的函数
from ppocr.metrics import build_metric
# 导入加载模型的函数
from ppocr.utils.save_load import load_model
# 导入程序工具
import tools.program as program

# 获取当前分布式环境的进程数量
dist.get_world_size()

# 获取需要剪枝的参数
def get_pruned_params(parameters):
    params = []

    for param in parameters:
        # 判断参数是否为4维，且不包含'depthwise'和'transpose'，并且不包含特定的卷积层名称
        if len(param.shape) == 4 and 'depthwise' not in param.name and 'transpose' not in param.name and "conv2d_57" not in param.name and "conv2d_56" not in param.name:
            params.append(param.name)
    return params

# 主函数
def main(config, device, logger, vdl_writer):
    # 初始化分布式环境
    if config['Global']['distributed']:
        dist.init_parallel_env()

    global_config = config['Global']

    # 构建数据加载器
    set_signal_handlers()
    # 构建训练数据加载器
    train_dataloader = build_dataloader(config, 'Train', device, logger)
    # 如果需要评估，则构建验证数据加载器，否则设为None
    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    else:
        valid_dataloader = None

    # 构建后处理类
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # 构建模型
    # 对于识别算法
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config['Architecture']["Head"]['out_channels'] = char_num
    model = build_model(config['Architecture'])
    # 根据模型类型设置输入形状
    if config['Architecture']['model_type'] == 'det':
        input_shape = [1, 3, 640, 640]
    elif config['Architecture']['model_type'] == 'rec':
        input_shape = [1, 3, 32, 320]
    # 计算模型的FLOPs
    flops = paddle.flops(model, input_shape)

    logger.info("FLOPs before pruning: {}".format(flops))

    # 导入FPGMFilterPruner
    from paddleslim.dygraph import FPGMFilterPruner
    model.train()

    # 构建剪枝器
    pruner = FPGMFilterPruner(model, input_shape)

    # 构建损失函数
    loss_class = build_loss(config['Loss'])

    # 构建优化器和学习率调度器
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader),
        model=model)

    # 构建评估指标
    eval_class = build_metric(config['Metric'])
    # 加载预训练模型
    pre_best_model_dict = load_model(config, model, optimizer)

    logger.info('train dataloader has {} iters, valid dataloader has {} iters'.
                format(len(train_dataloader), len(valid_dataloader)))
    # 构建评估指标
    eval_class = build_metric(config['Metric'])

    logger.info('train dataloader has {} iters, valid dataloader has {} iters'.
                format(len(train_dataloader), len(valid_dataloader)))
    # 定义一个函数用于评估模型性能
    def eval_fn():
        # 调用程序的评估函数，传入模型、验证数据加载器、后处理类、评估类和是否打印信息的标志
        metric = program.eval(model, valid_dataloader, post_process_class,
                              eval_class, False)
        # 根据模型类型设置主要指标
        if config['Architecture']['model_type'] == 'det':
            main_indicator = 'hmean'
        else:
            main_indicator = 'acc'

        # 打印主要指标的评估结果
        logger.info("metric[{}]: {}".format(main_indicator, metric[
            main_indicator]))
        # 返回主要指标的评估结果
        return metric[main_indicator]

    # 是否运行敏感性分析的标志
    run_sensitive_analysis = False
    """
    run_sensitive_analysis=True: 
        自动计算模型中卷积的敏感性。
        卷积的敏感性是指在不同修剪比例下测试数据集上的准确率损失。
        可以使用敏感性来得到一组最佳比例，满足某些条件。
    
    run_sensitive_analysis=False: 
        将修剪比例设置为固定值，例如10%。值越大，修剪的卷积权重越多。

    """

    # 如果运行敏感性分析
    if run_sensitive_analysis:
        # 计算参数的敏感性
        params_sensitive = pruner.sensitive(
            eval_func=eval_fn,
            sen_file="./deploy/slim/prune/sen.pickle",
            skip_vars=[
                "conv2d_57.w_0", "conv2d_transpose_2.w_0",
                "conv2d_transpose_3.w_0"
            ])
        # 打印模型参数的敏感性分析结果保存在sen.pickle中
        logger.info(
            "The sensitivity analysis results of model parameters saved in sen.pickle"
        )
        # 计算被修剪参数的比例
        params_sensitive = pruner._get_ratios_by_loss(
            params_sensitive, loss=0.02)
        # 遍历敏感性参数字典，打印每个参数的修剪比例
        for key in params_sensitive.keys():
            logger.info("{}, {}".format(key, params_sensitive[key]))
    else:
        # 如果不运行敏感性分析，将参数的敏感性设置为空字典
        params_sensitive = {}
        # 遍历模型的参数
        for param in model.parameters():
            # 如果参数名称中不包含'transpose'和'linear'
            if 'transpose' not in param.name and 'linear' not in param.name:
                # 将参数的修剪比例设置为10%
                params_sensitive[param.name] = 0.1
    # 使用敏感参数和需要保留的索引来剪枝模型，得到剪枝后的计划
    plan = pruner.prune_vars(params_sensitive, [0])

    # 计算模型的浮点运算次数（FLOPs）
    flops = paddle.flops(model, input_shape)
    # 打印剪枝后的FLOPs
    logger.info("FLOPs after pruning: {}".format(flops))

    # 开始训练

    # 使用配置、训练数据加载器、验证数据加载器、设备、模型、损失函数、优化器、学习率调度器、后处理类、评估类、之前最佳模型字典、日志记录器、可视化写入器来训练模型
    program.train(config, train_dataloader, valid_dataloader, device, model,
                  loss_class, optimizer, lr_scheduler, post_process_class,
                  eval_class, pre_best_model_dict, logger, vdl_writer)
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 调用 program 模块的 preprocess 函数，传入 is_train=True 参数，获取 config, device, logger, vdl_writer 四个变量
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    # 调用 main 函数，传入 config, device, logger, vdl_writer 四个变量作为参数
    main(config, device, logger, vdl_writer)
```