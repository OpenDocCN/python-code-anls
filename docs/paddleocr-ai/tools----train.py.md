# `.\PaddleOCR\tools\train.py`

```
# 版权声明和许可证信息
# 从未来模块导入绝对导入、除法和打印函数
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入操作系统和系统模块
import os
import sys

# 获取当前文件所在目录的绝对路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录添加到系统路径中
sys.path.append(__dir__)
# 将当前文件所在目录的上级目录添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

# 导入yaml模块
import yaml
# 导入paddle模块
import paddle
# 导入paddle分布式模块
import paddle.distributed as dist

# 从ppocr包中导入build_dataloader和set_signal_handlers函数
from ppocr.data import build_dataloader, set_signal_handlers
# 从ppocr.modeling.architectures包中导入build_model函数
from ppocr.modeling.architectures import build_model
# 从ppocr.losses包中导入build_loss函数
from ppocr.losses import build_loss
# 从ppocr.optimizer包中导入build_optimizer函数
from ppocr.optimizer import build_optimizer
# 从ppocr.postprocess包中导入build_post_process函数
from ppocr.postprocess import build_post_process
# 从ppocr.metrics包中导入build_metric函数
from ppocr.metrics import build_metric
# 从ppocr.utils.save_load包中导入load_model函数
from ppocr.utils.save_load import load_model
# 从ppocr.utils.utility包中导入set_seed函数
from ppocr.utils.utility import set_seed
# 从ppocr.modeling.architectures包中导入apply_to_static函数
from ppocr.modeling.architectures import apply_to_static
# 从tools.program模块中导入program函数
import tools.program as program

# 获取当前分布式环境的进程数量
dist.get_world_size()

# 主函数，接受配置、设备、日志和可视化写入器作为参数
def main(config, device, logger, vdl_writer):
    # 初始化分布式环境
    if config['Global']['distributed']:
        dist.init_parallel_env()

    # 获取全局配置
    global_config = config['Global']

    # 构建数据加载器
    set_signal_handlers()
    # 构建训练数据加载器
    train_dataloader = build_dataloader(config, 'Train', device, logger)
    # 检查训练数据加载器中是否有数据，如果没有则记录错误信息并返回
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n" +
            "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            +
            "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    # 根据配置文件中的 Eval 标志来构建验证数据加载器或者设置为 None
    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    else:
        valid_dataloader = None

    # 构建后处理类
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # 构建模型
    # 用于 rec 算法
    model = build_model(config['Architecture'])

    # 检查是否使用同步批量归一化
    use_sync_bn = config["Global"].get("use_sync_bn", False)
    if use_sync_bn:
        model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info('convert_sync_batchnorm')

    # 应用静态方法到模型
    model = apply_to_static(model, config, logger)

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

    logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    # 如果验证数据加载器不为空，则记录验证数据加载器的迭代次数
    if valid_dataloader is not None:
        logger.info('valid dataloader has {} iters'.format(
            len(valid_dataloader)))

    # 检查是否使用自动混合精度训练
    use_amp = config["Global"].get("use_amp", False)
    amp_level = config["Global"].get("amp_level", 'O2')
    amp_dtype = config["Global"].get("amp_dtype", 'float16')
    amp_custom_black_list = config['Global'].get('amp_custom_black_list', [])
    amp_custom_white_list = config['Global'].get('amp_custom_white_list', [])
    # 如果使用 AMP（Automatic Mixed Precision），设置相关的标志位
    if use_amp:
        # 定义与 AMP 相关的标志位设置
        AMP_RELATED_FLAGS_SETTING = {'FLAGS_max_inplace_grad_add': 8, }
        # 如果编译时使用了 CUDA，更新相关的标志位设置
        if paddle.is_compiled_with_cuda():
            AMP_RELATED_FLAGS_SETTING.update({
                'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
                'FLAGS_gemm_use_half_precision_compute_type': 0,
            })
        # 设置 AMP 相关的标志位
        paddle.set_flags(AMP_RELATED_FLAGS_SETTING)
        # 获取损失的缩放比例
        scale_loss = config["Global"].get("scale_loss", 1.0)
        # 获取是否使用动态损失缩放
        use_dynamic_loss_scaling = config["Global"].get(
            "use_dynamic_loss_scaling", False)
        # 创建 AMP 的 GradScaler 对象
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling)
        # 如果 AMP 级别为 "O2"，使用 AMP 装饰模型和优化器
        if amp_level == "O2":
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level=amp_level,
                master_weight=True,
                dtype=amp_dtype)
    else:
        # 如果不使用 AMP，将 scaler 设为 None
        scaler = None

    # 加载预训练模型
    pre_best_model_dict = load_model(config, model, optimizer,
                                     config['Architecture']["model_type"])

    # 如果配置中指定了分布式训练，使用 DataParallel 封装模型
    if config['Global']['distributed']:
        model = paddle.DataParallel(model)
    # 开始训练
    program.train(config, train_dataloader, valid_dataloader, device, model,
                  loss_class, optimizer, lr_scheduler, post_process_class,
                  eval_class, pre_best_model_dict, logger, vdl_writer, scaler,
                  amp_level, amp_custom_black_list, amp_custom_white_list,
                  amp_dtype)
# 定义一个测试数据读取函数，接受配置、设备和日志对象作为参数
def test_reader(config, device, logger):
    # 使用build_dataloader函数构建一个数据加载器，传入配置、训练标志、设备和日志对象
    loader = build_dataloader(config, 'Train', device, logger)
    # 导入时间模块
    import time
    # 记录开始时间
    starttime = time.time()
    # 初始化计数器
    count = 0
    try:
        # 遍历数据加载器
        for data in loader():
            # 每次迭代计数器加一
            count += 1
            # 每处理1个数据，计算批处理时间
            if count % 1 == 0:
                batch_time = time.time() - starttime
                starttime = time.time()
                # 记录日志，包括计数器、数据长度和批处理时间
                logger.info("reader: {}, {}, {}".format(
                    count, len(data[0]), batch_time))
    except Exception as e:
        # 捕获异常并记录到日志中
        logger.info(e)
    # 记录读取结束信息，包括计数器
    logger.info("finish reader: {}, Success!".format(count))


if __name__ == '__main__':
    # 预处理程序，获取配置、设备、日志对象和可视化写入器
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    # 获取全局配置中的种子值，如果不存在则使用默认值1024
    seed = config['Global']['seed'] if 'seed' in config['Global'] else 1024
    # 设置随机种子
    set_seed(seed)
    # 调用主函数，传入配置、设备、日志对象和可视化写入器
    main(config, device, logger, vdl_writer)
    # 调用测试数据读取函数，传入配置、设备和日志对象
    # test_reader(config, device, logger)
```