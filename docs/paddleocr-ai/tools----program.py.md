# `.\PaddleOCR\tools\program.py`

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
import platform
import yaml
import time
import datetime
import paddle
import paddle.distributed as dist
from tqdm import tqdm
import cv2
import numpy as np
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# 导入自定义模块
from ppocr.utils.stats import TrainingStats
from ppocr.utils.save_load import save_model
from ppocr.utils.utility import print_dict, AverageMeter
from ppocr.utils.logging import get_logger
from ppocr.utils.loggers import VDLLogger, WandbLogger, Loggers
from ppocr.utils import profiler
from ppocr.data import build_dataloader

# 定义参数解析类，继承自 ArgumentParser
class ArgsParser(ArgumentParser):
    def __init__(self):
        # 调用父类构造函数，设置 formatter_class 为 RawDescriptionHelpFormatter
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        # 添加配置文件参数
        self.add_argument("-c", "--config", help="configuration file to use")
        # 添加配置选项参数
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")
        # 添加性能分析器选项参数
        self.add_argument(
            '-p',
            '--profiler_options',
            type=str,
            default=None,
            help='The option of profiler, which should be in format ' \
                 '\"key1=value1;key2=value2;key3=value3\".'
        )
    # 解析命令行参数，如果未指定则抛出异常
    def parse_args(self, argv=None):
        # 调用父类方法解析参数
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        # 解析额外的选项参数
        args.opt = self._parse_opt(args.opt)
        # 返回解析后的参数
        return args
    
    # 解析额外的选项参数
    def _parse_opt(self, opts):
        config = {}
        # 如果选项参数为空，则返回空字典
        if not opts:
            return config
        # 遍历选项参数列表
        for s in opts:
            # 去除空格
            s = s.strip()
            # 根据等号分割键值对
            k, v = s.split('=')
            # 使用 yaml 加载值并存入字典
            config[k] = yaml.load(v, Loader=yaml.Loader)
        # 返回解析后的配置字典
        return config
# 从 yml/yaml 文件加载配置信息
def load_config(file_path):
    # 获取文件路径的扩展名
    _, ext = os.path.splitext(file_path)
    # 确保文件扩展名为 .yml 或 .yaml
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    # 读取并加载 yaml 文件内容
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    # 返回加载的配置信息
    return config


# 将配置信息合并到全局配置中
def merge_config(config, opts):
    # 遍历配置项
    for key, value in opts.items():
        # 如果配置项中不包含"."，直接更新配置信息
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            # 如果配置项包含"."，按照"."分割后逐级更新配置信息
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in config
            ), "the sub_keys can only be one of global_config: {}, but get: " \
               "{}, please check your running command".format(
                config.keys(), sub_keys[0])
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    # 返回合并后的配置信息
    return config


# 检查设备类型，当在 paddlepaddle 的 CPU 版本中设置 use_gpu=True 时，记录错误并退出
def check_device(use_gpu, use_xpu=False, use_npu=False, use_mlu=False):
    err = "Config {} cannot be set as true while your paddle " \
          "is not compiled with {} ! \nPlease try: \n" \
          "\t1. Install paddlepaddle to run model on {} \n" \
          "\t2. Set {} as false in config file to run " \
          "model on CPU"
    # 尝试检查是否使用了 GPU 和 XPU，如果两者都为真，则打印错误信息
    try:
        if use_gpu and use_xpu:
            print("use_xpu and use_gpu can not both be ture.")
        # 如果使用了 GPU 但未编译支持 CUDA，则打印错误信息并退出程序
        if use_gpu and not paddle.is_compiled_with_cuda():
            print(err.format("use_gpu", "cuda", "gpu", "use_gpu"))
            sys.exit(1)
        # 如果使用了 XPU 但未编译支持 XPU，则打印错误信息并退出程序
        if use_xpu and not paddle.device.is_compiled_with_xpu():
            print(err.format("use_xpu", "xpu", "xpu", "use_xpu"))
            sys.exit(1)
        # 如果使用了 NPU
        if use_npu:
            # 如果 PaddlePaddle 版本为 2.4 及以下且未编译支持 NPU，则打印错误信息并退出程序
            if int(paddle.version.major) != 0 and int(paddle.version.major) <= 2 and int(paddle.version.minor) <= 4:
                if not paddle.device.is_compiled_with_npu():
                    print(err.format("use_npu", "npu", "npu", "use_npu"))
                    sys.exit(1)
            # 在 paddle-2.4 之后，is_compiled_with_npu() 已更新
            else:
                # 如果未编译支持自定义设备 "npu"，则打印错误信息并退出程序
                if not paddle.device.is_compiled_with_custom_device("npu"):
                    print(err.format("use_npu", "npu", "npu", "use_npu"))
                    sys.exit(1)
        # 如果使用了 MLU 但未编译支持 MLU，则打印错误信息并退出程序
        if use_mlu and not paddle.device.is_compiled_with_mlu():
            print(err.format("use_mlu", "mlu", "mlu", "use_mlu"))
            sys.exit(1)
    # 捕获异常并忽略
    except Exception as e:
        pass
# 将输入的预测结果转换为 float32 类型
def to_float32(preds):
    # 如果 preds 是字典类型
    if isinstance(preds, dict):
        # 遍历字典中的每个键
        for k in preds:
            # 如果值是字典或列表类型，则递归调用 to_float32 函数
            if isinstance(preds[k], dict) or isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            # 如果值是 paddle.Tensor 类型，则转换为 float32 类型
            elif isinstance(preds[k], paddle.Tensor):
                preds[k] = preds[k].astype(paddle.float32)
    # 如果 preds 是列表类型
    elif isinstance(preds, list):
        # 遍历列表中的每个元素
        for k in range(len(preds)):
            # 如果元素是字典类型，则递归调用 to_float32 函数
            if isinstance(preds[k], dict):
                preds[k] = to_float32(preds[k])
            # 如果元素是列表类型，则递归调用 to_float32 函数
            elif isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            # 如果元素是 paddle.Tensor 类型，则转换为 float32 类型
            elif isinstance(preds[k], paddle.Tensor):
                preds[k] = preds[k].astype(paddle.float32)
    # 如果 preds 是 paddle.Tensor 类型
    elif isinstance(preds, paddle.Tensor):
        # 转换为 float32 类型
        preds = preds.astype(paddle.float32)
    # 返回转换后的结果
    return preds

# 训练函数，接受多个参数
def train(config,
          train_dataloader,
          valid_dataloader,
          device,
          model,
          loss_class,
          optimizer,
          lr_scheduler,
          post_process_class,
          eval_class,
          pre_best_model_dict,
          logger,
          log_writer=None,
          scaler=None,
          amp_level='O2',
          amp_custom_black_list=[],
          amp_custom_white_list=[],
          amp_dtype='float16'):
    # 获取配置中的参数
    cal_metric_during_train = config['Global'].get('cal_metric_during_train', False)
    calc_epoch_interval = config['Global'].get('calc_epoch_interval', 1)
    log_smooth_window = config['Global']['log_smooth_window']
    epoch_num = config['Global']['epoch_num']
    print_batch_step = config['Global']['print_batch_step']
    eval_batch_step = config['Global']['eval_batch_step']
    profiler_options = config['profiler_options']

    global_step = 0
    # 如果预训练模型字典中包含全局步数，则更新全局步数
    if 'global_step' in pre_best_model_dict:
        global_step = pre_best_model_dict['global_step']
    start_eval_step = 0
    # 检查 eval_batch_step 是否为列表且长度大于等于2
    if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
        # 将 eval_batch_step 列表中的第一个元素赋值给 start_eval_step
        start_eval_step = eval_batch_step[0]
        # 将 eval_batch_step 列表中的第二个元素赋值给 eval_batch_step
        eval_batch_step = eval_batch_step[1]
        # 如果 valid_dataloader 中没有数据
        if len(valid_dataloader) == 0:
            # 输出日志信息，表示 eval 数据集中没有图像，禁用训练过程中的评估
            logger.info(
                'No Images in eval dataset, evaluation during training ' \
                'will be disabled'
            )
            # 将 start_eval_step 设置为一个很大的值
            start_eval_step = 1e111
        # 输出日志信息，表示在训练过程中，每经过 eval_batch_step 次迭代后进行一次评估
        logger.info(
            "During the training process, after the {}th iteration, " \
            "an evaluation is run every {} iterations".
            format(start_eval_step, eval_batch_step))
    
    # 从配置中获取保存模型的间隔步数和保存模型的目录
    save_epoch_step = config['Global']['save_epoch_step']
    save_model_dir = config['Global']['save_model_dir']
    # 如果保存模型的目录不存在，则创建目录
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    
    # 获取评估类的主要指标
    main_indicator = eval_class.main_indicator
    # 创建包含主要指标的字典，并更新之前的最佳模型字典
    best_model_dict = {main_indicator: 0}
    best_model_dict.update(pre_best_model_dict)
    # 创建训练统计对象
    train_stats = TrainingStats(log_smooth_window, ['lr'])
    # 设置模型平均为 False
    model_average = False
    # 将模型设置为训练模式
    model.train()

    # 检查是否使用 SRN 算法
    use_srn = config['Architecture']['algorithm'] == "SRN"
    # 额外输入模型列表
    extra_input_models = [
        "SRN", "NRTR", "SAR", "SEED", "SVTR", "SVTR_LCNet", "SPIN", "VisionLAN",
        "RobustScanner", "RFL", 'DRRG', 'SATRN', 'SVTR_HGNet'
    ]
    # 是否有额外输入
    extra_input = False
    # 如果算法为 Distillation
    if config['Architecture']['algorithm'] == 'Distillation':
        # 遍历模型配置中的 Models 字段
        for key in config['Architecture']["Models"]:
            # 检查是否有额外输入模型
            extra_input = extra_input or config['Architecture']['Models'][key][
                'algorithm'] in extra_input_models
    else:
        # 检查是否有额外输入模型
        extra_input = config['Architecture']['algorithm'] in extra_input_models
    # 尝试获取模型类型，如果失败则设置为 None
    try:
        model_type = config['Architecture']['model_type']
    except:
        model_type = None

    # 获取算法类型
    algorithm = config['Architecture']['algorithm']

    # 获取开始训练的轮数
    start_epoch = best_model_dict[
        'start_epoch'] if 'start_epoch' in best_model_dict else 1

    # 初始化总样本数、训练读取器成本、训练批次成本
    total_samples = 0
    train_reader_cost = 0.0
    train_batch_cost = 0.0
    # 记录读取器开始时间
    reader_start = time.time()
    # 创建一个用于计算平均值的计量器
    eta_meter = AverageMeter()
    
    # 根据操作系统判断最大迭代次数
    max_iter = len(train_dataloader) - 1 if platform.system() == "Windows" else len(train_dataloader)
    
    # 根据最佳模型字典中的键值对生成字符串
    best_str = 'best metric, {}'.format(', '.join(['{}: {}'.format(k, v) for k, v in best_model_dict.items()]))
    # 在日志中记录最佳模型信息
    logger.info(best_str)
    
    # 如果当前进程是主进程且日志写入器不为空，则关闭日志写入器
    if dist.get_rank() == 0 and log_writer is not None:
        log_writer.close()
    
    # 返回
    return
# 对模型进行评估，返回评估指标
def eval(model,
         valid_dataloader,
         post_process_class,
         eval_class,
         model_type=None,
         extra_input=False,
         scaler=None,
         amp_level='O2',
         amp_custom_black_list=[],
         amp_custom_white_list=[],
         amp_dtype='float16'):
    # 将模型设置为评估模式
    model.eval()
    # 关闭进度条
    pbar.close()
    # 将模型设置为训练模式
    model.train()
    # 计算每秒处理的帧数
    metric['fps'] = total_frame / total_time
    # 返回评估指标
    return metric


# 更新字符中心
def update_center(char_center, post_result, preds):
    result, label = post_result
    feats, logits = preds
    # 将 logits 转换为 numpy 数组
    logits = paddle.argmax(logits, axis=-1)
    feats = feats.numpy()
    logits = logits.numpy()

    for idx_sample in range(len(label)):
        if result[idx_sample][0] == label[idx_sample][0]:
            feat = feats[idx_sample]
            logit = logits[idx_sample]
            for idx_time in range(len(logit)):
                index = logit[idx_time]
                if index in char_center.keys():
                    char_center[index][0] = (
                        char_center[index][0] * char_center[index][1] +
                        feat[idx_time]) / (char_center[index][1] + 1)
                    char_center[index][1] += 1
                else:
                    char_center[index] = [feat[idx_time], 1]
    return char_center


# 获取字符中心
def get_center(model, eval_dataloader, post_process_class):
    # 创建进度条
    pbar = tqdm(total=len(eval_dataloader), desc='get center:')
    # 根据操作系统设置最大迭代次数
    max_iter = len(eval_dataloader) - 1 if platform.system(
    ) == "Windows" else len(eval_dataloader)
    # 初始化字符中心字典
    char_center = dict()
    for idx, batch in enumerate(eval_dataloader):
        if idx >= max_iter:
            break
        images = batch[0]
        start = time.time()
        # 获取模型预测结果
        preds = model(images)

        batch = [item.numpy() for item in batch]
        # 从后处理方法中获取可用结果
        post_result = post_process_class(preds, batch[1])

        # 更新字符中心
        char_center = update_center(char_center, post_result, preds)
        pbar.update(1)
    # 关闭进度条
    pbar.close()
    # 遍历 char_center 字典的键
    for key in char_center.keys():
        # 将每个键对应的值改为其第一个元素
        char_center[key] = char_center[key][0]
    # 返回更新后的 char_center 字典
    return char_center
# 预处理函数，根据是否为训练模式来执行不同的操作
def preprocess(is_train=False):
    # 解析命令行参数
    FLAGS = ArgsParser().parse_args()
    # 获取性能分析器选项
    profiler_options = FLAGS.profiler_options
    # 加载配置文件
    config = load_config(FLAGS.config)
    # 合并配置文件和命令行参数
    config = merge_config(config, FLAGS.opt)
    # 创建性能分析器配置字典
    profile_dic = {"profiler_options": FLAGS.profiler_options}
    # 合并配置文件和性能分析器配置字典
    config = merge_config(config, profile_dic)

    if is_train:
        # 如果是训练模式，保存配置文件
        save_model_dir = config['Global']['save_model_dir']
        # 创建保存模型的目录
        os.makedirs(save_model_dir, exist_ok=True)
        # 将配置信息写入文件
        with open(os.path.join(save_model_dir, 'config.yml'), 'w') as f:
            yaml.dump(
                dict(config), f, default_flow_style=False, sort_keys=False)
        log_file = '{}/train.log'.format(save_model_dir)
    else:
        log_file = None
    # 获取日志记录器
    logger = get_logger(log_file=log_file)

    # 检查是否在 CPU 版本的 PaddlePaddle 中设置了 use_gpu=True
    use_gpu = config['Global'].get('use_gpu', False)
    use_xpu = config['Global'].get('use_xpu', False)
    use_npu = config['Global'].get('use_npu', False)
    use_mlu = config['Global'].get('use_mlu', False)

    # 获取算法名称
    alg = config['Architecture']['algorithm']
    # 断言算法名称在指定的列表中
    assert alg in [
        'EAST', 'DB', 'SAST', 'Rosetta', 'CRNN', 'STARNet', 'RARE', 'SRN',
        'CLS', 'PGNet', 'Distillation', 'NRTR', 'TableAttn', 'SAR', 'PSE',
        'SEED', 'SDMGR', 'LayoutXLM', 'LayoutLM', 'LayoutLMv2', 'PREN', 'FCE',
        'SVTR', 'SVTR_LCNet', 'ViTSTR', 'ABINet', 'DB++', 'TableMaster', 'SPIN',
        'VisionLAN', 'Gestalt', 'SLANet', 'RobustScanner', 'CT', 'RFL', 'DRRG',
        'CAN', 'Telescope', 'SATRN', 'SVTR_HGNet'
    ]

    # 根据不同的设备类型设置设备字符串
    if use_xpu:
        device = 'xpu:{0}'.format(os.getenv('FLAGS_selected_xpus', 0))
    elif use_npu:
        device = 'npu:{0}'.format(os.getenv('FLAGS_selected_npus', 0))
    elif use_mlu:
        device = 'mlu:{0}'.format(os.getenv('FLAGS_selected_mlus', 0))
    else:
        device = 'gpu:{}'.format(dist.ParallelEnv()
                                 .dev_id) if use_gpu else 'cpu'
    # 检查设备类型
    check_device(use_gpu, use_xpu, use_npu, use_mlu)
    # 设置 PaddlePaddle 使用的设备
    device = paddle.set_device(device)
    
    # 检查是否为分布式训练，更新配置中的 distributed 字段
    config['Global']['distributed'] = dist.get_world_size() != 1
    
    # 初始化日志记录器列表
    loggers = []
    
    # 检查是否使用 VisualDL，若是则创建 VisualDL 日志记录器
    if 'use_visualdl' in config['Global'] and config['Global']['use_visualdl']:
        save_model_dir = config['Global']['save_model_dir']
        vdl_writer_path = save_model_dir
        log_writer = VDLLogger(vdl_writer_path)
        loggers.append(log_writer)
    
    # 检查是否使用 WandB，若是则创建 WandB 日志记录器
    if ('use_wandb' in config['Global'] and
            config['Global']['use_wandb']) or 'wandb' in config:
        save_dir = config['Global']['save_model_dir']
        wandb_writer_path = "{}/wandb".format(save_dir)
        if "wandb" in config:
            wandb_params = config['wandb']
        else:
            wandb_params = dict()
        wandb_params.update({'save_dir': save_dir})
        log_writer = WandbLogger(**wandb_params, config=config)
        loggers.append(log_writer)
    else:
        log_writer = None
    
    # 打印配置信息
    print_dict(config, logger)
    
    # 根据是否有日志记录器，创建 Loggers 对象
    if loggers:
        log_writer = Loggers(loggers)
    else:
        log_writer = None
    
    # 打印使用的 PaddlePaddle 版本和设备信息
    logger.info('train with paddle {} and device {}'.format(paddle.__version__,
                                                            device))
    # 返回配置、设备、日志记录器和日志写入器
    return config, device, logger, log_writer
```