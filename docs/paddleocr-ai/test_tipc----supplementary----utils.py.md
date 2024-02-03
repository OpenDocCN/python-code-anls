# `.\PaddleOCR\test_tipc\supplementary\utils.py`

```
# 导入所需的库
import os
import sys
import logging
import functools
import paddle.distributed as dist

# 初始化记录器字典
logger_initialized = {}

# 定义一个函数，用于递归打印字典内容
def print_dict(d, logger, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    # 遍历字典的键值对
    for k, v in sorted(d.items()):
        # 如果值是字典类型，则递归打印
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, logger, delimiter + 4)
        # 如果值是列表类型且第一个元素是字典类型，则递归打印
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, logger, delimiter + 4)
        # 否则直接打印键值对
        else:
            logger.info("{}{} : {}".format(delimiter * " ", k, v))

# 使用 functools.lru_cache 装饰器缓存结果，初始化并获取记录器
@functools.lru_cache()
def get_logger(name='root', log_file=None, log_level=logging.DEBUG):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified a FileHandler will also be added.
    def get_logger(name, log_file, log_level):
        """
        创建并配置一个日志记录器
    
        Args:
            name (str): 日志记录器名称
            log_file (str | None): 日志文件名。如果指定了文件名，将会添加一个 FileHandler 到日志记录器中
            log_level (int): 日志记录器级别。注意只有进程 0 的日志记录器级别会受影响，其他进程将会被设置为"Error"级别，因此大部分时间会保持静默
    
        Returns:
            logging.Logger: 期望的日志记录器
        """
        # 获取指定名称的日志记录器
        logger = logging.getLogger(name)
        # 如果该名称的日志记录器已经初始化过，则直接返回
        if name in logger_initialized:
            return logger
        # 遍历已初始化的日志记录器名称列表
        for logger_name in logger_initialized:
            # 如果指定的名称以已初始化的日志记录器名称开头，则直接返回
            if name.startswith(logger_name):
                return logger
    
        # 设置日志格式
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
            datefmt="%Y/%m/%d %H:%M:%S")
    
        # 创建一个输出到控制台的 StreamHandler
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
        # 如果指定了日志文件名且当前进程为进程 0，则添加一个输出到文件的 FileHandler
        if log_file is not None and dist.get_rank() == 0:
            log_file_folder = os.path.split(log_file)[0]
            os.makedirs(log_file_folder, exist_ok=True)
            file_handler = logging.FileHandler(log_file, 'a')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
        # 如果当前进程为进程 0，则设置日志记录器级别为指定的级别，否则设置为 ERROR 级别
        if dist.get_rank() == 0:
            logger.setLevel(log_level)
        else:
            logger.setLevel(logging.ERROR)
    
        # 标记该日志记录器已经初始化
        logger_initialized[name] = True
        return logger
# 从检查点或预训练模型加载模型
def load_model(config, model, optimizer=None):
    """
    load model from checkpoint or pretrained_model
    """
    # 获取日志记录器
    logger = get_logger()
    # 获取配置中的检查点路径
    checkpoints = config.get('checkpoints')
    # 获取配置中的预训练模型路径
    pretrained_model = config.get('pretrained_model')
    # 创建空字典用于存储最佳模型参数
    best_model_dict = {}
    # 如果存在检查点
    if checkpoints:
        # 如果检查点以'.pdparams'结尾，则去掉'.pdparams'
        if checkpoints.endswith('.pdparams'):
            checkpoints = checkpoints.replace('.pdparams', '')
        # 断言检查点文件是否存在
        assert os.path.exists(checkpoints + ".pdparams"), \
            "The {}.pdparams does not exists!".format(checkpoints)

        # 从训练好的模型中加载参数
        params = paddle.load(checkpoints + '.pdparams')
        state_dict = model.state_dict()
        new_state_dict = {}
        # 遍历模型的状态字典
        for key, value in state_dict.items():
            # 如果键不在加载的参数中，则发出警告
            if key not in params:
                logger.warning("{} not in loaded params {} !".format(
                    key, params.keys()))
                continue
            pre_value = params[key]
            # 如果模型参数的形状与加载的参数形状相同，则更新状态字典
            if list(value.shape) == list(pre_value.shape):
                new_state_dict[key] = pre_value
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params shape {} !".
                    format(key, value.shape, pre_value.shape))
        # 设置模型的状态字典
        model.set_state_dict(new_state_dict)

        # 如果存在优化器
        if optimizer is not None:
            # 如果优化器参数文件存在，则加载优化器参数
            if os.path.exists(checkpoints + '.pdopt'):
                optim_dict = paddle.load(checkpoints + '.pdopt')
                optimizer.set_state_dict(optim_dict)
            else:
                logger.warning(
                    "{}.pdopt is not exists, params of optimizer is not loaded".
                    format(checkpoints))

        # 如果存在状态文件
        if os.path.exists(checkpoints + '.states'):
            with open(checkpoints + '.states', 'rb') as f:
                # 加载状态字典
                states_dict = pickle.load(f) if six.PY2 else pickle.load(
                    f, encoding='latin1')
                best_model_dict = states_dict.get('best_model_dict', {})
                # 如果状态字典中包含'epoch'，则更新开始的epoch
                if 'epoch' in states_dict:
                    best_model_dict['start_epoch'] = states_dict['epoch'] + 1
        # 打印日志，表示从检查点中恢复
        logger.info("resume from {}".format(checkpoints))
    # 如果存在预训练模型
    elif pretrained_model:
        # 加载预训练参数
        load_pretrained_params(model, pretrained_model)
    # 如果没有找到预训练模型，则记录日志信息“从头开始训练”
    else:
        logger.info('train from scratch')
    # 返回最佳模型字典
    return best_model_dict
# 加载预训练参数到模型中
def load_pretrained_params(model, path):
    # 获取日志记录器
    logger = get_logger()
    # 如果路径以'.pdparams'结尾，则去掉'.pdparams'
    if path.endswith('.pdparams'):
        path = path.replace('.pdparams', '')
    # 断言检查路径是否存在'.pdparams'文件
    assert os.path.exists(path + ".pdparams"), \
        "The {}.pdparams does not exists!".format(path)

    # 加载参数文件
    params = paddle.load(path + '.pdparams')
    # 获取模型的状态字典
    state_dict = model.state_dict()
    # 创建新的状态字典
    new_state_dict = {}
    # 遍历参数字典的键
    for k1 in params.keys():
        # 如果参数键不在模型状态字典中，则记录警告
        if k1 not in state_dict.keys():
            logger.warning("The pretrained params {} not in model".format(k1))
        else:
            # 如果模型参数和加载的参数形状相同，则添加到新的状态字典中
            if list(state_dict[k1].shape) == list(params[k1].shape):
                new_state_dict[k1] = params[k1]
            else:
                # 如果形状不匹配，则记录警告
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params {} {} !".
                    format(k1, state_dict[k1].shape, k1, params[k1].shape))
    # 将新的状态字典设置到模型中
    model.set_state_dict(new_state_dict)
    # 记录加载预训练参数成功的信息
    logger.info("load pretrain successful from {}".format(path))
    # 返回加载了预训练参数的模型
    return model
```