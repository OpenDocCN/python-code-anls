# `.\PaddleOCR\ppocr\utils\save_load.py`

```
# 版权声明和许可信息
# 该代码受 Apache 许可证版本 2.0 保护
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件。请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import errno
import os
import pickle
import six
import paddle
from ppocr.utils.logging import get_logger
from ppocr.utils.network import maybe_download_params

# 定义模块的公开接口
__all__ = ['load_model']

# 定义一个函数，如果路径不存在则创建目录，当多个进程一起创建目录时忽略异常
def _mkdir_if_not_exist(path, logger):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))

# 加载模型的函数，从检查点或预训练模型加载模型
def load_model(config, model, optimizer=None, model_type='det'):
    logger = get_logger()
    global_config = config['Global']
    checkpoints = global_config.get('checkpoints')
    pretrained_model = global_config.get('pretrained_model')
    best_model_dict = {}
    is_float16 = False
    is_nlp_model = model_type == 'kie' and config["Architecture"][
        "algorithm"] not in ["SDMGR"]
    # 如果是 NLP 模型
    if is_nlp_model is True:
        # 如果使用知识蒸馏算法，则不支持恢复训练，直接返回最佳模型字典
        if config["Architecture"]["algorithm"] in ["Distillation"]:
            return best_model_dict
        # 获取模型的检查点路径
        checkpoints = config['Architecture']['Backbone']['checkpoints']
        # 加载知识蒸馏方法的度量
        if checkpoints:
            # 如果存在度量文件，则加载其中的状态字典
            if os.path.exists(os.path.join(checkpoints, 'metric.states')):
                with open(os.path.join(checkpoints, 'metric.states'),
                          'rb') as f:
                    # 根据 Python 版本选择不同的 pickle 加载方式
                    states_dict = pickle.load(f) if six.PY2 else pickle.load(
                        f, encoding='latin1')
                # 获取最佳模型字典
                best_model_dict = states_dict.get('best_model_dict', {})
                # 如果状态字典中包含 'epoch'，则更新开始训练的 epoch
                if 'epoch' in states_dict:
                    best_model_dict['start_epoch'] = states_dict['epoch'] + 1
            logger.info("resume from {}".format(checkpoints))

            # 如果存在优化器，则加载优化器状态
            if optimizer is not None:
                # 处理检查点路径末尾的斜杠
                if checkpoints[-1] in ['/', '\\']:
                    checkpoints = checkpoints[:-1]
                # 如果存在优化器状态文件，则加载优化器状态
                if os.path.exists(checkpoints + '.pdopt'):
                    optim_dict = paddle.load(checkpoints + '.pdopt')
                    optimizer.set_state_dict(optim_dict)
                else:
                    logger.warning(
                        "{}.pdopt is not exists, params of optimizer is not loaded".
                        format(checkpoints))

        return best_model_dict

    # 如果存在预训练模型
    elif pretrained_model:
        # 加载预训练参数到模型中
        is_float16 = load_pretrained_params(model, pretrained_model)
    else:
        logger.info('train from scratch')
    # 更新最佳模型字典中的 'is_float16' 字段
    best_model_dict['is_float16'] = is_float16
    return best_model_dict
# 加载预训练参数到模型中
def load_pretrained_params(model, path):
    # 获取日志记录器
    logger = get_logger()
    # 可能下载参数文件
    path = maybe_download_params(path)
    # 如果路径以'.pdparams'结尾，则去掉后缀
    if path.endswith('.pdparams'):
        path = path.replace('.pdparams', '')
    # 断言检查路径是否存在
    assert os.path.exists(path + ".pdparams"), \
        "The {}.pdparams does not exists!".format(path)

    # 加载参数文件
    params = paddle.load(path + '.pdparams')

    # 获取模型的状态字典
    state_dict = model.state_dict()

    # 新的状态字典
    new_state_dict = {}
    # 是否为float16类型
    is_float16 = False

    # 遍历参数字典
    for k1 in params.keys():

        # 如果参数不在模型的状态字典中，则记录警告
        if k1 not in state_dict.keys():
            logger.warning("The pretrained params {} not in model".format(k1))
        else:
            # 如果参数类型为float16，则设置is_float16为True
            if params[k1].dtype == paddle.float16:
                is_float16 = True
            # 如果参数类型与模型状态字典中的参数类型不一致，则转换类型
            if params[k1].dtype != state_dict[k1].dtype:
                params[k1] = params[k1].astype(state_dict[k1].dtype)
            # 如果参数形状与模型状态字典中的参数形状一致，则添加到新的状态字典中
            if list(state_dict[k1].shape) == list(params[k1].shape):
                new_state_dict[k1] = params[k1]
            else:
                # 记录警告，参数形状不匹配
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params {} {} !".
                    format(k1, state_dict[k1].shape, k1, params[k1].shape))

    # 设置模型的状态字典
    model.set_state_dict(new_state_dict)
    # 如果参数类型为float16，则记录信息，转换为float32
    if is_float16:
        logger.info(
            "The parameter type is float16, which is converted to float32 when loading"
        )
    # 记录信息，加载预训练参数成功
    logger.info("load pretrain successful from {}".format(path))
    return is_float16


def save_model(model,
               optimizer,
               model_path,
               logger,
               config,
               is_best=False,
               prefix='ppocr',
               **kwargs):
    """
    save model to the target path
    """
    # 如果目标路径不存在，则创建
    _mkdir_if_not_exist(model_path, logger)
    # 模型前缀
    model_prefix = os.path.join(model_path, prefix)

    # 如果前缀为'best_accuracy'，则创建最佳模型路径
    if prefix == 'best_accuracy':
        best_model_path = os.path.join(model_path, 'best_model')
        _mkdir_if_not_exist(best_model_path, logger)

    # 保存优化器的状态字典
    paddle.save(optimizer.state_dict(), model_prefix + '.pdopt')
    # 如果前缀为'best_accuracy'，则保存优化器的状态字典到指定路径下的'model.pdopt'文件中
    if prefix == 'best_accuracy':
        paddle.save(optimizer.state_dict(),
                    os.path.join(best_model_path, 'model.pdopt'))

    # 判断是否为自然语言处理模型
    is_nlp_model = config['Architecture']["model_type"] == 'kie' and config[
        "Architecture"]["algorithm"] not in ["SDMGR"]
    # 如果不是自然语言处理模型
    if is_nlp_model is not True:
        # 保存模型的状态字典到指定路径下的'.pdparams'文件中
        paddle.save(model.state_dict(), model_prefix + '.pdparams')
        metric_prefix = model_prefix

        # 如果前缀为'best_accuracy'，则保存模型的状态字典到指定路径下的'model.pdparams'文件中
        if prefix == 'best_accuracy':
            paddle.save(model.state_dict(),
                        os.path.join(best_model_path, 'model.pdparams'))

    else:  # 对于kie系统，遵循NLP中的保存/加载规则
        # 如果全局配置中指定了分布式训练
        if config['Global']['distributed']:
            arch = model._layers
        else:
            arch = model
        # 如果算法为Distillation，则将arch指定为arch.Student
        if config["Architecture"]["algorithm"] in ["Distillation"]:
            arch = arch.Student
        # 保存模型的预训练参数到指定路径下
        arch.backbone.model.save_pretrained(model_prefix)
        metric_prefix = os.path.join(model_prefix, 'metric')

        # 如果前缀为'best_accuracy'，则保存模型的预训练参数到指定路径下
        if prefix == 'best_accuracy':
            arch.backbone.model.save_pretrained(best_model_path)

    # 保存指标和配置信息
    with open(metric_prefix + '.states', 'wb') as f:
        pickle.dump(kwargs, f, protocol=2)
    # 如果是最佳模型，则打印保存最佳模型的信息
    if is_best:
        logger.info('save best model is to {}'.format(model_prefix))
    else:
        logger.info("save model in {}".format(model_prefix))
```