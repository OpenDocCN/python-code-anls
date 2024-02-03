# `.\PaddleOCR\test_tipc\supplementary\train.py`

```py
# 导入 paddle 库
import paddle
# 导入 numpy 库，并重命名为 np
import numpy as np
# 导入 os 库
import os
# 导入 paddle 中的神经网络模块
import paddle.nn as nn
# 导入 paddle 分布式训练模块
import paddle.distributed as dist
# 获取当前进程组的进程数量
dist.get_world_size()
# 初始化并行环境
dist.init_parallel_env()

# 从 loss.py 文件中导入 build_loss、LossDistill、DMLLoss、KLJSLoss 函数
from loss import build_loss, LossDistill, DMLLoss, KLJSLoss
# 从 optimizer.py 文件中导入 create_optimizer 函数
from optimizer import create_optimizer
# 从 data_loader.py 文件中导入 build_dataloader 函数
from data_loader import build_dataloader
# 从 metric.py 文件中导入 create_metric 函数
from metric import create_metric
# 从 mv3.py 文件中导入 MobileNetV3_large_x0_5、distillmv3_large_x0_5、build_model 函数
from mv3 import MobileNetV3_large_x0_5, distillmv3_large_x0_5, build_model
# 从 config.py 文件中导入 preprocess 函数
from config import preprocess
# 导入 time 库

# 从 paddleslim.dygraph.quant 模块中导入 QAT 类
from paddleslim.dygraph.quant import QAT
# 从 slim.slim_quant 模块中导入 PACT、quant_config 函数
from slim.slim_quant import PACT, quant_config
# 从 slim.slim_fpgm 模块中导入 prune_model 函数
from slim.slim_fpgm import prune_model
# 从 utils.py 文件中导入 load_model 函数

# 定义一个函数 _mkdir_if_not_exist，用于创建目录
def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    # 如果目录不存在，则创建目录
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            # 如果目录已存在，则忽略异常
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))

# 定义一个函数 save_model，用于保存模型
def save_model(model,
               optimizer,
               model_path,
               logger,
               is_best=False,
               prefix='ppocr',
               **kwargs):
    """
    save model to the target path
    """
    # 创建目录
    _mkdir_if_not_exist(model_path, logger)
    # 拼接模型保存路径
    model_prefix = os.path.join(model_path, prefix)
    # 保存模型参数
    paddle.save(model.state_dict(), model_prefix + '.pdparams')
    # 如果优化器是列表，则保存两个优化器的参数
    if type(optimizer) is list:
        paddle.save(optimizer[0].state_dict(), model_prefix + '.pdopt')
        paddle.save(optimizer[1].state_dict(), model_prefix + "_1" + '.pdopt')
    # 如果优化器不是列表，则保存单个优化器的参数
    else:
        paddle.save(optimizer.state_dict(), model_prefix + '.pdopt')

    # # save metric and config
    # with open(model_prefix + '.states', 'wb') as f:
    #     pickle.dump(kwargs, f, protocol=2)
    # 如果是最佳模型，则打印信息
    if is_best:
        logger.info('save best model is to {}'.format(model_prefix))
    # 如果条件不满足，即模型保存路径不存在，则记录日志信息
    else:
        logger.info("save model in {}".format(model_prefix))
# 根据配置信息进行自动混合精度训练，返回梯度缩放器对象
def amp_scaler(config):
    # 如果配置中包含 AMP 并且启用了自动混合精度
    if 'AMP' in config and config['AMP']['use_amp'] is True:
        # 设置与 AMP 相关的标志
        AMP_RELATED_FLAGS_SETTING = {
            'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
            'FLAGS_max_inplace_grad_add': 8,
        }
        # 设置 PaddlePaddle 的标志
        paddle.set_flags(AMP_RELATED_FLAGS_SETTING)
        # 获取梯度缩放比例
        scale_loss = config["AMP"].get("scale_loss", 1.0)
        # 获取是否使用动态损失缩放
        use_dynamic_loss_scaling = config["AMP"].get("use_dynamic_loss_scaling", False)
        # 创建梯度缩放器对象
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling)
        return scaler
    else:
        return None


# 设置随机种子
def set_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)


# 训练函数
def train(config, scaler=None):
    # 获取配置中的训练轮数和 topk 值
    EPOCH = config['epoch']
    topk = config['topk']

    # 获取配置中的批量大小和工作线程数，构建训练数据加载器
    batch_size = config['TRAIN']['batch_size']
    num_workers = config['TRAIN']['num_workers']
    train_loader = build_dataloader(
        'train', batch_size=batch_size, num_workers=num_workers)

    # 构建评估指标函数
    metric_func = create_metric

    # 构建模型
    model = build_model(config)

    # 构建优化器和学习率调度器
    optimizer, lr_scheduler = create_optimizer(
        config, parameter_list=model.parameters())

    # 加载模型
    pre_best_model_dict = load_model(config, model, optimizer)
    # 如果加载了预训练模型，则打印加载的指标信息
    if len(pre_best_model_dict) > 0:
        pre_str = 'The metric of loaded metric as follows {}'.format(', '.join(
            ['{}: {}'.format(k, v) for k, v in pre_best_model_dict.items()]))
        logger.info(pre_str)

    # 关于剪枝和量化
    if "quant_train" in config and config['quant_train'] is True:
        # 如果配置中包含量化训练并且启用了量化训练
        quanter = QAT(config=quant_config, act_preprocess=PACT)
        quanter.quantize(model)
    elif "prune_train" in config and config['prune_train'] is True:
        # 如果配置中包含剪枝训练并且启用了剪枝训练
        model = prune_model(model, [1, 3, 32, 32], 0.1)
    else:
        pass

    # 分布式训练
    # 设置模型为训练模式
    model.train()
    # 使用 paddle 提供的 DataParallel 函数将模型转换为多 GPU 训练模式
    model = paddle.DataParallel(model)
    # 构建损失函数
    loss_func = build_loss(config)
    
    # 获取训练数据集的长度
    data_num = len(train_loader)
    
    # 初始化最佳准确率字典
    best_acc = {}
# 定义训练函数，接受配置和缩放器作为参数
def train_distill(config, scaler=None):
    # 从配置中获取训练轮数和topk值
    EPOCH = config['epoch']
    topk = config['topk']

    # 从配置中获取批量大小和工作进程数，构建训练数据加载器
    batch_size = config['TRAIN']['batch_size']
    num_workers = config['TRAIN']['num_workers']
    train_loader = build_dataloader(
        'train', batch_size=batch_size, num_workers=num_workers)

    # 构建度量函数
    metric_func = create_metric

    # 根据配置构建模型
    model = build_model(config)

    # 如果配置中包含"quant_train"并且为True，则进行量化训练
    if "quant_train" in config and config['quant_train'] is True:
        quanter = QAT(config=quant_config, act_preprocess=PACT)
        quanter.quantize(model)
    # 如果配置中包含"prune_train"并且为True，则进行剪枝训练
    elif "prune_train" in config and config['prune_train'] is True:
        model = prune_model(model, [1, 3, 32, 32], 0.1)
    else:
        pass

    # 构建优化器和学习率调度器
    optimizer, lr_scheduler = create_optimizer(
        config, parameter_list=model.parameters())

    # 加载模型
    pre_best_model_dict = load_model(config, model, optimizer)
    if len(pre_best_model_dict) > 0:
        pre_str = 'The metric of loaded metric as follows {}'.format(', '.join(
            ['{}: {}'.format(k, v) for k, v in pre_best_model_dict.items()]))
        logger.info(pre_str)

    # 将模型设置为训练模式，并使用DataParallel进行多GPU训练
    model.train()
    model = paddle.DataParallel(model)

    # 构建损失函数
    loss_func_distill = LossDistill(model_name_list=['student', 'student1'])
    loss_func_dml = DMLLoss(model_name_pairs=['student', 'student1'])
    loss_func_js = KLJSLoss(mode='js')

    # 获取训练数据集的长度
    data_num = len(train_loader)

    # 初始化最佳准确率字典
    best_acc = {}

# 定义多优化器训练函数，接受配置和缩放器作为参数
def train_distill_multiopt(config, scaler=None):
    # 从配置中获取训练轮数和topk值
    EPOCH = config['epoch']
    topk = config['topk']

    # 从配置中获取批量大小和工作进程数，构建训练数据加载器
    batch_size = config['TRAIN']['batch_size']
    num_workers = config['TRAIN']['num_workers']
    train_loader = build_dataloader(
        'train', batch_size=batch_size, num_workers=num_workers)

    # 构建度量函数
    metric_func = create_metric

    # 根据配置构建模型
    model = build_model(config)

    # 构建优化器
    # 创建优化器和学习率调度器，针对 model.student 的参数列表
    optimizer, lr_scheduler = create_optimizer(
        config, parameter_list=model.student.parameters())
    # 创建另一个优化器和学习率调度器，针对 model.student1 的参数列表
    optimizer1, lr_scheduler1 = create_optimizer(
        config, parameter_list=model.student1.parameters())

    # 加载模型
    pre_best_model_dict = load_model(config, model, optimizer)
    # 如果加载的模型字典不为空，则打印加载的模型指标信息
    if len(pre_best_model_dict) > 0:
        pre_str = 'The metric of loaded metric as follows {}'.format(', '.join(
            ['{}: {}'.format(k, v) for k, v in pre_best_model_dict.items()]))
        logger.info(pre_str)

    # 如果配置中包含 quant_train 并且为 True，则进行量化训练
    if "quant_train" in config and config['quant_train'] is True:
        # 创建量化器对象，并进行量化模型
        quanter = QAT(config=quant_config, act_preprocess=PACT)
        quanter.quantize(model)
    # 如果配置中包含 prune_train 并且为 True，则进行剪枝训练
    elif "prune_train" in config and config['prune_train'] is True:
        # 对模型进行剪枝
        model = prune_model(model, [1, 3, 32, 32], 0.1)
    else:
        pass

    # 将模型设置为训练模式
    model.train()

    # 使用 paddle 提供的 DataParallel 方法将模型转换为多 GPU 训练模式
    model = paddle.DataParallel(model)

    # 构建损失函数
    loss_func_distill = LossDistill(model_name_list=['student', 'student1'])
    loss_func_dml = DMLLoss(model_name_pairs=['student', 'student1'])
    loss_func_js = KLJSLoss(mode='js')

    # 获取训练数据集的样本数量
    data_num = len(train_loader)
    # 初始化最佳准确率字典
    best_acc = {}
# 定义一个评估函数，接受配置和模型作为参数
def eval(config, model):
    # 从配置中获取验证时的批量大小和工作进程数
    batch_size = config['VALID']['batch_size']
    num_workers = config['VALID']['num_workers']
    # 构建验证数据加载器
    valid_loader = build_dataloader(
        'test', batch_size=batch_size, num_workers=num_workers)

    # 构建评估指标函数
    metric_func = create_metric

    # 初始化空列表用于存储模型输出和标签
    outs = []
    labels = []
    # 遍历验证数据加载器
    for idx, data in enumerate(valid_loader):
        # 获取图像批次和标签
        img_batch, label = data
        # 调整图像批次的维度顺序
        img_batch = paddle.transpose(img_batch, [0, 3, 1, 2])
        label = paddle.unsqueeze(label, -1)
        # 使用模型进行推理
        out = model(img_batch)

        # 将模型输出和标签添加到列表中
        outs.append(out)
        labels.append(label)

    # 将模型输出和标签拼接在一起
    outs = paddle.concat(outs, axis=0)
    labels = paddle.concat(labels, axis=0)
    # 计算评估指标
    acc = metric_func(outs, labels)

    # 构建输出字符串，包含评估指标的值
    strs = f"The metric are as follows: acc_topk1: {float(acc['top1'])}, acc_top5: {float(acc['top5'])}"
    # 记录日志信息
    logger.info(strs)
    # 返回评估指标
    return acc


if __name__ == "__main__":

    # 预处理配置和日志记录器
    config, logger = preprocess(is_train=False)

    # 初始化 AMP 缩放器
    scaler = amp_scaler(config)

    # 获取模型类型
    model_type = config['model_type']

    # 根据模型类型选择训练函数
    if model_type == "cls":
        train(config)
    elif model_type == "cls_distill":
        train_distill(config)
    elif model_type == "cls_distill_multiopt":
        train_distill_multiopt(config)
    else:
        # 如果模型类型不在指定范围内，则引发数值错误
        raise ValueError("model_type should be one of ['cls', 'cls_distill', 'cls_distill_multiopt']")
```