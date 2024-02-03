# `.\PaddleOCR\test_tipc\supplementary\optimizer.py`

```
# 导入 sys 模块，用于访问与 Python 解释器交互的变量和函数
import sys
# 导入 math 模块，提供数学函数
import math
# 从 paddle.optimizer.lr 模块中导入 LinearWarmup 类
from paddle.optimizer.lr import LinearWarmup
# 从 paddle.optimizer.lr 模块中导入 PiecewiseDecay 类
from paddle.optimizer.lr import PiecewiseDecay
# 从 paddle.optimizer.lr 模块中导入 CosineAnnealingDecay 类
from paddle.optimizer.lr import CosineAnnealingDecay
# 从 paddle.optimizer.lr 模块中导入 ExponentialDecay 类
from paddle.optimizer.lr import ExponentialDecay
# 导入 paddle 模块，提供深度学习框架
import paddle
# 从 paddle.regularizer 模块中导入 regularizer 类
import paddle.regularizer as regularizer
# 从 copy 模块中导入 deepcopy 函数，用于深拷贝对象
from copy import deepcopy

# 定义 Cosine 类，继承自 CosineAnnealingDecay 类
class Cosine(CosineAnnealingDecay):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
    """

    # 初始化方法，接收 lr、step_each_epoch、epochs 等参数
    def __init__(self, lr, step_each_epoch, epochs, **kwargs):
        # 调用父类的初始化方法
        super(Cosine, self).__init__(
            learning_rate=lr,
            T_max=step_each_epoch * epochs, )

        # 设置更新标志为 False
        self.update_specified = False

# 定义 Piecewise 类，继承自 PiecewiseDecay 类
class Piecewise(PiecewiseDecay):
    """
    Piecewise learning rate decay
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        decay_epochs(list): piecewise decay epochs
        gamma(float): decay factor
    """

    # 初始化方法，接收 lr、step_each_epoch、decay_epochs、gamma 等参数
    def __init__(self, lr, step_each_epoch, decay_epochs, gamma=0.1, **kwargs):
        # 计算分段衰减的边界
        boundaries = [step_each_epoch * e for e in decay_epochs]
        # 计算每个阶段的学习率值
        lr_values = [lr * (gamma**i) for i in range(len(boundaries) + 1)]
        # 调用父类的初始化方法
        super(Piecewise, self).__init__(boundaries=boundaries, values=lr_values)

        # 设置更新标志为 False
        self.update_specified = False

# 定义 CosineWarmup 类，继承自 LinearWarmup 类
class CosineWarmup(LinearWarmup):
    """
    Cosine learning rate decay with warmup
    [0, warmup_epoch): linear warmup
    [warmup_epoch, epochs): cosine decay
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        warmup_epoch(int): epoch num of warmup
    """
    # 初始化函数，接受学习率lr、每个epoch的步数step_each_epoch、总epoch数epochs、热身epoch数warmup_epoch等参数
    def __init__(self, lr, step_each_epoch, epochs, warmup_epoch=5, **kwargs):
        # 断言总epoch数大于热身epoch数，否则抛出异常
        assert epochs > warmup_epoch, "total epoch({}) should be larger than warmup_epoch({}) in CosineWarmup.".format(
            epochs, warmup_epoch)
        # 计算热身步数
        warmup_step = warmup_epoch * step_each_epoch
        # 初始学习率为0.0
        start_lr = 0.0
        # 结束学习率为lr
        end_lr = lr
        # 创建Cosine学习率调度对象lr_sch
        lr_sch = Cosine(lr, step_each_epoch, epochs - warmup_epoch)

        # 调用父类的初始化函数，传入学习率lr_sch、热身步数warmup_step、初始学习率start_lr、结束学习率end_lr等参数
        super(CosineWarmup, self).__init__(
            learning_rate=lr_sch,
            warmup_steps=warmup_step,
            start_lr=start_lr,
            end_lr=end_lr)

        # 初始化self.update_specified为False
        self.update_specified = False
# 定义一个继承自LinearWarmup的类ExponentialWarmup，实现指数学习率衰减和热身
class ExponentialWarmup(LinearWarmup):
    """
    Exponential learning rate decay with warmup
    [0, warmup_epoch): linear warmup
    [warmup_epoch, epochs): Exponential decay
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        decay_epochs(float): decay epochs
        decay_rate(float): decay rate
        warmup_epoch(int): epoch num of warmup
    """

    # 初始化方法，设置初始学习率、每个epoch的步数、衰减的epoch数、衰减率、热身的epoch数等参数
    def __init__(self,
                 lr,
                 step_each_epoch,
                 decay_epochs=2.4,
                 decay_rate=0.97,
                 warmup_epoch=5,
                 **kwargs):
        # 计算热身步数
        warmup_step = warmup_epoch * step_each_epoch
        start_lr = 0.0
        end_lr = lr
        # 创建指数衰减的学习率调度器
        lr_sch = ExponentialDecay(lr, decay_rate)

        # 调用父类的初始化方法
        super(ExponentialWarmup, self).__init__(
            learning_rate=lr_sch,
            warmup_steps=warmup_step,
            start_lr=start_lr,
            end_lr=end_lr)

        # 设置更新指定学习率调度器的方法
        self.update_specified = True
        self.update_start_step = warmup_step
        self.update_step_interval = int(decay_epochs * step_each_epoch)
        self.step_each_epoch = step_each_epoch


# 定义一个类LearningRateBuilder，用于构建学习率变量
class LearningRateBuilder():
    """
    Build learning rate variable
    https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/layers_cn.html
    Args:
        function(str): class name of learning rate
        params(dict): parameters used for init the class
    """

    # 初始化方法，设置学习率类名和初始化类的参数
    def __init__(self,
                 function='Linear',
                 params={'lr': 0.1,
                         'steps': 100,
                         'end_lr': 0.0}):
        self.function = function
        self.params = params

    # 调用方法，根据类名和参数创建学习率对象
    def __call__(self):
        mod = sys.modules[__name__]
        lr = getattr(mod, self.function)(**self.params)
        return lr


# 定义一个类L1Decay，实现L1权重衰减正则化，鼓励权重稀疏化
class L1Decay(object):
    """
    L1 Weight Decay Regularization, which encourages the weights to be sparse.
    Args:
        factor(float): regularization coeff. Default:0.0.
    """
    
    # L1Decay 类的构造函数，初始化 L1 正则化系数
    def __init__(self, factor=0.0):
        # 调用父类的构造函数
        super(L1Decay, self).__init__()
        # 设置 L1 正则化系数
        self.factor = factor

    # 调用 L1Decay 对象时执行的方法
    def __call__(self):
        # 创建 L1 正则化对象，使用指定的系数
        reg = regularizer.L1Decay(self.factor)
        # 返回 L1 正则化对象
        return reg
class L2Decay(object):
    """
    L2 Weight Decay Regularization, which encourages the weights to be sparse.
    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    def __init__(self, factor=0.0):
        # 初始化 L2 正则化对象，设置正则化系数
        super(L2Decay, self).__init__()
        self.factor = factor

    def __call__(self):
        # 创建 L2 正则化对象
        reg = regularizer.L2Decay(self.factor)
        return reg


class Momentum(object):
    """
    Simple Momentum optimizer with velocity state.
    Args:
        learning_rate (float|Variable) - The learning rate used to update parameters.
            Can be a float value or a Variable with one float value as data element.
        momentum (float) - Momentum factor.
        regularization (WeightDecayRegularizer, optional) - The strategy of regularization.
    """

    def __init__(self,
                 learning_rate,
                 momentum,
                 parameter_list=None,
                 regularization=None,
                 **args):
        # 初始化 Momentum 优化器，设置学习率、动量、参数列表和正则化策略
        super(Momentum, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.parameter_list = parameter_list
        self.regularization = regularization

    def __call__(self):
        # 创建 Momentum 优化器对象
        opt = paddle.optimizer.Momentum(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            parameters=self.parameter_list,
            weight_decay=self.regularization)
        return opt


class RMSProp(object):
    """
    Root Mean Squared Propagation (RMSProp) is an unpublished, adaptive learning rate method.
    Args:
        learning_rate (float|Variable) - The learning rate used to update parameters.
            Can be a float value or a Variable with one float value as data element.
        momentum (float) - Momentum factor.
        rho (float) - rho value in equation.
        epsilon (float) - avoid division by zero, default is 1e-6.
        regularization (WeightDecayRegularizer, optional) - The strategy of regularization.
    """
    """
    
    # 初始化 RMSProp 优化器对象
    def __init__(self,
                 learning_rate,
                 momentum,
                 rho=0.95,
                 epsilon=1e-6,
                 parameter_list=None,
                 regularization=None,
                 **args):
        # 调用父类的初始化方法
        super(RMSProp, self).__init__()
        # 设置学习率
        self.learning_rate = learning_rate
        # 设置动量
        self.momentum = momentum
        # 设置 rho 参数，默认为 0.95
        self.rho = rho
        # 设置 epsilon 参数，默认为 1e-6
        self.epsilon = epsilon
        # 设置参数列表，默认为 None
        self.parameter_list = parameter_list
        # 设置正则化方法，默认为 None
        self.regularization = regularization

    # 调用对象时返回 RMSProp 优化器对象
    def __call__(self):
        # 创建 RMSProp 优化器对象
        opt = paddle.optimizer.RMSProp(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            rho=self.rho,
            epsilon=self.epsilon,
            parameters=self.parameter_list,
            weight_decay=self.regularization)
        # 返回创建的优化器对象
        return opt
class OptimizerBuilder(object):
    """
    Build optimizer
    Args:
        function(str): optimizer name of learning rate
        params(dict): parameters used for init the class
        regularizer (dict): parameters used for create regularization
    """

    def __init__(self,
                 function='Momentum',
                 params={'momentum': 0.9},
                 regularizer=None):
        # 初始化优化器构建器，设置默认参数
        self.function = function
        self.params = params
        # 如果存在正则化参数，则创建正则化对象
        if regularizer is not None:
            mod = sys.modules[__name__]
            reg_func = regularizer['function'] + 'Decay'
            del regularizer['function']
            reg = getattr(mod, reg_func)(**regularizer)()
            self.params['regularization'] = reg

    def __call__(self, learning_rate, parameter_list=None):
        mod = sys.modules[__name__]
        opt = getattr(mod, self.function)
        # 调用优化器类并返回实例
        return opt(learning_rate=learning_rate,
                   parameter_list=parameter_list,
                   **self.params)()


def create_optimizer(config, parameter_list=None):
    """
    Create an optimizer using config, usually including
    learning rate and regularization.

    Args:
        config(dict):  such as
        {
            'LEARNING_RATE':
                {'function': 'Cosine',
                 'params': {'lr': 0.1}
                },
            'OPTIMIZER':
                {'function': 'Momentum',
                 'params':{'momentum': 0.9},
                 'regularizer':
                    {'function': 'L2', 'factor': 0.0001}
                }
        }

    Returns:
        an optimizer instance
    """
    # 创建学习率实例
    lr_config = config['LEARNING_RATE']
    lr_config['params'].update({
        'epochs': config['epoch'],
        'step_each_epoch':
        config['total_images'] // config['TRAIN']['batch_size'],
    })
    lr = LearningRateBuilder(**lr_config)()

    # 创建优化器实例
    # 深度复制配置文件中的优化器配置，避免对原配置的修改
    opt_config = deepcopy(config['OPTIMIZER'])

    # 使用优化器构建器和优化器配置创建优化器对象
    opt = OptimizerBuilder(**opt_config)
    # 返回优化器对象对学习率和参数列表进行优化后的结果，以及学习率
    return opt(lr, parameter_list), lr
# 创建多个优化器的函数
def create_multi_optimizer(config, parameter_list=None):
    """
    """
    # 创建学习率实例
    lr_config = config['LEARNING_RATE']
    lr_config['params'].update({
        'epochs': config['epoch'],
        'step_each_epoch':
        config['total_images'] // config['TRAIN']['batch_size'],
    })
    lr = LearningRateBuilder(**lr_config)()

    # 创建优化器实例
    opt_config = deepcopy.copy(config['OPTIMIZER'])
    opt = OptimizerBuilder(**opt_config)
    # 返回优化器和学习率实例
    return opt(lr, parameter_list), lr
```