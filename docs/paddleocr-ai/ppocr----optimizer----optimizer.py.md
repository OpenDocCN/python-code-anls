# `.\PaddleOCR\ppocr\optimizer\optimizer.py`

```py
# 版权声明
#
# 版权所有 (c) 2020 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请参阅许可证以获取特定语言下的权限和限制。

# 导入必要的模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from paddle import optimizer as optim

# 定义 Momentum 类
class Momentum(object):
    """
    Simple Momentum optimizer with velocity state.
    Args:
        learning_rate (float|Variable) - The learning rate used to update parameters.
            Can be a float value or a Variable with one float value as data element.
        momentum (float) - Momentum factor.
        regularization (WeightDecayRegularizer, optional) - The strategy of regularization.
    """

    # 初始化 Momentum 类
    def __init__(self,
                 learning_rate,
                 momentum,
                 weight_decay=None,
                 grad_clip=None,
                 **args):
        super(Momentum, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    # 定义调用方法
    def __call__(self, model):
        # 获取可训练参数
        train_params = [
            param for param in model.parameters() if param.trainable is True
        ]
        # 创建 Momentum 优化器
        opt = optim.Momentum(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            parameters=train_params)
        return opt

# 定义 Adam 类
class Adam(object):
    # 初始化优化器对象，设置默认参数值
    def __init__(self,
                 learning_rate=0.001,  # 学习率，默认为0.001
                 beta1=0.9,  # Adam优化器参数beta1，默认为0.9
                 beta2=0.999,  # Adam优化器参数beta2，默认为0.999
                 epsilon=1e-08,  # Adam优化器参数epsilon，默认为1e-08
                 parameter_list=None,  # 参数列表，默认为None
                 weight_decay=None,  # 权重衰减，默认为None
                 grad_clip=None,  # 梯度裁剪，默认为None
                 name=None,  # 优化器名称，默认为None
                 lazy_mode=False,  # 懒惰模式，默认为False
                 **kwargs):  # 其他关键字参数
        self.learning_rate = learning_rate  # 设置学习率
        self.beta1 = beta1  # 设置beta1
        self.beta2 = beta2  # 设置beta2
        self.epsilon = epsilon  # 设置epsilon
        self.parameter_list = parameter_list  # 设置参数列表
        self.learning_rate = learning_rate  # 重新设置学习率（此处可能有误，应该是设置权重衰减）
        self.weight_decay = weight_decay  # 设置权重衰减
        self.grad_clip = grad_clip  # 设置梯度裁剪
        self.name = name  # 设置优化器名称
        self.lazy_mode = lazy_mode  # 设置懒惰模式
        self.group_lr = kwargs.get('group_lr', False)  # 获取关键字参数中的group_lr，默认为False
        self.training_step = kwargs.get('training_step', None)  # 获取关键字参数中的training_step，默认为None
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

    # 初始化 RMSProp 类
    def __init__(self,
                 learning_rate,
                 momentum=0.0,
                 rho=0.95,
                 epsilon=1e-6,
                 weight_decay=None,
                 grad_clip=None,
                 **args):
        # 调用父类的初始化方法
        super(RMSProp, self).__init__()
        # 设置学习率
        self.learning_rate = learning_rate
        # 设置动量因子
        self.momentum = momentum
        # 设置 rho 值
        self.rho = rho
        # 设置 epsilon 值，避免除零
        self.epsilon = epsilon
        # 设置权重衰减策略
        self.weight_decay = weight_decay
        # 设置梯度裁剪策略
        self.grad_clip = grad_clip

    # 调用 RMSProp 类时执行的方法
    def __call__(self, model):
        # 获取模型中可训练的参数
        train_params = [
            param for param in model.parameters() if param.trainable is True
        ]
        # 创建 RMSProp 优化器对象
        opt = optim.RMSProp(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            rho=self.rho,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            parameters=train_params)
        # 返回优化器对象
        return opt


class Adadelta(object):
    # 初始化函数，设置默认学习率、epsilon、rho等参数，并接收额外的关键字参数
    def __init__(self,
                 learning_rate=0.001,
                 epsilon=1e-08,
                 rho=0.95,
                 parameter_list=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None,
                 **kwargs):
        # 设置学习率
        self.learning_rate = learning_rate
        # 设置epsilon
        self.epsilon = epsilon
        # 设置rho
        self.rho = rho
        # 设置参数列表
        self.parameter_list = parameter_list
        # 设置权重衰减
        self.weight_decay = weight_decay
        # 设置梯度裁剪
        self.grad_clip = grad_clip
        # 设置名称
        self.name = name

    # 调用函数，传入模型，获取可训练参数列表，创建Adadelta优化器并返回
    def __call__(self, model):
        # 获取模型中可训练的参数列表
        train_params = [
            param for param in model.parameters() if param.trainable is True
        ]
        # 创建Adadelta优化器，设置学习率、epsilon、rho、权重衰减、梯度裁剪、名称和参数列表
        opt = optim.Adadelta(
            learning_rate=self.learning_rate,
            epsilon=self.epsilon,
            rho=self.rho,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            name=self.name,
            parameters=train_params)
        # 返回创建的优化器
        return opt
# 定义 AdamW 类
class AdamW(object):
    # 初始化函数，设置 AdamW 类的参数
    def __init__(self,
                 learning_rate=0.001,  # 学习率，默认为 0.001
                 beta1=0.9,  # Adam 算法的参数 beta1，默认为 0.9
                 beta2=0.999,  # Adam 算法的参数 beta2，默认为 0.999
                 epsilon=1e-8,  # Adam 算法的参数 epsilon，默认为 1e-8
                 weight_decay=0.01,  # 权重衰减，默认为 0.01
                 multi_precision=False,  # 是否使用多精度，默认为 False
                 grad_clip=None,  # 梯度裁剪，默认为 None
                 no_weight_decay_name=None,  # 不进行权重衰减的参数名，默认为 None
                 one_dim_param_no_weight_decay=False,  # 是否对一维参数不进行权重衰减，默认为 False
                 name=None,  # 名称，默认为 None
                 lazy_mode=False,  # 是否启用懒惰模式，默认为 False
                 **args):  # 其他参数
        super().__init__()  # 调用父类的初始化函数
        # 设置 AdamW 类的各个参数
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.grad_clip = grad_clip
        self.weight_decay = 0.01 if weight_decay is None else weight_decay
        self.grad_clip = grad_clip
        self.name = name
        self.lazy_mode = lazy_mode
        self.multi_precision = multi_precision
        # 将不进行权重衰减的参数名转换为列表，如果没有指定则为空列表
        self.no_weight_decay_name_list = no_weight_decay_name.split() if no_weight_decay_name else []
        self.one_dim_param_no_weight_decay = one_dim_param_no_weight_decay
    # 定义一个方法，用于为给定模型创建优化器对象
    def __call__(self, model):
        # 获取模型中可训练的参数列表
        parameters = [
            param for param in model.parameters() if param.trainable is True
        ]

        # 根据指定的名称列表，获取不需要进行权重衰减的参数名列表
        self.no_weight_decay_param_name_list = [
            p.name for n, p in model.named_parameters()
            if any(nd in n for nd in self.no_weight_decay_name_list)
        ]

        # 如果设置了需要单维度参数不进行权重衰减，则将这些参数名添加到不进行权重衰减的参数名列表中
        if self.one_dim_param_no_weight_decay:
            self.no_weight_decay_param_name_list += [
                p.name for n, p in model.named_parameters() if len(p.shape) == 1
            ]

        # 创建 AdamW 优化器对象，设置学习率、beta 值、参数、权重衰减等参数
        opt = optim.AdamW(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            parameters=parameters,
            weight_decay=self.weight_decay,
            multi_precision=self.multi_precision,
            grad_clip=self.grad_clip,
            name=self.name,
            lazy_mode=self.lazy_mode,
            apply_decay_param_fun=self._apply_decay_param_fun)
        # 返回创建的优化器对象
        return opt

    # 定义一个方法，用于判断参数是否需要进行权重衰减
    def _apply_decay_param_fun(self, name):
        # 返回参数名是否在不进行权重衰减的参数名列表中
        return name not in self.no_weight_decay_param_name_list
```