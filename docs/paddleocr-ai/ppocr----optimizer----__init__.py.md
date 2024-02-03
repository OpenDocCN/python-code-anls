# `.\PaddleOCR\ppocr\optimizer\__init__.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
import paddle

# 定义模块中公开的函数
__all__ = ['build_optimizer']

# 构建学习率调度器
def build_lr_scheduler(lr_config, epochs, step_each_epoch):
    # 导入 learning_rate 模块
    from . import learning_rate
    # 更新 lr_config 字典，添加 epochs 和 step_each_epoch 键值对
    lr_config.update({'epochs': epochs, 'step_each_epoch': step_each_epoch})
    # 获取 lr_config 中的 name 键值对，如果不存在则默认为 'Const'
    lr_name = lr_config.pop('name', 'Const')
    # 根据 lr_name 获取对应的学习率调度器对象，并调用生成学习率调度器
    lr = getattr(learning_rate, lr_name)(**lr_config)()
    return lr

# 构建优化器
def build_optimizer(config, epochs, step_each_epoch, model):
    # 导入 regularizer 和 optimizer 模块
    from . import regularizer, optimizer
    # 深拷贝 config 字典
    config = copy.deepcopy(config)
    
    # 步骤1：构建学习率调度器
    lr = build_lr_scheduler(config.pop('lr'), epochs, step_each_epoch)

    # 步骤2：构建正则化器
    if 'regularizer' in config and config['regularizer'] is not None:
        reg_config = config.pop('regularizer')
        reg_name = reg_config.pop('name')
        # 如果 regularizer 模块中没有对应的正则化器，则添加 'Decay' 后缀
        if not hasattr(regularizer, reg_name):
            reg_name += 'Decay'
        # 根据 reg_name 获取对应的正则化器对象，并调用生成正则化器
        reg = getattr(regularizer, reg_name)(**reg_config)()
    elif 'weight_decay' in config:
        # 如果 config 中有 'weight_decay' 键，则将其作为正则化器
        reg = config.pop('weight_decay')
    else:
        reg = None

    # 步骤3：构建优化器
    optim_name = config.pop('name')
    if 'clip_norm' in config:
        # 如果 config 中有 'clip_norm' 键，则获取 clip_norm 值
        clip_norm = config.pop('clip_norm')
        # 创建梯度裁剪对象，根据 clip_norm 进行梯度裁剪
        grad_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    # 如果配置中包含'clip_norm_global'，则将其从配置中弹出并赋值给clip_norm
    elif 'clip_norm_global' in config:
        clip_norm = config.pop('clip_norm_global')
        # 使用全局范数对梯度进行裁剪
        grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    # 如果配置中不包含'clip_norm_global'，则将grad_clip设为None
    else:
        grad_clip = None
    # 根据配置中的优化器名称实例化优化器对象，设置学习率、正则化参数、梯度裁剪参数等
    optim = getattr(optimizer, optim_name)(learning_rate=lr,
                                           weight_decay=reg,
                                           grad_clip=grad_clip,
                                           **config)
    # 返回优化器对象和学习率
    return optim(model), lr
```