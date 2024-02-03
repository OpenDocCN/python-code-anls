# `.\PaddleOCR\StyleText\utils\load_params.py`

```py
# 版权声明和许可信息
# 2020年PaddlePaddle作者保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证“按原样”分发，
# 没有任何明示或暗示的保证或条件。
# 有关特定语言的权限和
# 限制，请参阅许可证
import os
import paddle

# 导出的函数列表
__all__ = ['load_dygraph_pretrain']

# 加载动态图预训练模型
def load_dygraph_pretrain(model, logger, path=None, load_static_weights=False):
    # 如果指定路径的参数文件不存在，则抛出异常
    if not os.path.exists(path + '.pdparams'):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    # 加载参数状态字典
    param_state_dict = paddle.load(path + '.pdparams')
    # 设置模型状态字典
    model.set_state_dict(param_state_dict)
    # 记录日志，显示从指定路径加载预训练模型
    logger.info("load pretrained model from {}".format(path))
    return
```