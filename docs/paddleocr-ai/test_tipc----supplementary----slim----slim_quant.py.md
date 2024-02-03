# `.\PaddleOCR\test_tipc\supplementary\slim\slim_quant.py`

```
# 导入 paddle 模块
import paddle
# 导入 numpy 模块，并重命名为 np
import numpy as np
# 导入 os 模块
import os
# 导入 paddle.nn 模块，并重命名为 nn
import paddle.nn as nn
# 导入 paddleslim 模块

# 定义 PACT 类，继承自 paddle.nn.Layer
class PACT(paddle.nn.Layer):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super(PACT, self).__init__()
        # 定义 alpha 参数的属性
        alpha_attr = paddle.ParamAttr(
            # 参数名
            name=self.full_name() + ".pact",
            # 初始化器，常量值为 20
            initializer=paddle.nn.initializer.Constant(value=20),
            # 学习率为 1.0
            learning_rate=1.0,
            # 正则化器，L2 衰减为 2e-5
            regularizer=paddle.regularizer.L2Decay(2e-5))

        # 创建参数 alpha
        self.alpha = self.create_parameter(
            # 参数形状为 [1]
            shape=[1], 
            # 参数属性为 alpha_attr
            attr=alpha_attr, 
            # 参数类型为 float32
            dtype='float32')

    # 前向传播方法
    def forward(self, x):
        # 计算左侧输出
        out_left = paddle.nn.functional.relu(x - self.alpha)
        # 计算右侧输出
        out_right = paddle.nn.functional.relu(-self.alpha - x)
        # 更新 x
        x = x - out_left + out_right
        # 返回 x
        return x

# 定义量化配置字典 quant_config
quant_config = {
    # 权重预处理类型，默认为 None，不进行预处理
    'weight_preprocess_type': None,
    # 激活预处理类型，默认为 None，不进行预处理
    'activation_preprocess_type': None,
    # 权重量化类型，默认为 'channel_wise_abs_max'
    'weight_quantize_type': 'channel_wise_abs_max',
    # 激活量化类型，默认为 'moving_average_abs_max'
    'activation_quantize_type': 'moving_average_abs_max',
    # 权重量化位数，默认为 8
    'weight_bits': 8,
    # 激活量化位数，默认为 8
    'activation_bits': 8,
    # 量化后的数据类型，如 'uint8'、'int8'，默认为 'int8'
    'dtype': 'int8',
    # 'range_abs_max' 量化的窗口大小，默认为 10000
    'window_size': 10000,
    # 移动平均的衰减系数，默认为 0.9
    'moving_rate': 0.9,
    # 对于动态图量化，quantizable_layer_type 中的层将被量化
    'quantizable_layer_type': ['Conv2D', 'Linear'],
}
```