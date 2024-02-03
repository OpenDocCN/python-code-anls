# `.\PaddleOCR\ppocr\modeling\backbones\rec_mobilenet_v3.py`

```
# 导入 paddle.nn 模块
from paddle import nn

# 从 ppocr.modeling.backbones.det_mobilenet_v3 模块中导入 ResidualUnit, ConvBNLayer, make_divisible 函数
from ppocr.modeling.backbones.det_mobilenet_v3 import ResidualUnit, ConvBNLayer, make_divisible

# 定义 MobileNetV3 类，继承自 nn.Layer
class MobileNetV3(nn.Layer):
    # 定义前向传播函数
    def forward(self, x):
        # 对输入 x 进行卷积操作
        x = self.conv1(x)
        # 对 x 进行残差块操作
        x = self.blocks(x)
        # 再次对 x 进行卷积操作
        x = self.conv2(x)
        # 对 x 进行池化操作
        x = self.pool(x)
        # 返回处理后的 x
        return x
```