# `.\PaddleOCR\StyleText\arch\encoder.py`

```
# 导入 paddle 库中的 nn 模块
import paddle
import paddle.nn as nn

# 从 arch.base_module 模块中导入 SNConv, SNConvTranspose, ResBlock 类
from arch.base_module import SNConv, SNConvTranspose, ResBlock

# 定义 Encoder 类，继承自 nn.Layer 类
class Encoder(nn.Layer):
    # 定义 forward 方法
    def forward(self, x):
        # 创建一个空字典 out_dict
        out_dict = dict()
        # 对输入 x 进行二维填充
        x = self._pad2d(x)
        # 将填充后的 x 传入 in_conv 模块，并将结果存入 out_dict 中的 "in_conv" 键
        out_dict["in_conv"] = self._in_conv.forward(x)
        # 将 "in_conv" 的结果传入 down1 模块，并将结果存入 out_dict 中的 "down1" 键
        out_dict["down1"] = self._down1.forward(out_dict["in_conv"])
        # 将 "down1" 的结果传入 down2 模块，并将结果存入 out_dict 中的 "down2" 键
        out_dict["down2"] = self._down2.forward(out_dict["down1"])
        # 将 "down2" 的结果传入 down3 模块，并将结果存入 out_dict 中的 "down3" 键
        out_dict["down3"] = self._down3.forward(out_dict["down2"])
        # 将 "down3" 的结果传入 conv_blocks 模块，并将结果存入 out_dict 中的 "res_blocks" 键
        out_dict["res_blocks"] = self._conv_blocks.forward(out_dict["down3"])
        # 返回 out_dict 字典
        return out_dict

# 定义 EncoderUnet 类，继承自 nn.Layer 类
class EncoderUnet(nn.Layer):
    # 定义 forward 方法
    def forward(self, x):
        # 创建一个空字典 output_dict
        output_dict = dict()
        # 对输入 x 进行二维填充
        x = self._pad2d(x)
        # 将填充后的 x 传入 in_conv 模块，并将结果存入 output_dict 中的 "in_conv" 键
        output_dict['in_conv'] = self._in_conv.forward(x)
        # 将 "in_conv" 的结果传入 down1 模块，并将结果存入 output_dict 中的 "down1" 键
        output_dict['down1'] = self._down1.forward(output_dict['in_conv'])
        # 将 "down1" 的结果传入 down2 模块，并将结果存入 output_dict 中的 "down2" 键
        output_dict['down2'] = self._down2.forward(output_dict['down1'])
        # 将 "down2" 的结果传入 down3 模块，并将结果存入 output_dict 中的 "down3" 键
        output_dict['down3'] = self._down3.forward(output_dict['down2'])
        # 将 "down3" 的结果传入 down4 模块，并将结果存入 output_dict 中的 "down4" 键
        output_dict['down4'] = self._down4.forward(output_dict['down3'])
        # 将 "down4" 的结果传入 up1 模块，并将结果存入 output_dict 中的 "up1" 键
        output_dict['up1'] = self._up1.forward(output_dict['down4'])
        # 将 "down3" 和 "up1" 的结果在 axis=1 方向上拼接，传入 up2 模块，并将结果存入 output_dict 中的 "up2" 键
        output_dict['up2'] = self._up2.forward(
            paddle.concat(
                (output_dict['down3'], output_dict['up1']), axis=1))
        # 将 "down2" 和 "up2" 的结果在 axis=1 方向上拼接，存入 output_dict 中的 "concat" 键
        output_dict['concat'] = paddle.concat(
            (output_dict['down2'], output_dict['up2']), axis=1)
        # 返回 output_dict 字典
        return output_dict
```