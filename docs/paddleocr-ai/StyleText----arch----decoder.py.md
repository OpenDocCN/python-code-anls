# `.\PaddleOCR\StyleText\arch\decoder.py`

```
# 版权声明
# 2020年PaddlePaddle作者保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证分发在“按原样”基础上，
# 没有任何明示或暗示的保证或条件。
# 有关特定语言的权限和限制，请参阅许可证。
import paddle
import paddle.nn as nn

from arch.base_module import SNConv, SNConvTranspose, ResBlock

# 定义Decoder类，继承自nn.Layer
class Decoder(nn.Layer):
    # 定义前向传播函数
    def forward(self, x):
        # 如果输入x是列表或元组，则沿着axis=1进行拼接
        if isinstance(x, (list, tuple)):
            x = paddle.concat(x, axis=1)
        # 初始化输出字典
        output_dict = dict()
        # 将输入x传入conv_blocks模块，并保存到output_dict中
        output_dict["conv_blocks"] = self.conv_blocks.forward(x)
        # 将output_dict["conv_blocks"]传入_up1模块，并保存到output_dict中
        output_dict["up1"] = self._up1.forward(output_dict["conv_blocks"])
        # 将output_dict["up1"]传入_up2模块，并保存到output_dict中
        output_dict["up2"] = self._up2.forward(output_dict["up1"])
        # 将output_dict["up2"]传入_up3模块，并保存到output_dict中
        output_dict["up3"] = self._up3.forward(output_dict["up2"])
        # 将output_dict["up3"]传入_pad2d模块，并保存到output_dict中
        output_dict["pad2d"] = self._pad2d.forward(output_dict["up3"])
        # 将output_dict["pad2d"]传入_out_conv模块，并保存到output_dict中
        output_dict["out_conv"] = self._out_conv.forward(output_dict["pad2d"])
        # 返回输出字典
        return output_dict

# 定义DecoderUnet类，继承自nn.Layer
class DecoderUnet(nn.Layer):
    # 定义一个前向传播函数，接受输入 x, y, feature2, feature1
    def forward(self, x, y, feature2, feature1):
        # 初始化一个空字典用于存储输出结果
        output_dict = dict()
        # 对输入 x 和 y 进行拼接，然后通过卷积块处理
        output_dict["conv_blocks"] = self._conv_blocks(
            paddle.concat(
                (x, y), axis=1))
        # 将卷积块的输出传递给上采样模块 up1 进行处理
        output_dict["up1"] = self._up1.forward(output_dict["conv_blocks"])
        # 将 up1 的输出与 feature2 进行拼接，然后通过上采样模块 up2 处理
        output_dict["up2"] = self._up2.forward(
            paddle.concat(
                (output_dict["up1"], feature2), axis=1))
        # 将 up2 的输出与 feature1 进行拼接，然后通过上采样模块 up3 处理
        output_dict["up3"] = self._up3.forward(
            paddle.concat(
                (output_dict["up2"], feature1), axis=1))
        # 将 up3 的输出传递给填充模块 pad2d 进行处理
        output_dict["pad2d"] = self._pad2d.forward(output_dict["up3"])
        # 将 pad2d 的输出传递给输出卷积模块 out_conv 进行处理
        output_dict["out_conv"] = self._out_conv.forward(output_dict["pad2d"])
        # 返回包含所有输出结果的字典
        return output_dict
# 定义一个单一解码器类，继承自 nn.Layer
class SingleDecoder(nn.Layer):
    # 定义前向传播函数，接受输入 x、feature2 和 feature1
    def forward(self, x, feature2, feature1):
        # 创建一个空字典用于存储输出结果
        output_dict = dict()
        # 将输入 x 经过卷积块处理后的结果存入字典
        output_dict["conv_blocks"] = self._conv_blocks.forward(x)
        # 将上一步结果经过上采样1处理后的结果存入字典
        output_dict["up1"] = self._up1.forward(output_dict["conv_blocks"])
        # 将上一步结果与 feature2 拼接后经过上采样2处理后的结果存入字典
        output_dict["up2"] = self._up2.forward(
            paddle.concat(
                (output_dict["up1"], feature2), axis=1))
        # 将上一步结果与 feature1 拼接后经过上采样3处理后的结果存入字典
        output_dict["up3"] = self._up3.forward(
            paddle.concat(
                (output_dict["up2"], feature1), axis=1))
        # 将上一步结果经过填充处理后的结果存入字典
        output_dict["pad2d"] = self._pad2d.forward(output_dict["up3"])
        # 将上一步结果经过输出卷积处理后的结果存入字典
        output_dict["out_conv"] = self._out_conv.forward(output_dict["pad2d"])
        # 返回存储了所有处理结果的字典
        return output_dict
```