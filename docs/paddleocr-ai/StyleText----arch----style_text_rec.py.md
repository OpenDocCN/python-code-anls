# `.\PaddleOCR\StyleText\arch\style_text_rec.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证以“原样”分发，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和
# 限制
# 导入 paddle 库
import paddle
# 导入 paddle.nn 模块
import paddle.nn as nn

# 从 arch.base_module 模块导入 MiddleNet 和 ResBlock 类
from arch.base_module import MiddleNet, ResBlock
# 从 arch.encoder 模块导入 Encoder 类
from arch.encoder import Encoder
# 从 arch.decoder 模块导入 Decoder, DecoderUnet, SingleDecoder 类
from arch.decoder import Decoder, DecoderUnet, SingleDecoder
# 从 utils.load_params 模块导入 load_dygraph_pretrain 函数
from utils.load_params import load_dygraph_pretrain
# 从 utils.logging 模块导入 get_logger 函数
from utils.logging import get_logger

# 定义 StyleTextRec 类，继承自 nn.Layer 类
class StyleTextRec(nn.Layer):
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super(StyleTextRec, self).__init__()
        # 获取日志记录器
        self.logger = get_logger()
        # 创建文本生成器对象
        self.text_generator = TextGenerator(config["Predictor"]["text_generator"])
        # 创建带蒙版的背景生成器对象
        self.bg_generator = BgGeneratorWithMask(config["Predictor"]["bg_generator"])
        # 创建简单融合生成器对象
        self.fusion_generator = FusionGeneratorSimple(config["Predictor"]["fusion_generator"])
        # 获取背景生成器的预训练路径
        bg_generator_pretrain = config["Predictor"]["bg_generator"]["pretrain"]
        # 获取文本生成器的预训练路径
        text_generator_pretrain = config["Predictor"]["text_generator"]["pretrain"]
        # 获取融合生成器的预训练路径
        fusion_generator_pretrain = config["Predictor"]["fusion_generator"]["pretrain"]
        # 加载背景生成器的预训练参数
        load_dygraph_pretrain(
            self.bg_generator,
            self.logger,
            path=bg_generator_pretrain,
            load_static_weights=False)
        # 加载文本生成器的预训练参数
        load_dygraph_pretrain(
            self.text_generator,
            self.logger,
            path=text_generator_pretrain,
            load_static_weights=False)
        # 加载融合生成器的预训练参数
        load_dygraph_pretrain(
            self.fusion_generator,
            self.logger,
            path=fusion_generator_pretrain,
            load_static_weights=False)
    # 前向传播函数，接收风格输入和文本输入，返回生成的假文本、假背景和融合结果
    def forward(self, style_input, text_input):
        # 调用文本生成器的前向传播函数，生成假文本和假关键点
        text_gen_output = self.text_generator.forward(style_input, text_input)
        fake_text = text_gen_output["fake_text"]
        fake_sk = text_gen_output["fake_sk"]
        # 调用背景生成器的前向传播函数，生成编码特征和解码特征
        bg_gen_output = self.bg_generator.forward(style_input)
        bg_encode_feature = bg_gen_output["bg_encode_feature"]
        bg_decode_feature1 = bg_gen_output["bg_decode_feature1"]
        bg_decode_feature2 = bg_gen_output["bg_decode_feature2"]
        fake_bg = bg_gen_output["fake_bg"]

        # 调用融合生成器的前向传播函数，生成融合结果
        fusion_gen_output = self.fusion_generator.forward(fake_text, fake_bg)
        fake_fusion = fusion_gen_output["fake_fusion"]
        # 返回生成的结果字典
        return {
            "fake_fusion": fake_fusion,
            "fake_text": fake_text,
            "fake_sk": fake_sk,
            "fake_bg": fake_bg,
        }
class TextGenerator(nn.Layer):
    # 定义文本生成器类
    def forward(self, style_input, text_input):
        # 前向传播函数，接收风格输入和文本输入
        style_feature = self.encoder_style.forward(style_input)["res_blocks"]
        # 使用编码器对风格输入进行编码，获取风格特征
        text_feature = self.encoder_text.forward(text_input)["res_blocks"]
        # 使用编码器对文本输入进行编码，获取文本特征
        fake_c_temp = self.decoder_text.forward([text_feature,
                                                 style_feature])["out_conv"]
        # 使用文本解码器生成临时输出
        fake_sk = self.decoder_sk.forward([text_feature,
                                           style_feature])["out_conv"]
        # 使用风格解码器生成风格输出
        fake_text = self.middle(paddle.concat((fake_c_temp, fake_sk), axis=1))
        # 将临时输出和风格输出拼接后通过中间层处理
        return {"fake_sk": fake_sk, "fake_text": fake_text}
        # 返回生成的风格输出和文本输出的字典


class BgGeneratorWithMask(nn.Layer):
    # 定义带蒙版的背景生成器类
    def forward(self, style_input):
        # 前向传播函数，接收风格输入
        encode_bg_output = self.encoder_bg(style_input)
        # 使用背景编码器对风格输入进行编码
        decode_bg_output = self.decoder_bg(encode_bg_output["res_blocks"],
                                           encode_bg_output["down2"],
                                           encode_bg_output["down1"])
        # 使用背景解码器对编码后的特征进行解码

        fake_c_temp = decode_bg_output["out_conv"]
        # 获取解码后的临时输出
        fake_bg_mask = self.decoder_mask.forward(encode_bg_output[
            "res_blocks"])["out_conv"]
        # 使用蒙版解码器生成背景蒙版
        fake_bg = self.middle(
            paddle.concat(
                (fake_c_temp, fake_bg_mask), axis=1))
        # 将临时输出和背景蒙版拼接后通过中间层处理
        return {
            "bg_encode_feature": encode_bg_output["res_blocks"],
            "bg_decode_feature1": decode_bg_output["up1"],
            "bg_decode_feature2": decode_bg_output["up2"],
            "fake_bg": fake_bg,
            "fake_bg_mask": fake_bg_mask,
        }
        # 返回生成的背景编码特征、解码特征1、解码特征2、背景输出和背景蒙版


class FusionGeneratorSimple(nn.Layer):
    # 定义简单融合生成器类
    # FusionGeneratorSimple 类的初始化方法，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super(FusionGeneratorSimple, self).__init__()
        # 从配置参数中获取模块名称、编码维度、规范化层、卷积块丢弃率、卷积块膨胀率等信息
        name = config["module_name"]
        encode_dim = config["encode_dim"]
        norm_layer = config["norm_layer"]
        conv_block_dropout = config["conv_block_dropout"]
        conv_block_dilation = config["conv_block_dilation"]
        # 根据规范化层类型判断是否使用偏置
        if norm_layer == "InstanceNorm2D":
            use_bias = True
        else:
            use_bias = False

        # 创建一个卷积层，输入通道数为6，输出通道数为编码维度，卷积核大小为3x3
        self._conv = nn.Conv2D(
            in_channels=6,
            out_channels=encode_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            weight_attr=paddle.ParamAttr(name=name + "_conv_weights"),
            bias_attr=False)

        # 创建一个残差块，用于特征提取和转换
        self._res_block = ResBlock(
            name="{}_conv_block".format(name),
            channels=encode_dim,
            norm_layer=norm_layer,
            use_dropout=conv_block_dropout,
            use_dilation=conv_block_dilation,
            use_bias=use_bias)

        # 创建一个降维卷积层，将特征图维度降至3
        self._reduce_conv = nn.Conv2D(
            in_channels=encode_dim,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            weight_attr=paddle.ParamAttr(name=name + "_reduce_conv_weights"),
            bias_attr=False)

    # FusionGeneratorSimple 类的前向传播方法，接受两个输入 fake_text 和 fake_bg
    def forward(self, fake_text, fake_bg):
        # 将两个输入在通道维度上拼接
        fake_concat = paddle.concat((fake_text, fake_bg), axis=1)
        # 将拼接后的输入通过卷积层处理
        fake_concat_tmp = self._conv(fake_concat)
        # 将卷积层的输出通过残差块处理
        output_res = self._res_block(fake_concat_tmp)
        # 将残差块的输出通过降维卷积层处理，得到融合后的结果
        fake_fusion = self._reduce_conv(output_res)
        # 返回融合后的结果
        return {"fake_fusion": fake_fusion}
```