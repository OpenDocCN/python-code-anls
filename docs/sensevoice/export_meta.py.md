# `.\SenseVoiceSmall-src\export_meta.py`

```
# 指定解释器路径和编码格式
#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# 版权所有 FunASR (https://github.com/alibaba-damo-academy/FunASR)，保留所有权利
#  MIT 许可证 (https://opensource.org/licenses/MIT)

# 导入所需模块
import types
import torch
from funasr.utils.torch_function import sequence_mask

# 定义导出重建模型的函数
def export_rebuild_model(model, **kwargs):
    # 从关键字参数获取设备信息并设置模型设备
    model.device = kwargs.get("device")
    # 创建填充掩码，指定最大序列长度
    model.make_pad_mask = sequence_mask(kwargs["max_seq_len"], flip=False)
    # 将自定义前向传播函数绑定到模型上
    model.forward = types.MethodType(export_forward, model)
    # 将导出虚拟输入函数绑定到模型上
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    # 将导出输入名称函数绑定到模型上
    model.export_input_names = types.MethodType(export_input_names, model)
    # 将导出输出名称函数绑定到模型上
    model.export_output_names = types.MethodType(export_output_names, model)
    # 将导出动态轴函数绑定到模型上
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)
    # 将导出模型名称函数绑定到模型上
    model.export_name = types.MethodType(export_name, model)
    # 返回修改后的模型
    return model

# 定义自定义前向传播函数
def export_forward(
    self,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
    language: torch.Tensor,
    textnorm: torch.Tensor,
    **kwargs,
):
    # 将输入的 speech 张量移动到 CUDA 设备上（已注释）
    # speech = speech.to(device="cuda")
    # 将输入的 speech_lengths 张量移动到 CUDA 设备上（已注释）
    # speech_lengths = speech_lengths.to(device="cuda")
    # 嵌入语言张量并增加一个维度
    language_query = self.embed(language.to(speech.device)).unsqueeze(1)
    # 嵌入文本归一化张量并增加一个维度
    textnorm_query = self.embed(textnorm.to(speech.device)).unsqueeze(1)
    # 打印 textnorm_query 和 speech 的形状
    print(textnorm_query.shape, speech.shape)
    # 将 textnorm_query 与 speech 在维度 1 上拼接
    speech = torch.cat((textnorm_query, speech), dim=1)
    # 更新 speech_lengths，增加 1
    speech_lengths += 1
    
    # 嵌入事件情感查询并重复以匹配 batch size
    event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
        speech.size(0), 1, 1
    )
    # 将语言查询与事件情感查询在维度 1 上拼接
    input_query = torch.cat((language_query, event_emo_query), dim=1)
    # 将 input_query 与 speech 在维度 1 上拼接
    speech = torch.cat((input_query, speech), dim=1)
    # 更新 speech_lengths，增加 3
    speech_lengths += 3
    
    # 将 speech 和 speech_lengths 传入编码器，获取输出和长度
    encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
    # 如果 encoder_out 是元组，则取第一个元素
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]

    # 计算 CTC logits
    ctc_logits = self.ctc.ctc_lo(encoder_out)
    
    # 返回 CTC logits 和编码器输出的长度
    return ctc_logits, encoder_out_lens

# 定义导出虚拟输入的函数
def export_dummy_inputs(self):
    # 生成随机的 speech 张量
    speech = torch.randn(2, 30, 560)
    # 定义对应的 speech_lengths 张量
    speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
    # 定义语言张量
    language = torch.tensor([0, 0], dtype=torch.int32)
    # 定义文本归一化张量
    textnorm = torch.tensor([15, 15], dtype=torch.int32)
    # 返回所有虚拟输入张量
    return (speech, speech_lengths, language, textnorm)

# 定义导出输入名称的函数
def export_input_names(self):
    # 返回输入名称列表
    return ["speech", "speech_lengths", "language", "textnorm"]

# 定义导出输出名称的函数
def export_output_names(self):
    # 返回输出名称列表
    return ["ctc_logits", "encoder_out_lens"]

# 定义导出动态轴的函数
def export_dynamic_axes(self):
    # 返回动态轴的字典映射
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_lengths": {0: "batch_size"},
        "language": {0: "batch_size"},
        "textnorm": {0: "batch_size"},
        "ctc_logits": {0: "batch_size", 1: "logits_length"},
        "encoder_out_lens":  {0: "batch_size"},
    }

# 定义导出模型名称的函数
def export_name(self):
    # 返回模型名称
    return "model.onnx"
```