# `.\SenseVoiceSmall-src\export.py`

```
#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# 版权声明，FunASR 所有权利保留，使用 MIT 许可证
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# 导入操作系统模块
import os
# 导入 PyTorch 库
import torch
# 从自定义模型模块导入 SenseVoiceSmall 类
from model import SenseVoiceSmall
# 从实用工具模块导入 export_utils
from utils import export_utils
# 从模型二进制模块导入 SenseVoiceSmallONNX
from utils.model_bin import SenseVoiceSmallONNX
# 从后处理模块导入 rich_transcription_postprocess 函数
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 设置量化标志为假
quantize = False

# 指定模型目录
model_dir = "iic/SenseVoiceSmall"
# 从预训练模型加载模型和参数
model, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")

# 将模型导出为 ONNX 格式，且不进行量化
rebuilt_model = model.export(type="onnx", quantize=False)
# 获取输出目录，默认为初始化参数所在的目录
model_path = kwargs.get("output_dir", os.path.dirname(kwargs.get("init_param")))

# 构造最终的模型文件路径
model_file = os.path.join(model_path, "model.onnx")
# 如果量化标志为真，构造量化后的模型文件路径
if quantize:
    model_file = os.path.join(model_path, "model_quant.onnx")

# 导出模型
# 如果模型文件不存在，则进行导出
if not os.path.exists(model_file):
    # 在不计算梯度的情况下进行操作
    with torch.no_grad():
        # 从参数中删除模型
        del kwargs['model']
        # 调用导出工具进行模型导出
        export_dir = export_utils.export(model=rebuilt_model, **kwargs)
        # 打印导出成功的消息
        print("Export model onnx to {}".format(model_file))
        
# 导出模型初始化
# 从模型路径创建 SenseVoiceSmallONNX 实例
model_bin = SenseVoiceSmallONNX(model_path)

# 构建分词器
try:
    # 从分词器模块导入 SentencepiecesTokenizer
    from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
    # 初始化分词器，指定 BPE 模型文件路径
    tokenizer = SentencepiecesTokenizer(bpemodel=os.path.join(model_path, "chn_jpn_yue_eng_ko_spectok.bpe.model"))
# 捕获异常，如果导入失败则将 tokenizer 设置为 None
except:
    tokenizer = None

# 进行推理
# 指定音频文件路径
wav_or_scp = "/Users/shixian/Downloads/asr_example_hotword.wav"
# 指定语言列表
language_list = [0]
# 指定文本规范化列表
textnorm_list = [15]
# 调用模型进行推理，返回结果
res = model_bin(wav_or_scp, language_list, textnorm_list, tokenizer=tokenizer)
# 打印后处理后的结果
print([rich_transcription_postprocess(i) for i in res])
```