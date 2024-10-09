# `.\SenseVoiceSmall-src\demo1.py`

```
#!/usr/bin/env python3  # 指定脚本使用的解释器为 Python 3
# -*- encoding: utf-8 -*-  # 设置源文件的编码为 UTF-8
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.  # 版权声明
#  MIT License  (https://opensource.org/licenses/MIT)  # 许可证信息

from funasr import AutoModel  # 从 funasr 库导入 AutoModel 类
from funasr.utils.postprocess_utils import rich_transcription_postprocess  # 从工具包导入后处理函数

model_dir = "iic/SenseVoiceSmall"  # 指定模型目录

# 初始化模型
model = AutoModel(
    model=model_dir,  # 指定模型路径
    trust_remote_code=True,  # 允许使用远程代码
    remote_code="./model.py",  # 远程代码文件路径
    vad_model="fsmn-vad",  # 指定语音活动检测模型
    vad_kwargs={"max_single_segment_time": 30000},  # 设置 VAD 参数，单段最大时长为 30000 毫秒
    device="cuda:0",  # 指定使用的设备为第一个 CUDA 设备
)

# en
res = model.generate(  # 调用模型生成方法进行推理
    input=f"{model.model_path}/example/en.mp3",  # 输入音频文件路径
    cache={},  # 使用空缓存
    language="auto",  # 自动检测语言
    use_itn=True,  # 使用文本到语音的转录
    batch_size_s=60,  # 设置批处理大小为 60 秒
    merge_vad=True,  # 启用合并语音活动检测
    merge_length_s=15,  # 合并的时长设置为 15 秒
)
text = rich_transcription_postprocess(res[0]["text"])  # 对结果进行后处理
print(text)  # 打印处理后的文本

# zh
res = model.generate(  # 重复模型推理流程，处理中文音频
    input=f"{model.model_path}/example/zh.mp3",  # 输入中文音频文件路径
    cache={},  # 使用空缓存
    language="auto",  # 自动检测语言
    use_itn=True,  # 使用文本到语音的转录
    batch_size_s=60,  # 设置批处理大小为 60 秒
    merge_vad=True,  # 启用合并语音活动检测
    merge_length_s=15,  # 合并的时长设置为 15 秒
)
text = rich_transcription_postprocess(res[0]["text"])  # 对结果进行后处理
print(text)  # 打印处理后的文本

# yue
res = model.generate(  # 处理粤语音频
    input=f"{model.model_path}/example/yue.mp3",  # 输入粤语音频文件路径
    cache={},  # 使用空缓存
    language="auto",  # 自动检测语言
    use_itn=True,  # 使用文本到语音的转录
    batch_size_s=60,  # 设置批处理大小为 60 秒
    merge_vad=True,  # 启用合并语音活动检测
    merge_length_s=15,  # 合并的时长设置为 15 秒
)
text = rich_transcription_postprocess(res[0]["text"])  # 对结果进行后处理
print(text)  # 打印处理后的文本

# ja
res = model.generate(  # 处理日语音频
    input=f"{model.model_path}/example/ja.mp3",  # 输入日语音频文件路径
    cache={},  # 使用空缓存
    language="auto",  # 自动检测语言
    use_itn=True,  # 使用文本到语音的转录
    batch_size_s=60,  # 设置批处理大小为 60 秒
    merge_vad=True,  # 启用合并语音活动检测
    merge_length_s=15,  # 合并的时长设置为 15 秒
)
text = rich_transcription_postprocess(res[0]["text"])  # 对结果进行后处理
print(text)  # 打印处理后的文本

# ko
res = model.generate(  # 处理韩语音频
    input=f"{model.model_path}/example/ko.mp3",  # 输入韩语音频文件路径
    cache={},  # 使用空缓存
    language="auto",  # 自动检测语言
    use_itn=True,  # 使用文本到语音的转录
    batch_size_s=60,  # 设置批处理大小为 60 秒
    merge_vad=True,  # 启用合并语音活动检测
    merge_length_s=15,  # 合并的时长设置为 15 秒
)
text = rich_transcription_postprocess(res[0]["text"])  # 对结果进行后处理
print(text)  # 打印处理后的文本
```