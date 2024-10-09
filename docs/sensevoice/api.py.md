# `.\SenseVoiceSmall-src\api.py`

```
# 设置设备环境，默认为 cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re  # 导入操作系统和正则表达式模块
from fastapi import FastAPI, File, Form  # 从 FastAPI 导入创建应用和处理文件、表单数据的功能
from fastapi.responses import HTMLResponse  # 导入 HTML 响应类
from typing_extensions import Annotated  # 导入注解类型扩展
from typing import List  # 导入列表类型
from enum import Enum  # 导入枚举类
import torchaudio  # 导入音频处理库
from model import SenseVoiceSmall  # 从模型模块导入 SenseVoiceSmall 类
from funasr.utils.postprocess_utils import rich_transcription_postprocess  # 导入后处理工具
from io import BytesIO  # 导入字节流模块


class Language(str, Enum):  # 定义语言枚举类，继承自字符串和枚举
    auto = "auto"  # 自动语言识别
    zh = "zh"  # 中文
    en = "en"  # 英文
    yue = "yue"  # 粤语
    ja = "ja"  # 日语
    ko = "ko"  # 韩语
    nospeech = "nospeech"  # 无语音


model_dir = "iic/SenseVoiceSmall"  # 定义模型目录
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=os.getenv("SENSEVOICE_DEVICE", "cuda:0"))  # 从预训练模型加载模型，使用指定设备或默认设备
m.eval()  # 设置模型为评估模式

regex = r"<\|.*\|>"  # 定义用于文本处理的正则表达式

app = FastAPI()  # 创建 FastAPI 应用实例


@app.get("/", response_class=HTMLResponse)  # 定义根路径的 GET 请求处理函数，响应为 HTML
async def root():  # 异步处理函数
    return """  # 返回 HTML 文档
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>  # 设置网页标题
        </head>
        <body>
            <a href='./docs'>Documents of API</a>  # 添加链接到 API 文档
        </body>
    </html>
    """


@app.post("/api/v1/asr")  # 定义音频转文本的 POST 请求处理函数
async def turn_audio_to_text(files: Annotated[List[bytes], File(description="wav or mp3 audios in 16KHz")],  # 接收音频文件
                             keys: Annotated[str, Form(description="name of each audio joined with comma")],  # 接收音频文件名称
                             lang: Annotated[Language, Form(description="language of audio content")] = "auto"):  # 接收语言参数，默认为自动
    audios = []  # 初始化音频数据列表
    audio_fs = 0  # 初始化音频采样率
    for file in files:  # 遍历上传的每个文件
        file_io = BytesIO(file)  # 将文件数据封装为字节流
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)  # 使用 torchaudio 加载音频数据和采样率
        data_or_path_or_list = data_or_path_or_list.mean(0)  # 计算多通道音频的均值
        audios.append(data_or_path_or_list)  # 将处理后的音频数据添加到列表中
        file_io.close()  # 关闭字节流
    if lang == "":  # 检查语言参数是否为空
        lang = "auto"  # 若为空，设为自动语言识别
    if keys == "":  # 检查音频名称是否为空
        key = ["wav_file_tmp_name"]  # 若为空，使用默认名称
    else:
        key = keys.split(",")  # 否则根据逗号分隔音频名称
    res = m.inference(  # 调用模型进行推断
        data_in=audios,  # 输入音频数据
        language=lang,  # 语言参数
        use_itn=False,  # 是否使用音频转换
        ban_emo_unk=False,  # 是否禁止情感未知
        key=key,  # 音频名称列表
        fs=audio_fs,  # 音频采样率
        **kwargs,  # 其他参数
    )
    if len(res) == 0:  # 检查结果是否为空
        return {"result": []}  # 若为空，返回空结果
    for it in res[0]:  # 遍历第一个结果
        it["raw_text"] = it["text"]  # 将原始文本存储
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)  # 使用正则表达式清洗文本
        it["text"] = rich_transcription_postprocess(it["text"])  # 进行后处理以优化文本
    return {"result": res[0]}  # 返回处理后的结果
```