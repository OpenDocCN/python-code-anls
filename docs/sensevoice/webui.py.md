# `.\SenseVoiceSmall-src\webui.py`

```
# coding=utf-8  # 指定文件编码为 UTF-8

import os  # 导入操作系统接口模块
import librosa  # 导入 librosa 音频处理库
import base64  # 导入 base64 编码和解码模块
import io  # 导入输入输出流模块
import gradio as gr  # 导入 Gradio 库，用于创建交互式界面
import re  # 导入正则表达式模块

import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 库，用于深度学习
import torchaudio  # 导入 Torchaudio 库，处理音频数据

from funasr import AutoModel  # 从 funasr 库导入自动模型类

model = "iic/SenseVoiceSmall"  # 定义要使用的模型名称
model = AutoModel(model=model,  # 创建 AutoModel 实例
                  vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",  # 设置语音活动检测模型
                  vad_kwargs={"max_single_segment_time": 30000},  # 设置语音活动检测参数
                  trust_remote_code=True,  # 允许使用远程代码
                  )

import re  # 再次导入正则表达式模块（可能为冗余）

# 定义情绪与表情符号的映射字典
emo_dict = {
    "<|HAPPY|>": "😊",  # 快乐
    "<|SAD|>": "😔",  # 悲伤
    "<|ANGRY|>": "😡",  # 生气
    "<|NEUTRAL|>": "",  # 中立
    "<|FEARFUL|>": "😰",  # 害怕
    "<|DISGUSTED|>": "🤢",  # 厌恶
    "<|SURPRISED|>": "😮",  # 惊讶
}

# 定义事件与对应表情符号的映射字典
event_dict = {
    "<|BGM|>": "🎼",  # 背景音乐
    "<|Speech|>": "",  # 讲话
    "<|Applause|>": "👏",  # 掌声
    "<|Laughter|>": "😀",  # 笑声
    "<|Cry|>": "😭",  # 哭泣
    "<|Sneeze|>": "🤧",  # 打喷嚏
    "<|Breath|>": "",  # 呼吸
    "<|Cough|>": "🤧",  # 咳嗽
}

# 定义多种标记与对应表情符号的映射字典
emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓",  # 未知事件
    "<|zh|>": "",  # 中文
    "<|en|>": "",  # 英文
    "<|yue|>": "",  # 粤语
    "<|ja|>": "",  # 日语
    "<|ko|>": "",  # 韩语
    "<|nospeech|>": "",  # 没有语音
    "<|HAPPY|>": "😊",  # 快乐
    "<|SAD|>": "😔",  # 悲伤
    "<|ANGRY|>": "😡",  # 生气
    "<|NEUTRAL|>": "",  # 中立
    "<|BGM|>": "🎼",  # 背景音乐
    "<|Speech|>": "",  # 讲话
    "<|Applause|>": "👏",  # 掌声
    "<|Laughter|>": "😀",  # 笑声
    "<|FEARFUL|>": "😰",  # 害怕
    "<|DISGUSTED|>": "🤢",  # 厌恶
    "<|SURPRISED|>": "😮",  # 惊讶
    "<|Cry|>": "😭",  # 哭泣
    "<|EMO_UNKNOWN|>": "",  # 未知情绪
    "<|Sneeze|>": "🤧",  # 打喷嚏
    "<|Breath|>": "",  # 呼吸
    "<|Cough|>": "😷",  # 咳嗽
    "<|Sing|>": "",  # 唱歌
    "<|Speech_Noise|>": "",  # 讲话噪声
    "<|withitn|>": "",  # 不明标记
    "<|woitn|>": "",  # 不明标记
    "<|GBG|>": "",  # 不明标记
    "<|Event_UNK|>": "",  # 未知事件
}

# 定义语言标记的映射字典
lang_dict =  {
    "<|zh|>": "<|lang|>",  # 中文标记映射
    "<|en|>": "<|lang|>",  # 英文标记映射
    "<|yue|>": "<|lang|>",  # 粤语标记映射
    "<|ja|>": "<|lang|>",  # 日语标记映射
    "<|ko|>": "<|lang|>",  # 韩语标记映射
    "<|nospeech|>": "<|lang|>",  # 没有语音标记映射
}

# 定义情绪集合
emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}  # 包含所有情绪表情符号
# 定义事件集合
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}  # 包含所有事件表情符号

# 定义字符串格式化函数
def format_str(s):
    for sptk in emoji_dict:  # 遍历每个表情符号标记
        s = s.replace(sptk, emoji_dict[sptk])  # 替换为对应的表情符号
    return s  # 返回格式化后的字符串

# 定义字符串格式化函数版本 2
def format_str_v2(s):
    sptk_dict = {}  # 初始化空字典用于统计标记出现次数
    for sptk in emoji_dict:  # 遍历每个表情符号标记
        sptk_dict[sptk] = s.count(sptk)  # 统计标记在字符串中出现的次数
        s = s.replace(sptk, "")  # 从字符串中移除该标记
    emo = "<|NEUTRAL|>"  # 初始化情绪为中立
    for e in emo_dict:  # 遍历情绪标记
        if sptk_dict[e] > sptk_dict[emo]:  # 如果当前情绪出现次数更多
            emo = e  # 更新情绪
    for e in event_dict:  # 遍历事件标记
        if sptk_dict[e] > 0:  # 如果当前事件出现次数大于0
            s = event_dict[e] + s  # 将事件表情符号添加到字符串前
    s = s + emo_dict[emo]  # 将最终情绪表情符号添加到字符串末尾

    for emoji in emo_set.union(event_set):  # 遍历情绪和事件表情符号的并集
        s = s.replace(" " + emoji, emoji)  # 去除表情符号前的空格
        s = s.replace(emoji + " ", emoji)  # 去除表情符号后的空格
    return s.strip()  # 返回去掉首尾空格的字符串

# 定义字符串格式化函数版本 3
def format_str_v3(s):
    def get_emo(s):  # 定义获取情绪的辅助函数
        return s[-1] if s[-1] in emo_set else None  # 返回字符串最后一个字符作为情绪，如果在情绪集合中
    def get_event(s):  # 定义获取事件的辅助函数
        return s[0] if s[0] in event_set else None  # 返回字符串第一个字符作为事件，如果在事件集合中

    s = s.replace("<|nospeech|><|Event_UNK|>", "❓")  # 替换特定标记为表情符号
    for lang in lang_dict:  # 遍历语言标记
        s = s.replace(lang, "<|lang|>")  # 替换为通用的语言标记
    s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]  # 分割字符串并格式化每个部分
    new_s = " " + s_list[0]  # 初始化新字符串，前面加上空格
    cur_ent_event = get_event(new_s)  # 获取当前事件
    # 遍历 s_list 列表，从第二个元素开始
        for i in range(1, len(s_list)):
            # 如果当前元素为空字符串，则跳过该次循环
            if len(s_list[i]) == 0:
                continue
            # 如果当前元素的事件与当前事件相同且不为 None，则移除当前元素的第一个字符
            if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
                s_list[i] = s_list[i][1:]
            # 更新当前事件为当前元素的事件
            cur_ent_event = get_event(s_list[i])
            # 如果当前元素的情感不为 None 且与 new_s 的情感相同，则移除 new_s 的最后一个字符
            if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
                new_s = new_s[:-1]
            # 将当前元素的处理结果添加到 new_s 中，去除前后空白
            new_s += s_list[i].strip().lstrip()
        # 将 new_s 中的 "The." 替换为空格
        new_s = new_s.replace("The.", " ")
        # 返回处理后的 new_s，去除前后空白
        return new_s.strip()
# 定义模型推理函数，接收输入音频、语言和采样率
def model_inference(input_wav, language, fs=16000):
    # 语言缩写字典，映射不同语言的标识
    language_abbr = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko",
                     "nospeech": "nospeech"}
    
    # 语言为 auto 如果提供的语言字符串为空
    language = "auto" if len(language) < 1 else language
    # 选择对应的语言缩写
    selected_language = language_abbr[language]
    # 选择任务的缩写（注释掉的代码）
    
    # 打印输入音频的信息（注释掉的代码）
    # print(f"input_wav: {type(input_wav)}, {input_wav[1].shape}, {input_wav}")
    
    # 检查输入音频是否为元组，提取采样率和音频数据
    if isinstance(input_wav, tuple):
        fs, input_wav = input_wav
        # 将音频数据标准化为浮点数
        input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
        # 如果音频有多个声道，取平均值
        if len(input_wav.shape) > 1:
            input_wav = input_wav.mean(-1)
        # 如果采样率不是 16000，进行重采样
        if fs != 16000:
            print(f"audio_fs: {fs}")
            resampler = torchaudio.transforms.Resample(fs, 16000)
            # 将 NumPy 数组转换为 PyTorch 张量
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            # 重采样并转换回 NumPy 数组
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
    
    # 合并 VAD 标志（注释掉的代码）
    merge_vad = True #False if selected_task == "ASR" else True
    # 打印当前语言和合并 VAD 状态
    print(f"language: {language}, merge_vad: {merge_vad}")
    # 生成文本，调用模型进行推理
    text = model.generate(input=input_wav,
                          cache={},
                          language=language,
                          use_itn=True,
                          batch_size_s=60, merge_vad=merge_vad)
    
    # 打印生成的文本
    print(text)
    # 提取生成文本的内容
    text = text[0]["text"]
    # 格式化文本
    text = format_str_v3(text)
    
    # 打印格式化后的文本
    print(text)
    
    # 返回生成的文本
    return text


# 音频示例列表，每个示例包含音频文件路径和语言
audio_examples = [
    ["example/zh.mp3", "zh"],
    ["example/yue.mp3", "yue"],
    ["example/en.mp3", "en"],
    ["example/ja.mp3", "ja"],
    ["example/ko.mp3", "ko"],
    ["example/emo_1.wav", "auto"],
    ["example/emo_2.wav", "auto"],
    ["example/emo_3.wav", "auto"],
    #["example/emo_4.wav", "auto"],
    #["example/event_1.wav", "auto"],
    #["example/event_2.wav", "auto"],
    #["example/event_3.wav", "auto"],
    ["example/rich_1.wav", "auto"],
    ["example/rich_2.wav", "auto"],
    #["example/rich_3.wav", "auto"],
    ["example/longwav_1.wav", "auto"],
    ["example/longwav_2.wav", "auto"],
    ["example/longwav_3.wav", "auto"],
    #["example/longwav_4.wav", "auto"],
]

# 定义 HTML 内容，描述模型信息
html_content = """
<div>
    <h2 style="font-size: 22px;margin-left: 0px;">Voice Understanding Model: SenseVoice-Small</h2>
    <p style="font-size: 18px;margin-left: 20px;">SenseVoice-Small is an encoder-only speech foundation model designed for rapid voice understanding. It encompasses a variety of features including automatic speech recognition (ASR), spoken language identification (LID), speech emotion recognition (SER), and acoustic event detection (AED). SenseVoice-Small supports multilingual recognition for Chinese, English, Cantonese, Japanese, and Korean. Additionally, it offers exceptionally low inference latency, performing 7 times faster than Whisper-small and 17 times faster than Whisper-large.</p>
    # 创建一个二级标题，字体大小为 22px，左边距为 0px，内容为“Usage”
    <h2 style="font-size: 22px;margin-left: 0px;">Usage</h2> 
    # 创建一个段落，字体大小为 18px，左边距为 20px，描述音频文件上传和转录的功能
    <p style="font-size: 18px;margin-left: 20px;">Upload an audio file or input through a microphone, then select the task and language. the audio is transcribed into corresponding text along with associated emotions (😊 happy, 😡 angry/exicting, 😔 sad) and types of sound events (😀 laughter, 🎼 music, 👏 applause, 🤧 cough&sneeze, 😭 cry). The event labels are placed in the front of the text and the emotion are in the back of the text.</p>
    # 创建一个段落，字体大小为 18px，左边距为 20px，推荐音频输入时长信息
    <p style="font-size: 18px;margin-left: 20px;">Recommended audio input duration is below 30 seconds. For audio longer than 30 seconds, local deployment is recommended.</p>
    # 创建一个二级标题，字体大小为 22px，左边距为 0px，内容为“Repo”
    <h2 style="font-size: 22px;margin-left: 0px;">Repo</h2>
    # 创建一个段落，字体大小为 18px，左边距为 20px，链接到“SenseVoice”项目，描述其功能
    <p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/FunAudioLLM/SenseVoice" target="_blank">SenseVoice</a>: multilingual speech understanding model</p>
    # 创建一个段落，字体大小为 18px，左边距为 20px，链接到“FunASR”项目，描述其功能
    <p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/modelscope/FunASR" target="_blank">FunASR</a>: fundamental speech recognition toolkit</p>
    # 创建一个段落，字体大小为 18px，左边距为 20px，链接到“CosyVoice”项目，描述其功能
    <p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/FunAudioLLM/CosyVoice" target="_blank">CosyVoice</a>: high-quality multilingual TTS model</p>
</div>
"""  # 结束 HTML 内容的标签

def launch():  # 定义一个名为 launch 的函数
    with gr.Blocks(theme=gr.themes.Soft()) as demo:  # 创建一个 Gradio 块，使用 Soft 主题并命名为 demo
        # gr.Markdown(description)  # 可能用于显示描述的 Markdown 文本（注释掉了）
        gr.HTML(html_content)  # 显示 HTML 内容，内容来自 html_content 变量
        with gr.Row():  # 创建一个水平布局的行
            with gr.Column():  # 创建一个垂直布局的列
                audio_inputs = gr.Audio(label="Upload audio or use the microphone")  # 创建音频输入组件，标签为“上传音频或使用麦克风”
                
                with gr.Accordion("Configuration"):  # 创建一个可折叠的面板，标题为“配置”
                    language_inputs = gr.Dropdown(choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],  # 创建下拉菜单，选择语言
                                                  value="auto",  # 默认值为“自动”
                                                  label="Language")  # 下拉菜单标签为“语言”
                fn_button = gr.Button("Start", variant="primary")  # 创建一个按钮，标签为“开始”，并设置为主要变体
                text_outputs = gr.Textbox(label="Results")  # 创建一个文本框，标签为“结果”
            gr.Examples(examples=audio_examples, inputs=[audio_inputs, language_inputs], examples_per_page=20)  # 创建示例输入，提供音频示例和语言选择，示例每页显示20个
        
        fn_button.click(model_inference, inputs=[audio_inputs, language_inputs], outputs=text_outputs)  # 设置按钮点击事件，调用 model_inference 函数，输入为音频和语言，输出为结果文本框

    demo.launch()  # 启动 Gradio 应用

if __name__ == "__main__":  # 如果该脚本是主程序
    # iface.launch()  # 可能用于启动接口（注释掉了）
    launch()  # 调用 launch 函数启动应用
```