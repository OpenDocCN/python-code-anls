# `d:/src/tocomm/Bert-VITS2\webui.py`

```
# flake8: noqa: E402
```
这是一个特殊的注释，用于告诉 flake8 工具忽略 E402 错误。E402 错误表示在导入模块时出现了错误的顺序。

```
import os
```
导入 os 模块，用于与操作系统进行交互。

```
import logging
```
导入 logging 模块，用于记录日志信息。

```
import re_matching
```
导入 re_matching 模块，这是一个自定义的模块，用于进行正则表达式匹配。

```
from tools.sentence import split_by_language
```
从 tools.sentence 模块中导入 split_by_language 函数。

```
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
```
设置 numba、markdown_it、urllib3 和 matplotlib 的日志级别为 WARNING，表示只记录警告级别及以上的日志信息。

```
logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)
```
配置 logging 模块的基本日志记录设置，将日志级别设置为 INFO，格式为 "| 模块名 | 日志级别 | 日志信息"。

```
logger = logging.getLogger(__name__)
```
创建一个名为 logger 的日志记录器，用于记录当前模块的日志信息。

```
import torch
```
导入 torch 模块，用于进行深度学习相关的操作。

```
import utils
```
导入 utils 模块，这是一个自定义的模块，包含了一些常用的工具函数。

```
from infer import infer, latest_version, get_net_g, infer_multilang
```
从 infer 模块中导入 infer、latest_version、get_net_g 和 infer_multilang 函数。
import gradio as gr  # 导入 gradio 库，用于构建交互式界面
import webbrowser  # 导入 webbrowser 库，用于在浏览器中打开网页
import numpy as np  # 导入 numpy 库，用于进行数值计算
from config import config  # 导入 config 模块，用于读取配置信息
from tools.translate import translate  # 导入 translate 模块，用于进行翻译
import librosa  # 导入 librosa 库，用于音频处理

net_g = None  # 初始化变量 net_g

device = config.webui_config.device  # 从配置文件中读取设备信息，并赋值给变量 device
if device == "mps":  # 如果设备为 "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 设置环境变量 PYTORCH_ENABLE_MPS_FALLBACK 为 "1"

# 定义函数 generate_audio，接受参数 slices, sdp_ratio, noise_scale, noise_scale_w, length_scale
def generate_audio(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,  # 输入参数：说话人
    language,  # 输入参数：语言
    reference_audio,  # 输入参数：参考音频
    emotion,  # 输入参数：情感
    style_text,  # 输入参数：风格文本
    style_weight,  # 输入参数：风格权重
    skip_start=False,  # 输入参数：是否跳过开头
    skip_end=False,  # 输入参数：是否跳过结尾
):
    audio_list = []  # 创建一个空列表用于存储音频
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)  # 创建一个长度为采样率一半的零数组，用于表示静音
    with torch.no_grad():  # 禁用梯度计算
        for idx, piece in enumerate(slices):  # 遍历slices列表中的元素，同时获取索引
            skip_start = idx != 0  # 如果索引不等于0，则将skip_start设置为True，否则为False
            skip_end = idx != len(slices) - 1  # 如果索引不等于slices列表长度减1，则将skip_end设置为True，否则为False
            audio = infer(  # 调用infer函数，返回音频
                piece,  # 输入参数：音频片段
                reference_audio=reference_audio,  # 输入参数：参考音频
                emotion=emotion,  # 输入参数：情感
                sdp_ratio=sdp_ratio,  # 输入参数：sdp比率
                noise_scale=noise_scale,  # 噪声的缩放比例
                noise_scale_w=noise_scale_w,  # 噪声的缩放比例（宽度）
                length_scale=length_scale,  # 音频长度的缩放比例
                sid=speaker,  # 说话者的标识
                language=language,  # 语言的标识
                hps=hps,  # 超参数
                net_g=net_g,  # 生成器网络
                device=device,  # 设备（CPU或GPU）
                skip_start=skip_start,  # 跳过音频开头的帧数
                skip_end=skip_end,  # 跳过音频结尾的帧数
                style_text=style_text,  # 风格文本
                style_weight=style_weight,  # 风格权重
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)  # 将音频转换为16位wav格式
            audio_list.append(audio16bit)  # 将音频添加到音频列表中
    return audio_list


def generate_audio_multilang(
    slices,  # 切片
    sdp_ratio,  # SDP 比例
    noise_scale,  # 噪声比例
    noise_scale_w,  # 噪声比例权重
    length_scale,  # 长度比例
    speaker,  # 说话人
    language,  # 语言
    reference_audio,  # 参考音频
    emotion,  # 情感
    skip_start=False,  # 是否跳过开头
    skip_end=False,  # 是否跳过结尾
):
    audio_list = []  # 音频列表
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)  # 静音
    with torch.no_grad():  # 禁用梯度计算
        for idx, piece in enumerate(slices):  # 遍历切片
            skip_start = idx != 0  # 是否跳过开头
            skip_end = idx != len(slices) - 1  # 是否跳过结尾
            audio = infer_multilang(  # 推断多语言音频
                piece,  # 切片
                reference_audio=reference_audio,  # 参考音频
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language[idx],
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
```
这段代码是一个函数调用，调用了一个名为tts_process的函数，并传递了一系列参数。这些参数包括情感(emotion)、sdp比例(sdp_ratio)、噪声比例(noise_scale)、噪声比例w(noise_scale_w)、长度比例(length_scale)、说话人ID(sid)、语言(language)、hps、net_g、设备(device)、跳过开始部分(skip_start)和跳过结束部分(skip_end)。这些参数将用于tts_process函数的计算和处理。

```
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
```
这段代码将通过调用gr.processing_utils.convert_to_16_bit_wav函数将音频数据(audio)转换为16位wav格式，并将转换后的音频数据添加到audio_list列表中。

```
    return audio_list
```
这段代码返回audio_list列表作为函数的结果。

```
def tts_split(
    text: str,
```
这段代码定义了一个名为tts_split的函数，该函数接受一个字符串类型的参数text作为输入。
    speaker,  # 说话者
    sdp_ratio,  # SDP 比例
    noise_scale,  # 噪声比例
    noise_scale_w,  # 噪声比例权重
    length_scale,  # 长度比例
    language,  # 语言
    cut_by_sent,  # 是否按句切分
    interval_between_para,  # 段落之间的间隔
    interval_between_sent,  # 句子之间的间隔
    reference_audio,  # 参考音频
    emotion,  # 情感
    style_text,  # 风格文本
    style_weight,  # 风格权重
):
    while text.find("\n\n") != -1:  # 当文本中存在连续两个换行符时
        text = text.replace("\n\n", "\n")  # 将连续两个换行符替换为一个换行符
    text = text.replace("|", "")  # 将文本中的竖线符号替换为空字符串
    para_list = re_matching.cut_para(text)  # 使用正则表达式切分文本为段落列表
    para_list = [p for p in para_list if p != ""]  # 去除空段落
    audio_list = []  # 初始化音频列表
    for p in para_list:
        # 如果不按句子切分，则调用 process_text 函数处理文本
        if not cut_by_sent:
            audio_list += process_text(
                p,
                speaker,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                language,
                reference_audio,
                emotion,
                style_text,
                style_weight,
            )
            # 在音频列表中添加一个间隔的静音段
            silence = np.zeros((int)(44100 * interval_between_para), dtype=np.int16)
            audio_list.append(silence)
        # 如果按句子切分，则先将段落切分成句子
        else:
            audio_list_sent = []
            sent_list = re_matching.cut_sent(p)
# 根据句子列表过滤掉空字符串
sent_list = [s for s in sent_list if s != ""]

# 遍历句子列表
for s in sent_list:
    # 将每个句子进行文本处理，并将处理后的音频添加到音频列表中
    audio_list_sent += process_text(
        s,
        speaker,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        length_scale,
        language,
        reference_audio,
        emotion,
        style_text,
        style_weight,
    )
    # 在句子之间添加间隔的静音
    silence = np.zeros((int)(44100 * interval_between_sent))
    audio_list_sent.append(silence)

# 如果段落之间的间隔大于句子之间的间隔
if (interval_between_para - interval_between_sent) > 0:
    # 在段落之间添加间隔的静音
    silence = np.zeros(
        (int)(44100 * (interval_between_para - interval_between_sent))
                )
                audio_list_sent.append(silence)
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(
                np.concatenate(audio_list_sent)
            )  # 对完整句子做音量归一
            audio_list.append(audio16bit)
    audio_concat = np.concatenate(audio_list)
    return ("Success", (hps.data.sampling_rate, audio_concat))
```

注释如下：

```
# 对每个句子进行处理
for sent in slice:
    # 获取句子的音频数据
    audio = sent[0]
    # 获取句子的静音数据
    silence = sent[1]
    # 如果句子为空，则跳过
    if len(audio) == 0:
        continue
    # 将静音数据添加到音频数据中
    audio_list_sent.append(audio)
    # 将静音数据添加到音频列表中
    audio_list_sent.append(silence)
    # 对完整句子做音量归一化处理
    audio16bit = gr.processing_utils.convert_to_16_bit_wav(
        np.concatenate(audio_list_sent)
    )
    # 将处理后的音频数据添加到音频列表中
    audio_list.append(audio16bit)
# 将所有音频数据连接起来
audio_concat = np.concatenate(audio_list)
# 返回处理结果
return ("Success", (hps.data.sampling_rate, audio_concat))
```

```
def process_mix(slice):
    # 弹出最后一个元素作为_speaker
    _speaker = slice.pop()
    # 初始化_text和_lang为空列表
    _text, _lang = [], []
    # 遍历slice中的每个元素
    for lang, content in slice:
        # 将content按照"|"分割成列表
        content = content.split("|")
        # 过滤掉空字符串
        content = [part for part in content if part != ""]
        # 如果content为空列表，则跳过
        if len(content) == 0:
            continue
        # 如果_text为空列表，则将content中的每个部分作为一个子列表添加到_text中
        if len(_text) == 0:
            _text = [[part] for part in content]
            _lang = [[lang] for part in content]
```
这行代码创建了一个名为`_lang`的列表，其中每个元素都是一个包含`lang`的列表。这个列表用于存储`content`的语言信息。

```
        else:
            _text[-1].append(content[0])
            _lang[-1].append(lang)
            if len(content) > 1:
                _text += [[part] for part in content[1:]]
                _lang += [[lang] for part in content[1:]]
```
这段代码是一个条件语句，当条件不满足时执行。它将`content`的内容添加到`_text`和`_lang`列表中。如果`content`的长度大于1，则将剩余的部分也添加到`_text`和`_lang`列表中。

```
    return _text, _lang, _speaker
```
这行代码返回`_text`、`_lang`和`_speaker`这三个变量作为函数的结果。

```
def process_auto(text):
```
这是一个名为`process_auto`的函数定义，它接受一个名为`text`的参数。

```
    _text, _lang = [], []
```
这行代码创建了两个空列表`_text`和`_lang`，用于存储处理后的文本和语言信息。

```
    for slice in text.split("|"):
```
这是一个`for`循环，它遍历通过`|`分隔的`text`字符串的每个部分，并将每个部分赋值给`slice`变量。

```
        if slice == "":
            continue
```
这是一个条件语句，当`slice`为空字符串时执行。`continue`关键字用于跳过当前循环的剩余部分，继续下一次循环。

```
        temp_text, temp_lang = [], []
```
这行代码创建了两个空列表`temp_text`和`temp_lang`，用于存储临时的文本和语言信息。

```
        sentences_list = split_by_language(slice, target_languages=["zh", "ja", "en"])
```
这行代码调用名为`split_by_language`的函数，将`slice`和`target_languages`作为参数传递给它，并将返回的结果赋值给`sentences_list`变量。这个函数用于将`slice`按照指定的目标语言进行分割。

```
        for sentence, lang in sentences_list:
```
这是一个`for`循环，它遍历`sentences_list`中的每个元素，并将每个元素的第一个值赋值给`sentence`变量，第二个值赋值给`lang`变量。

```
            if sentence == "":
                continue
```
这是一个条件语句，当`sentence`为空字符串时执行。`continue`关键字用于跳过当前循环的剩余部分，继续下一次循环。
temp_text.append(sentence)
```
将句子添加到临时文本列表中。

```
if lang == "ja":
    lang = "jp"
```
如果语言是日语（"ja"），则将语言更改为"jp"。

```
temp_lang.append(lang.upper())
```
将语言转换为大写形式，并将其添加到临时语言列表中。

```
_text.append(temp_text)
_lang.append(temp_lang)
```
将临时文本列表和临时语言列表添加到主文本列表和主语言列表中。

```
return _text, _lang
```
返回主文本列表和主语言列表作为结果。

```
def process_text(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    reference_audio,
    emotion,
    style_text=None,
```
定义一个名为process_text的函数，该函数接受多个参数，包括文本、说话者、SDP比例、噪声比例、噪声比例w、长度比例、语言、参考音频、情感和样式文本。
    style_weight=0,
):
    # 创建一个空列表，用于存储音频数据
    audio_list = []
    # 如果语言是"mix"
    if language == "mix":
        # 使用正则表达式验证文本的有效性，并返回验证结果和验证信息
        bool_valid, str_valid = re_matching.validate_text(text)
        # 如果验证结果为False
        if not bool_valid:
            # 返回验证信息和一个包含空音频数据的元组
            return str_valid, (
                hps.data.sampling_rate,
                np.concatenate([np.zeros(hps.data.sampling_rate // 2)]),
            )
        # 对于每个切片在文本匹配结果中
        for slice in re_matching.text_matching(text):
            # 处理混合语言的切片，获取处理后的文本、语言和说话者信息
            _text, _lang, _speaker = process_mix(slice)
            # 如果说话者信息为空
            if _speaker is None:
                # 继续下一个切片的处理
                continue
            # 打印处理后的文本和语言信息
            print(f"Text: {_text}\nLang: {_lang}")
            # 将生成的多语言音频数据添加到音频列表中
            audio_list.extend(
                generate_audio_multilang(
                    _text,
                    sdp_ratio,
                    noise_scale,
```

这段代码是一个函数的一部分，主要功能是根据给定的文本生成多语言音频数据。具体注释如下：

- `style_weight=0,`：函数的参数，表示样式权重，默认为0。
- `audio_list = []`：创建一个空列表，用于存储音频数据。
- `if language == "mix":`：如果语言是"mix"。
- `bool_valid, str_valid = re_matching.validate_text(text)`：使用正则表达式验证文本的有效性，并返回验证结果和验证信息。
- `if not bool_valid:`：如果验证结果为False。
- `return str_valid, (hps.data.sampling_rate, np.concatenate([np.zeros(hps.data.sampling_rate // 2)]))`：返回验证信息和一个包含空音频数据的元组。
- `for slice in re_matching.text_matching(text):`：对于每个切片在文本匹配结果中。
- `_text, _lang, _speaker = process_mix(slice)`：处理混合语言的切片，获取处理后的文本、语言和说话者信息。
- `if _speaker is None:`：如果说话者信息为空。
- `continue`：继续下一个切片的处理。
- `print(f"Text: {_text}\nLang: {_lang}")`：打印处理后的文本和语言信息。
- `audio_list.extend(generate_audio_multilang(_text, sdp_ratio, noise_scale`：将生成的多语言音频数据添加到音频列表中。
# 如果语言参数为 "auto"，则调用 process_auto 函数处理文本，获取处理后的文本和语言信息
elif language.lower() == "auto":
    _text, _lang = process_auto(text)
    # 打印处理后的文本和语言信息
    print(f"Text: {_text}\nLang: {_lang}")
    # 调用 generate_audio_multilang 函数生成多语言音频，并将结果添加到音频列表中
    audio_list.extend(
        generate_audio_multilang(
            _text,
            sdp_ratio,
            noise_scale,
            noise_scale_w,
            length_scale,
            speaker,
            _lang,
            reference_audio,
            emotion,
        )
    )
# 如果条件成立，执行以下代码块
if condition:
    # 将生成的音频列表扩展到现有的音频列表中
    audio_list.extend(
        # 调用generate_audio函数生成音频，并将返回的音频列表扩展到现有的音频列表中
        generate_audio(
            # 将文本按照"|"分割成多个部分，并作为参数传递给generate_audio函数
            text.split("|"),
            # 将sdp_ratio作为参数传递给generate_audio函数
            sdp_ratio,
            # 将noise_scale作为参数传递给generate_audio函数
            noise_scale,
            # 将noise_scale_w作为参数传递给generate_audio函数
            noise_scale_w,
            # 将length_scale作为参数传递给generate_audio函数
            length_scale,
            # 将speaker作为参数传递给generate_audio函数
            speaker,
            # 将language作为参数传递给generate_audio函数
            language,
            # 将reference_audio作为参数传递给generate_audio函数
            reference_audio,
            # 将emotion作为参数传递给generate_audio函数
            emotion,
            # 将style_text作为参数传递给generate_audio函数
            style_text,
            # 将style_weight作为参数传递给generate_audio函数
            style_weight,
        )
    )
# 如果条件不成立，执行以下代码块
else:
    # 将生成的音频列表扩展到现有的音频列表中
    audio_list.extend(
        # 调用generate_audio函数生成音频，并将返回的音频列表扩展到现有的音频列表中
        generate_audio(
            # 将文本按照"|"分割成多个部分，并作为参数传递给generate_audio函数
            text.split("|"),
            # 将sdp_ratio作为参数传递给generate_audio函数
            sdp_ratio,
            # 将noise_scale作为参数传递给generate_audio函数
            noise_scale,
            # 将noise_scale_w作为参数传递给generate_audio函数
            noise_scale_w,
            # 将length_scale作为参数传递给generate_audio函数
            length_scale,
            # 将speaker作为参数传递给generate_audio函数
            speaker,
            # 将language作为参数传递给generate_audio函数
            language,
            # 将reference_audio作为参数传递给generate_audio函数
            reference_audio,
            # 将emotion作为参数传递给generate_audio函数
            emotion,
        )
    )
    return audio_list
```
返回变量`audio_list`。

```
def tts_fn(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    reference_audio,
    emotion,
    prompt_mode,
    style_text=None,
    style_weight=0,
):
```
定义一个名为`tts_fn`的函数，接受多个参数：`text`（字符串类型）、`speaker`、`sdp_ratio`、`noise_scale`、`noise_scale_w`、`length_scale`、`language`、`reference_audio`、`emotion`、`prompt_mode`、`style_text`（默认值为`None`）和`style_weight`（默认值为0）。

```
    if style_text == "":
        style_text = None
```
如果`style_text`的值为空字符串，则将其赋值为`None`。

```
    if prompt_mode == "Audio prompt":
```
如果`prompt_mode`的值等于"Audio prompt"，则执行以下代码块。
        if reference_audio == None:
            return ("Invalid audio prompt", None)
```
如果`reference_audio`为空，则返回一个元组，第一个元素是字符串"Invalid audio prompt"，第二个元素是`None`。

```
        else:
            reference_audio = load_audio(reference_audio)[1]
```
否则，调用`load_audio`函数加载`reference_audio`，并将返回结果的第二个元素赋值给`reference_audio`。

```
    else:
        reference_audio = None
```
如果`reference_audio`不为空，则将其赋值为`None`。

```
    audio_list = process_text(
        text,
        speaker,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        length_scale,
        language,
        reference_audio,
        emotion,
        style_text,
        style_weight,
    )
```
调用`process_text`函数，传入多个参数，生成一个音频列表，并将其赋值给`audio_list`变量。
audio_concat = np.concatenate(audio_list)
```
将`audio_list`中的音频数据连接起来，得到一个新的音频数据`audio_concat`。

```
return "Success", (hps.data.sampling_rate, audio_concat)
```
返回一个元组，第一个元素是字符串"Success"，第二个元素是一个元组，包含采样率`hps.data.sampling_rate`和连接后的音频数据`audio_concat`。

```
def format_utils(text, speaker):
```
定义一个名为`format_utils`的函数，接受两个参数`text`和`speaker`。

```
_text, _lang = process_auto(text)
```
调用`process_auto`函数处理`text`，得到两个结果`_text`和`_lang`。

```
res = f"[{speaker}]"
```
初始化一个字符串`res`，以`[speaker]`的格式开始。

```
for lang_s, content_s in zip(_lang, _text):
```
使用`zip`函数将`_lang`和`_text`进行迭代，每次迭代得到`lang_s`和`content_s`。

```
for lang, content in zip(lang_s, content_s):
```
使用`zip`函数将`lang_s`和`content_s`进行迭代，每次迭代得到`lang`和`content`。

```
res += f"<{lang.lower()}>{content}"
```
将`lang`转换为小写，并将`<lang>`和`content`拼接到`res`中。

```
res += "|"
```
在`res`的末尾添加一个竖线符号`|`。

```
return "mix", res[:-1]
```
返回一个元组，第一个元素是字符串"mix"，第二个元素是`res`去掉最后一个字符后的结果。

```
def load_audio(path):
```
定义一个名为`load_audio`的函数，接受一个参数`path`。

```
audio, sr = librosa.load(path, 48000)
```
使用`librosa.load`函数加载音频文件`path`，并指定采样率为48000，得到音频数据`audio`和采样率`sr`。

```
return sr, audio
```
返回一个元组，第一个元素是采样率`sr`，第二个元素是音频数据`audio`。
# 定义一个名为gr_util的函数，接受一个参数item
def gr_util(item):
    # 如果item等于"Text prompt"，返回一个字典和一个字典
    if item == "Text prompt":
        return {"visible": True, "__type__": "update"}, {
            "visible": False,
            "__type__": "update",
        }
    # 如果item不等于"Text prompt"，返回一个字典和一个字典
    else:
        return {"visible": False, "__type__": "update"}, {
            "visible": True,
            "__type__": "update",
        }


# 如果当前脚本被直接执行而不是被导入，则执行以下代码块
if __name__ == "__main__":
    # 如果config.webui_config.debug为真，则打印日志信息并设置日志级别为DEBUG
    if config.webui_config.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    # 从config.webui_config.config_path中获取超参数
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    # 如果config.json中未指定版本，则将版本设置为最新版本
version = hps.version if hasattr(hps, "version") else latest_version
```
这行代码根据条件判断选择要使用的版本号。如果`hps`对象有`version`属性，则使用`hps.version`作为版本号；否则使用`latest_version`作为版本号。

```
net_g = get_net_g(
    model_path=config.webui_config.model, version=version, device=device, hps=hps
)
```
这行代码调用`get_net_g`函数，传入模型路径、版本号、设备和`hps`对象作为参数，并将返回的结果赋值给`net_g`变量。

```
speaker_ids = hps.data.spk2id
```
这行代码将`hps.data.spk2id`赋值给`speaker_ids`变量。`hps.data.spk2id`是一个包含说话人ID的字典。

```
speakers = list(speaker_ids.keys())
```
这行代码将`speaker_ids`字典的键转换为列表，并将结果赋值给`speakers`变量。这样可以获取所有的说话人ID。

```
languages = ["ZH", "JP", "EN", "mix", "auto"]
```
这行代码创建一个包含不同语言选项的列表。列表中的元素分别表示中文、日语、英语、混合语言和自动检测语言。

```
with gr.Blocks() as app:
```
这行代码创建一个`gr.Blocks`对象，并将其赋值给`app`变量。`with`语句用于确保在代码块执行完毕后自动关闭`app`对象。

```
with gr.Row():
```
这行代码创建一个`gr.Row`对象，用于在界面中创建一行元素。

```
with gr.Column():
```
这行代码创建一个`gr.Column`对象，用于在界面中创建一列元素。

```
text = gr.TextArea(
    label="输入文本内容",
    placeholder="""
    如果你选择语言为\'mix\'，必须按照格式输入，否则报错:
        格式举例(zh是中文，jp是日语，不区分大小写；说话人举例:gongzi):
         [说话人1]<zh>你好，こんにちは！ <jp>こんにちは，世界。
         [说话人2]<zh>你好吗？<jp>元気ですか？
         [说话人3]<zh>谢谢。<jp>どういたしまして。
         ...
    另外，所有的语言选项都可以用'|'分割长段实现分句生成。
"""
)
```
这行代码创建一个`gr.TextArea`对象，用于在界面中创建一个文本输入框。`label`参数指定输入框的标签文本，`placeholder`参数指定输入框的占位文本，即在输入框为空时显示的提示信息。在这个示例中，输入框的标签文本是"输入文本内容"，占位文本是一段说明文本，提供了关于输入格式的示例和要求。

注：以上代码中的`gr`是一个模块或类的引用，具体的含义需要根据上下文来确定。
                    """,
                )
                # 创建一个名为trans的按钮，用于中翻日操作，样式为primary
                trans = gr.Button("中翻日", variant="primary")
                # 创建一个名为slicer的按钮，用于快速切分操作，样式为primary
                slicer = gr.Button("快速切分", variant="primary")
                # 创建一个名为formatter的按钮，用于检测语言并整理为MIX格式的操作，样式为primary
                formatter = gr.Button("检测语言，并整理为 MIX 格式", variant="primary")
                # 创建一个下拉菜单，用于选择说话者，初始值为speakers列表的第一个元素，标签为"Speaker"
                speaker = gr.Dropdown(
                    choices=speakers, value=speakers[0], label="Speaker"
                )
                # 创建一个Markdown组件，显示提示模式的说明文字，初始时不可见
                _ = gr.Markdown(
                    value="提示模式（Prompt mode）：可选文字提示或音频提示，用于生成文字或音频指定风格的声音。\n",
                    visible=False,
                )
                # 创建一个单选按钮组件，用于选择提示模式，初始值为"Text prompt"，初始时不可见
                prompt_mode = gr.Radio(
                    ["Text prompt", "Audio prompt"],
                    label="Prompt Mode",
                    value="Text prompt",
                    visible=False,
                )
                # 创建一个文本框组件，用于输入文字提示，初始时不可见
                text_prompt = gr.Textbox(
                    label="Text prompt",
# 创建一个文本输入框，用于用户输入生成风格的文字描述
placeholder="用文字描述生成风格。如：Happy",
value="Happy",
visible=False,
```
这段代码创建了一个文本输入框，用于用户输入生成风格的文字描述。`placeholder`参数设置了输入框的占位符，提示用户应该输入什么样的文字描述。`value`参数设置了输入框的默认值为"Happy"。`visible`参数设置了输入框的可见性为False，即初始状态下输入框不可见。

```
# 创建一个音频输入框，用于用户上传音频文件作为生成的音频的参考
audio_prompt = gr.Audio(
    label="Audio prompt", type="filepath", visible=False
)
```
这段代码创建了一个音频输入框，用于用户上传音频文件作为生成的音频的参考。`label`参数设置了输入框的标签为"Audio prompt"。`type`参数设置了输入框的类型为"filepath"，表示用户可以选择本地文件进行上传。`visible`参数设置了输入框的可见性为False，即初始状态下输入框不可见。

```
# 创建一个滑动条，用于调整生成音频的SDP比例
sdp_ratio = gr.Slider(
    minimum=0, maximum=1, value=0.5, step=0.1, label="SDP Ratio"
)
```
这段代码创建了一个滑动条，用于调整生成音频的SDP比例。`minimum`参数设置了滑动条的最小值为0，`maximum`参数设置了滑动条的最大值为1，`value`参数设置了滑动条的初始值为0.5，`step`参数设置了滑动条的步长为0.1，`label`参数设置了滑动条的标签为"SDP Ratio"。

```
# 创建一个滑动条，用于调整生成音频的噪声比例
noise_scale = gr.Slider(
    minimum=0.1, maximum=2, value=0.6, step=0.1, label="Noise"
)
```
这段代码创建了一个滑动条，用于调整生成音频的噪声比例。`minimum`参数设置了滑动条的最小值为0.1，`maximum`参数设置了滑动条的最大值为2，`value`参数设置了滑动条的初始值为0.6，`step`参数设置了滑动条的步长为0.1，`label`参数设置了滑动条的标签为"Noise"。

```
# 创建一个滑动条，用于调整生成音频的噪声权重
noise_scale_w = gr.Slider(
    minimum=0.1, maximum=2, value=0.9, step=0.1, label="Noise_W"
)
```
这段代码创建了一个滑动条，用于调整生成音频的噪声权重。`minimum`参数设置了滑动条的最小值为0.1，`maximum`参数设置了滑动条的最大值为2，`value`参数设置了滑动条的初始值为0.9，`step`参数设置了滑动条的步长为0.1，`label`参数设置了滑动条的标签为"Noise_W"。

```
# 创建一个滑动条，用于调整生成音频的长度比例
length_scale = gr.Slider(
    minimum=0.1, maximum=2, value=1.0, step=0.1, label="Length"
)
```
这段代码创建了一个滑动条，用于调整生成音频的长度比例。`minimum`参数设置了滑动条的最小值为0.1，`maximum`参数设置了滑动条的最大值为2，`value`参数设置了滑动条的初始值为1.0，`step`参数设置了滑动条的步长为0.1，`label`参数设置了滑动条的标签为"Length"。

```
# 创建一个下拉菜单，用于选择生成音频的语言
language = gr.Dropdown(
```
这段代码创建了一个下拉菜单，用于选择生成音频的语言。
# 创建下拉选择框，用于选择语言
language_select = gr.Select(
    choices=languages, value=languages[0], label="Language"
)
# 创建按钮，用于触发生成音频的操作
btn = gr.Button("生成音频！", variant="primary")
```

这段代码创建了一个下拉选择框和一个按钮。下拉选择框用于选择语言，按钮用于触发生成音频的操作。

```
with gr.Column():
    with gr.Accordion("融合文本语义", open=False):
        gr.Markdown(
            value="使用辅助文本的语意来辅助生成对话（语言保持与主文本相同）\n\n"
            "**注意**：不要使用**指令式文本**（如：开心），要使用**带有强烈情感的文本**（如：我好快乐！！！）\n\n"
            "效果较不明确，留空即为不使用该功能"
        )
        style_text = gr.Textbox(label="辅助文本")
        style_weight = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.7,
            step=0.1,
            label="Weight",
            info="主文本和辅助文本的bert混合比率，0表示仅主文本，1表示仅辅助文本",
        )
    with gr.Row():
```

这段代码创建了一个列布局，并在列布局中创建了一个手风琴组件。手风琴组件包含一个Markdown组件，用于显示关于使用辅助文本的说明。接下来，代码创建了一个文本框组件和一个滑动条组件，用于输入辅助文本和设置主文本与辅助文本的混合比率。最后，代码创建了一个行布局。
# 创建一个包含多个控件的列布局
with gr.Column():
    # 创建一个滑块控件，用于设置句间停顿的时间间隔
    interval_between_sent = gr.Slider(
        minimum=0,
        maximum=5,
        value=0.2,
        step=0.1,
        label="句间停顿(秒)，勾选按句切分才生效",
    )
    # 创建一个滑块控件，用于设置段间停顿的时间间隔
    interval_between_para = gr.Slider(
        minimum=0,
        maximum=10,
        value=1,
        step=0.1,
        label="段间停顿(秒)，需要大于句间停顿才有效",
    )
    # 创建一个复选框控件，用于选择是否按句子切分文本
    opt_cut_by_sent = gr.Checkbox(
        label="按句切分    在按段落切分的基础上再按句子切分文本"
    )
    # 创建一个按钮控件，用于触发切分生成操作
    slicer = gr.Button("切分生成", variant="primary")
# 创建一个文本框控件，用于显示状态信息
text_output = gr.Textbox(label="状态信息")
```

这段代码创建了一个包含多个控件的列布局。具体控件的作用如下：

- `interval_between_sent`：滑块控件，用于设置句间停顿的时间间隔，取值范围为0到5秒，默认值为0.2秒。
- `interval_between_para`：滑块控件，用于设置段间停顿的时间间隔，取值范围为0到10秒，默认值为1秒。
- `opt_cut_by_sent`：复选框控件，用于选择是否按句子切分文本。
- `slicer`：按钮控件，用于触发切分生成操作。
- `text_output`：文本框控件，用于显示状态信息。
audio_output = gr.Audio(label="输出音频")
```
创建一个名为`audio_output`的音频输出组件，用于显示和播放生成的音频。

```
# explain_image = gr.Image(
#     label="参数解释信息",
#     show_label=True,
#     show_share_button=False,
#     show_download_button=False,
#     value=os.path.abspath("./img/参数说明.png"),
# )
```
创建一个名为`explain_image`的图像组件，用于显示参数解释的信息。由于该代码被注释掉了，所以不会执行。

```
btn.click(
    tts_fn,
    inputs=[
        text,
        speaker,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        length_scale,
        language,
        audio_prompt,
        text_prompt,
```
当按钮`btn`被点击时，调用`tts_fn`函数，并将`text`、`speaker`、`sdp_ratio`、`noise_scale`、`noise_scale_w`、`length_scale`、`language`、`audio_prompt`和`text_prompt`作为输入传递给`tts_fn`函数。
prompt_mode,  # 设置模式参数，用于控制翻译的模式
style_text,  # 设置文本样式参数，用于控制翻译的文本样式
style_weight,  # 设置样式权重参数，用于控制翻译的样式权重
],
outputs=[text_output, audio_output],  # 设置输出参数，指定翻译结果的输出方式

trans.click(  # 点击翻译按钮，触发翻译操作
    translate,  # 指定翻译函数
    inputs=[text],  # 设置输入参数，指定待翻译的文本
    outputs=[text],  # 设置输出参数，指定翻译结果的输出方式
)

slicer.click(  # 点击切割按钮，触发切割操作
    tts_split,  # 指定切割函数
    inputs=[
        text,  # 设置输入参数，指定待切割的文本
        speaker,  # 设置说话者参数，用于指定切割的说话者
        sdp_ratio,  # 设置SDP比例参数，用于控制切割的SDP比例
        noise_scale,  # 设置噪声比例参数，用于控制切割的噪声比例
        noise_scale_w,  # 设置噪声比例权重参数，用于控制切割的噪声比例权重
                length_scale,
                language,
                opt_cut_by_sent,
                interval_between_para,
                interval_between_sent,
                audio_prompt,
                text_prompt,
                style_text,
                style_weight,
            ],
            outputs=[text_output, audio_output],
        )
```
这段代码是一个函数调用，调用了一个名为`gr_util`的函数，并传入了一系列参数`length_scale, language, opt_cut_by_sent, interval_between_para, interval_between_sent, audio_prompt, text_prompt, style_text, style_weight`。该函数的返回值会被赋值给`outputs`列表中的`text_output`和`audio_output`。

```
        prompt_mode.change(
            lambda x: gr_util(x),
            inputs=[prompt_mode],
            outputs=[text_prompt, audio_prompt],
        )
```
这段代码是一个函数调用，调用了`prompt_mode`对象的`change`方法。该方法接受一个函数作为参数，并将`prompt_mode`作为输入。函数`lambda x: gr_util(x)`会被调用，并将返回值赋值给`outputs`列表中的`text_prompt`和`audio_prompt`。

```
        audio_prompt.upload(
```
这段代码是一个函数调用，调用了`audio_prompt`对象的`upload`方法。
            lambda x: load_audio(x),
            inputs=[audio_prompt],
            outputs=[audio_prompt],
        )
```
这段代码是一个匿名函数，它接受一个参数x，并调用load_audio函数来处理x。它的输入是一个名为audio_prompt的变量，输出也是一个名为audio_prompt的变量。

```
        formatter.click(
            format_utils,
            inputs=[text, speaker],
            outputs=[language, text],
        )
```
这段代码调用了formatter对象的click方法，传入了format_utils作为参数。它的输入是text和speaker两个变量，输出是language和text两个变量。

```
    print("推理页面已开启!")
    webbrowser.open(f"http://127.0.0.1:{config.webui_config.port}")
    app.launch(share=config.webui_config.share, server_port=config.webui_config.port)
```
这段代码打印了一条消息，然后使用webbrowser模块打开了一个网页，网址是"http://127.0.0.1:{config.webui_config.port}"，其中config.webui_config.port是一个端口号。最后，它调用了app对象的launch方法，传入了一些参数来启动应用程序。
```