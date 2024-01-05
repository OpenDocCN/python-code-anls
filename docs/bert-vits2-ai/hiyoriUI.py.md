# `d:/src/tocomm/Bert-VITS2\hiyoriUI.py`

```
"""
api服务，网页后端 多版本多模型 fastapi实现
原 server_fastapi
"""
```
这是一个多版本多模型的FastAPI实现的网页后端API服务的注释。

```
import logging
```
导入logging模块，用于记录日志。

```
import gc
```
导入gc模块，用于垃圾回收。

```
import random
```
导入random模块，用于生成随机数。

```
import librosa
```
导入librosa模块，用于音频处理。

```
import gradio
```
导入gradio模块，用于构建交互式界面。

```
import numpy as np
```
导入numpy模块，并将其命名为np，用于进行数值计算。

```
import utils
```
导入utils模块，可能是自定义的工具函数或类。

```
from fastapi import FastAPI, Query, Request, File, UploadFile, Form
```
从fastapi模块中导入FastAPI、Query、Request、File、UploadFile和Form类。

```
from fastapi.responses import Response, FileResponse
```
从fastapi.responses模块中导入Response和FileResponse类。

```
from fastapi.staticfiles import StaticFiles
```
从fastapi.staticfiles模块中导入StaticFiles类，用于处理静态文件。

```
from io import BytesIO
```
从io模块中导入BytesIO类，用于处理二进制数据。

```
from scipy.io import wavfile
```
从scipy.io模块中导入wavfile类，用于处理音频文件。

```
import uvicorn
```
导入uvicorn模块，用于运行FastAPI应用。

```
import torch
```
导入torch模块，用于深度学习任务。

```
import webbrowser
```
导入webbrowser模块，用于在浏览器中打开网页。

```
import psutil
```
导入psutil模块，用于获取系统信息。
import GPUtil
```
导入GPUtil模块，用于获取GPU的相关信息。

```
from typing import Dict, Optional, List, Set, Union, Tuple
```
从typing模块中导入Dict、Optional、List、Set、Union和Tuple等类型，用于类型注解。

```
import os
```
导入os模块，用于与操作系统进行交互。

```
from tools.log import logger
```
从tools.log模块中导入logger对象，用于记录日志。

```
from urllib.parse import unquote
```
从urllib.parse模块中导入unquote函数，用于解码URL编码的字符串。

```
from infer import infer, get_net_g, latest_version
```
从infer模块中导入infer、get_net_g和latest_version函数。

```
import tools.translate as trans
```
导入tools.translate模块，并将其命名为trans，用于进行翻译操作。

```
from tools.sentence import split_by_language
```
从tools.sentence模块中导入split_by_language函数，用于根据语言切分句子。

```
from re_matching import cut_sent
```
从re_matching模块中导入cut_sent函数，用于根据正则表达式切分句子。

```
from config import config
```
从config模块中导入config对象，用于获取配置信息。

```
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```
设置环境变量TOKENIZERS_PARALLELISM为"false"，用于禁用tokenizers的并行处理。

```
class Model:
    """模型封装类"""
```
定义一个名为Model的类，用于封装模型的相关操作。
    def __init__(self, config_path: str, model_path: str, device: str, language: str):
        # 初始化函数，接收配置文件路径、模型文件路径、设备类型和语言类型作为参数
        self.config_path: str = os.path.normpath(config_path)
        # 将配置文件路径规范化，并赋值给实例变量config_path
        self.model_path: str = os.path.normpath(model_path)
        # 将模型文件路径规范化，并赋值给实例变量model_path
        self.device: str = device
        # 将设备类型赋值给实例变量device
        self.language: str = language
        # 将语言类型赋值给实例变量language
        self.hps = utils.get_hparams_from_file(config_path)
        # 调用utils模块的get_hparams_from_file函数，根据配置文件路径获取超参数，并赋值给实例变量hps
        self.spk2id: Dict[str, int] = self.hps.data.spk2id  # spk - id 映射字典
        # 将超参数中的spk2id映射字典赋值给实例变量spk2id
        self.id2spk: Dict[int, str] = dict()  # id - spk 映射字典
        # 创建一个空的id - spk映射字典，并赋值给实例变量id2spk
        for speaker, speaker_id in self.hps.data.spk2id.items():
            self.id2spk[speaker_id] = speaker
        # 遍历超参数中的spk2id映射字典，将键值对反转后赋值给实例变量id2spk
        self.version: str = (
            self.hps.version if hasattr(self.hps, "version") else latest_version
        )
        # 如果超参数中存在version属性，则将其赋值给实例变量version，否则赋值为latest_version
        self.net_g = get_net_g(
            model_path=model_path,
            version=self.version,
            device=device,
            hps=self.hps,
        )
        # 调用get_net_g函数，根据模型文件路径、版本号、设备类型和超参数获取生成器网络，并赋值给实例变量net_g
    def to_dict(self) -> Dict[str, any]:
        # 将对象的属性转换为字典形式
        return {
            "config_path": self.config_path,  # 配置文件路径
            "model_path": self.model_path,  # 模型文件路径
            "device": self.device,  # 设备
            "language": self.language,  # 语言
            "spk2id": self.spk2id,  # 说话人到ID的映射字典
            "id2spk": self.id2spk,  # ID到说话人的映射字典
            "version": self.version,  # 版本号
        }


class Models:
    def __init__(self):
        self.models: Dict[int, Model] = dict()  # 模型ID到模型对象的映射字典
        self.num = 0  # 模型数量
        # spkInfo[角色名][模型id] = 角色id
        self.spk_info: Dict[str, Dict[int, int]] = dict()  # 角色名到模型ID到角色ID的映射字典
        self.path2ids: Dict[str, Set[int]] = dict()  # 路径到模型ID集合的映射字典
```

这段代码是一个类的定义，包含了两个类：`to_dict`方法和`Models`类。

`to_dict`方法将对象的属性转换为字典形式，并返回该字典。

`Models`类有一个构造函数`__init__`，用于初始化类的实例。它包含了四个属性：

- `models`：一个字典，将模型ID映射到模型对象。
- `num`：一个整数，表示模型的数量。
- `spk_info`：一个嵌套字典，将角色名映射到模型ID到角色ID的映射字典。
- `path2ids`：一个嵌套字典，将路径映射到模型ID集合。
    def init_model(
        self, config_path: str, model_path: str, device: str, language: str
    ) -> int:
        """
        初始化并添加一个模型

        :param config_path: 模型config.json路径
        :param model_path: 模型路径
        :param device: 模型推理使用设备
        :param language: 模型推理默认语言
        """
        # 若文件不存在则不进行加载
        # 检查模型文件是否存在，如果不存在则不进行初始化
        if not os.path.isfile(model_path):
            if model_path != "":
                logger.warning(f"模型文件{model_path} 不存在，不进行初始化")
            return self.num
        # 检查配置文件是否存在，如果不存在则不进行初始化
        if not os.path.isfile(config_path):
            if config_path != "":
                logger.warning(f"配置文件{config_path} 不存在，不进行初始化")
            return self.num
```

这段代码是一个初始化并添加模型的方法。注释解释了每个参数的含义。代码首先检查模型文件和配置文件是否存在，如果不存在则不进行初始化，并返回一个整数值。
# 若路径中的模型已存在，则不添加模型，若不存在，则进行初始化。
model_path = os.path.realpath(model_path)  # 获取模型路径的绝对路径
if model_path not in self.path2ids.keys():  # 判断模型路径是否已存在于字典self.path2ids的键中
    self.path2ids[model_path] = {self.num}  # 将模型路径添加到字典self.path2ids的键中，并将其对应的值设为一个包含self.num的集合
    self.models[self.num] = Model(  # 在字典self.models中以self.num为键，创建一个Model对象作为值
        config_path=config_path,  # 使用给定的配置文件路径创建Model对象
        model_path=model_path,  # 使用给定的模型路径创建Model对象
        device=device,  # 设置Model对象的设备
        language=language,  # 设置Model对象的语言
    )
    logger.success(f"添加模型{model_path}，使用配置文件{os.path.realpath(config_path)}")  # 打印成功添加模型的日志信息
else:
    # 获取一个指向id
    m_id = next(iter(self.path2ids[model_path]))  # 从字典self.path2ids中获取模型路径对应的id，并将其赋值给变量m_id
    self.models[self.num] = self.models[m_id]  # 在字典self.models中以self.num为键，将值设为与模型路径相同的已存在的Model对象
    self.path2ids[model_path].add(self.num)  # 将self.num添加到字典self.path2ids中模型路径对应的集合中
    logger.success("模型已存在，添加模型引用。")  # 打印模型已存在的日志信息
# 添加角色信息
for speaker, speaker_id in self.models[self.num].spk2id.items():  # 遍历当前Model对象的spk2id字典的键值对
            if speaker not in self.spk_info.keys():
                self.spk_info[speaker] = {self.num: speaker_id}
            else:
                self.spk_info[speaker][self.num] = speaker_id
```
这段代码的作用是将角色信息添加到`self.spk_info`字典中。如果`speaker`不在`self.spk_info`的键中，就将其作为键，创建一个新的字典作为值，并将`self.num`作为键，`speaker_id`作为值添加到新的字典中。如果`speaker`已经在`self.spk_info`的键中，就将`self.num`作为键，`speaker_id`作为值添加到对应的字典中。

```
        # 修改计数
        self.num += 1
        return self.num - 1
```
这段代码的作用是将`self.num`的值加1，并返回加1之前的值。

```
    def del_model(self, index: int) -> Optional[int]:
        """删除对应序号的模型，若不存在则返回None"""
        if index not in self.models.keys():
            return None
        # 删除角色信息
        for speaker, speaker_id in self.models[index].spk2id.items():
            self.spk_info[speaker].pop(index)
            if len(self.spk_info[speaker]) == 0:
                # 若对应角色的所有模型都被删除，则清除该角色信息
                self.spk_info.pop(speaker)
        # 删除路径信息
        model_path = os.path.realpath(self.models[index].model_path)
```
这段代码的作用是删除指定序号的模型。首先，判断`index`是否在`self.models`的键中，如果不在，则返回`None`。然后，遍历`self.models[index].spk2id`字典的键值对，将对应的角色信息从`self.spk_info`字典中删除。如果删除后对应角色的模型数量为0，则将该角色信息从`self.spk_info`字典中清除。最后，获取指定序号模型的路径信息，并赋值给`model_path`变量。
        self.path2ids[model_path].remove(index)
```
这行代码从self.path2ids字典中删除指定模型路径model_path对应的索引index。

```
        if len(self.path2ids[model_path]) == 0:
```
这行代码判断self.path2ids字典中指定模型路径model_path对应的索引列表是否为空。

```
            self.path2ids.pop(model_path)
```
这行代码从self.path2ids字典中删除指定模型路径model_path。

```
            logger.success(f"删除模型{model_path}, id = {index}")
```
这行代码使用logger记录成功删除模型的日志信息。

```
        else:
            logger.success(f"删除模型引用{model_path}, id = {index}")
```
这行代码使用logger记录成功删除模型引用的日志信息。

```
        self.models.pop(index)
```
这行代码从self.models列表中删除指定索引index对应的模型。

```
        gc.collect()
```
这行代码手动触发垃圾回收，释放不再使用的内存。

```
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```
这行代码如果CUDA可用，则清空CUDA缓存，释放GPU显存。

```
        return index
```
这行代码返回删除的模型索引index。

```
    def get_models(self):
        """获取所有模型"""
        return self.models
```
这个函数用于获取所有的模型，返回self.models列表。

```
if __name__ == "__main__":
    app = FastAPI()
```
这段代码是程序的入口点，当直接运行该脚本时，创建一个FastAPI应用实例app。
    app.logger = logger
    # 将logger对象赋值给app.logger，用于记录日志信息

    logger.info("开始挂载网页页面")
    # 记录日志信息，表示开始挂载网页页面

    StaticDir: str = "./Web"
    # 设置静态文件目录的路径

    if not os.path.isdir(StaticDir):
        # 判断静态文件目录是否存在
        logger.warning(
            "缺少网页资源，无法开启网页页面，如有需要请在 https://github.com/jiangyuxiaoxiao/Bert-VITS2-UI 或者Bert-VITS对应版本的release页面下载"
        )
        # 如果静态文件目录不存在，则记录警告日志，提示缺少网页资源
    else:
        dirs = [fir.name for fir in os.scandir(StaticDir) if fir.is_dir()]
        # 获取静态文件目录下的所有子目录名
        files = [fir.name for fir in os.scandir(StaticDir) if fir.is_dir()]
        # 获取静态文件目录下的所有文件名
        for dirName in dirs:
            # 遍历每个子目录名
            app.mount(
                f"/{dirName}",
                StaticFiles(directory=f"./{StaticDir}/{dirName}"),
                name=dirName,
            )
            # 将每个子目录挂载到应用程序中，使其可以通过URL访问

    loaded_models = Models()
    # 创建Models对象，用于加载模型

    logger.info("开始加载模型")
    # 记录日志信息，表示开始加载模型
    models_info = config.server_config.models
    # 遍历models_info列表中的每个元素，将其作为参数传递给loaded_models.init_model()函数
    for model_info in models_info:
        loaded_models.init_model(
            config_path=model_info["config"],
            model_path=model_info["model"],
            device=model_info["device"],
            language=model_info["language"],
        )

    @app.get("/")
    async def index():
        # 返回一个文件响应，将"./Web/index.html"文件作为内容
        return FileResponse("./Web/index.html")

    async def _voice(
        text: str,
        model_id: int,
        speaker_name: str,
        speaker_id: int,
        sdp_ratio: float,
        noise: float,
        ):
        # 这是一个异步函数，接受多个参数，用于处理语音相关的操作
        noisew: float,  # 噪声权重，类型为浮点数
        length: float,  # 音频长度，类型为浮点数
        language: str,  # 语言，类型为字符串
        auto_translate: bool,  # 是否自动翻译，类型为布尔值
        auto_split: bool,  # 是否自动分割，类型为布尔值
        emotion: Optional[Union[int, str]] = None,  # 情感，类型为可选的整数或字符串，默认值为None
        reference_audio=None,  # 参考音频，默认值为None
        style_text: Optional[str] = None,  # 风格文本，类型为可选的字符串，默认值为None
        style_weight: float = 0.7,  # 风格权重，类型为浮点数，默认值为0.7
    ) -> Union[Response, Dict[str, any]]:  # 返回值类型为Response或字典

        """TTS实现函数"""

        # 检查
        # 检查模型是否存在
        if model_id not in loaded_models.models.keys():  # 如果模型ID不在已加载的模型字典中
            logger.error(f"/voice 请求错误：模型model_id={model_id}未加载")  # 记录错误日志
            return {"status": 10, "detail": f"模型model_id={model_id}未加载"}  # 返回错误信息字典
        # 检查是否提供speaker
        if speaker_name is None and speaker_id is None:  # 如果未提供speaker_name和speaker_id
            logger.error("/voice 请求错误：推理请求未提供speaker_name或speaker_id")  # 记录错误日志
# 检查是否提供了speaker_name或speaker_id，如果没有则返回错误信息
if speaker_name is None and speaker_id is None:
    return {"status": 11, "detail": "请提供speaker_name或speaker_id"}

# 如果没有提供speaker_name，则检查speaker_id是否存在于加载的模型中
if speaker_name is None:
    if speaker_id not in loaded_models.models[model_id].id2spk.keys():
        logger.error(f"/voice 请求错误：角色speaker_id={speaker_id}不存在")
        return {"status": 12, "detail": f"角色speaker_id={speaker_id}不存在"}
    # 如果存在，则根据speaker_id获取对应的speaker_name
    speaker_name = loaded_models.models[model_id].id2spk[speaker_id]

# 检查speaker_name是否存在于加载的模型中
if speaker_name not in loaded_models.models[model_id].spk2id.keys():
    logger.error(f"/voice 请求错误：角色speaker_name={speaker_name}不存在")
    return {"status": 13, "detail": f"角色speaker_name={speaker_name}不存在"}

# 如果没有传入language，则使用默认语言
if language is None:
    language = loaded_models.models[model_id].language

# 如果auto_translate为True，则检查language是否为"auto"或"mix"，如果是则返回错误信息
if auto_translate:
    if language == "auto" or language == "mix":
        logger.error(
            f"/voice 请求错误：请勿同时使用language = {language}与auto_translate模式"
        )
                return {
                    "status": 20,
                    "detail": f"请勿同时使用language = {language}与auto_translate模式",
                }
```
这段代码是一个条件判断语句的一部分。如果满足条件 `language = {language}` 与 `auto_translate` 模式同时使用，那么会返回一个字典，其中包含键值对 `"status": 20` 和 `"detail": f"请勿同时使用language = {language}与auto_translate模式"`。

```
            text = trans.translate(Sentence=text, to_Language=language.lower())
```
这行代码调用了一个名为 `trans.translate` 的函数，将 `text` 作为参数传入，并将翻译结果赋值给 `text` 变量。

```
        if reference_audio is not None:
            ref_audio = BytesIO(await reference_audio.read())
            # 2.2 适配
            if loaded_models.models[model_id].version == "2.2":
                ref_audio, _ = librosa.load(ref_audio, 48000)
        else:
            ref_audio = reference_audio
```
这段代码是一个条件判断语句的一部分。如果 `reference_audio` 不为空，则将其读取为字节流，并将结果赋值给 `ref_audio` 变量。然后，如果 `loaded_models.models[model_id].version` 的值为 "2.2"，则使用 `librosa.load` 函数加载 `ref_audio`，采样率为 48000，并将结果赋值给 `ref_audio` 变量。如果 `reference_audio` 为空，则将 `reference_audio` 的值赋值给 `ref_audio` 变量。

```
        text2 = text.replace("\n", "").lstrip()
        texts: List[str] = text2.split("||")
```
这段代码对 `text` 进行处理，首先使用 `replace` 函数将换行符替换为空字符串，然后使用 `lstrip` 函数去除字符串开头的空格。将处理后的结果赋值给 `text2` 变量。接着，使用 `split` 函数将 `text2` 按照 "||" 进行切分，并将切分后的结果赋值给 `texts` 变量。

```
        if language == "MIX":
```
这段代码是一个条件判断语句的一部分。如果 `language` 的值等于 "MIX"，则执行下面的代码块。
# 创建一个空列表，用于存储元组，每个元组包含三个字符串
text_language_speakers: List[Tuple[str, str, str]] = []

# 遍历texts列表中的每个元素
for _text in texts:
    # 按照"["字符分割_text字符串，得到多个块
    speaker_pieces = _text.split("[")  # 按说话人分割多块
    
    # 遍历speaker_pieces列表中的每个块
    for speaker_piece in speaker_pieces:
        # 如果块为空字符串，则跳过本次循环
        if speaker_piece == "":
            continue
        
        # 按照"]"字符分割speaker_piece字符串，得到两个块
        speaker_piece2 = speaker_piece.split("]")
        
        # 如果块的数量不等于2，则返回一个包含错误信息的字典
        if len(speaker_piece2) != 2:
            return {
                "status": 21,
                "detail": f"MIX语法错误",
            }
        
        # 获取第一个块，并去除首尾的空格，作为说话人的字符串
        speaker = speaker_piece2[0].strip()
        
        # 按照"<"字符分割第二个块，得到多个块
        lang_pieces = speaker_piece2[1].split("<")
        
        # 遍历lang_pieces列表中的每个块
        for lang_piece in lang_pieces:
            # 如果块为空字符串，则跳过本次循环
            if lang_piece == "":
                continue
            
            # 按照">"字符分割lang_piece字符串，得到两个块
            lang_piece2 = lang_piece.split(">")
            
            # 如果块的数量不等于2，则返回一个包含错误信息的字典
            if len(lang_piece2) != 2:
                return {
                    "status": 21,
                    "detail": f"MIX语法错误",
                }
            
            # 将说话人、语言和说话内容作为元组添加到text_language_speakers列表中
            text_language_speakers.append((speaker, lang_piece2[0], lang_piece2[1]))

# 如果代码执行到这里，表示没有发生错误，返回text_language_speakers列表
return text_language_speakers
                                "status": 21,
                                "detail": f"MIX语法错误",
                            }
```
这段代码是一个字典，用于表示一个错误的状态和详细信息。

```
                        lang = lang_piece2[0].strip()
                        if lang.upper() not in ["ZH", "EN", "JP"]:
                            return {
                                "status": 21,
                                "detail": f"MIX语法错误",
                            }
```
这段代码用于检查语言是否为"ZH"、"EN"或"JP"，如果不是，则返回一个错误的状态和详细信息的字典。

```
                        t = lang_piece2[1]
                        text_language_speakers.append((t, lang.upper(), speaker))
```
这段代码将语言、文本和说话者添加到一个列表中。

```
        elif language == "AUTO":
            text_language_speakers: List[Tuple[str, str, str]] = [
                (final_text, language.upper().replace("JA", "JP"), speaker_name)
                for sub_list in [
                    split_by_language(_text, target_languages=["zh", "ja", "en"])
                    for _text in texts
                    if _text != ""
                ]
```
这段代码是一个条件语句，如果语言是"AUTO"，则根据文本内容和目标语言列表创建一个包含文本、语言和说话者的元组的列表。
# 如果 sub_list 不为空，则遍历 sub_list 中的每个元素，将 final_text 和 language 组成元组，并添加到列表中
# 这里使用了列表推导式和条件判断语句
text_language_speakers: List[Tuple[str, str, str]] = [
    (final_text, language, speaker_name) for final_text, language in sub_list
    if final_text != ""
]
# 如果 sub_list 为空，则遍历 texts 列表中的每个元素，将 _text 和 language 和 speaker_name 组成元组，并添加到列表中
# 这里使用了列表推导式和条件判断语句
else:
    text_language_speakers: List[Tuple[str, str, str]] = [
        (_text, language, speaker_name) for _text in texts if _text != ""
    ]

# 如果 auto_split 为真，则遍历 text_language_speakers 列表中的每个元素，将 _text 切分成多个句子，并将切分后的句子和 lang 和 speaker 组成元组，并添加到列表中
# 这里使用了列表推导式和嵌套的 for 循环
if auto_split:
    text_language_speakers: List[Tuple[str, str, str]] = [
        (final_text, lang, speaker)
        for _text, lang, speaker in text_language_speakers
        for final_text in cut_sent(_text)
    ]

# 创建一个空列表 audios
audios = []
# 使用 torch.no_grad() 上下文管理器，禁用梯度计算
with torch.no_grad():
    # 遍历 text_language_speakers 列表中的每个元素，将 _text、lang 和 speaker 作为参数传递给 infer 函数，并将返回的结果添加到 audios 列表中
    for _text, lang, speaker in text_language_speakers:
        audios.append(
            infer(
                _text, lang, speaker
            )
        )
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```

需要注释的代码：

```
# 调用某个函数，传入多个参数，其中参数的含义如下：
# _text: 文本数据
# sdp_ratio: sdp 比例
# noise_scale: 噪声比例
# noise_scale_w: 噪声比例（w）
# length_scale: 长度比例
# sid: 说话人 ID
# language: 语言
# hps: 加载的模型的超参数
# net_g: 加载的模型的生成器网络
# device: 加载的模型所在的设备
# emotion: 情感
# reference_audio: 参考音频
# style_text: 风格文本
# style_weight: 风格权重
audio = gradio.processing_utils.convert_to_16_bit_wav(
    gradio.processing_utils.postprocess(
        gradio.processing_utils.preprocess(
            text=_text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise,
            noise_scale_w=noisew,
            length_scale=length,
            sid=speaker,
            language=lang,
            hps=loaded_models.models[model_id].hps,
            net_g=loaded_models.models[model_id].net_g,
            device=loaded_models.models[model_id].device,
            emotion=emotion,
            reference_audio=ref_audio,
            style_text=style_text,
            style_weight=style_weight,
        )
    )
)
# audios.append(np.zeros(int(44100 * 0.2)))
# audios.pop()
# 将多个音频片段连接起来
audio = np.concatenate(audios)
# 将音频转换为 16 位 WAV 格式
audio = gradio.processing_utils.convert_to_16_bit_wav(audio)
        with BytesIO() as wavContent:
```
使用`BytesIO()`创建一个字节流对象`wavContent`，并将其赋值给`with`语句的上下文管理器。

```
            wavfile.write(
                wavContent, loaded_models.models[model_id].hps.data.sampling_rate, audio
            )
```
使用`wavfile.write()`函数将音频数据`audio`写入到字节流对象`wavContent`中。`loaded_models.models[model_id].hps.data.sampling_rate`表示模型的采样率。

```
            response = Response(content=wavContent.getvalue(), media_type="audio/wav")
```
创建一个`Response`对象`response`，将字节流对象`wavContent`的值作为内容，媒体类型设置为"audio/wav"。

```
            return response
```
返回`response`对象作为响应结果。

```
    @app.post("/voice")
    async def voice(
        request: Request,  # fastapi自动注入
        text: str = Form(...),
        model_id: int = Query(..., description="模型ID"),  # 模型序号
        speaker_name: str = Query(
            None, description="说话人名"
        ),  # speaker_name与 speaker_id二者选其一
        speaker_id: int = Query(None, description="说话人id，与speaker_name二选一"),
        sdp_ratio: float = Query(0.2, description="SDP/DP混合比"),
        noise: float = Query(0.2, description="感情"),
        noisew: float = Query(0.9, description="音素长度"),
        length: float = Query(1, description="语速"),
```
定义一个名为`voice`的异步函数，用于处理POST请求。函数的参数包括`request`（自动注入的Request对象）、`text`（字符串类型，通过Form方式传递）、`model_id`（整数类型，通过Query方式传递，表示模型ID）、`speaker_name`（字符串类型，通过Query方式传递，表示说话人名，与speaker_id二选一）、`speaker_id`（整数类型，通过Query方式传递，表示说话人ID，与speaker_name二选一）、`sdp_ratio`（浮点数类型，通过Query方式传递，表示SDP/DP混合比，默认值为0.2）、`noise`（浮点数类型，通过Query方式传递，表示感情，默认值为0.2）、`noisew`（浮点数类型，通过Query方式传递，表示音素长度，默认值为0.9）、`length`（浮点数类型，通过Query方式传递，表示语速，默认值为1）。
language: str = Query(None, description="语言"),  # 若不指定使用语言则使用默认值
```
这行代码定义了一个名为`language`的变量，类型为字符串（`str`），并使用`Query`装饰器指定了默认值为`None`，描述为"语言"。

```
auto_translate: bool = Query(False, description="自动翻译"),
```
这行代码定义了一个名为`auto_translate`的变量，类型为布尔值（`bool`），并使用`Query`装饰器指定了默认值为`False`，描述为"自动翻译"。

```
auto_split: bool = Query(False, description="自动切分"),
```
这行代码定义了一个名为`auto_split`的变量，类型为布尔值（`bool`），并使用`Query`装饰器指定了默认值为`False`，描述为"自动切分"。

```
emotion: Optional[Union[int, str]] = Query(None, description="emo"),
```
这行代码定义了一个名为`emotion`的变量，类型为可选的整数或字符串（`Optional[Union[int, str]]`），并使用`Query`装饰器指定了默认值为`None`，描述为"emo"。

```
reference_audio: UploadFile = File(None),
```
这行代码定义了一个名为`reference_audio`的变量，类型为`UploadFile`，并使用`File`装饰器指定了默认值为`None`。

```
style_text: Optional[str] = Form(None, description="风格文本"),
```
这行代码定义了一个名为`style_text`的变量，类型为可选的字符串（`Optional[str]`），并使用`Form`装饰器指定了默认值为`None`，描述为"风格文本"。

```
style_weight: float = Query(0.7, description="风格权重"),
```
这行代码定义了一个名为`style_weight`的变量，类型为浮点数（`float`），并使用`Query`装饰器指定了默认值为`0.7`，描述为"风格权重"。

```
"""语音接口，若需要上传参考音频请仅使用post请求"""
```
这是一个函数的文档字符串（docstring），用于描述函数的功能和使用方法。

```
logger.info(
    f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )} text={text}"
)
```
这行代码使用日志记录器（`logger`）记录一条信息。它包含了请求的客户端主机和端口，请求的查询参数，以及一个名为`text`的变量的值。

```
return await _voice(
    text=text,
    model_id=model_id,
    speaker_name=speaker_name,
    speaker_id=speaker_id,
    sdp_ratio=sdp_ratio,
    noise=noise,
    noisew=noisew,
)
```
这行代码调用名为`_voice`的函数，并传递了多个参数。函数调用的结果将作为返回值。
@app.get("/voice")
async def voice(
    request: Request,  # fastapi自动注入，表示请求对象
    text: str = Query(..., description="输入文字"),  # 输入的文字内容
    model_id: int = Query(..., description="模型ID"),  # 模型的ID序号
    speaker_name: str = Query(
        None, description="说话人名"
    ),  # 说话人的姓名，可选参数
    speaker_id: int = Query(None, description="说话人id，与speaker_name二选一"),  # 说话人的ID，可选参数
    sdp_ratio: float = Query(0.2, description="SDP/DP混合比"),  # SDP/DP混合比例，默认为0.2
)
        noise: float = Query(0.2, description="感情"),
        noisew: float = Query(0.9, description="音素长度"),
        length: float = Query(1, description="语速"),
        language: str = Query(None, description="语言"),  # 若不指定使用语言则使用默认值
        auto_translate: bool = Query(False, description="自动翻译"),
        auto_split: bool = Query(False, description="自动切分"),
        emotion: Optional[Union[int, str]] = Query(None, description="emo"),
        style_text: Optional[str] = Query(None, description="风格文本"),
        style_weight: float = Query(0.7, description="风格权重"),
    ):
```
这段代码是一个函数的参数列表。它定义了函数的输入参数，并为每个参数指定了默认值和描述。参数的含义如下：
- `noise`：浮点数类型，表示感情的程度，默认值为0.2。
- `noisew`：浮点数类型，表示音素长度，默认值为0.9。
- `length`：浮点数类型，表示语速，默认值为1。
- `language`：字符串类型，表示语言，默认值为None。如果不指定语言，则使用默认值。
- `auto_translate`：布尔类型，表示是否自动翻译，默认值为False。
- `auto_split`：布尔类型，表示是否自动切分，默认值为False。
- `emotion`：可选的整数或字符串类型，表示情感，默认值为None。
- `style_text`：可选的字符串类型，表示风格文本，默认值为None。
- `style_weight`：浮点数类型，表示风格权重，默认值为0.7。

```
        """语音接口，不建议使用"""
        logger.info(
            f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )}"
        )
```
这是一个函数的文档字符串，用于描述函数的功能和使用方法。在这个例子中，它说明了这个函数是一个语音接口，但不建议使用。文档字符串通常用来提供函数的说明和帮助信息。

```
        return await _voice(
            text=text,
            model_id=model_id,
            speaker_name=speaker_name,
            speaker_id=speaker_id,
            sdp_ratio=sdp_ratio,
```
这是函数的返回语句。它调用了名为`_voice`的函数，并传递了一些参数。函数将在异步模式下执行，并返回结果。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```

```
@app.get("/models/info")
def get_loaded_models_info(request: Request):
    """获取已加载模型信息"""
    
    result: Dict[str, Dict] = dict()
    for key, model in loaded_models.models.items():
        result[str(key)] = model.to_dict()
    return result
```

注释解释：

1. `noise=noise, noisew=noisew, length=length, language=language, auto_translate=auto_translate, auto_split=auto_split, emotion=emotion, style_text=style_text, style_weight=style_weight,` - 这是一个函数调用，传递了多个参数给函数。参数的具体含义需要根据上下文来确定。
2. `@app.get("/models/info")` - 这是一个装饰器，将下面的函数注册为一个 GET 请求的处理函数。路径为"/models/info"。
3. `def get_loaded_models_info(request: Request):` - 这是一个函数定义，定义了一个名为`get_loaded_models_info`的函数，接受一个`Request`类型的参数`request`。
4. `"""获取已加载模型信息"""` - 这是函数的文档字符串，用于描述函数的功能和用法。
5. `result: Dict[str, Dict] = dict()` - 这是一个变量定义，定义了一个名为`result`的字典变量，键的类型为字符串，值的类型为字典，并初始化为空字典。
6. `for key, model in loaded_models.models.items():` - 这是一个循环语句，遍历`loaded_models.models`字典的键值对，将键赋值给`key`变量，将值赋值给`model`变量。
7. `result[str(key)] = model.to_dict()` - 将`model`对象转换为字典，并将其添加到`result`字典中，键为`key`的字符串形式。
8. `return result` - 返回`result`字典作为函数的结果。
    @app.get("/models/delete")
    def delete_model(
        request: Request, model_id: int = Query(..., description="删除模型id")
    ):
        """删除指定模型"""
        # 记录请求的客户端信息和查询参数
        logger.info(
            f"{request.client.host}:{request.client.port}/models/delete  { unquote(str(request.query_params) )}"
        )
        # 调用loaded_models.del_model()方法删除指定模型
        result = loaded_models.del_model(model_id)
        # 如果删除结果为None，表示模型不存在，返回错误信息
        if result is None:
            logger.error(f"/models/delete 模型删除错误：模型{model_id}不存在，删除失败")
            return {"status": 14, "detail": f"模型{model_id}不存在，删除失败"}

        # 返回删除成功的信息
        return {"status": 0, "detail": "删除成功"}

    @app.get("/models/add")
    def add_model(
        request: Request,
        model_path: str = Query(..., description="添加模型路径"),
        config_path: str = Query(
```

- `@app.get("/models/delete")`: 定义一个GET请求的路由，路径为"/models/delete"。
- `def delete_model(request: Request, model_id: int = Query(..., description="删除模型id")):`: 定义一个名为`delete_model`的函数，接受一个`Request`对象和一个`model_id`参数，`model_id`参数是一个整数类型，使用`Query`装饰器指定了参数的描述信息为"删除模型id"。
- `"""删除指定模型"""`: 函数的文档字符串，用于描述函数的功能。
- `logger.info(...)`: 使用日志记录器记录请求的客户端信息和查询参数。
- `result = loaded_models.del_model(model_id)`: 调用`loaded_models`对象的`del_model`方法删除指定模型，并将结果赋值给`result`变量。
- `if result is None:`: 如果删除结果为None，表示模型不存在。
- `logger.error(...)`: 使用日志记录器记录模型删除错误的信息。
- `return {"status": 14, "detail": f"模型{model_id}不存在，删除失败"}`: 返回一个字典，表示模型删除失败的详细信息。
- `return {"status": 0, "detail": "删除成功"}`: 返回一个字典，表示模型删除成功的信息。

```
    @app.get("/models/add")
    def add_model(
        request: Request,
        model_path: str = Query(..., description="添加模型路径"),
        config_path: str = Query(
```

- `@app.get("/models/add")`: 定义一个GET请求的路由，路径为"/models/add"。
- `def add_model(request: Request, model_path: str = Query(..., description="添加模型路径"), config_path: str = Query(`: 定义一个名为`add_model`的函数，接受一个`Request`对象和两个参数`model_path`和`config_path`，`model_path`和`config_path`都是字符串类型，使用`Query`装饰器指定了参数的描述信息。
- `"""添加指定模型"""`: 函数的文档字符串，用于描述函数的功能。
None, description="添加模型配置文件路径，不填则使用./config.json或../config.json"
```
这是一个函数参数，用于指定模型的配置文件路径。如果不填，则默认使用当前目录下的`config.json`文件或上级目录下的`config.json`文件。

```
device: str = Query("cuda", description="推理使用设备")
```
这是一个函数参数，用于指定推理使用的设备。默认值为"cuda"，表示使用GPU进行推理。

```
language: str = Query("ZH", description="模型默认语言")
```
这是一个函数参数，用于指定模型的默认语言。默认值为"ZH"，表示中文。

```
"""添加指定模型：允许重复添加相同路径模型，且不重复占用内存"""
```
这是函数的文档字符串，用于描述函数的功能。该函数用于添加指定的模型，允许重复添加相同路径的模型，并且不会重复占用内存。

```
logger.info(
    f"{request.client.host}:{request.client.port}/models/add  { unquote(str(request.query_params) )}"
)
```
这行代码使用日志记录器记录了一个信息级别的日志。它打印了请求的客户端主机和端口，以及请求的查询参数。

```
if config_path is None:
    model_dir = os.path.dirname(model_path)
    if os.path.isfile(os.path.join(model_dir, "config.json")):
        config_path = os.path.join(model_dir, "config.json")
    elif os.path.isfile(os.path.join(model_dir, "../config.json")):
        config_path = os.path.join(model_dir, "../config.json")
    else:
        logger.error("/models/add 模型添加失败：未在模型所在目录以及上级目录找到config.json文件")
        return {
            "status": 15,
            "detail": "查询未传入配置文件路径，同时默认路径./与../中不存在配置文件config.json。",
```
这段代码用于确定模型的配置文件路径。如果`config_path`参数为空，则根据`model_path`确定模型所在的目录，并检查该目录下是否存在`config.json`文件或上级目录中是否存在`config.json`文件。如果都不存在，则记录一个错误日志并返回一个包含错误信息的字典。

以上是对给定代码的注释解释。
        }
```
这是一个代码块的结束标志。

```
try:
    model_id = loaded_models.init_model(
        config_path=config_path,
        model_path=model_path,
        device=device,
        language=language,
    )
```
尝试初始化模型，并将返回的模型ID赋值给变量model_id。init_model函数接受config_path、model_path、device和language作为参数。

```
except Exception:
    logging.exception("模型加载出错")
    return {
        "status": 16,
        "detail": "模型加载出错，详细查看日志",
    }
```
如果初始化模型时发生异常，记录异常信息并返回一个包含错误状态码和详细信息的字典。

```
return {
    "status": 0,
    "detail": "模型添加成功",
    "Data": {
        "model_id": model_id,
        "model_info": loaded_models.models[model_id].to_dict(),
    }
```
如果模型初始化成功，返回一个包含成功状态码、详细信息和模型ID以及模型信息的字典。模型信息通过调用loaded_models.models[model_id].to_dict()方法获取。
            },
        }

    def _get_all_models(root_dir: str = "Data", only_unloaded: bool = False):
        """从root_dir搜索获取所有可用模型"""
        result: Dict[str, List[str]] = dict()
        files = os.listdir(root_dir) + ["."]
        # 遍历root_dir目录下的文件和文件夹
        for file in files:
            # 判断是否是文件夹
            if os.path.isdir(os.path.join(root_dir, file)):
                sub_dir = os.path.join(root_dir, file)
                # 搜索 "sub_dir" 、 "sub_dir/models" 两个路径
                result[file] = list()
                sub_files = os.listdir(sub_dir)
                model_files = []
                # 遍历sub_dir目录下的文件和文件夹
                for sub_file in sub_files:
                    relpath = os.path.realpath(os.path.join(sub_dir, sub_file))
                    # 如果only_unloaded为True，并且文件路径已经加载过，则跳过
                    if only_unloaded and relpath in loaded_models.path2ids.keys():
                        continue
                    # 如果文件以".pth"结尾，并且以"G_"开头，并且是一个文件
                    if sub_file.endswith(".pth") and sub_file.startswith("G_") and os.path.isfile(relpath):
# 创建一个空列表，用于存储模型文件名
model_files = []

# 对模型文件按步数排序
# 使用lambda函数作为排序的key，将文件名中的数字部分提取出来进行排序
# 如果文件名中的数字部分是一个整数，则按照整数进行排序
# 如果文件名中的数字部分不是一个整数，则将其视为最大值进行排序
model_files = sorted(
    model_files,
    key=lambda pth: int(pth.lstrip("G_").rstrip(".pth"))
    if pth.lstrip("G_").rstrip(".pth").isdigit()
    else 10**10,
)

# 将排序后的模型文件列表存入结果字典中
result[file] = model_files

# 拼接出模型文件所在的目录路径
models_dir = os.path.join(sub_dir, "models")

# 创建一个空列表，用于存储模型文件名
model_files = []

# 判断模型文件所在的目录是否存在
if os.path.isdir(models_dir):
    # 获取模型文件目录下的所有文件名
    sub_files = os.listdir(models_dir)
    # 遍历模型文件目录下的所有文件名
    for sub_file in sub_files:
        # 获取模型文件的绝对路径
        relpath = os.path.realpath(os.path.join(models_dir, sub_file))
        # 如果only_unloaded为True，并且模型文件已经加载过，则跳过当前文件
        if only_unloaded and relpath in loaded_models.path2ids.keys():
            continue
        # 如果模型文件以".pth"结尾，并且以"G_"开头，则将其添加到模型文件列表中
        if sub_file.endswith(".pth") and sub_file.startswith("G_"):
            # 判断模型文件是否存在
            if os.path.isfile(os.path.join(models_dir, sub_file)):
                # 将模型文件名添加到模型文件列表中
                model_files.append(f"models/{sub_file}")
# 对模型文件按步数排序
model_files = sorted(
    model_files,
    key=lambda pth: int(pth.lstrip("models/G_").rstrip(".pth"))
    if pth.lstrip("models/G_").rstrip(".pth").isdigit()
    else 10**10,
)
result[file] += model_files
```
这段代码的作用是对模型文件进行排序。首先，使用`sorted()`函数对`model_files`列表进行排序。排序的依据是一个lambda函数，该函数将文件名进行处理，去掉开头的"models/G_"和结尾的".pth"，然后将剩下的部分转换为整数。如果转换成功，则使用转换后的整数作为排序依据；如果转换失败，则使用10的10次方作为排序依据。排序后的结果赋值给`model_files`。然后，将排序后的`model_files`列表添加到`result[file]`列表中。

```
if len(result[file]) == 0:
    result.pop(file)
```
这段代码的作用是判断`result[file]`列表是否为空。如果为空，则使用`result.pop(file)`将`result`字典中的`file`键值对删除。

```
return result
```
这行代码的作用是返回`result`字典作为函数的结果。

```
@app.get("/models/get_unloaded")
def get_unloaded_models_info(
    request: Request, root_dir: str = Query("Data", description="搜索根目录")
):
    """获取未加载模型"""
    logger.info(
        f"{request.client.host}:{request.client.port}/models/get_unloaded  { unquote(str(request.query_params) )}"
    )
```
这段代码定义了一个名为`get_unloaded_models_info`的函数，并将其绑定到`/models/get_unloaded`路径上。该函数接受一个`Request`对象和一个名为`root_dir`的字符串参数，默认值为"Data"。函数的作用是获取未加载的模型信息。在函数体内，使用`logger.info()`函数记录了请求的客户端地址和端口以及查询参数的字符串表示。函数的注释说明了函数的作用是获取未加载模型。
@app.get("/models/get_local")
def get_local_models_info(
    request: Request, root_dir: str = Query("Data", description="搜索根目录")
):
    """获取全部本地模型"""
    # 记录请求的客户端IP和端口以及查询参数
    logger.info(
        f"{request.client.host}:{request.client.port}/models/get_local  { unquote(str(request.query_params) )}"
    )
    # 调用_get_all_models函数获取全部本地模型信息
    return _get_all_models(root_dir, only_unloaded=False)
```

```
@app.get("/status")
def get_status():
    """获取电脑运行状态"""
    # 获取CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    # 获取内存信息
    memory_info = psutil.virtual_memory()
    # 获取总内存大小
    memory_total = memory_info.total
    # 获取可用内存大小
    memory_available = memory_info.available
```

这段代码是一个基于FastAPI框架的Web应用程序。其中的注释解释了每个函数的作用。

`get_local_models_info`函数是一个GET请求的处理函数，用于获取全部本地模型的信息。它接受一个`Request`对象和一个`root_dir`参数作为输入。`root_dir`参数是一个字符串类型的路径，用于指定搜索根目录。函数内部使用`logger.info`记录了请求的客户端IP和端口以及查询参数。然后调用`_get_all_models`函数获取全部本地模型的信息，并返回结果。

`get_status`函数是另一个GET请求的处理函数，用于获取电脑的运行状态。函数内部使用`psutil`库获取了CPU的使用率和内存的信息，包括总内存大小和可用内存大小。
        memory_used = memory_info.used
```
这行代码将`memory_info.used`的值赋给变量`memory_used`，表示当前内存使用量。

```
        memory_percent = memory_info.percent
```
这行代码将`memory_info.percent`的值赋给变量`memory_percent`，表示当前内存使用百分比。

```
        gpuInfo = []
```
这行代码创建一个空列表`gpuInfo`，用于存储GPU信息。

```
        devices = ["cpu"]
```
这行代码创建一个包含字符串"cpu"的列表`devices`，表示使用CPU。

```
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
```
这段代码使用`torch.cuda.device_count()`获取当前可用的GPU数量，并将每个GPU的名称（"cuda:0"、"cuda:1"等）添加到`devices`列表中。

```
        gpus = GPUtil.getGPUs()
```
这行代码调用`GPUtil.getGPUs()`函数获取当前系统中的GPU信息，并将其赋给变量`gpus`。

```
        for gpu in gpus:
            gpuInfo.append(
                {
                    "gpu_id": gpu.id,
                    "gpu_load": gpu.load,
                    "gpu_memory": {
                        "total": gpu.memoryTotal,
                        "used": gpu.memoryUsed,
                        "free": gpu.memoryFree,
                    },
                }
            )
```
这段代码遍历`gpus`列表中的每个GPU对象，将每个GPU的ID、负载情况和内存信息（总内存、已使用内存、可用内存）以字典的形式添加到`gpuInfo`列表中。

```
        return {
            "memory_used": memory_used,
            "memory_percent": memory_percent,
            "gpu_info": gpuInfo,
            "devices": devices,
        }
```
这段代码返回一个字典，包含内存使用量、内存使用百分比、GPU信息和设备列表。
"devices": devices,
"cpu_percent": cpu_percent,
"memory_total": memory_total,
"memory_available": memory_available,
"memory_used": memory_used,
"memory_percent": memory_percent,
"gpu": gpuInfo,
```
这段代码是一个字典，包含了一些设备信息和性能指标。它将这些信息存储在对应的键值对中。

```
@app.get("/tools/translate")
def translate(
    request: Request,
    texts: str = Query(..., description="待翻译文本"),
    to_language: str = Query(..., description="翻译目标语言"),
):
    """翻译"""
    logger.info(
        f"{request.client.host}:{request.client.port}/tools/translate  { unquote(str(request.query_params) )}"
    )
    return {"texts": trans.translate(Sentence=texts, to_Language=to_language)}
```
这段代码定义了一个名为`translate`的函数，它是一个路由处理函数。当收到`/tools/translate`的GET请求时，会调用这个函数来处理请求。函数的参数包括`request`、`texts`和`to_language`。`request`参数是一个Request对象，用于获取请求的相关信息。`texts`和`to_language`是从请求的查询参数中获取的待翻译文本和翻译目标语言。函数内部使用了一个logger来记录请求的相关信息，并调用`trans.translate`函数进行翻译。最后，返回一个包含翻译结果的字典。
all_examples: Dict[str, Dict[str, List]] = dict()  # 存放示例
```
这行代码定义了一个变量`all_examples`，它是一个字典类型，键是字符串类型，值是字典类型，字典的键是字符串类型，值是列表类型。这个变量用于存储示例数据。

```
@app.get("/tools/random_example")
```
这行代码使用`app.get`装饰器将下面的函数`random_example`注册为一个GET请求的处理函数。它指定了一个路径`/tools/random_example`，当客户端发送GET请求到这个路径时，将会调用这个函数。

```
def random_example(
    request: Request,
    language: str = Query(None, description="指定语言，未指定则随机返回"),
    root_dir: str = Query("Data", description="搜索根目录"),
):
```
这是一个函数定义，函数名为`random_example`，它有三个参数：`request`、`language`和`root_dir`。`request`参数的类型是`Request`，`language`和`root_dir`参数的类型是字符串。`language`和`root_dir`参数都有默认值，`language`的默认值是`None`，`root_dir`的默认值是`"Data"`。这个函数用于处理`/tools/random_example`路径的GET请求。

```
"""
获取一个随机音频+文本，用于对比，音频会从本地目录随机选择。
"""
```
这是一个多行字符串，用于对`random_example`函数进行文档注释。它描述了这个函数的功能，即获取一个随机音频和文本用于对比，音频会从本地目录随机选择。

```
logger.info(
    f"{request.client.host}:{request.client.port}/tools/random_example  { unquote(str(request.query_params) )}"
)
```
这行代码使用`logger.info`函数记录一条日志信息。日志信息包括客户端的主机和端口，以及请求的路径和查询参数。

```
global all_examples
```
这行代码声明`all_examples`是一个全局变量，即在函数内部可以访问和修改这个变量。

```
if root_dir not in all_examples.keys():
    all_examples[root_dir] = {"ZH": [], "JP": [], "EN": []}
```
这是一个条件语句，判断`root_dir`是否在`all_examples`字典的键中。如果不在，则将`root_dir`作为键，对应的值是一个字典，字典的键是`"ZH"`、`"JP"`和`"EN"`，对应的值是空列表。这段代码用于初始化`all_examples`字典中指定键的值。
            examples = all_examples[root_dir]  # 从字典中获取指定键的值，并赋给变量examples

            # 从项目Data目录中搜索train/val.list
            for root, directories, _files in os.walk(root_dir):  # 遍历root_dir目录及其子目录下的所有文件和文件夹
                for file in _files:  # 遍历_files列表中的每个文件
                    if file in ["train.list", "val.list"]:  # 判断文件名是否为"train.list"或"val.list"
                        with open(
                            os.path.join(root, file), mode="r", encoding="utf-8"
                        ) as f:  # 打开文件，并将文件对象赋给变量f
                            lines = f.readlines()  # 读取文件的所有行，并将其赋给列表lines
                            for line in lines:  # 遍历lines列表中的每一行
                                data = line.split("|")  # 将行按"|"分割，并将分割后的结果赋给列表data
                                if len(data) != 7:  # 判断data列表的长度是否为7
                                    continue  # 如果长度不为7，则跳过当前循环，继续下一次循环
                                # 音频存在 且语言为ZH/EN/JP
                                if os.path.isfile(data[0]) and data[2] in [
                                    "ZH",
                                    "JP",
                                    "EN",
                                ]:  # 判断data列表的第一个元素是否为一个文件路径，并且data列表的第三个元素是否为"ZH"、"JP"或"EN"
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```

需要注释的代码：

```
# 将数据添加到示例字典中的对应语言列表中
examples[data[2]].append(
    {
        "text": data[3],
        "audio": data[0],
        "speaker": data[1],
    }
)

# 获取所有示例数据的字典
examples = all_examples[root_dir]
# 如果语言参数为空
if language is None:
    # 如果示例数据字典中的中文、日文和英文示例数据都没有加载
    if len(examples["ZH"]) + len(examples["JP"]) + len(examples["EN"]) == 0:
        # 返回错误信息
        return {"status": 17, "detail": "没有加载任何示例数据"}
    else:
        # 随机生成一个数字
        rand_num = random.randint(
            0,
            len(examples["ZH"]) + len(examples["JP"]) + len(examples["EN"]) - 1,
        )
        # 如果随机数字小于中文示例数据的数量
        if rand_num < len(examples["ZH"]):
            # 选择中文示例数据
# 如果随机数小于中文例句的数量，则返回中文例句
if rand_num < len(examples["ZH"]):
    return {"status": 0, "Data": examples["ZH"][rand_num]}
# 如果随机数小于中文例句和日文例句的数量之和，则返回日文例句
if rand_num < len(examples["ZH"]) + len(examples["JP"]):
    return {
        "status": 0,
        "Data": examples["JP"][rand_num - len(examples["ZH"])],
    }
# 否则返回英文例句
return {
    "status": 0,
    "Data": examples["EN"][
        rand_num - len(examples["ZH"]) - len(examples["JP"])
    ],
}
# 如果没有加载任何指定语言的数据，则返回错误状态码和详细信息
if len(examples[language]) == 0:
    return {"status": 17, "detail": f"没有加载任何{language}数据"}
# 否则返回成功状态码和结果数据
return {
    "status": 0,
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```

需要注释的代码：

```
# 导入必要的模块和函数
from fastapi import FastAPI, Request, Query
from fastapi.responses import FileResponse
from urllib.parse import unquote
import os
import random

# 创建 FastAPI 应用
app = FastAPI()

# 定义 GET 请求的处理函数
@app.get("/tools/get_audio")
def get_audio(request: Request, path: str = Query(..., description="本地音频路径")):
    # 记录请求日志
    logger.info(
        f"{request.client.host}:{request.client.port}/tools/get_audio  { unquote(str(request.query_params) )}"
    )
    # 检查音频文件是否存在
    if not os.path.isfile(path):
        logger.error(f"/tools/get_audio 获取音频错误：指定音频{path}不存在")
        return {"status": 18, "detail": "指定音频不存在"}
    # 检查音频文件是否为 WAV 格式
    if not path.lower().endswith(".wav"):
        logger.error(f"/tools/get_audio 获取音频错误：音频{path}非wav文件")
        return {"status": 19, "detail": "非wav格式文件"}
    # 返回音频文件
    return FileResponse(path=path)

# 输出警告信息
logger.warning("本地服务，请勿将服务端口暴露于外网")
# 输出 API 文档地址
logger.info(f"api文档地址 http://127.0.0.1:{config.server_config.port}/docs")
```

注释解释了每个代码块的作用，使得代码更易读和理解。
    if os.path.isdir(StaticDir):
        # 如果 StaticDir 是一个目录，则打开浏览器并访问本地服务器的地址
        webbrowser.open(f"http://127.0.0.1:{config.server_config.port}")
    # 运行 uvicorn 服务器
    uvicorn.run(
        app, port=config.server_config.port, host="0.0.0.0", log_level="warning"
    )
```

这段代码的作用是判断 `StaticDir` 是否是一个目录，如果是目录，则打开浏览器并访问本地服务器的地址。然后运行 `uvicorn` 服务器，将 `app` 对象作为参数传入，指定端口号、主机地址和日志级别。
```