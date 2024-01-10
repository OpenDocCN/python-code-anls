# `Bert-VITS2\hiyoriUI.py`

```
"""
api服务，网页后端 多版本多模型 fastapi实现
原 server_fastapi
"""
# 导入所需的库和模块
import logging
import gc
import random
import librosa
import gradio
import numpy as np
import utils
from fastapi import FastAPI, Query, Request, File, UploadFile, Form
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from scipy.io import wavfile
import uvicorn
import torch
import webbrowser
import psutil
import GPUtil
from typing import Dict, Optional, List, Set, Union, Tuple
import os
from tools.log import logger
from urllib.parse import unquote

from infer import infer, get_net_g, latest_version
import tools.translate as trans
from tools.sentence import split_by_language
from re_matching import cut_sent

from config import config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 定义模型封装类
class Model:
    """模型封装类"""

    def __init__(self, config_path: str, model_path: str, device: str, language: str):
        # 初始化模型封装类的属性
        self.config_path: str = os.path.normpath(config_path)  # 规范化配置文件路径
        self.model_path: str = os.path.normpath(model_path)  # 规范化模型文件路径
        self.device: str = device  # 设备类型
        self.language: str = language  # 语言类型
        self.hps = utils.get_hparams_from_file(config_path)  # 从配置文件中获取超参数
        self.spk2id: Dict[str, int] = self.hps.data.spk2id  # spk - id 映射字典
        self.id2spk: Dict[int, str] = dict()  # id - spk 映射字典
        # 构建 id - spk 映射字典
        for speaker, speaker_id in self.hps.data.spk2id.items():
            self.id2spk[speaker_id] = speaker
        # 获取模型版本号
        self.version: str = (
            self.hps.version if hasattr(self.hps, "version") else latest_version
        )
        # 获取生成器网络
        self.net_g = get_net_g(
            model_path=model_path,
            version=self.version,
            device=device,
            hps=self.hps,
        )
    # 将对象的属性转换为字典形式并返回
    def to_dict(self) -> Dict[str, any]:
        # 返回包含配置路径的键值对
        "config_path": self.config_path,
        # 返回包含模型路径的键值对
        "model_path": self.model_path,
        # 返回包含设备信息的键值对
        "device": self.device,
        # 返回包含语言信息的键值对
        "language": self.language,
        # 返回包含spk2id映射的键值对
        "spk2id": self.spk2id,
        # 返回包含id2spk映射的键值对
        "id2spk": self.id2spk,
        # 返回包含版本信息的键值对
        "version": self.version,
        # 返回字典对象
        return {
class Models:
    # 初始化Models类
    def __init__(self):
        # 初始化models属性为一个空字典，键为整数，值为Model对象
        self.models: Dict[int, Model] = dict()
        # 初始化num属性为0
        self.num = 0
        # 初始化spk_info属性为一个空字典，键为字符串，值为字典，字典的键为整数，值为整数
        # spkInfo[角色名][模型id] = 角色id
        self.spk_info: Dict[str, Dict[int, int]] = dict()
        # 初始化path2ids属性为一个空字典，键为字符串，值为整数集合
        # 路径指向的model的id
        self.path2ids: Dict[str, Set[int]] = dict()
    
    # 初始化模型
    def init_model(
        self, config_path: str, model_path: str, device: str, language: str
    # 初始化并添加一个模型，返回添加的模型编号
    def add_model(self, config_path: str, model_path: str, device: str, language: str) -> int:
        """
        初始化并添加一个模型

        :param config_path: 模型config.json路径
        :param model_path: 模型路径
        :param device: 模型推理使用设备
        :param language: 模型推理默认语言
        """
        # 若文件不存在则不进行加载
        if not os.path.isfile(model_path):
            if model_path != "":
                logger.warning(f"模型文件{model_path} 不存在，不进行初始化")
            return self.num
        if not os.path.isfile(config_path):
            if config_path != "":
                logger.warning(f"配置文件{config_path} 不存在，不进行初始化")
            return self.num

        # 若路径中的模型已存在，则不添加模型，若不存在，则进行初始化。
        model_path = os.path.realpath(model_path)
        if model_path not in self.path2ids.keys():
            self.path2ids[model_path] = {self.num}
            self.models[self.num] = Model(
                config_path=config_path,
                model_path=model_path,
                device=device,
                language=language,
            )
            logger.success(f"添加模型{model_path}，使用配置文件{os.path.realpath(config_path)}")
        else:
            # 获取一个指向id
            m_id = next(iter(self.path2ids[model_path]))
            self.models[self.num] = self.models[m_id]
            self.path2ids[model_path].add(self.num)
            logger.success("模型已存在，添加模型引用。")
        # 添加角色信息
        for speaker, speaker_id in self.models[self.num].spk2id.items():
            if speaker not in self.spk_info.keys():
                self.spk_info[speaker] = {self.num: speaker_id}
            else:
                self.spk_info[speaker][self.num] = speaker_id
        # 修改计数
        self.num += 1
        return self.num - 1
    # 删除对应序号的模型，若不存在则返回None
    def del_model(self, index: int) -> Optional[int]:
        # 如果给定序号不在模型字典的键中，则返回None
        if index not in self.models.keys():
            return None
        # 删除角色信息
        for speaker, speaker_id in self.models[index].spk2id.items():
            # 从角色信息中删除对应序号的模型
            self.spk_info[speaker].pop(index)
            # 如果该角色的所有模型都被删除，则清除该角色信息
            if len(self.spk_info[speaker]) == 0:
                self.spk_info.pop(speaker)
        # 删除路径信息
        model_path = os.path.realpath(self.models[index].model_path)
        # 从路径到ID的映射中删除对应路径的ID
        self.path2ids[model_path].remove(index)
        # 如果该路径没有对应的ID了，则从映射中删除该路径
        if len(self.path2ids[model_path]) == 0:
            self.path2ids.pop(model_path)
            logger.success(f"删除模型{model_path}, id = {index}")
        else:
            logger.success(f"删除模型引用{model_path}, id = {index}")
        # 从模型字典中删除对应序号的模型
        self.models.pop(index)
        # 执行垃圾回收
        gc.collect()
        # 如果有可用的CUDA设备，则清空CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 返回被删除的模型的序号
        return index

    # 获取所有模型
    def get_models(self):
        return self.models
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建 FastAPI 应用
    app = FastAPI()
    # 将日志对象赋值给应用的 logger 属性
    app.logger = logger
    # 输出日志信息，开始挂载网页页面
    logger.info("开始挂载网页页面")
    # 设置静态文件目录
    StaticDir: str = "./Web"
    # 如果静态文件目录不存在
    if not os.path.isdir(StaticDir):
        # 输出警告日志，缺少网页资源，无法开启网页页面
        logger.warning(
            "缺少网页资源，无法开启网页页面，如有需要请在 https://github.com/jiangyuxiaoxiao/Bert-VITS2-UI 或者Bert-VITS对应版本的release页面下载"
        )
    else:
        # 获取静态文件目录下的所有子目录和文件
        dirs = [fir.name for fir in os.scandir(StaticDir) if fir.is_dir()]
        files = [fir.name for fir in os.scandir(StaticDir) if fir.is_dir()]
        # 遍历子目录
        for dirName in dirs:
            # 挂载静态文件目录到应用
            app.mount(
                f"/{dirName}",
                StaticFiles(directory=f"./{StaticDir}/{dirName}"),
                name=dirName,
            )
    # 创建 Models 对象
    loaded_models = Models()
    # 输出日志信息，开始加载模型
    logger.info("开始加载模型")
    # 获取模型配置信息
    models_info = config.server_config.models
    # 遍历模型配置信息
    for model_info in models_info:
        # 初始化模型
        loaded_models.init_model(
            config_path=model_info["config"],
            model_path=model_info["model"],
            device=model_info["device"],
            language=model_info["language"],
        )

    # 定义根路径的异步处理函数，返回 index.html 文件
    @app.get("/")
    async def index():
        return FileResponse("./Web/index.html")

    # 定义 voice 路径的异步处理函数，接收多个参数
    async def _voice(
        text: str,
        model_id: int,
        speaker_name: str,
        speaker_id: int,
        sdp_ratio: float,
        noise: float,
        noisew: float,
        length: float,
        language: str,
        auto_translate: bool,
        auto_split: bool,
        emotion: Optional[Union[int, str]] = None,
        reference_audio=None,
        style_text: Optional[str] = None,
        style_weight: float = 0.7,
    # 创建 voice 路径的 POST 请求处理函数
    @app.post("/voice")
    # 定义一个异步函数voice，接收Request对象和一系列参数
    async def voice(
        request: Request,  # fastapi自动注入
        text: str = Form(...),  # 接收文本参数
        model_id: int = Query(..., description="模型ID"),  # 模型序号
        speaker_name: str = Query(
            None, description="说话人名"
        ),  # speaker_name与 speaker_id二者选其一
        speaker_id: int = Query(None, description="说话人id，与speaker_name二选一"),
        sdp_ratio: float = Query(0.2, description="SDP/DP混合比"),
        noise: float = Query(0.2, description="感情"),
        noisew: float = Query(0.9, description="音素长度"),
        length: float = Query(1, description="语速"),
        language: str = Query(None, description="语言"),  # 若不指定使用语言则使用默认值
        auto_translate: bool = Query(False, description="自动翻译"),
        auto_split: bool = Query(False, description="自动切分"),
        emotion: Optional[Union[int, str]] = Query(None, description="emo"),
        reference_audio: UploadFile = File(None),  # 接收上传的参考音频文件
        style_text: Optional[str] = Form(None, description="风格文本"),
        style_weight: float = Query(0.7, description="风格权重"),
    ):
        """语音接口，若需要上传参考音频请仅使用post请求"""
        # 记录请求日志
        logger.info(
            f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )} text={text}"
        )
        # 调用_voice函数处理参数并返回结果
        return await _voice(
            text=text,
            model_id=model_id,
            speaker_name=speaker_name,
            speaker_id=speaker_id,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noisew=noisew,
            length=length,
            language=language,
            auto_translate=auto_translate,
            auto_split=auto_split,
            emotion=emotion,
            reference_audio=reference_audio,
            style_text=style_text,
            style_weight=style_weight,
        )

    @app.get("/voice")  # 定义一个GET请求的路由
    # 异步函数，处理语音相关请求
    async def voice(
        request: Request,  # fastapi自动注入
        text: str = Query(..., description="输入文字"),
        model_id: int = Query(..., description="模型ID"),  # 模型序号
        speaker_name: str = Query(
            None, description="说话人名"
        ),  # speaker_name与 speaker_id二者选其一
        speaker_id: int = Query(None, description="说话人id，与speaker_name二选一"),
        sdp_ratio: float = Query(0.2, description="SDP/DP混合比"),
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
        """语音接口，不建议使用"""
        # 记录请求日志
        logger.info(
            f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )}"
        )
        # 调用内部函数处理语音请求
        return await _voice(
            text=text,
            model_id=model_id,
            speaker_name=speaker_name,
            speaker_id=speaker_id,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noisew=noisew,
            length=length,
            language=language,
            auto_translate=auto_translate,
            auto_split=auto_split,
            emotion=emotion,
            style_text=style_text,
            style_weight=style_weight,
        )

    # 获取已加载模型信息
    @app.get("/models/info")
    def get_loaded_models_info(request: Request):
        """获取已加载模型信息"""
        # 创建空字典用于存储模型信息
        result: Dict[str, Dict] = dict()
        # 遍历已加载模型，将模型信息存入字典
        for key, model in loaded_models.models.items():
            result[str(key)] = model.to_dict()
        # 返回模型信息字典
        return result

    # 删除模型
    @app.get("/models/delete")
    # 删除指定模型
    def delete_model(
        request: Request, model_id: int = Query(..., description="删除模型id")
    ):
        """删除指定模型"""
        # 记录请求信息
        logger.info(
            f"{request.client.host}:{request.client.port}/models/delete  { unquote(str(request.query_params) )}"
        )
        # 调用loaded_models.del_model()删除指定模型
        result = loaded_models.del_model(model_id)
        # 如果删除结果为None，记录错误信息并返回删除失败的状态和详情
        if result is None:
            logger.error(f"/models/delete 模型删除错误：模型{model_id}不存在，删除失败")
            return {"status": 14, "detail": f"模型{model_id}不存在，删除失败"}
        # 返回删除成功的状态和详情
        return {"status": 0, "detail": "删除成功"}

    # 添加模型
    @app.get("/models/add")
    def add_model(
        request: Request,
        model_path: str = Query(..., description="添加模型路径"),
        config_path: str = Query(
            None, description="添加模型配置文件路径，不填则使用./config.json或../config.json"
        ),
        device: str = Query("cuda", description="推理使用设备"),
        language: str = Query("ZH", description="模型默认语言"),
    ):
        """添加指定模型：允许重复添加相同路径模型，且不重复占用内存"""
        # 记录日志，记录客户端IP和端口以及请求参数
        logger.info(
            f"{request.client.host}:{request.client.port}/models/add  { unquote(str(request.query_params) )}"
        )
        # 如果配置路径为空
        if config_path is None:
            # 获取模型路径的目录
            model_dir = os.path.dirname(model_path)
            # 如果模型目录下存在config.json文件
            if os.path.isfile(os.path.join(model_dir, "config.json")):
                config_path = os.path.join(model_dir, "config.json")
            # 如果模型目录的上级目录存在config.json文件
            elif os.path.isfile(os.path.join(model_dir, "../config.json")):
                config_path = os.path.join(model_dir, "../config.json")
            else:
                # 记录错误日志并返回错误信息
                logger.error("/models/add 模型添加失败：未在模型所在目录以及上级目录找到config.json文件")
                return {
                    "status": 15,
                    "detail": "查询未传入配置文件路径，同时默认路径./与../中不存在配置文件config.json。",
                }
        try:
            # 初始化模型并获取模型ID
            model_id = loaded_models.init_model(
                config_path=config_path,
                model_path=model_path,
                device=device,
                language=language,
            )
        except Exception:
            # 记录异常日志并返回错误信息
            logging.exception("模型加载出错")
            return {
                "status": 16,
                "detail": "模型加载出错，详细查看日志",
            }
        # 返回成功添加模型的信息
        return {
            "status": 0,
            "detail": "模型添加成功",
            "Data": {
                "model_id": model_id,
                "model_info": loaded_models.models[model_id].to_dict(),
            },
        }

    @app.get("/models/get_unloaded")
    def get_unloaded_models_info(
        request: Request, root_dir: str = Query("Data", description="搜索根目录")
    ):
        """获取未加载模型"""
        # 记录日志，记录客户端IP和端口以及请求参数
        logger.info(
            f"{request.client.host}:{request.client.port}/models/get_unloaded  { unquote(str(request.query_params) )}"
        )
        # 返回未加载模型的信息
        return _get_all_models(root_dir, only_unloaded=True)

    @app.get("/models/get_local")
    def get_local_models_info(
        request: Request, root_dir: str = Query("Data", description="搜索根目录")
    ):
        """获取全部本地模型"""
        # 记录请求的客户端地址和端口以及查询参数
        logger.info(
            f"{request.client.host}:{request.client.port}/models/get_local  { unquote(str(request.query_params) )}"
        )
        # 返回所有模型的信息
        return _get_all_models(root_dir, only_unloaded=False)

    @app.get("/status")
    def get_status():
        """获取电脑运行状态"""
        # 获取 CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        # 获取内存信息
        memory_info = psutil.virtual_memory()
        memory_total = memory_info.total
        memory_available = memory_info.available
        memory_used = memory_info.used
        memory_percent = memory_info.percent
        gpuInfo = []
        devices = ["cpu"]
        # 获取所有 GPU 设备信息
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
        gpus = GPUtil.getGPUs()
        # 遍历所有 GPU 设备，获取相关信息
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
        # 返回电脑运行状态信息
        return {
            "devices": devices,
            "cpu_percent": cpu_percent,
            "memory_total": memory_total,
            "memory_available": memory_available,
            "memory_used": memory_used,
            "memory_percent": memory_percent,
            "gpu": gpuInfo,
        }

    @app.get("/tools/translate")
    def translate(
        request: Request,
        texts: str = Query(..., description="待翻译文本"),
        to_language: str = Query(..., description="翻译目标语言"),
    ):
        """翻译"""
        # 记录请求的客户端地址和端口以及查询参数
        logger.info(
            f"{request.client.host}:{request.client.port}/tools/translate  { unquote(str(request.query_params) )}"
        )
        # 返回翻译结果
        return {"texts": trans.translate(Sentence=texts, to_Language=to_language)}

    all_examples: Dict[str, Dict[str, List]] = dict()  # 存放示例

    @app.get("/tools/random_example")
    # 定义一个接受请求的函数，接受三个参数：Request对象、语言参数（默认为None）、根目录参数（默认为"Data"）
    def random_example(
        request: Request,
        language: str = Query(None, description="指定语言，未指定则随机返回"),
        root_dir: str = Query("Data", description="搜索根目录"),
    @app.get("/tools/get_audio")
    # 定义一个获取音频的接口函数，接受两个参数：Request对象、音频路径参数
    def get_audio(request: Request, path: str = Query(..., description="本地音频路径")):
        # 记录请求的客户端信息和查询参数
        logger.info(
            f"{request.client.host}:{request.client.port}/tools/get_audio  { unquote(str(request.query_params) )}"
        )
        # 如果指定的音频路径不存在，则返回错误信息
        if not os.path.isfile(path):
            logger.error(f"/tools/get_audio 获取音频错误：指定音频{path}不存在")
            return {"status": 18, "detail": "指定音频不存在"}
        # 如果音频不是wav格式文件，则返回错误信息
        if not path.lower().endswith(".wav"):
            logger.error(f"/tools/get_audio 获取音频错误：音频{path}非wav文件")
            return {"status": 19, "detail": "非wav格式文件"}
        # 返回音频文件的响应
        return FileResponse(path=path)
    
    # 记录警告信息，提醒不要将服务端口暴露于外网
    logger.warning("本地服务，请勿将服务端口暴露于外网")
    # 记录信息，显示API文档地址
    logger.info(f"api文档地址 http://127.0.0.1:{config.server_config.port}/docs")
    # 如果静态目录存在，则在浏览器中打开对应的URL
    if os.path.isdir(StaticDir):
        webbrowser.open(f"http://127.0.0.1:{config.server_config.port}")
    # 运行应用程序，监听指定端口，允许外部访问，记录警告级别的日志
    uvicorn.run(
        app, port=config.server_config.port, host="0.0.0.0", log_level="warning"
    )
```