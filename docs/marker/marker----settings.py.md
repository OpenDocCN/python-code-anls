# `.\marker\marker\settings.py`

```py
import os
from typing import Optional, List, Dict

from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
import fitz as pymupdf
import torch

# 定义一个设置类，继承自BaseSettings
class Settings(BaseSettings):
    # General
    TORCH_DEVICE: Optional[str] = None

    # 计算属性，返回TORCH_DEVICE_MODEL
    @computed_field
    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        # 如果TORCH_DEVICE不为None，则返回TORCH_DEVICE
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        # 如果CUDA可用，则返回"cuda"
        if torch.cuda.is_available():
            return "cuda"

        # 如果MPS可用，则返回"mps"
        if torch.backends.mps.is_available():
            return "mps"

        # 否则返回"cpu"
        return "cpu"

    INFERENCE_RAM: int = 40 # 每个GPU的VRAM量（以GB为单位）。
    VRAM_PER_TASK: float = 2.5 # 每个任务分配的VRAM量（以GB为单位）。 峰值标记VRAM使用量约为3GB，但工作程序的平均值较低。
    DEFAULT_LANG: str = "English" # 我们假设文件所在的默认语言，应该是TESSERACT_LANGUAGES中的一个键

    SUPPORTED_FILETYPES: Dict = {
        "application/pdf": "pdf",
        "application/epub+zip": "epub",
        "application/x-mobipocket-ebook": "mobi",
        "application/vnd.ms-xpsdocument": "xps",
        "application/x-fictionbook+xml": "fb2"
    }

    # PyMuPDF
    TEXT_FLAGS: int = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_LIGATURES & ~pymupdf.TEXT_PRESERVE_IMAGES

    # OCR
    INVALID_CHARS: List[str] = [chr(0xfffd), "�"]
    OCR_DPI: int = 400
    TESSDATA_PREFIX: str = ""
    TESSERACT_LANGUAGES: Dict = {
        "English": "eng",
        "Spanish": "spa",
        "Portuguese": "por",
        "French": "fra",
        "German": "deu",
        "Russian": "rus",
        "Chinese": "chi_sim",
        "Japanese": "jpn",
        "Korean": "kor",
        "Hindi": "hin",
    }
    TESSERACT_TIMEOUT: int = 20 # 何时放弃OCR
    # 定义拼写检查语言对应的字典
    SPELLCHECK_LANGUAGES: Dict = {
        "English": "en",
        "Spanish": "es",
        "Portuguese": "pt",
        "French": "fr",
        "German": "de",
        "Russian": "ru",
        "Chinese": None,
        "Japanese": None,
        "Korean": None,
        "Hindi": None,
    }
    # 是否在每一页运行 OCR，即使可以提取文本
    OCR_ALL_PAGES: bool = False
    # 用于 OCR 的并行 CPU 工作线程数
    OCR_PARALLEL_WORKERS: int = 2
    # 使用的 OCR 引擎，可以是 "tesseract" 或 "ocrmypdf"，ocrmypdf 质量更高但速度较慢
    OCR_ENGINE: str = "ocrmypdf"

    # Texify 模型相关参数
    TEXIFY_MODEL_MAX: int = 384 # Texify 的最大推理长度
    TEXIFY_TOKEN_BUFFER: int = 256 # Texify 的 token 缓冲区大小
    TEXIFY_DPI: int = 96 # 渲染图像的 DPI
    TEXIFY_BATCH_SIZE: int = 2 if TORCH_DEVICE_MODEL == "cpu" else 6 # Texify 的批处理大小，CPU 上较低因为使用 float32
    TEXIFY_MODEL_NAME: str = "vikp/texify"

    # Layout 模型相关参数
    BAD_SPAN_TYPES: List[str] = ["Caption", "Footnote", "Page-footer", "Page-header", "Picture"]
    LAYOUT_MODEL_MAX: int = 512
    LAYOUT_CHUNK_OVERLAP: int = 64
    LAYOUT_DPI: int = 96
    LAYOUT_MODEL_NAME: str = "vikp/layout_segmenter"
    LAYOUT_BATCH_SIZE: int = 8 # 最大 512 个 token 意味着较高的批处理大小

    # Ordering 模型相关参数
    ORDERER_BATCH_SIZE: int = 32 # 可以较高，因为最大 token 数为 128
    ORDERER_MODEL_NAME: str = "vikp/column_detector"

    # 最终编辑模型相关参数
    EDITOR_BATCH_SIZE: int = 4
    EDITOR_MAX_LENGTH: int = 1024
    EDITOR_MODEL_NAME: str = "vikp/pdf_postprocessor_t5"
    ENABLE_EDITOR_MODEL: bool = False # 编辑模型可能会产生误报
    EDITOR_CUTOFF_THRESH: float = 0.9 # 忽略概率低于此阈值的预测

    # Ray 相关参数
    RAY_CACHE_PATH: Optional[str] = None # 保存 Ray 缓存的路径
    RAY_CORES_PER_WORKER: int = 1 # 每个 worker 分配的 CPU 核心数

    # 调试相关参数
    DEBUG: bool = False # 启用调试日志
    # 调试数据文件夹路径，默认为 None
    DEBUG_DATA_FOLDER: Optional[str] = None
    # 调试级别，范围从 0 到 2，2 表示记录所有信息
    DEBUG_LEVEL: int = 0
    
    # 计算属性，返回是否使用 CUDA
    @computed_field
    @property
    def CUDA(self) -> bool:
        return "cuda" in self.TORCH_DEVICE
    
    # 计算属性，返回模型数据类型
    @computed_field
    @property
    def MODEL_DTYPE(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cuda":
            return torch.bfloat16
        else:
            return torch.float32
    
    # 计算属性，返回用于转换的数据类型
    @computed_field
    @property
    def TEXIFY_DTYPE(self) -> torch.dtype:
        return torch.float32 if self.TORCH_DEVICE_MODEL == "cpu" else torch.float16
    
    # 类配置
    class Config:
        # 从环境文件中查找 local.env 文件
        env_file = find_dotenv("local.env")
        # 额外配置，忽略错误
        extra = "ignore"
# 创建一个 Settings 对象实例
settings = Settings()
```