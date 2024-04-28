# `.\transformers\dependency_versions_table.py`

```py
# 定义依赖字典，包含各个库的版本要求
deps = {
    "Pillow": "Pillow>=10.0.1,<=15.0",  # 图像处理库 Pillow 版本要求
    "accelerate": "accelerate>=0.21.0",  # 加速库版本要求
    "av": "av==9.2.0",  # 多媒体处理库 av 版本要求
    "beautifulsoup4": "beautifulsoup4",  # HTML 解析库版本要求
    "codecarbon": "codecarbon==1.2.0",  # 代码碳足迹库版本要求
    "cookiecutter": "cookiecutter==1.7.3",  # 项目模板生成工具版本要求
    "dataclasses": "dataclasses",  # 数据类库版本要求
    "datasets": "datasets!=2.5.0",  # 数据集处理库版本要求
    "decord": "decord==0.6.0",  # 多媒体处理库 decord 版本要求
    "deepspeed": "deepspeed>=0.9.3",  # 深度学习加速库 deepspeed 版本要求
    "diffusers": "diffusers",  # 扩散库版本要求
    "dill": "dill<0.3.5",  # 数据序列化库 dill 版本要求
    "evaluate": "evaluate>=0.2.0",  # 评估库版本要求
    "faiss-cpu": "faiss-cpu",  # Faiss CPU 版本要求
    "fastapi": "fastapi",  # Web 框架 FastAPI 版本要求
    "filelock": "filelock",  # 文件锁定库版本要求
    "flax": "flax>=0.4.1,<=0.7.0",  # 深度学习库 flax 版本要求
    "fsspec": "fsspec<2023.10.0",  # 文件系统库 fsspec 版本要求
    "ftfy": "ftfy",  # 文本处理库 ftfy 版本要求
    "fugashi": "fugashi>=1.0",  # 日语分词库 fugashi 版本要求
    "GitPython": "GitPython<3.1.19",  # Git 操作库 GitPython 版本要求
    "hf-doc-builder": "hf-doc-builder>=0.3.0",  # Hugging Face 文档生成工具版本要求
    "huggingface-hub": "huggingface-hub>=0.19.3,<1.0",  # Hugging Face 模型中心版本要求
    "importlib_metadata": "importlib_metadata",  # 元数据导入库版本要求
    "ipadic": "ipadic>=1.0.0,<2.0",  # 日语分词库 ipadic 版本要求
    "isort": "isort>=5.5.4",  # 代码排序工具 isort 版本要求
    "jax": "jax>=0.4.1,<=0.4.13",  # 数值计算库 jax 版本要求
    "jaxlib": "jaxlib>=0.4.1,<=0.4.13",  # 数值计算库 jaxlib 版本要求
    "jieba": "jieba",  # 中文分词库 jieba 版本要求
    "kenlm": "kenlm",  # 语言模型库 kenlm 版本要求
    "keras": "keras<2.16",  # 深度学习库 keras 版本要求
    "keras-nlp": "keras-nlp>=0.3.1",  # 自然语言处理库 keras-nlp 版本要求
    "librosa": "librosa",  # 音频处理库 librosa 版本要求
    "nltk": "nltk",  # 自然语言处理库 nltk 版本要求
    "natten": "natten>=0.14.6,<0.15.0",  # 注意���库 natten 版本要求
    "numpy": "numpy>=1.17",  # 数值计算库 numpy 版本要求
    "onnxconverter-common": "onnxconverter-common",  # ONNX 转换库版本要求
    "onnxruntime-tools": "onnxruntime-tools>=1.4.2",  # ONNX 运行工具版本要求
    "onnxruntime": "onnxruntime>=1.4.0",  # ONNX 运行库版本要求
    "opencv-python": "opencv-python",  # 图像处理库 OpenCV 版本要求
    "optuna": "optuna",  # 超参数优化库 optuna 版本要求
    "optax": "optax>=0.0.8,<=0.1.4",  # 优化库 optax 版本要求
    "packaging": "packaging>=20.0",  # 打包库 packaging 版本要求
    "parameterized": "parameterized",  # 参数化库版本要求
    "phonemizer": "phonemizer",  # 语音合成库 phonemizer 版本要求
    "protobuf": "protobuf",  # 序列化库 protobuf 版本要求
    "psutil": "psutil",  # 系统进程管理库 psutil 版本要求
    "pyyaml": "pyyaml>=5.1",  # YAML 解析库 pyyaml 版本要求
    "pydantic": "pydantic<2",  # 数据验证库 pydantic 版本要求
    "pytest": "pytest>=7.2.0",  # 测试框架 pytest 版本要求
    "pytest-timeout": "pytest-timeout",  # 测试框架 pytest-timeout 版本要求
    "pytest-xdist": "pytest-xdist",  # 测试框架 pytest-xdist 版本要求
    "python": "python>=3.8.0",  # Python 版本要求
    "ray[tune]": "ray[tune]>=2.7.0",  # 分布式计算库 ray 版本要求
    "regex": "regex!=2019.12.17",  # 正则表达式库 regex 版本要求
    "requests": "requests",  # HTTP 请求库 requests 版本要求
    "rhoknp": "rhoknp>=1.1.0,<1.3.1",  # 日语分词库 rhoknp 版本要求
    "rjieba": "rjieba",  # 中文分词库 rjieba 版本要求
    "rouge-score": "rouge-score!=0.0.7,!=0.0.8,!=0.1,!=0.1.1",  # 文本评估库 rouge-score 版本要求
    "ruff": "ruff==0.1.5",  # Ruff 版本要求
    "sacrebleu": "sacrebleu>=1.4.12,<2.0.0",  # BLEU 评估库 sacrebleu 版本要求
    "sacremoses": "sacremoses",  # 文本处理库 sacremoses 版本要求
    "safetensors": "safetensors>=0.3.1",  # 安全张量库 safetensors 版本要求
    "sagemaker": "sagemaker>=2.31.0",  # 机器学习服务库 sagemaker 版本要求
    "scikit-learn": "scikit-learn",  # 机器学习库 scikit-learn 版本要求
    "sentencepiece": "sentencepiece>=0.1.91,!=0.1.92",  # 分词库 sentencepiece 版本要求
    "sigopt": "sigopt",  # 超参数优化库 sigopt 版本要求
    "starlette": "starlette",  # Web 框架 starlette 版本要求
    "sudachipy": "sudachipy>=0.6.6",  # 日语分词库 sudachipy 版本要求
    "sudachidict_core": "sudachidict_core>=20220729",  # 日语分词库 sudachidict_core 版本要求
    "tensorboard": "tensorboard",  # TensorFlow 可视化工具 tensorboard 版本要求
    "tensorflow-cpu": "tensorflow-cpu>=2.6,<2.16",  # TensorFlow CPU 版本要求
    "tensorflow": "tensorflow>=2.6,<2.16",  # TensorFlow 版本要求
    "tensorflow-text": "tensorflow-text<2.16",  # TensorFlow 文本处理库版本要求
    "tf2onnx": "tf2onnx",  # TensorFlow 转 ONNX 工具 tf2onnx 版本要求
    "timeout-decorator": "timeout-decorator",  # 超时装饰器库版本要求
    "timm": "timm",  # 计算机视觉库 timm 版本要求
    "tokenizers": "tokenizers>=0.14,<0.19",  # 分词库 tokenizers 版本要求
    "torch": "torch>=1.11,!=1.12.0",  # 深度学习库 PyTorch 版本要求
    "torchaudio": "torchaudio",  # 音频处理库 torchaudio 版本要求
    "torchvision": "torchvision",  # 计算机视觉库 torchvision 版本要求
    # 定义依赖包 "pyctcdecode"，版本需大于等于 0.4.0
    "pyctcdecode": "pyctcdecode>=0.4.0",
    # 定义依赖包 "tqdm"，版本需大于等于 4.27
    "tqdm": "tqdm>=4.27",
    # 定义依赖包 "unidic"，版本需大于等于 1.0.2
    "unidic": "unidic>=1.0.2",
    # 定义依赖包 "unidic_lite"，版本需大于等于 1.0.7
    "unidic_lite": "unidic_lite>=1.0.7",
    # 定义依赖包 "urllib3"，版本需小于 2.0.0
    "urllib3": "urllib3<2.0.0",
    # 定义依赖包 "uvicorn"
    "uvicorn": "uvicorn",
# 闭合大括号，表示代码块的结束
```