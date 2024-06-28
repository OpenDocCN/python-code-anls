# `.\dependency_versions_table.py`

```py
# 自动化生成的依赖字典，用于设置项目的依赖关系和版本限制
deps = {
    "Pillow": "Pillow>=10.0.1,<=15.0",  # 图像处理库 Pillow 的版本要求在 10.0.1 到 15.0 之间
    "accelerate": "accelerate>=0.21.0",  # 加速计算库 accelerate 的版本要求至少为 0.21.0
    "av": "av==9.2.0",  # 多媒体处理库 av 的版本要求为精确匹配 9.2.0
    "beautifulsoup4": "beautifulsoup4",  # 解析 HTML 和 XML 文档的库 beautifulsoup4，版本不做限制
    "codecarbon": "codecarbon==1.2.0",  # 计算代码碳足迹的库 codecarbon 的版本要求为精确匹配 1.2.0
    "cookiecutter": "cookiecutter==1.7.3",  # 项目模板生成工具 cookiecutter 的版本要求为精确匹配 1.7.3
    "dataclasses": "dataclasses",  # Python 3.7 引入的 dataclasses 库，版本不做限制
    "datasets": "datasets!=2.5.0",  # 数据集处理库 datasets 的版本要求排除 2.5.0
    "decord": "decord==0.6.0",  # 多媒体处理库 decord 的版本要求为精确匹配 0.6.0
    "deepspeed": "deepspeed>=0.9.3",  # 分布式训练加速库 deepspeed 的版本要求至少为 0.9.3
    "diffusers": "diffusers",  # 数据扰动库 diffusers，版本不做限制
    "dill": "dill<0.3.5",  # 对象序列化库 dill 的版本要求小于 0.3.5
    "evaluate": "evaluate>=0.2.0",  # 评估工具库 evaluate 的版本要求至少为 0.2.0
    "faiss-cpu": "faiss-cpu",  # 向量相似度搜索库 faiss-cpu，版本不做限制
    "fastapi": "fastapi",  # 高性能 API 框架 fastapi，版本不做限制
    "filelock": "filelock",  # 文件锁定库 filelock，版本不做限制
    "flax": "flax>=0.4.1,<=0.7.0",  # JAX 的神经网络库 flax 的版本要求在 0.4.1 到 0.7.0 之间
    "fsspec": "fsspec<2023.10.0",  # 文件系统库 fsspec 的版本要求小于 2023.10.0
    "ftfy": "ftfy",  # 处理 Unicode 文本的库 ftfy，版本不做限制
    "fugashi": "fugashi>=1.0",  # 日语分词器 fugashi 的版本要求至少为 1.0
    "GitPython": "GitPython<3.1.19",  # Git 操作库 GitPython 的版本要求小于 3.1.19
    "hf-doc-builder": "hf-doc-builder>=0.3.0",  # Hugging Face 文档构建工具的版本要求至少为 0.3.0
    "huggingface-hub": "huggingface-hub>=0.19.3,<1.0",  # Hugging Face 模型中心库的版本要求在 0.19.3 到 1.0 之间
    "importlib_metadata": "importlib_metadata",  # 导入库信息的元数据库 importlib_metadata，版本不做限制
    "ipadic": "ipadic>=1.0.0,<2.0",  # 日语词典 ipadic 的版本要求在 1.0.0 到 2.0 之间
    "isort": "isort>=5.5.4",  # Python 代码排序工具 isort 的版本要求至少为 5.5.4
    "jax": "jax>=0.4.1,<=0.4.13",  # 数值计算库 JAX 的版本要求在 0.4.1 到 0.4.13 之间
    "jaxlib": "jaxlib>=0.4.1,<=0.4.13",  # JAX 的线性代数库 jaxlib 的版本要求在 0.4.1 到 0.4.13 之间
    "jieba": "jieba",  # 中文分词库 jieba，版本不做限制
    "kenlm": "kenlm",  # 语言模型工具 kenlm，版本不做限制
    "keras": "keras<2.16",  # 深度学习库 Keras 的版本要求小于 2.16
    "keras-nlp": "keras-nlp>=0.3.1",  # Keras 自然语言处理库 keras-nlp 的版本要求至少为 0.3.1
    "librosa": "librosa",  # 音频处理库 librosa，版本不做限制
    "nltk": "nltk",  # 自然语言工具包 NLTK，版本不做限制
    "natten": "natten>=0.14.6,<0.15.0",  # 多头自注意力模型库 natten 的版本要求在 0.14.6 到 0.15.0 之间
    "numpy": "numpy>=1.17",  # 数值计算库 numpy 的版本要求至少为 1.17
    "onnxconverter-common": "onnxconverter-common",  # ONNX 模型转换通用库 onnxconverter-common，版本不做限制
    "onnxruntime-tools": "onnxruntime-tools>=1.4.2",  # ONNX 运行时工具库 onnxruntime-tools 的版本要求至少为 1.4.2
    "onnxruntime": "onnxruntime>=1.4.0",  # ONNX 运行时库 onnxruntime 的版本要求至少为 1.4.0
    "opencv-python": "opencv-python",  # 计算机视觉库 opencv-python，版本不做限制
    "optuna": "optuna",  # 自动机器学习工具 optuna，版本不做限制
    "optax": "optax>=0.0.8,<=0.1.4",  # 优化库 optax 的版本要求在 0.0.8 到 0.1.4 之间
    "packaging": "packaging>=20.0",  # 打包工具库 packaging 的版本要求至少为 20.0
    "parameterized": "parameterized",  # 参数化测试工具 parameterized，版本不做限制
    "phonemizer": "phonemizer",  # 文本到音素转换库 phonemizer，版本不做限制
    "protobuf": "protobuf",  # Google 的序列化库 protobuf，版本不做限制
    "psutil": "psutil",  # 进程和系统工具库 psutil，版本不做限制
    "pyyaml": "pyyaml>=5.1",  # YAML 解析器库 pyyaml 的版本要求至少为 5.1
    "pydantic": "pydantic",  # 数据验证库 pydantic，版本不做限制
    "pytest": "pytest>=7.2.0,<8.0.0",  # 测试框架 pytest 的版本要求在 7.2.0 到 8.0.0 之间
    "pytest-timeout": "pytest-timeout",  # pytest 插件 pytest-timeout，版本不做限制
    "pytest-xdist": "pytest-xdist",  # pytest 插件 pytest-xdist，版本不做限制
    "python": "python>=3.8.0",  # Python 解释器的版本要求至少为 3.8.0
    "ray[t
    "pyctcdecode": "pyctcdecode>=0.4.0",
    # 定义依赖项：pyctcdecode 库，版本需大于或等于 0.4.0

    "tqdm": "tqdm>=4.27",
    # 定义依赖项：tqdm 库，版本需大于或等于 4.27

    "unidic": "unidic>=1.0.2",
    # 定义依赖项：unidic 库，版本需大于或等于 1.0.2

    "unidic_lite": "unidic_lite>=1.0.7",
    # 定义依赖项：unidic_lite 库，版本需大于或等于 1.0.7

    "urllib3": "urllib3<2.0.0",
    # 定义依赖项：urllib3 库，版本需小于 2.0.0

    "uvicorn": "uvicorn",
    # 定义依赖项：uvicorn 库，无指定版本要求
}


注释：


# 这行代码表示一个代码块的结束，对应于一个以 '{' 开始的代码块的结束
```