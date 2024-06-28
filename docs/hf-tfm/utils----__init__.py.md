# `.\utils\__init__.py`

```
# 指定 Python 解释器的路径，使得脚本可以在环境中独立运行
#!/usr/bin/env python
# 设置脚本的字符编码为 UTF-8
# coding=utf-8

# 版权声明和许可证信息，该脚本遵循 Apache 许可证版本 2.0
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入所需模块和函数

from huggingface_hub import get_full_repo_name  # 用于向后兼容
from huggingface_hub.constants import HF_HUB_DISABLE_TELEMETRY as DISABLE_TELEMETRY  # 用于向后兼容
from packaging import version

from .. import __version__  # 导入上层包的版本信息
from .backbone_utils import BackboneConfigMixin, BackboneMixin  # 导入本地模块中的类和函数
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD  # 导入图像标准化常量
from .doc import (
    add_code_sample_docstrings,  # 导入用于文档注释的函数
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    copy_func,
    replace_return_docstrings,
)
from .generic import (
    ContextManagers,  # 导入上下文管理器类
    ExplicitEnum,
    ModelOutput,
    PaddingStrategy,
    TensorType,
    add_model_info_to_auto_map,
    cached_property,
    can_return_loss,
    expand_dims,
    find_labels,
    flatten_dict,
    infer_framework,
    is_jax_tensor,
    is_numpy_array,
    is_tensor,
    is_tf_symbolic_tensor,
    is_tf_tensor,
    is_torch_device,
    is_torch_dtype,
    is_torch_tensor,
    reshape,
    squeeze,
    strtobool,
    tensor_size,
    to_numpy,
    to_py_obj,
    transpose,
    working_or_temp_dir,
)
from .hub import (
    CLOUDFRONT_DISTRIB_PREFIX,  # 导入与模型存储和缓存相关的常量和函数
    HF_MODULES_CACHE,
    HUGGINGFACE_CO_PREFIX,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    S3_BUCKET_PREFIX,
    TRANSFORMERS_CACHE,
    TRANSFORMERS_DYNAMIC_MODULE_NAME,
    EntryNotFoundError,
    PushInProgress,
    PushToHubMixin,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    cached_file,
    default_cache_path,
    define_sagemaker_information,
    download_url,
    extract_commit_hash,
    get_cached_models,
    get_file_from_repo,
    has_file,
    http_user_agent,
    is_offline_mode,
    is_remote_url,
    move_cache,
    send_example_telemetry,
    try_to_load_from_cache,
)
from .import_utils import (
    ACCELERATE_MIN_VERSION,  # 导入与外部库依赖相关的版本要求和工具函数
    ENV_VARS_TRUE_AND_AUTO_VALUES,
    ENV_VARS_TRUE_VALUES,
    TORCH_FX_REQUIRED_VERSION,
    USE_JAX,
    USE_TF,
    USE_TORCH,
    DummyObject,
    OptionalDependencyNotAvailable,
    _LazyModule,
    ccl_version,
    direct_transformers_import,
    get_torch_version,
    is_accelerate_available,
    is_apex_available,
    is_aqlm_available,
    is_auto_awq_available,
    # 检查是否自动加载 GPT-3.5 或更高版本的问答功能可用
    is_auto_gptq_available,
    # 检查是否安装了 bitsandbytes 库
    is_bitsandbytes_available,
    # 检查是否安装了 BeautifulSoup4 库
    is_bs4_available,
    # 检查是否安装了 coloredlogs 库
    is_coloredlogs_available,
    # 检查是否安装了 OpenCV2 库
    is_cv2_available,
    # 检查是否安装了 Cython 编译器
    is_cython_available,
    # 检查是否安装了 datasets 库
    is_datasets_available,
    # 检查是否安装了 Decord 库
    is_decord_available,
    # 检查是否安装了 Detectron2 库
    is_detectron2_available,
    # 检查是否安装了 Essentia 库
    is_essentia_available,
    # 检查是否安装了 Faiss 库
    is_faiss_available,
    # 检查是否安装了 Flash Attn 2.x 库
    is_flash_attn_2_available,
    # 检查是否安装了 Flash Attn 2.10 或更高版本
    is_flash_attn_greater_or_equal_2_10,
    # 检查是否安装了 Flax 库
    is_flax_available,
    # 检查是否安装了 FSDP 库
    is_fsdp_available,
    # 检查是否安装了 ftfy 库
    is_ftfy_available,
    # 检查是否安装了 g2p_en 库
    is_g2p_en_available,
    # 检查是否安装了 GaloreTorch 库
    is_galore_torch_available,
    # 检查是否在笔记本环境中运行
    is_in_notebook,
    # 检查是否安装了 Intel Extension for PyTorch (IPEX)
    is_ipex_available,
    # 检查是否安装了 Jinja2 模板引擎
    is_jinja_available,
    # 检查是否安装了 Juman++ 库
    is_jumanpp_available,
    # 检查是否安装了 KenLM 库
    is_kenlm_available,
    # 检查是否安装了 Keras NLP 库
    is_keras_nlp_available,
    # 检查是否安装了 Levenshtein 库
    is_levenshtein_available,
    # 检查是否安装了 librosa 库
    is_librosa_available,
    # 检查是否安装了 MLX 库
    is_mlx_available,
    # 检查是否安装了 N-ATTEN 库
    is_natten_available,
    # 检查是否安装了 Ninja 编译器
    is_ninja_available,
    # 检查是否安装了 NLTK 库
    is_nltk_available,
    # 检查是否安装了 ONNX 运行时
    is_onnx_available,
    # 检查是否安装了 OpenAI 库
    is_openai_available,
    # 检查是否安装了 Optimum 库
    is_optimum_available,
    # 检查是否安装了 pandas 库
    is_pandas_available,
    # 检查是否安装了 PEFT 库
    is_peft_available,
    # 检查是否安装了 Phonemizer 库
    is_phonemizer_available,
    # 检查是否安装了 PrettyMIDI 库
    is_pretty_midi_available,
    # 检查是否安装了 Protocol Buffers 库
    is_protobuf_available,
    # 检查是否安装了 psutil 库
    is_psutil_available,
    # 检查是否安装了 Py3nvml 库
    is_py3nvml_available,
    # 检查是否安装了 PyCTCDecode 库
    is_pyctcdecode_available,
    # 检查是否安装了 PyTesseract 库
    is_pytesseract_available,
    # 检查是否安装了 pytest 测试框架
    is_pytest_available,
    # 检查是否支持 PyTorch 量化
    is_pytorch_quantization_available,
    # 检查是否安装了 Quanto 库
    is_quanto_available,
    # 检查是否安装了 rjieba 库
    is_rjieba_available,
    # 检查是否安装了 Sacremoses 库
    is_sacremoses_available,
    # 检查是否支持 SafeTensors 库
    is_safetensors_available,
    # 检查 SageMaker 是否启用了分布式训练
    is_sagemaker_dp_enabled,
    # 检查 SageMaker 是否启用了多进程训练
    is_sagemaker_mp_enabled,
    # 检查是否安装了 SciPy 库
    is_scipy_available,
    # 检查是否安装了 SentencePiece 库
    is_sentencepiece_available,
    # 检查是否安装了 SeqIO 库
    is_seqio_available,
    # 检查是否安装了 scikit-learn 库
    is_sklearn_available,
    # 检查是否安装了 SoundFile 库
    is_soundfile_availble,
    # 检查是否安装了 spaCy 库
    is_spacy_available,
    # 检查是否支持语音处理库
    is_speech_available,
    # 检查是否安装了 Sudachi 分词器
    is_sudachi_available,
    # 检查是否安装了 SudachiProjection 库
    is_sudachi_projection_available,
    # 检查是否支持 TensorFlow Probability 库
    is_tensorflow_probability_available,
    # 检查是否支持 TensorFlow Text 库
    is_tensorflow_text_available,
    # 检查是否支持 TF2ONNX 库
    is_tf2onnx_available,
    # 检查是否支持 TensorFlow 库
    is_tf_available,
    # 检查是否安装了 timm 库
    is_timm_available,
    # 检查是否支持 Tokenizers 库
    is_tokenizers_available,
    # 检查是否支持 PyTorch 库
    is_torch_available,
    # 检查是否在设备上支持 BF16 操作
    is_torch_bf16_available_on_device,
    # 检查是否在 CPU 上支持 BF16 操作
    is_torch_bf16_cpu_available,
    # 检查是否在 GPU 上支持 BF16 操作
    is_torch_bf16_gpu_available,
    # 检查是否支持 PyTorch 编译器
    is_torch_compile_available,
    # 检查是否支持 PyTorch CUDA
    is_torch_cuda_available,
    # 检查是否在设备上支持 FP16 操作
    is_torch_fp16_available_on_device,
    # 检查是否支持 PyTorch FX
    is_torch_fx_available,
    # 检查是否支持 PyTorch FX 代理
    is_torch_fx_proxy,
    # 检查是否支持 PyTorch MPS
    is_torch_mps_available,
    # 检查是否支持 PyTorch NeuronCore
    is_torch_neuroncore_available,
    # 检查是否支持 PyTorch NPU
    is_torch_npu_available,
    # 检查是否支持 PyTorch SDPA
    is_torch_sdpa_available,
    # 检查是否支持 PyTorch TensorRT FX
    is_torch_tensorrt_fx_available,
    # 检查是否支持 PyTorch TF32
    is_torch_tf32_available,
    # 检查是否支持 PyTorch TPU
    is_torch_tpu_available,
    # 检查是否支持 PyTorch XLA
    is_torch_xla_available,
    # 检查是否支持 PyTorch XPU
    is_torch_xpu_available,
    # 检查是否安装了 torchaudio 库
    is_torchaudio_available,
    # 检查是否支持 TorchDistX 库
    is_torchdistx_available,
    # 检查是否支持 TorchDynamo 库
    is_torchdynamo_available,
    # 检查 TorchDynamo 是否正在编译
    is_torchdynamo_compiling,
    # 检查是否安装了 torchvision 库
    is_torchvision_available,
    # 检查是否在 SageMaker 上运行训练任务
    is_training_run_on_sagemaker,
    # 检查所需的后端是否安装
    requires_backends,
    # 检查是否为 Torch 专用方法
    torch_only_method,
# 导入所需模块和变量
from .peft_utils import (
    ADAPTER_CONFIG_NAME,      # 导入适配器配置名称常量
    ADAPTER_SAFE_WEIGHTS_NAME,  # 导入适配器安全权重名称常量
    ADAPTER_WEIGHTS_NAME,     # 导入适配器权重名称常量
    check_peft_version,       # 导入检查 PEFT 版本函数
    find_adapter_config_file,  # 导入查找适配器配置文件函数
)

# 定义模型权重文件名常量
WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF2_WEIGHTS_INDEX_NAME = "tf_model.h5.index.json"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
FLAX_WEIGHTS_INDEX_NAME = "flax_model.msgpack.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
PROCESSOR_NAME = "processor_config.json"
GENERATION_CONFIG_NAME = "generation_config.json"
MODEL_CARD_NAME = "modelcard.json"

SENTENCEPIECE_UNDERLINE = "▁"
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # 为了向后兼容保留了这个变量

# 定义用于多选问题的虚拟输入数据，XLM 使用它来表示语言
MULTIPLE_CHOICE_DUMMY_INPUTS = [
    [[0, 1, 0, 1], [1, 0, 0, 1]]
] * 2  # 需要仅包含 0 和 1，因为 XLM 用它来表示语言
# 定义虚拟输入数据
DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
# 定义虚拟掩码数据
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]


def check_min_version(min_version):
    # 检查当前安装的 Transformers 版本是否满足最小版本要求
    if version.parse(__version__) < version.parse(min_version):
        if "dev" in min_version:
            # 如果是开发版，生成错误信息
            error_message = (
                "This example requires a source install from HuggingFace Transformers (see "
                "`https://huggingface.co/docs/transformers/installation#install-from-source`),"
            )
        else:
            # 如果是稳定版，生成错误信息
            error_message = f"This example requires a minimum version of {min_version},"
        # 添加当前版本信息到错误信息
        error_message += f" but the version found is {__version__}.\n"
        # 抛出导入错误，包含详细信息和引导链接
        raise ImportError(
            error_message
            + "Check out https://github.com/huggingface/transformers/tree/main/examples#important-note for the examples corresponding to other "
            "versions of HuggingFace Transformers."
        )
```