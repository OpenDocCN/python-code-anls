# `.\transformers\utils\__init__.py`

```
#!/usr/bin/env python
# coding=utf-8

# 版权声明
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息

from huggingface_hub import get_full_repo_name  # 用于向后兼容
from huggingface_hub.constants import HF_HUB_DISABLE_TELEMETRY as DISABLE_TELEMETRY  # 用于向后兼容
from packaging import version

from .. import __version__
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from .doc import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    copy_func,
    replace_return_docstrings,
)
from .generic import (
    ContextManagers,
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
    CLOUDFRONT_DISTRIB_PREFIX,
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
    ACCELERATE_MIN_VERSION,
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
    is_auto_awq_available,
    is_auto_gptq_available,
    is_bitsandbytes_available,
    is_bs4_available,
    is_coloredlogs_available,
    # 检查是否安装了 OpenCV
    is_cv2_available,
    # 检查是否安装了 Cython
    is_cython_available,
    # 检查是否安装了 datasets 库
    is_datasets_available,
    # 检查是否安装了 Decord
    is_decord_available,
    # 检查是否安装了 Detectron2
    is_detectron2_available,
    # 检查是否安装了 Essentia
    is_essentia_available,
    # 检查是否安装了 Faiss
    is_faiss_available,
    # 检查是否安装了 Flash-Attn 2
    is_flash_attn_2_available,
    # 检查是否安装了 Flash-Attn
    is_flash_attn_available,
    # 检查 Flash-Attn 版本是否大于等于 2.10
    is_flash_attn_greater_or_equal_2_10,
    # 检查是否安装了 Flax
    is_flax_available,
    # 检查是否安装了 FSDP
    is_fsdp_available,
    # 检查是否安装了 ftfy
    is_ftfy_available,
    # 检查是否安装了 g2p-en
    is_g2p_en_available,
    # 检查是否在 notebook 环境中
    is_in_notebook,
    # 检查是否安装了 Intel PyTorch Extension (IPEX)
    is_ipex_available,
    # 检查是否安装了 jieba
    is_jieba_available,
    # 检查是否安装了 Jinja
    is_jinja_available,
    # 检查是否安装了 Juman++
    is_jumanpp_available,
    # 检查是否安装了 KenLM
    is_kenlm_available,
    # 检查是否安装了 Keras NLP
    is_keras_nlp_available,
    # 检查是否安装了 Levenshtein
    is_levenshtein_available,
    # 检查是否安装了 librosa
    is_librosa_available,
    # 检查是否安装了 N-Autoregressive Transformer (NATTEN)
    is_natten_available,
    # 检查是否安装了 Ninja
    is_ninja_available,
    # 检查是否安装了 NLTK
    is_nltk_available,
    # 检查是否安装了 ONNX
    is_onnx_available,
    # 检查是否安装了 OpenAI
    is_openai_available,
    # 检查是否安装了 Optimum
    is_optimum_available,
    # 检查是否安装了 pandas
    is_pandas_available,
    # 检查是否安装了 PEFT
    is_peft_available,
    # 检查是否安装了 Phonemizer
    is_phonemizer_available,
    # 检查是否安装了 pretty_midi
    is_pretty_midi_available,
    # 检查是否安装了 Protobuf
    is_protobuf_available,
    # 检查是否安装了 psutil
    is_psutil_available,
    # 检查是否安装了 Py3nvml
    is_py3nvml_available,
    # 检查是否安装了 PyCTCDecode
    is_pyctcdecode_available,
    # 检查是否安装了 PyTesseract
    is_pytesseract_available,
    # 检查是否安装了 pytest
    is_pytest_available,
    # 检查是否安装了 PyTorch Quantization
    is_pytorch_quantization_available,
    # 检查是否安装了 Rjieba
    is_rjieba_available,
    # 检查是否安装了 Sacremoses
    is_sacremoses_available,
    # 检查是否安装了 SafeTensors
    is_safetensors_available,
    # 检查 SageMaker 是否启用了 Data Parallelism
    is_sagemaker_dp_enabled,
    # 检查 SageMaker 是否启用了 Model Parallelism
    is_sagemaker_mp_enabled,
    # 检查是否安装了 SciPy
    is_scipy_available,
    # 检查是否安装了 SentencePiece
    is_sentencepiece_available,
    # 检查是否安装了 SeqIO
    is_seqio_available,
    # 检查是否安装了 scikit-learn
    is_sklearn_available,
    # 检查是否安装了 SoundFile
    is_soundfile_availble,
    # 检查是否安装了 spaCy
    is_spacy_available,
    # 检查是否安装了 Speech
    is_speech_available,
    # 检查是否安装了 Sudachi
    is_sudachi_available,
    # 检查是否安装了 TensorFlow Probability
    is_tensorflow_probability_available,
    # 检查是否安装了 TensorFlow Text
    is_tensorflow_text_available,
    # 检查是否安装了 tf2onnx
    is_tf2onnx_available,
    # 检查是否安装了 TensorFlow
    is_tf_available,
    # 检查是否安装了 timm
    is_timm_available,
    # 检查是否安装了 Tokenizers
    is_tokenizers_available,
    # 检查是否安装了 PyTorch
    is_torch_available,
    # 检查是否在设备上安装了 Torch BF16
    is_torch_bf16_available_on_device,
    # 检查是否在 CPU 上安装了 Torch BF16
    is_torch_bf16_cpu_available,
    # 检查是否在 GPU 上安装了 Torch BF16
    is_torch_bf16_gpu_available,
    # 检查是否安装了 Torch BF16
    is_torch_bf16_available,
    # 检查是否安装了 Torch Compile
    is_torch_compile_available,
    # 检查是否在设备上安装了 Torch CUDA
    is_torch_cuda_available,
    # 检查是否在设备上安装了 Torch FP16
    is_torch_fp16_available_on_device,
    # 检查是否安装了 Torch FX
    is_torch_fx_available,
    # 检查是否安装了 Torch FX 代理
    is_torch_fx_proxy,
    # 检查是否安装了 Torch MPS
    is_torch_mps_available,
    # 检查是否安装了 Torch NeuronCore
    is_torch_neuroncore_available,
    # 检查是否在设备上安装了 Torch NPU
    is_torch_npu_available,
    # 检查是否安装了 Torch SDPA
    is_torch_sdpa_available,
    # 检查是否安装了 Torch TensorRT FX
    is_torch_tensorrt_fx_available,
    # 检查是否安装了 Torch TF32
    is_torch_tf32_available,
    # 检查是否在设备上安装了 Torch TPU
    is_torch_tpu_available,
    # 检查是否在设备上安装了 Torch XPU
    is_torch_xpu_available,
    # 检查是否安装了 torchaudio
    is_torchaudio_available,
    # 检查是否安装了 Torch Distributed X
    is_torchdistx_available,
    # 检查是否安装了 Torch Dynamo
    is_torchdynamo_available,
    # 检查是否安装了 torchvision
    is_torchvision_available,
    # 检查是否在 SageMaker 上运行训练
    is_training_run_on_sagemaker,
    # 检查是否安装了 Vision
    is_vision_available,
    # 指定所需的后端
    requires_backends,
    # 仅适用于 Torch 的方法
    torch_only_method,
# 导入必要的模块和函数
)
from .peft_utils import (
    ADAPTER_CONFIG_NAME,  # 导入适配器配置文件名
    ADAPTER_SAFE_WEIGHTS_NAME,  # 导入适配器安全权重文件名
    ADAPTER_WEIGHTS_NAME,  # 导入适配器权重文件名
    check_peft_version,  # 导入检查 PEFT 版本的函数
    find_adapter_config_file,  # 导入查找适配器配置文件的函数
)

# 定义各种文件名常量
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
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # 为了向后兼容而保留

# 定义多项选择任务的虚拟输入
MULTIPLE_CHOICE_DUMMY_INPUTS = [
    [[0, 1, 0, 1], [1, 0, 0, 1]]
] * 2  # 需要只包含 0 和 1，因为 XLM 也用于语言
DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]

# 检查最小版本号
def check_min_version(min_version):
    if version.parse(__version__) < version.parse(min_version):
        if "dev" in min_version:
            error_message = (
                "This example requires a source install from HuggingFace Transformers (see "
                "`https://huggingface.co/docs/transformers/installation#install-from-source`),"
            )
        else:
            error_message = f"This example requires a minimum version of {min_version},"
        error_message += f" but the version found is {__version__}.\n"
        raise ImportError(
            error_message
            + "Check out https://github.com/huggingface/transformers/tree/main/examples#important-note for the examples corresponding to other "
            "versions of HuggingFace Transformers."
        )
```