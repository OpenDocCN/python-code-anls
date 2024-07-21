# `.\pytorch\benchmarks\dynamo\huggingface.py`

```py
#!/usr/bin/env python3
import importlib  # 导入模块，用于动态导入其他模块
import logging  # 导入日志模块，用于记录程序运行日志
import os  # 导入操作系统功能模块，用于与操作系统交互
import re  # 导入正则表达式模块，用于处理字符串匹配
import subprocess  # 导入子进程管理模块，用于执行外部命令
import sys  # 导入系统相关模块，用于访问系统相关变量和函数
import warnings  # 导入警告模块，用于管理警告消息的显示和过滤

try:
    from .common import BenchmarkRunner, download_retry_decorator, main, reset_rng_state  # 尝试从当前包中导入指定模块和函数
except ImportError:
    from common import BenchmarkRunner, download_retry_decorator, main, reset_rng_state  # 导入当前目录下的指定模块和函数

import torch  # 导入PyTorch库

from torch._dynamo.testing import collect_results  # 从PyTorch私有模块中导入函数
from torch._dynamo.utils import clone_inputs  # 从PyTorch私有模块中导入函数

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# Enable FX graph caching
if "TORCHINDUCTOR_FX_GRAPH_CACHE" not in os.environ:  # 检查环境变量是否包含指定项
    torch._inductor.config.fx_graph_cache = True  # 如果环境变量中不存在，则启用FX图缓存功能


def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])  # 使用pip安装指定的Python包


# Disable the flake warnings for the imports. Flake8 does not provide a way to
# disable just warning for the entire file. Disabling flake8 entirely.
# flake8: noqa
imports = [
    "AlbertForPreTraining",  # 导入Hugging Face Transformers中的预训练模型
    "AutoConfig",  # 导入Hugging Face Transformers中的自动配置类
    "AutoModelForCausalLM",  # 导入Hugging Face Transformers中的自动模型类
    "AutoModelForMaskedLM",  # 导入Hugging Face Transformers中的自动模型类
    "AutoModelForSeq2SeqLM",  # 导入Hugging Face Transformers中的自动模型类
    "BigBirdConfig",  # 导入Hugging Face Transformers中的配置类
    "BlenderbotForConditionalGeneration",  # 导入Hugging Face Transformers中的模型类
    "BlenderbotModel",  # 导入Hugging Face Transformers中的模型类
    "BlenderbotSmallForConditionalGeneration",  # 导入Hugging Face Transformers中的模型类
    "BlenderbotSmallModel",  # 导入Hugging Face Transformers中的模型类
    "CLIPModel",  # 导入Hugging Face Transformers中的模型类
    "CLIPVisionModel",  # 导入Hugging Face Transformers中的模型类
    "ElectraForPreTraining",  # 导入Hugging Face Transformers中的预训练模型
    "GPT2ForSequenceClassification",  # 导入Hugging Face Transformers中的模型类
    "GPTJForSequenceClassification",  # 导入Hugging Face Transformers中的模型类
    "GPTNeoForSequenceClassification",  # 导入Hugging Face Transformers中的模型类
    "HubertForSequenceClassification",  # 导入Hugging Face Transformers中的模型类
    "LxmertForPreTraining",  # 导入Hugging Face Transformers中的预训练模型
    "LxmertForQuestionAnswering",  # 导入Hugging Face Transformers中的模型类
    "MarianForCausalLM",  # 导入Hugging Face Transformers中的模型类
    "MarianModel",  # 导入Hugging Face Transformers中的模型类
    "MarianMTModel",  # 导入Hugging Face Transformers中的模型类
    "PegasusForConditionalGeneration",  # 导入Hugging Face Transformers中的模型类
    "PegasusModel",  # 导入Hugging Face Transformers中的模型类
    "ReformerConfig",  # 导入Hugging Face Transformers中的配置类
    "ViTForImageClassification",  # 导入Hugging Face Transformers中的模型类
    "ViTForMaskedImageModeling",  # 导入Hugging Face Transformers中的模型类
    "ViTModel",  # 导入Hugging Face Transformers中的模型类
]


def process_hf_reformer_output(out):
    assert isinstance(out, list)  # 断言out是一个列表
    # second output is unstable
    return [elem for i, elem in enumerate(out) if i != 1]  # 返回out列表中除第二个元素外的所有元素


try:
    mod = importlib.import_module("transformers")  # 尝试动态导入transformers模块
    for cls in imports:
        if not hasattr(mod, cls):  # 检查transformers模块中是否包含指定的类
            raise ModuleNotFoundError  # 如果没有找到指定类，则抛出ModuleNotFoundError异常
except ModuleNotFoundError:
    print("Installing HuggingFace Transformers...")
    pip_install("git+https://github.com/huggingface/transformers.git#egg=transformers")  # 安装Hugging Face Transformers库
finally:
    for cls in imports:
        exec(f"from transformers import {cls}")  # 动态导入Hugging Face Transformers中指定的类


# These models contain the models present in huggingface_models_list. It is a
# combination of models supported by HF Fx parser and some manually supplied
# models. For these models, we already know the largest batch size that can fit
# on A100 GPUs - 40 GB.
BATCH_SIZE_KNOWN_MODELS = dict()  # 创建一个空字典，用于存储已知模型的最大批处理大小


# Get the list of models and their batch sizes
MODELS_FILENAME = os.path.join(os.path.dirname(__file__), "huggingface_models_list.txt")  # 获取模型列表文件的完整路径
assert os.path.exists(MODELS_FILENAME)  # 断言模型列表文件存在
with open(MODELS_FILENAME, "r") as fh:
    lines = fh.readlines()  # 读取模型列表文件的所有行
    lines = [line.rstrip() for line in lines]  # 去除每行末尾的换行符
    for line in lines:
        model_name, batch_size = line.split(",")  # 根据逗号分隔每行内容，获取模型名称和批处理大小
        batch_size = int(batch_size)  # 将批处理大小转换为整数类型
        BATCH_SIZE_KNOWN_MODELS[model_name] = batch_size  # 将模型名称和批处理大小添加到字典中
# 检查已知模型批处理大小是否为空
assert len(BATCH_SIZE_KNOWN_MODELS)

# 定义跳过的模型集合，这些模型由于特定原因被跳过测试
SKIP = {
    # 由于不支持 .eval()，设置准确性测试困难
    "Reformer",
    # 深拷贝失败
    "BlenderbotForConditionalGeneration",
    "GPTNeoForCausalLM",
    "GPTNeoForSequenceClassification",
    # 即使批处理大小为 1 也会失败
    "GPTJForCausalLM",
    "GPTJForQuestionAnswering",
}

# 定义每个模型的批处理大小除数，用于测试和推断
BATCH_SIZE_DIVISORS = {
    "AlbertForMaskedLM": 2,
    "AlbertForQuestionAnswering": 2,
    "AllenaiLongformerBase": 2,
    "BartForCausalLM": 2,
    "BartForConditionalGeneration": 2,
    "BertForMaskedLM": 2,
    "BertForQuestionAnswering": 2,
    "BlenderbotForCausalLM": 8,
    # "BlenderbotForConditionalGeneration" : 16,
    "BlenderbotSmallForCausalLM": 4,
    "BlenderbotSmallForConditionalGeneration": 2,
    "CamemBert": 2,
    "DebertaForMaskedLM": 4,
    "DebertaForQuestionAnswering": 2,
    "DebertaV2ForMaskedLM": 4,
    "DebertaV2ForQuestionAnswering": 8,
    "DistilBertForMaskedLM": 2,
    "DistilBertForQuestionAnswering": 2,
    "DistillGPT2": 2,
    "ElectraForCausalLM": 2,
    "ElectraForQuestionAnswering": 2,
    "GPT2ForSequenceClassification": 2,
    # "GPTJForCausalLM" : 2,
    # "GPTJForQuestionAnswering" : 2,
    # "GPTNeoForCausalLM" : 32,
    # "GPTNeoForSequenceClassification" : 2,
    "GoogleFnet": 2,
    "LayoutLMForMaskedLM": 2,
    "LayoutLMForSequenceClassification": 2,
    "M2M100ForConditionalGeneration": 4,
    "MBartForCausalLM": 2,
    "MBartForConditionalGeneration": 2,
    "MT5ForConditionalGeneration": 2,
    "MegatronBertForCausalLM": 4,
    "MegatronBertForQuestionAnswering": 2,
    "MobileBertForMaskedLM": 2,
    "MobileBertForQuestionAnswering": 2,
    "OPTForCausalLM": 2,
    "PLBartForCausalLM": 2,
    "PLBartForConditionalGeneration": 2,
    "PegasusForCausalLM": 4,
    "PegasusForConditionalGeneration": 2,
    "RobertaForCausalLM": 2,
    "RobertaForQuestionAnswering": 2,
    "Speech2Text2ForCausalLM": 4,
    "T5ForConditionalGeneration": 2,
    "T5Small": 2,
    "TrOCRForCausalLM": 2,
    "XGLMForCausalLM": 4,
    "XLNetLMHeadModel": 2,
    "YituTechConvBert": 2,
}

# 需要跳过准确性检查的模型集合，这些模型过大或配置复杂，不支持同时使用 eager、dynamo 和 fp64_numbers
SKIP_ACCURACY_CHECK_MODELS = {
    "DebertaV2ForMaskedLM",
    "BlenderbotForCausalLM",
}

# 由于控制流问题需要跳过的模型集合
SKIP_DUE_TO_CONTROL_FLOW = {"AllenaiLongformerBase"}

# 需要更高容忍度训练的模型集合
REQUIRE_HIGHER_TOLERANCE_TRAINING = {
    "MT5ForConditionalGeneration",
    # 在 CI GCP A100 上 AlbertForQuestionAnswering 失败，但错误似乎不会影响
    "AlbertForQuestionAnswering",
}

# 需要更高容忍度推断的模型集合
REQUIRE_HIGHER_TOLERANCE_INFERENCE = {
    "GPT2ForSequenceClassification",
    "RobertaForQuestionAnswering",
}

# 需要在 CPU 模式下运行的模型集合，例如由于内存溢出问题
SKIP_FOR_CPU = {
    "OPTForCausalLM",  # OOMs
}

# 仅支持评估模式的模型集合，例如在训练模式下失败
ONLY_EVAL_MODE = {
    "M2M100ForConditionalGeneration",  # 使用 dynamo 在训练模式下失败
}

# 仅支持 FP32 的模型集合
FP32_ONLY_MODELS = {
    "GoogleFnet",
}

# 根据模型类名获取模型模块的函数定义，尚未实现其功能
def get_module_cls_by_model_name(model_cls_name):
    # 根据模型类名查找对应的模块路径映射字典
    _module_by_model_name = {
        "Speech2Text2Decoder": "transformers.models.speech_to_text_2.modeling_speech_to_text_2",
        "TrOCRDecoder": "transformers.models.trocr.modeling_trocr",
    }
    
    # 根据模型类名获取对应的模块路径，如果找不到，默认使用 "transformers" 模块
    module_name = _module_by_model_name.get(model_cls_name, "transformers")
    
    # 使用 importlib 动态导入指定名称的模块
    module = importlib.import_module(module_name)
    
    # 返回指定模块中的指定类名对应的对象或方法
    return getattr(module, model_cls_name)
def get_sequence_length(model_cls, model_name):
    # 根据模型名称确定序列长度
    if model_name.startswith(("Blenderbot",)):
        seq_length = 128  # 如果模型名以 "Blenderbot" 开头，设定序列长度为128
    elif model_name.startswith(("GPT2", "Bart", "T5", "PLBart", "MBart")):
        seq_length = 1024  # 如果模型名以指定的几种开头之一，设定序列长度为1024
    elif model_name in ("AllenaiLongformerBase", "BigBird"):
        seq_length = 1024  # 如果模型名在指定的模型名列表中，设定序列长度为1024
    elif model_name.startswith("OPT"):
        seq_length = 2048  # 如果模型名以 "OPT" 开头，设定序列长度为2048
    elif "Reformer" in model_name:
        seq_length = 4096  # 如果模型名包含 "Reformer"，设定序列长度为4096
    elif model_name.startswith(
        (
            "Albert",
            "Deberta",
            "Layout",
            "Electra",
            "XLNet",
            "MegatronBert",
            "Bert",
            "Roberta",
        )
    ) or model_name in ("DistillGPT2", "GoogleFnet", "YituTechConvBert", "CamemBert"):
        seq_length = 512  # 如果模型名以指定的几种开头之一，或者在指定的模型名列表中，设定序列长度为512
    elif model_name in ("TrOCRForCausalLM"):
        seq_length = 256  # 如果模型名为 "TrOCRForCausalLM"，设定序列长度为256
    elif model_name.startswith("MobileBert"):
        seq_length = 128  # 如果模型名以 "MobileBert" 开头，设定序列长度为128
    elif model_name.startswith("Wav2Vec2"):
        # 如果模型名以 "Wav2Vec2" 开头，设定序列长度为10000
        # 注意：10000 是一个更现实的大小选择是 155136
        seq_length = 10000
    else:
        log.info(
            f"Sequence Length not defined for {model_name}. Choosing 128 arbitrarily"
        )
        seq_length = 128  # 如果找不到模型名对应的序列长度，记录日志并且默认设定序列长度为128
    return seq_length


def generate_inputs_for_model(
    model_cls, model, model_name, bs, device, include_loss_args=False
):
    # TODO - Check if following values are representative
    num_choices = 3  # 多选题的选择个数
    num_visual_features = 42  # 视觉特征的数量
    seq_length = get_sequence_length(model_cls, model_name)  # 获取模型的序列长度
    vocab_size = model.config.vocab_size  # 获取模型的词汇表大小

    if model_name.startswith("Wav2Vec2"):
        # 如果模型名以 "Wav2Vec2" 开头，设定目标长度为100，并返回对应的输入字典
        target_length = 100
        return {
            "input_values": torch.randn((bs, seq_length), device=device),
            # 添加因为示例训练脚本中有的内容
            "attention_mask": rand_int_tensor(device, 0, 2, (bs, seq_length)),
            "labels": rand_int_tensor(device, 0, vocab_size, (bs, target_length)),
        }

    if model_name.endswith("MultipleChoice"):
        # 如果模型名以 "MultipleChoice" 结尾，生成多选题的输入张量
        input = rand_int_tensor(device, 0, vocab_size, (bs, num_choices, seq_length))
    elif model_name.startswith("Roberta"):
        # 如果模型名以 "Roberta" 开头，生成特定的输入张量
        input = rand_int_tensor(device, 0, 1, (bs, seq_length))
    else:
        # 否则，生成通用的输入张量
        input = rand_int_tensor(device, 0, vocab_size, (bs, seq_length))

    if "Bart" in model_name:
        input[:, -1] = model.config.eos_token_id  # 如果模型名包含 "Bart"，在输入张量的末尾添加结束标记

    input_dict = {"input_ids": input}  # 创建包含输入张量的字典
    # 检查模型名称是否以特定字符串开头，或者模型类是否属于特定类型之一
    if (
        model_name.startswith("T5")
        or model_name.startswith("M2M100")
        or model_name.startswith("MT5")
        or model_cls
        in [
            BlenderbotModel,
            BlenderbotSmallModel,
            BlenderbotForConditionalGeneration,
            BlenderbotSmallForConditionalGeneration,
            PegasusModel,
            PegasusForConditionalGeneration,
            MarianModel,
            MarianMTModel,
        ]
    ):
        # 如果条件成立，将输入数据作为解码器的输入标识存储在输入字典中
        input_dict["decoder_input_ids"] = input

    # 检查模型名称是否以特定字符串开头
    if model_name.startswith("Lxmert"):
        # 如果条件成立，获取模型配置中的视觉特征维度和视觉位置编码维度
        visual_feat_dim, visual_pos_dim = (
            model.config.visual_feat_dim,
            model.config.visual_pos_dim,
        )
        # 使用随机生成的张量作为视觉特征数据，存储在输入字典中
        input_dict["visual_feats"] = torch.randn(
            bs, num_visual_features, visual_feat_dim
        )
        # 使用随机生成的张量作为视觉位置编码数据，存储在输入字典中
        input_dict["visual_pos"] = torch.randn(bs, num_visual_features, visual_pos_dim)
    # 如果需要包含损失函数参数
    if include_loss_args:
        # 如果模型名以 "PreTraining" 结尾
        if model_name.endswith("PreTraining"):
            # 如果模型类是 ElectraForPreTraining 或 LxmertForPreTraining
            if model_cls in [ElectraForPreTraining, LxmertForPreTraining]:
                # 创建标签键 "labels"，并生成随机整数张量，形状为 (bs, seq_length)，值在 0 到 1 之间，设备为指定的 device
                input_dict["labels"] = rand_int_tensor(device, 0, 1, (bs, seq_length))
            else:
                # 否则，根据模型类确定标签名称
                label_name = (
                    "sentence_order_label"
                    if model_cls in [AlbertForPreTraining]
                    else "next_sentence_label"
                )
                # 创建标签键 "labels"，并生成随机整数张量，形状为 (bs, seq_length)，值在 0 到 vocab_size 之间，设备为指定的 device
                input_dict["labels"] = (
                    rand_int_tensor(device, 0, vocab_size, (bs, seq_length)),
                )
                # 创建相应的标签键，生成随机整数张量，形状为 (bs,)，值在 0 到 1 之间，设备为指定的 device
                input_dict[label_name] = rand_int_tensor(device, 0, 1, (bs,))
        # 如果模型名以 "QuestionAnswering" 结尾
        elif model_name.endswith("QuestionAnswering"):
            # 创建标签键 "start_positions"，并生成随机整数张量，形状为 (bs,)，值在 0 到 seq_length 之间，设备为指定的 device
            input_dict["start_positions"] = rand_int_tensor(device, 0, seq_length, (bs,))
            # 创建标签键 "end_positions"，并生成随机整数张量，形状为 (bs,)，值在 0 到 seq_length 之间，设备为指定的 device
            input_dict["end_positions"] = rand_int_tensor(device, 0, seq_length, (bs,))
        # 如果模型名以 "MaskedLM", "HeadModel", "CausalLM" 或 "DoubleHeadsModel" 结尾
        elif (
            model_name.endswith("MaskedLM")
            or model_name.endswith("HeadModel")
            or model_name.endswith("CausalLM")
            or model_name.endswith("DoubleHeadsModel")
        ):
            # 创建标签键 "labels"，并生成随机整数张量，形状为 (bs, seq_length)，值在 0 到 vocab_size 之间，设备为指定的 device
            input_dict["labels"] = rand_int_tensor(device, 0, vocab_size, (bs, seq_length))
        # 如果模型名以 "TokenClassification" 结尾
        elif model_name.endswith("TokenClassification"):
            # 创建标签键 "labels"，并生成随机整数张量，形状为 (bs, seq_length)，值在 0 到 model.config.num_labels - 1 之间，设备为指定的 device
            input_dict["labels"] = rand_int_tensor(device, 0, model.config.num_labels - 1, (bs, seq_length))
        # 如果模型名以 "MultipleChoice" 结尾
        elif model_name.endswith("MultipleChoice"):
            # 创建标签键 "labels"，并生成随机整数张量，形状为 (bs,)，值在 0 到 num_choices 之间，设备为指定的 device
            input_dict["labels"] = rand_int_tensor(device, 0, num_choices, (bs,))
        # 如果模型名以 "SequenceClassification" 结尾
        elif model_name.endswith("SequenceClassification"):
            # 创建标签键 "labels"，并生成随机整数张量，形状为 (bs,)，值在 0 到 model.config.num_labels - 1 之间，设备为指定的 device
            input_dict["labels"] = rand_int_tensor(device, 0, model.config.num_labels - 1, (bs,))
        # 如果模型名以 "NextSentencePrediction" 结尾
        elif model_name.endswith("NextSentencePrediction"):
            # 创建标签键 "labels"，并生成随机整数张量，形状为 (bs,)，值在 0 到 1 之间，设备为指定的 device
            input_dict["labels"] = rand_int_tensor(device, 0, 1, (bs,))
        # 如果模型名以 "ForConditionalGeneration" 结尾
        elif model_name.endswith("ForConditionalGeneration"):
            # 创建标签键 "labels"，并生成随机整数张量，形状为 (bs, seq_length)，值在 0 到 vocab_size - 1 之间，设备为指定的 device
            input_dict["labels"] = rand_int_tensor(device, 0, vocab_size - 1, (bs, seq_length))
        # 如果模型名在 EXTRA_MODELS 中
        elif model_name in EXTRA_MODELS:
            # 创建标签键 "labels"，并生成随机整数张量，形状为 (bs, seq_length)，值在 0 到 vocab_size 之间，设备为指定的 device
            input_dict["labels"] = rand_int_tensor(device, 0, vocab_size, (bs, seq_length))
        else:
            # 如果以上条件都不满足，抛出未实现的错误
            raise NotImplementedError(
                f"Class {model_name} unsupported for training test "
            )

    # 返回输入字典
    return input_dict
def rand_int_tensor(device, low, high, shape):
    return torch.randint(
        low,  # 最小值
        high,  # 最大值（不包括）
        shape,  # 张量的形状
        device=device,  # 设备类型，如CPU或GPU
        dtype=torch.int64,  # 数据类型为64位整数
        requires_grad=False,  # 不需要计算梯度
    )


EXTRA_MODELS = {
    "AllenaiLongformerBase": (
        AutoConfig.from_pretrained("allenai/longformer-base-4096"),  # 加载预训练模型配置
        AutoModelForMaskedLM,  # 自动模型用于掩码语言建模
    ),
    "Reformer": (
        ReformerConfig(),  # Reformer模型配置
        AutoModelForMaskedLM,  # 自动模型用于掩码语言建模
    ),
    "T5Small": (
        AutoConfig.from_pretrained("t5-small"),  # 加载预训练模型配置
        AutoModelForSeq2SeqLM,  # 自动模型用于序列到序列的语言建模
    ),
    # "BigBird": (
    #     BigBirdConfig(attention_type="block_sparse"),  # BigBird模型配置，注意类型为块稀疏
    #     AutoModelForMaskedLM,  # 自动模型用于掩码语言建模
    # ),
    "DistillGPT2": (
        AutoConfig.from_pretrained("distilgpt2"),  # 加载预训练模型配置
        AutoModelForCausalLM,  # 自动模型用于因果语言建模
    ),
    "GoogleFnet": (
        AutoConfig.from_pretrained("google/fnet-base"),  # 加载预训练模型配置
        AutoModelForMaskedLM,  # 自动模型用于掩码语言建模
    ),
    "YituTechConvBert": (
        AutoConfig.from_pretrained("YituTech/conv-bert-base"),  # 加载预训练模型配置
        AutoModelForMaskedLM,  # 自动模型用于掩码语言建模
    ),
    "CamemBert": (
        AutoConfig.from_pretrained("camembert-base"),  # 加载预训练模型配置
        AutoModelForMaskedLM,  # 自动模型用于掩码语言建模
    ),
}


class HuggingfaceRunner(BenchmarkRunner):
    def __init__(self):
        super().__init__()
        self.suite_name = "huggingface"  # 设置基准测试套件名称

    @property
    def skip_models_for_cpu(self):
        return SKIP_FOR_CPU  # 返回需要在CPU上跳过的模型列表

    @property
    def fp32_only_models(self):
        return FP32_ONLY_MODELS  # 返回仅支持单精度浮点数运算的模型列表

    @property
    def skip_models_due_to_control_flow(self):
        return SKIP_DUE_TO_CONTROL_FLOW  # 返回由于控制流问题需要跳过的模型列表

    def _get_model_cls_and_config(self, model_name):
        if model_name not in EXTRA_MODELS:
            model_cls = get_module_cls_by_model_name(model_name)  # 获取模型类别名对应的模型类
            config_cls = model_cls.config_class  # 获取模型的配置类
            config = config_cls()

            # NB: some models need a pad token defined to handle BS > 1
            if (
                model_cls
                in [
                    GPT2ForSequenceClassification,
                    GPTNeoForSequenceClassification,
                    GPTJForSequenceClassification,
                ]
                or model_cls.__name__.startswith("Roberta")
                or model_cls.__name__.startswith("Marian")
            ):
                config.pad_token_id = 0  # 设置填充标记ID为0以处理批量大小大于1的情况

        else:
            config, model_cls = EXTRA_MODELS[model_name]  # 获取额外模型中的配置和模型类

        return model_cls, config  # 返回模型类和配置

    @download_retry_decorator
    def _download_model(self, model_name):
        model_cls, config = self._get_model_cls_and_config(model_name)
        if "auto" in model_cls.__module__:
            # Handle auto classes
            model = model_cls.from_config(config)  # 根据配置创建自动加载的模型实例
        else:
            model = model_cls(config)  # 根据配置创建模型实例
        return model  # 返回下载的模型实例

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
        extra_args=None,
        ):
            # 从 self.args.training 中获取是否处于训练模式
            is_training = self.args.training
            # 从 self.args.use_eval_mode 中获取是否使用评估模式
            use_eval_mode = self.args.use_eval_mode
            # 设置张量的数据类型为 float32
            dtype = torch.float32
            # 重置随机数生成器的状态
            reset_rng_state()
            # 根据模型名称获取模型类和配置
            model_cls, config = self._get_model_cls_and_config(model_name)
            # 下载模型
            model = self._download_model(model_name)
            # 将模型移动到指定的设备并设置数据类型
            model = model.to(device, dtype=dtype)
            # 如果启用激活检查点，则启用模型的梯度检查点
            if self.args.enable_activation_checkpointing:
                model.gradient_checkpointing_enable()
            # 如果模型名称在已知的批量大小模型中
            if model_name in BATCH_SIZE_KNOWN_MODELS:
                # 使用已知模型的默认批量大小
                batch_size_default = BATCH_SIZE_KNOWN_MODELS[model_name]
            # 否则，如果未指定批量大小
            elif batch_size is None:
                # 设置默认批量大小为 16
                batch_size_default = 16
                log.info(
                    f"Batch size not specified for {model_name}. Setting batch_size=16"
                )

            # 如果批量大小未指定
            if batch_size is None:
                # 使用默认批量大小
                batch_size = batch_size_default
                # 如果模型名称在批量大小除数中
                if model_name in BATCH_SIZE_DIVISORS:
                    # 调整批量大小为原批量大小除以除数的最大值，最小为1
                    batch_size = max(int(batch_size / BATCH_SIZE_DIVISORS[model_name]), 1)
                    log.info(
                        f"Running smaller batch size={batch_size} for {model_name}, orig batch_size={batch_size_default}"
                    )

            # 为模型生成输入示例
            example_inputs = generate_inputs_for_model(
                model_cls, model, model_name, batch_size, device, include_loss_args=True
            )

            # 用于检查正确梯度，但不会消除 dropout 计算
            for attr in dir(config):
                if "drop" in attr and isinstance(getattr(config, attr), float):
                    setattr(config, attr, 1e-30)

            # 如果处于训练状态且未使用评估模式，并且未要求准确性检查和模型名称不在仅评估模式列表中
            if (
                is_training
                and not use_eval_mode
                and not (self.args.accuracy and model_name in ONLY_EVAL_MODE)
            ):
                # 设置模型为训练模式
                model.train()
            else:
                # 否则设置模型为评估模式
                model.eval()

            # 验证模型
            self.validate_model(model, example_inputs)
            # 返回设备、模型名称、模型、示例输入、批量大小
            return device, model_name, model, example_inputs, batch_size

        def iter_model_names(self, args):
            # 获取已知批量大小模型和额外模型的所有模型名称列表，并排序
            model_names = list(BATCH_SIZE_KNOWN_MODELS.keys()) + list(EXTRA_MODELS.keys())
            model_names = set(model_names)
            model_names = sorted(model_names)

            # 获取用于基准测试的起始和结束索引
            start, end = self.get_benchmark_indices(len(model_names))
            # 遍历模型名称列表
            for index, model_name in enumerate(model_names):
                # 如果索引小于起始索引或大于等于结束索引，则继续下一个模型
                if index < start or index >= end:
                    continue
                # 如果模型名称不匹配过滤条件、在排除列表中、在精确排除列表中或在跳过列表中，则继续下一个模型
                if (
                    not re.search("|".join(args.filter), model_name, re.I)
                    or re.search("|".join(args.exclude), model_name, re.I)
                    or model_name in args.exclude_exact
                    or model_name in SKIP
                ):
                    continue
                # 返回符合条件的模型名称
                yield model_name

        @property
        def skip_accuracy_checks_large_models_dashboard(self):
            # 如果启用了仪表板或准确性检查，则返回跳过准确性检查的模型集合
            if self.args.dashboard or self.args.accuracy:
                return SKIP_ACCURACY_CHECK_MODELS
            # 否则返回空集合
            return set()

        @property
        def get_output_amp_train_process_func(self):
            # 返回空字典
            return {}

        def pick_grad(self, name, is_training):
            # 如果处于训练状态，则启用梯度
            if is_training:
                return torch.enable_grad()
            else:
                # 否则禁用梯度
                return torch.no_grad()
    # 根据训练状态、设备和名称获取容差和余弦标志
    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        # 从参数中获取余弦标志
        cosine = self.args.cosine
        # 如果是训练阶段
        if is_training:
            # 如果名称在需要更高容差的训练列表中
            if name in REQUIRE_HIGHER_TOLERANCE_TRAINING:
                # 返回更高的容差和当前余弦标志
                return 2e-2, cosine
            else:
                # 否则返回较低的容差和当前余弦标志
                return 1e-2, cosine
        else:
            # 如果是推断阶段
            # 如果名称在需要更高容差的推断列表中
            if name in REQUIRE_HIGHER_TOLERANCE_INFERENCE:
                # 返回更高的容差和当前余弦标志
                return 4e-3, cosine
        # 默认返回较低的容差和当前余弦标志
        return 1e-3, cosine

    # 计算损失函数，返回预测的第一个元素
    def compute_loss(self, pred):
        return pred[0]

    # 执行模型的前向传播，支持自动混合精度
    def forward_pass(self, mod, inputs, collect_outputs=True):
        # 使用自动混合精度上下文管理器
        with self.autocast(**self.autocast_arg):
            # 调用模型的前向传播方法，传入输入参数
            return mod(**inputs)

    # 执行模型的前向和反向传播，支持自动混合精度
    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        # 克隆输入以避免原始输入修改
        cloned_inputs = clone_inputs(inputs)
        # 优化器清空梯度
        self.optimizer_zero_grad(mod)
        # 使用自动混合精度上下文管理器
        with self.autocast(**self.autocast_arg):
            # 调用模型的前向传播方法，传入克隆后的输入
            pred = mod(**cloned_inputs)
            # 计算预测结果的损失
            loss = self.compute_loss(pred)
        # 使用梯度放大器对损失进行梯度计算
        self.grad_scaler.scale(loss).backward()
        # 优化器执行单步参数更新
        self.optimizer_step()
        # 如果需要收集输出结果，则返回模型的结果、预测、损失和克隆后的输入
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        # 否则返回空值
        return None
# 定义一个函数，用于刷新 HF Fx 追踪器支持的模型名称和批处理大小
def refresh_model_names_and_batch_sizes():
    """
    This function reads the HF Fx tracer supported models and finds the largest
    batch size that could fit on the GPU with PyTorch eager.

    The resulting data is written in huggingface_models_list.txt.

    Note - We only need to run this function if we believe that HF Fx tracer now
    supports more models.
    """
    # 导入 transformers.utils.fx 模块作为 hf_fx
    import transformers.utils.fx as hf_fx

    # 用于存储不同家族的模型
    family = dict()
    # 用于记录已经看到的语言模型
    lm_seen = set()
    # 用于记录已经看到的家族
    family_seen = set()
    
    # 遍历 HF Fx 追踪器支持的所有模型类名
    for cls_name in hf_fx._SUPPORTED_MODELS:
        # 如果模型类名中不包含 "For"，则跳过
        if "For" not in cls_name:
            continue
        
        # 根据模型类名获取模型类对象
        model_cls = get_module_cls_by_model_name(cls_name)

        # TODO: AttributeError: '*Config' object has no attribute 'vocab_size'
        # 如果模型类属于以下类别，则跳过
        if model_cls in [
            CLIPModel,
            CLIPVisionModel,
            # SwinForImageClassification,
            # SwinForImageClassification,
            # SwinForMaskedImageModeling,
            # SwinModel,
            ViTForImageClassification,
            ViTForMaskedImageModeling,
            ViTModel,
        ]:
            continue

        # TODO: AssertionError: Padding_idx must be within num_embeddings
        # 如果模型类属于以下类别，则跳过
        if model_cls in [MarianForCausalLM, MarianMTModel, MarianModel]:
            continue

        # TODO: "model is not supported yet" from HFTracer
        # 如果模型类属于以下类别，则跳过
        if model_cls in [HubertForSequenceClassification]:
            continue

        # TODO: shape mismatch in loss calculation
        # 如果模型类属于以下类别，则跳过
        if model_cls in [LxmertForQuestionAnswering]:
            continue

        # 获取家族名称（模型类名去掉 "For" 后的部分）
        family_name = cls_name.split("For")[0]
        # 如果家族名称不在 family 字典中，则初始化空列表
        if family_name not in family:
            family[family_name] = []
        
        # 根据模型类名的后缀类型将模型类名添加到相应的家族列表中
        if cls_name.endswith(("MaskedLM", "CausalLM")) and family_name not in lm_seen:
            family[family_name].append(cls_name)
            lm_seen.add(family_name)
        elif (
            cls_name.endswith(
                ("SequenceClassification", "ConditionalGeneration", "QuestionAnswering")
            )
            and family_name not in family_seen
        ):
            family[family_name].append(cls_name)
            family_seen.add(family_name)
        elif cls_name.endswith("ImageClassification"):
            family[family_name].append(cls_name)

    # 创建一个集合来存储被选中的模型名称
    chosen_models = set()
    # 将所有家族中的模型类名加入到 chosen_models 中
    for members in family.values():
        chosen_models.update(set(members))

    # 将 EXTRA_MODELS 中的模型名称加入到 chosen_models 中
    chosen_models.update(set(EXTRA_MODELS.keys()))

    # 对选择的模型名称进行排序并逐个处理
    for model_name in sorted(chosen_models):
        try:
            # 调用 subprocess 模块执行命令来查找适合的批处理大小
            subprocess.check_call(
                [sys.executable]
                + sys.argv
                + ["--find-batch-sizes"]
                + [f"--only={model_name}"]
                + [f"--output={MODELS_FILENAME}"]
            )
        except subprocess.SubprocessError:
            # 记录警告信息，指出未能找到适合模型名称的批处理大小
            log.warning(f"Failed to find suitable batch size for {model_name}")


# 定义一个函数，用于主程序入口点执行
def huggingface_main():
    # 如果命令行参数中不包含 "--find-batch-sizes"，则调用刷新模型名称和批处理大小的函数
    # refresh_model_names_and_batch_sizes 函数的调用被注释掉，可能是为了避免不必要的调用
    # if "--find-batch-sizes" not in sys.argv:
    #     refresh_model_names_and_batch_sizes()
    # 设置日志系统的基本配置，将日志级别设为警告级别
    logging.basicConfig(level=logging.WARNING)
    # 忽略警告信息，不输出到标准输出或日志系统中
    warnings.filterwarnings("ignore")
    # 调用主函数，并传入一个 HuggingfaceRunner 的实例作为参数
    main(HuggingfaceRunner())
# 如果当前脚本作为主程序运行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 huggingface_main 的函数来执行主要逻辑
    huggingface_main()
```