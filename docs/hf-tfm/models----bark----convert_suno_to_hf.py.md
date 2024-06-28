# `.\models\bark\convert_suno_to_hf.py`

```
"""Convert Bark checkpoint."""
# 导入所需的库和模块
import argparse
import os
from pathlib import Path

import torch
from bark.generation import _load_model as _bark_load_model
from huggingface_hub import hf_hub_download

# 导入 Transformers 库中相关的类和函数
from transformers import EncodecConfig, EncodecModel, set_seed
from transformers.models.bark.configuration_bark import (
    BarkCoarseConfig,
    BarkConfig,
    BarkFineConfig,
    BarkSemanticConfig,
)
from transformers.models.bark.generation_configuration_bark import (
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
    BarkGenerationConfig,
    BarkSemanticGenerationConfig,
)
from transformers.models.bark.modeling_bark import BarkCoarseModel, BarkFineModel, BarkModel, BarkSemanticModel
from transformers.utils import logging


# 设置日志级别为 info
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# 设置随机种子
set_seed(770)

# 定义一个字典，用于将模型层次结构中的旧层次名称映射到新名称
new_layer_name_dict = {
    "c_attn": "att_proj",
    "c_proj": "out_proj",
    "c_fc": "in_proj",
    "transformer.": "",
    "h.": "layers.",
    "ln_1": "layernorm_1",
    "ln_2": "layernorm_2",
    "ln_f": "layernorm_final",
    "wpe": "position_embeds_layer",
    "wte": "input_embeds_layer",
}

# 定义远程模型路径的字典
REMOTE_MODEL_PATHS = {
    "text_small": {
        "repo_id": "suno/bark",
        "file_name": "text.pt",
    },
    "coarse_small": {
        "repo_id": "suno/bark",
        "file_name": "coarse.pt",
    },
    "fine_small": {
        "repo_id": "suno/bark",
        "file_name": "fine.pt",
    },
    "text": {
        "repo_id": "suno/bark",
        "file_name": "text_2.pt",
    },
    "coarse": {
        "repo_id": "suno/bark",
        "file_name": "coarse_2.pt",
    },
    "fine": {
        "repo_id": "suno/bark",
        "file_name": "fine_2.pt",
    },
}

# 获取当前文件的路径
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
# 设置默认缓存目录
default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
# 设置最终的缓存目录路径
CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "suno", "bark_v0")


# 根据模型类型和是否使用小模型返回对应的检查点文件路径
def _get_ckpt_path(model_type, use_small=False):
    key = model_type
    if use_small:
        key += "_small"
    return os.path.join(CACHE_DIR, REMOTE_MODEL_PATHS[key]["file_name"])


# 下载模型文件到本地缓存目录
def _download(from_hf_path, file_name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    hf_hub_download(repo_id=from_hf_path, filename=file_name, local_dir=CACHE_DIR)


# 加载模型的函数，根据模型类型和大小选择对应的模型类和配置类
def _load_model(ckpt_path, device, use_small=False, model_type="text"):
    if model_type == "text":
        ModelClass = BarkSemanticModel
        ConfigClass = BarkSemanticConfig
        GenerationConfigClass = BarkSemanticGenerationConfig
    elif model_type == "coarse":
        ModelClass = BarkCoarseModel
        ConfigClass = BarkCoarseConfig
        GenerationConfigClass = BarkCoarseGenerationConfig
    elif model_type == "fine":
        ModelClass = BarkFineModel
        ConfigClass = BarkFineConfig
        GenerationConfigClass = BarkFineGenerationConfig
    else:
        raise NotImplementedError()
    model_key = f"{model_type}_small" if use_small else model_type
    # 获取远程模型路径中与模型键对应的模型信息
    model_info = REMOTE_MODEL_PATHS[model_key]
    
    # 如果检查点路径不存在
    if not os.path.exists(ckpt_path):
        # 输出日志信息，指示模型类型未找到，并下载到 `CACHE_DIR` 中
        logger.info(f"{model_type} model not found, downloading into `{CACHE_DIR}`.")
        # 下载模型文件
        _download(model_info["repo_id"], model_info["file_name"])
    
    # 加载检查点文件到 torch 中
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 从检查点中获取模型参数
    # 这是一个临时解决方案
    model_args = checkpoint["model_args"]
    
    # 如果模型参数中没有 `input_vocab_size`
    if "input_vocab_size" not in model_args:
        # 使用 `vocab_size` 来填充 `input_vocab_size` 和 `output_vocab_size`
        model_args["input_vocab_size"] = model_args["vocab_size"]
        model_args["output_vocab_size"] = model_args["vocab_size"]
        # 删除原来的 `vocab_size` 键
        del model_args["vocab_size"]
    
    # 将 Bark 模型参数转换为 HF Bark 模型参数
    model_args["num_heads"] = model_args.pop("n_head")
    model_args["hidden_size"] = model_args.pop("n_embd")
    model_args["num_layers"] = model_args.pop("n_layer")
    
    # 使用模型参数创建配置对象
    model_config = ConfigClass(**checkpoint["model_args"])
    
    # 使用配置对象实例化模型
    model = ModelClass(config=model_config)
    
    # 创建模型生成配置对象
    model_generation_config = GenerationConfigClass()
    
    # 将生成配置对象赋值给模型的生成配置
    model.generation_config = model_generation_config
    
    # 获取模型的状态字典
    state_dict = checkpoint["model"]
    
    # 修复检查点中的问题
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        # 如果键以不需要的前缀开头
        if k.startswith(unwanted_prefix):
            # 替换键的一部分与 HF 实现中的相应层名称
            new_k = k[len(unwanted_prefix):]
            for old_layer_name in new_layer_name_dict:
                new_k = new_k.replace(old_layer_name, new_layer_name_dict[old_layer_name])
            # 替换原始键
            state_dict[new_k] = state_dict.pop(k)
    
    # 查找额外的键
    extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
    extra_keys = {k for k in extra_keys if not k.endswith(".attn.bias")}
    
    # 查找丢失的键
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = {k for k in missing_keys if not k.endswith(".attn.bias")}
    
    # 如果有额外的键存在，则引发值错误
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    
    # 如果有丢失的键存在，则引发值错误
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    
    # 加载状态字典到模型中（允许部分匹配）
    model.load_state_dict(state_dict, strict=False)
    
    # 计算模型参数数量（不包括嵌入层）
    n_params = model.num_parameters(exclude_embeddings=True)
    
    # 获取最佳验证损失值
    val_loss = checkpoint["best_val_loss"].item()
    
    # 输出日志信息，指示模型已加载，包括参数数量和验证损失值
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params, {round(val_loss,3)} loss")
    
    # 将模型设置为评估模式
    model.eval()
    
    # 将模型移动到指定设备
    model.to(device)
    
    # 删除检查点和状态字典，释放内存
    del checkpoint, state_dict
    
    # 返回加载并配置好的模型
    return model
# 定义函数，用于加载特定类型的 PyTorch 模型到指定路径下的文件夹
def load_model(pytorch_dump_folder_path, use_small=False, model_type="text"):
    # 检查模型类型是否合法，只允许 "text", "coarse", "fine" 三种类型
    if model_type not in ("text", "coarse", "fine"):
        raise NotImplementedError()

    # 设定设备为 CPU，执行模型转换操作
    device = "cpu"  # do conversion on cpu

    # 获取模型的检查点路径
    ckpt_path = _get_ckpt_path(model_type, use_small=use_small)
    # 载入模型，返回加载的模型对象
    model = _load_model(ckpt_path, device, model_type=model_type, use_small=use_small)

    # 加载 bark 初始模型
    bark_model = _bark_load_model(ckpt_path, "cpu", model_type=model_type, use_small=use_small)

    # 如果模型类型为 "text"，则从 bark_model 字典中获取 "model" 键对应的值
    if model_type == "text":
        bark_model = bark_model["model"]

    # 检查初始化的模型和新模型的参数数量是否相同
    if model.num_parameters(exclude_embeddings=True) != bark_model.get_num_params():
        raise ValueError("initial and new models don't have the same number of parameters")

    # 检查新模型和 bark 模型的输出是否相同
    batch_size = 5
    sequence_length = 10

    # 根据模型类型不同，生成不同形状的随机张量 vec，并计算模型的输出
    if model_type in ["text", "coarse"]:
        vec = torch.randint(256, (batch_size, sequence_length), dtype=torch.int)
        output_old_model = bark_model(vec)[0]
        output_new_model_total = model(vec)
        # 取最后一个时间步的输出 logits
        output_new_model = output_new_model_total.logits[:, [-1], :]
    else:
        prediction_codeboook_channel = 3
        n_codes_total = 8
        vec = torch.randint(256, (batch_size, sequence_length, n_codes_total), dtype=torch.int)
        output_new_model_total = model(prediction_codeboook_channel, vec)
        output_old_model = bark_model(prediction_codeboook_channel, vec)
        output_new_model = output_new_model_total.logits

    # 检查新旧模型输出的形状是否一致
    if output_new_model.shape != output_old_model.shape:
        raise ValueError("initial and new outputs don't have the same shape")
    # 检查新旧模型输出的数值差异是否在阈值内
    if (output_new_model - output_old_model).abs().max().item() > 1e-3:
        raise ValueError("initial and new outputs are not equal")

    # 创建存储 PyTorch 模型的文件夹，如果不存在则创建
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)


# 加载完整的 Bark 模型
def load_whole_bark_model(
    semantic_path,
    coarse_path,
    fine_path,
    append_text,
    hub_path,
    folder_path,
):
    # 构建 PyTorch 模型保存的文件夹路径
    pytorch_dump_folder_path = os.path.join(folder_path, append_text)

    # 从预训练的配置文件加载 BarkSemanticConfig
    semanticConfig = BarkSemanticConfig.from_pretrained(os.path.join(semantic_path, "config.json"))
    # 从预训练的配置文件加载 BarkCoarseConfig
    coarseAcousticConfig = BarkCoarseConfig.from_pretrained(os.path.join(coarse_path, "config.json"))
    # 从预训练的配置文件加载 BarkFineConfig
    fineAcousticConfig = BarkFineConfig.from_pretrained(os.path.join(fine_path, "config.json"))
    # 从预训练模型加载 EncodecConfig
    codecConfig = EncodecConfig.from_pretrained("facebook/encodec_24khz")

    # 从预训练模型加载 BarkSemanticModel
    semantic = BarkSemanticModel.from_pretrained(semantic_path)
    # 从预训练模型加载 BarkCoarseModel
    coarseAcoustic = BarkCoarseModel.from_pretrained(coarse_path)
    # 从预训练模型加载 BarkFineModel
    fineAcoustic = BarkFineModel.from_pretrained(fine_path)
    # 从预训练模型加载 EncodecModel
    codec = EncodecModel.from_pretrained("facebook/encodec_24khz")

    # 根据子模型的配置创建 BarkConfig 对象
    bark_config = BarkConfig.from_sub_model_configs(
        semanticConfig, coarseAcousticConfig, fineAcousticConfig, codecConfig
    )
    # 使用多个子模型配置参数创建 BarkGenerationConfig 对象
    bark_generation_config = BarkGenerationConfig.from_sub_model_configs(
        semantic.generation_config, coarseAcoustic.generation_config, fineAcoustic.generation_config
    )

    # 创建 BarkModel 对象，并传入 bark_config 参数
    bark = BarkModel(bark_config)

    # 将各个子模型的实例赋给 BarkModel 对象的属性
    bark.semantic = semantic
    bark.coarse_acoustics = coarseAcoustic
    bark.fine_acoustics = fineAcoustic
    bark.codec_model = codec

    # 将之前创建的 BarkGenerationConfig 对象赋给 BarkModel 对象的 generation_config 属性
    bark.generation_config = bark_generation_config

    # 创建目录 pytorch_dump_folder_path（如果不存在的话）
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    
    # 将 BarkModel 对象保存到指定路径 pytorch_dump_folder_path，并推送到模型中心（hub）
    bark.save_pretrained(pytorch_dump_folder_path, repo_id=hub_path, push_to_hub=True)
if __name__ == "__main__":
    # 如果脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必需的参数
    parser.add_argument("model_type", type=str, help="text, coarse or fine.")
    # 添加一个必需的参数，用于指定模型类型，接受字符串类型输入，例如 "text", "coarse" 或 "fine"

    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加一个必需的参数，用于指定 PyTorch 模型输出路径，接受字符串类型输入

    parser.add_argument("--is_small", action="store_true", help="convert the small version instead of the large.")
    # 添加一个可选的布尔类型参数，用于指定是否使用小版本而非大版本模型

    args = parser.parse_args()
    # 解析命令行参数，并将结果存储在 args 变量中

    load_model(args.pytorch_dump_folder_path, model_type=args.model_type, use_small=args.is_small)
    # 调用 load_model 函数，传入解析得到的参数：模型输出路径、模型类型和是否使用小版本模型
```