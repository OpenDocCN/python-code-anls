# `.\transformers\models\bark\convert_suno_to_hf.py`

```py
"""Convert Bark checkpoint."""
# 导入所需模块
import argparse  # 用于解析命令行参数
import os  # 用于处理操作系统相关功能
from pathlib import Path  # 用于处理文件路径

import torch  # PyTorch库，用于深度学习
from bark.generation import _load_model as _bark_load_model  # 导入Bark生成模块的_load_model函数
from huggingface_hub import hf_hub_download  # 从Hugging Face Hub导入模型下载函数

from transformers import EncodecConfig, EncodecModel, set_seed  # 从transformers库导入一些模块
from transformers.models.bark.configuration_bark import (  # 导入Bark模型的配置类
    BarkCoarseConfig,
    BarkConfig,
    BarkFineConfig,
    BarkSemanticConfig,
)
from transformers.models.bark.generation_configuration_bark import (  # 导入Bark模型生成配置类
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
    BarkGenerationConfig,
    BarkSemanticGenerationConfig,
)
from transformers.models.bark.modeling_bark import (  # 导入Bark模型类
    BarkCoarseModel,
    BarkFineModel,
    BarkModel,
    BarkSemanticModel,
)
from transformers.utils import logging  # 导入日志记录工具

# 设置日志记录级别为信息级别
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 设置随机种子
set_seed(770)

# 定义新层名称字典，用于将旧模型的层名称映射为新模型的层名称
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

# 定义远程模型路径字典，包含不同模型类型和大小的远程模型文件路径信息
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

# 获取当前模块的路径
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
# 获取默认缓存目录
default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
# 设置缓存目录
CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "suno", "bark_v0")

# 定义函数，根据模型类型和大小获取检查点文件路径
def _get_ckpt_path(model_type, use_small=False):
    key = model_type
    if use_small:
        key += "_small"
    return os.path.join(CACHE_DIR, REMOTE_MODEL_PATHS[key]["file_name"])

# 定义函数，下载模型文件
def _download(from_hf_path, file_name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    hf_hub_download(repo_id=from_hf_path, filename=file_name, local_dir=CACHE_DIR)

# 定义函数，加载模型
def _load_model(ckpt_path, device, use_small=False, model_type="text"):
    # 根据模型类型选择相应的模型类、配置类和生成配置类
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
    # 根据模型类型和大小选择正确的模型键
    model_key = f"{model_type}_small" if use_small else model_type
    # 获取指定模型键对应的模型信息
    model_info = REMOTE_MODEL_PATHS[model_key]
    # 如果检查点路径不存在
    if not os.path.exists(ckpt_path):
        # 输出信息：模型类型找不到，正在下载至`CACHE_DIR`
        logger.info(f"{model_type} model not found, downloading into `{CACHE_DIR}`.")
        # 下载模型文件
        _download(model_info["repo_id"], model_info["file_name"])
    # 加载检查点
    checkpoint = torch.load(ckpt_path, map_location=device)
    # 获取模型参数
    model_args = checkpoint["model_args"]
    # 如果模型参数中不存在输入词汇表大小
    if "input_vocab_size" not in model_args:
        # 使用原有词汇表大小填充输入和输出词汇表大小
        model_args["input_vocab_size"] = model_args["vocab_size"]
        model_args["output_vocab_size"] = model_args["vocab_size"]
        # 删除旧的词汇表大小键
        del model_args["vocab_size"]

    # 将Bark模型参数转换为HF Bark模型参数
    model_args["num_heads"] = model_args.pop("n_head")
    model_args["hidden_size"] = model_args.pop("n_embd")
    model_args["num_layers"] = model_args.pop("n_layer")

    # 根据模型参数创建模型配置对象
    model_config = ConfigClass(**checkpoint["model_args"])
    # 根据模型配置创建模型对象
    model = ModelClass(config=model_config)
    # 创建模型生成配置对象
    model_generation_config = GenerationConfigClass()

    # 将模型生成配置对象赋值给模型
    model.generation_config = model_generation_config
    # 获取模型状态字典
    state_dict = checkpoint["model"]
    # 修复检查点
    # 不需要的前缀
    unwanted_prefix = "_orig_mod."
    # 遍历状态字典中的键值对
    for k, v in list(state_dict.items()):
        # 如果键以不需要的前缀开头
        if k.startswith(unwanted_prefix):
            # 替换部分键名为HF实现中相应层的名称
            new_k = k[len(unwanted_prefix) :]
            for old_layer_name in new_layer_name_dict:
                new_k = new_k.replace(old_layer_name, new_layer_name_dict[old_layer_name])

            # 替换状态字典中的键名
            state_dict[new_k] = state_dict.pop(k)

    # 获取额外的键（在模型状态字典中但不在模型状态中的键）
    extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
    extra_keys = {k for k in extra_keys if not k.endswith(".attn.bias")}
    # 获取丢失的键（在模型状态中但不在模型状态字典中的键）
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = {k for k in missing_keys if not k.endswith(".attn.bias")}
    # 如果存在额外的键
    if len(extra_keys) != 0:
        # 抛出异常：发现额外的键
        raise ValueError(f"extra keys found: {extra_keys}")
    # 如果存在丢失的键
    if len(missing_keys) != 0:
        # 抛出异常：丢失的键
        raise ValueError(f"missing keys: {missing_keys}")
    # 加载模型状态字典（允许不匹配）
    model.load_state_dict(state_dict, strict=False)
    # 计算模型参数数量（排除嵌入层）
    n_params = model.num_parameters(exclude_embeddings=True)
    # 获取最佳验证损失值
    val_loss = checkpoint["best_val_loss"].item()
    # 输出信息：模型已加载，显示模型参数数量和最佳验证损失值
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params, {round(val_loss,3)} loss")
    # 将模型设置为评估模式
    model.eval()
    # 将模型移到指定的设备上
    model.to(device)
    # 删除检查点和状态字典以释放内存
    del checkpoint, state_dict

    # 返回加载的模型
    return model
# 加载模型函数，从指定路径加载 PyTorch 模型
def load_model(pytorch_dump_folder_path, use_small=False, model_type="text"):
    # 如果模型类型不在 "text", "coarse", "fine" 中，则抛出 NotImplementedError 异常
    if model_type not in ("text", "coarse", "fine"):
        raise NotImplementedError()

    # 设备选择为 CPU，用于模型转换
    device = "cpu"  # do conversion on cpu

    # 获取模型检查点路径
    ckpt_path = _get_ckpt_path(model_type, use_small=use_small)
    # 载入模型
    model = _load_model(ckpt_path, device, model_type=model_type, use_small=use_small)

    # 载入 Bark 初始模型
    bark_model = _bark_load_model(ckpt_path, "cpu", model_type=model_type, use_small=use_small)

    # 如果模型类型为 "text"，则选择其中的 "model" 部分
    if model_type == "text":
        bark_model = bark_model["model"]

    # 检查初始模型和新模型的参数数量是否相同
    if model.num_parameters(exclude_embeddings=True) != bark_model.get_num_params():
        raise ValueError("initial and new models don't have the same number of parameters")

    # 检查新模型输出是否与 Bark 模型输出相同
    batch_size = 5
    sequence_length = 10

    # 如果模型类型为 "text" 或 "coarse"，则生成随机整数张量进行测试
    if model_type in ["text", "coarse"]:
        vec = torch.randint(256, (batch_size, sequence_length), dtype=torch.int)
        output_old_model = bark_model(vec)[0]

        output_new_model_total = model(vec)

        # 取最后一步的 logits
        output_new_model = output_new_model_total.logits[:, [-1], :]

    # 如果模型类型为 "fine"，则生成具有特定形状的随机整数张量进行测试
    else:
        prediction_codeboook_channel = 3
        n_codes_total = 8
        vec = torch.randint(256, (batch_size, sequence_length, n_codes_total), dtype=torch.int)

        output_new_model_total = model(prediction_codeboook_channel, vec)
        output_old_model = bark_model(prediction_codeboook_channel, vec)

        output_new_model = output_new_model_total.logits

    # 检查新旧模型输出的形状是否相同
    if output_new_model.shape != output_old_model.shape:
        raise ValueError("initial and new outputs don't have the same shape")
    # 检查新旧模型输出的差异是否在一定范围内
    if (output_new_model - output_old_model).abs().max().item() > 1e-3:
        raise ValueError("initial and new outputs are not equal")

    # 创建保存 PyTorch 模型的文件夹
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 保存 PyTorch 模型
    model.save_pretrained(pytorch_dump_folder_path)


# 加载整个 Bark 模型
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

    # 加载 BarkSemanticModel 配置
    semanticConfig = BarkSemanticConfig.from_pretrained(os.path.join(semantic_path, "config.json"))
    # 加载 BarkCoarseModel 配置
    coarseAcousticConfig = BarkCoarseConfig.from_pretrained(os.path.join(coarse_path, "config.json"))
    # 加载 BarkFineModel 配置
    fineAcousticConfig = BarkFineConfig.from_pretrained(os.path.join(fine_path, "config.json"))
    # 加载 EncodecModel 配置
    codecConfig = EncodecConfig.from_pretrained("facebook/encodec_24khz")

    # 加载 BarkSemanticModel
    semantic = BarkSemanticModel.from_pretrained(semantic_path)
    # 加载 BarkCoarseModel
    coarseAcoustic = BarkCoarseModel.from_pretrained(coarse_path)
    # 加载 BarkFineModel
    fineAcoustic = BarkFineModel.from_pretrained(fine_path)
    # 加载 EncodecModel
    codec = EncodecModel.from_pretrained("facebook/encodec_24khz")

    # 构建 BarkConfig
    bark_config = BarkConfig.from_sub_model_configs(
        semanticConfig, coarseAcousticConfig, fineAcousticConfig, codecConfig
    )
    # 从子模型配置中生成 Bark 生成配置对象，用于生成 Bark 模型
    bark_generation_config = BarkGenerationConfig.from_sub_model_configs(
        semantic.generation_config, coarseAcoustic.generation_config, fineAcoustic.generation_config
    )
    
    # 创建 Bark 模型对象
    bark = BarkModel(bark_config)
    
    # 将语义模型配置赋值给 Bark 模型的语义属性
    bark.semantic = semantic
    # 将粗略声学模型配置赋值给 Bark 模型的粗略声学属性
    bark.coarse_acoustics = coarseAcoustic
    # 将细致声学模型配置赋值给 Bark 模型的细致声学属性
    bark.fine_acoustics = fineAcoustic
    # 将编解码模型配置赋值给 Bark 模型的编解码模型属性
    bark.codec_model = codec
    
    # 将生成配置对象赋值给 Bark 模型的生成配置属性
    bark.generation_config = bark_generation_config
    
    # 创建目录以保存 PyTorch 模型，并确保目录存在
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 将 Bark 模型保存到指定路径，并可选择将其推送到 Hub 上
    bark.save_pretrained(pytorch_dump_folder_path, repo_id=hub_path, push_to_hub=True)
# 如果该脚本作为主程序执行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数

    # 添加参数，指定模型类型，类型为字符串
    parser.add_argument("model_type", type=str, help="text, coarse or fine.")
    # 添加参数，指定输出 PyTorch 模型的文件夹路径，类型为字符串，默认为 None
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加可选参数，如果存在则设为 True，指示转换小版本而不是大版本
    parser.add_argument("--is_small", action="store_true", help="convert the small version instead of the large.")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数加载模型
    load_model(args.pytorch_dump_folder_path, model_type=args.model_type, use_small=args.is_small)
```