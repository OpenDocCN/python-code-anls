# `.\transformers\models\nllb_moe\convert_nllb_moe_sharded_original_checkpoint_to_pytorch.py`

```
# 导入必要的库和模块
import argparse
import json
import os

import torch
from torch import nn

from transformers import NllbMoeConfig, NllbMoeModel
from transformers.modeling_utils import dtype_byte_size
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME

# 定义一个函数用于删除指定的密钥
def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "decoder.output_projection.weight",
        "_float_tensor",
        "encoder.embed_positions._float_tensor",
        "decoder.embed_positions._float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)

# 定义一个函数用于从嵌入层创建线性层
def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

# 定义一个函数用于重命名 Fairseq 密钥
def rename_fairseq_keys(state_dict, expert_idx=None):
    new_dict = {}
    for old_key in state_dict.keys():
        key = old_key
        if "moe_layer.experts." in key:
            if expert_idx is not None:
                key = key.replace("moe_layer.experts.0", f"ffn.experts.expert_{expert_idx}")
            else:
                key = key.replace("moe_layer.experts.", "ffn.experts.expert_")
        if "gate" in key:
            key = key.replace(".moe_layer.gate.wg", ".ffn.router.classifier")
        if "fc2" and "experts" not in key:
            key = key.replace(".fc2.", ".ffn.fc2.")
        if "fc1" and "experts" not in key:
            key = key.replace(".fc1.", ".ffn.fc1.")
        if ".encoder_attn." in key:
            key = key.replace(".encoder_attn.", ".cross_attention.")
        if "encoder_attn_layer_norm" in key:
            key = key.replace("encoder_attn_layer_norm", "cross_attention_layer_norm")
        if "final_layer_norm" in key:
            key = key.replace("final_layer_norm", "ff_layer_norm")
        new_dict[key] = state_dict[old_key]
    return new_dict

# 定义一个函数用于将 Switch Transformer 模型分片
def shard_on_the_fly(switch_checkpoint_path, dump_path, num_experts, dtype, weights_name: str = WEIGHTS_NAME):
    sharded_state_dicts = []
    total_size = 0
    os.makedirs(dump_path, exist_ok=True)


这段代码定义了几个函数用于处理 Switch Transformer 模型的参数。其中包括:

1. `remove_ignore_keys_`函数用于从模型的状态字典中删除指定的密钥。
2. `make_linear_from_emb`函数用于从嵌入层创建线性层。
3. `rename_fairseq_keys`函数用于重命名 Fairseq 模型的密钥。
4. `shard_on_the_fly`函数用于将 Switch Transformer 模型分片。

总的来说,这些函数可以用于处理和转换 Switch Transformer 模型的参数,为后续的模型加载和使用做准备。
    # 遍历专家数量范围
    for expert in range(num_experts):
        # 构建每个专家的检查点路径
        expert_path = switch_checkpoint_path + f"-rank-{expert}.pt"
        # 如果专家的检查点文件存在
        if os.path.isfile(expert_path):
            # 加载专家模型状态字典
            expert_state = torch.load(expert_path)["model"]
            # 移除无需考虑的键
            remove_ignore_keys_(expert_state)
            # 重命名 Fairseq 模型的键
            expert_state = rename_fairseq_keys(expert_state, expert)
            # 构建保存路径，考虑到可能存在多个分片
            save_path = os.path.join(
                dump_path, weights_name.replace(".bin", f"-{len(sharded_state_dicts)+1:05d}-of-???.bin")
            )
            # 保存专家模型状态字典
            torch.save(expert_state, save_path)
            # 记录分片模型状态字典的键
            sharded_state_dicts.append(expert_state.keys())
            # 累加总参数大小，考虑数据类型的字节大小
            total_size += sum([value.numel() for key, value in expert_state.items()]) * dtype_byte_size(
                expert_state[list(expert_state)[0]].dtype
            )
    
    # 添加最后一个块
    save_path = os.path.join(dump_path, weights_name.replace(".bin", f"-{len(sharded_state_dicts)+1:05d}-of-???.bin"))
    # 加载共享权重模型状态字典
    shared_weights = torch.load(switch_checkpoint_path + "-shared.pt")["model"]
    # 移除无需考虑的键
    remove_ignore_keys_(shared_weights)
    # 重命名 Fairseq 模型的键
    shared_weights = rename_fairseq_keys(shared_weights, None)
    # 将共享权重映射到特定键
    shared_weights["shared.weight"] = shared_weights["decoder.embed_tokens.weight"]
    # 记录共享权重模型状态字典的键
    sharded_state_dicts.append(shared_weights.keys())
    
    # 如果只有共享权重模型状态字典（在同一文件上保存了虚拟模型/专家）
    if len(sharded_state_dicts) == 1:
        # 保存共享权重模型状态字典
        save_path = os.path.join(dump_path, weights_name)
        torch.save(shared_weights, save_path)
        # 返回权重文件名和对应的状态字典键的映射，以及空的索引
        return {weights_name: sharded_state_dicts[0]}, None
    else:
        # 保存共享权重模型状态字典
        torch.save(shared_weights, save_path)
    
    # 否则，构建权重索引
    weight_map = {}
    for idx, shard in enumerate(sharded_state_dicts):
        # 构建分片权重文件名
        shard_file = weights_name.replace(".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin")
        # 重命名临时文件为分片权重文件名
        temp_filename = os.path.join(dump_path, weights_name.replace(".bin", f"-{idx+1:05d}-of-???.bin"))
        os.rename(temp_filename, os.path.join(dump_path, shard_file))
        # 记录权重键和分片权重文件名的映射关系
        for key in shard:
            weight_map[key] = shard_file
    
    # 添加元数据
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    
    # 写入权重索引到文件
    with open(os.path.join(dump_path, WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
        # 序列化索引内容为 JSON 格式，并写入文件
        content = json.dumps(index, indent=2, sort_keys=True) + "\n"
        f.write(content)
    
    # 返回元数据和权重索引
    return metadata, index
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必填参数
    parser.add_argument(
        "--nllb_moe_checkpoint_path",
        default="/home/arthur_huggingface_co/fairseq/weights/checkpoints/model_moe_54b/checkpoint_2_300000",
        type=str,
        required=False,
        help="Path to a directory containing a folder per layer. Follows the original Google format.",
    )
    # 添加参数指定数据类型
    parser.add_argument("--dtype", default="float32", type=str, required=False, help="dtype of the saved model")
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/home/arthur_huggingface_co/fairseq/weights/checkpoints/hf-converted-moe-54b",
        type=str,
        required=False,
        help="Path to the output pytorch model.",
    )
    # 解析命令行参数，存储在 args 对象中
    args = parser.parse_args()
    # 在运行期间利用函数 shard_on_the_fly() 进行分片处理
    metadata, index = shard_on_the_fly(
        args.nllb_moe_checkpoint_path,
        args.pytorch_dump_folder_path,
        128,
        args.dtype,
    )

    # 对 NllbMoeConfig 进行配置
    config = NllbMoeConfig.from_pretrained(
        "facebook/nllb-200-3.3B", encoder_sparse_step=4, decoder_sparse_step=4, num_experts=128
    )
    # 将配置保存到指定路径
    config.save_pretrained(args.pytorch_dump_folder_path)
    # 从指定路径加载 NllbMoeModel 模型
    model = NllbMoeModel.from_pretrained(args.pytorch_dump_folder_path)
    # 打印信息
    print("Done")
    # 将模型保存到指定路径
    model.save_pretrained(args.pytorch_dump_folder_path)
```  
```