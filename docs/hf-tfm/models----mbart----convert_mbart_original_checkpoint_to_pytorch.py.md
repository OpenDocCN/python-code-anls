# `.\transformers\models\mbart\convert_mbart_original_checkpoint_to_pytorch.py`

```
# 版权声明和许可信息
# 该代码版权归 HuggingFace 团队所有，受 Apache 许可证 2.0 版本保护
# 只有在遵守许可证的情况下才能使用该文件
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的库
import argparse
import torch
from torch import nn
from transformers import MBartConfig, MBartForConditionalGeneration

# 从状态字典中移除指定的键
def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)

# 从嵌入层创建线性层
def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

# 从磁盘中加载 Fairseq MBart 模型检查点并转换为 HuggingFace 模型
def convert_fairseq_mbart_checkpoint_from_disk(
    checkpoint_path, hf_config_path="facebook/mbart-large-en-ro", finetuned=False, mbart_50=False
):
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    remove_ignore_keys_(state_dict)
    vocab_size = state_dict["encoder.embed_tokens.weight"].shape[0]

    mbart_config = MBartConfig.from_pretrained(hf_config_path, vocab_size=vocab_size)
    if mbart_50 and finetuned:
        mbart_config.activation_function = "relu"

    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    model = MBartForConditionalGeneration(mbart_config)
    model.model.load_state_dict(state_dict)

    if finetuned:
        model.lm_head = make_linear_from_emb(model.model.shared)

    return model

# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument(
        "fairseq_path", type=str, help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem."
    )
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--hf_config",
        default="facebook/mbart-large-cc25",
        type=str,
        help="Which huggingface architecture to use: mbart-large",
    )
    parser.add_argument("--mbart_50", action="store_true", help="whether the model is mMART-50 checkpoint")
    parser.add_argument("--finetuned", action="store_true", help="whether the model is a fine-tuned checkpoint")
    args = parser.parse_args()
    model = convert_fairseq_mbart_checkpoint_from_disk(
        args.fairseq_path, hf_config_path=args.hf_config, finetuned=args.finetuned, mbart_50=args.mbart_50
    )
    # 保存模型到指定路径
    model.save_pretrained(args.pytorch_dump_folder_path)
```