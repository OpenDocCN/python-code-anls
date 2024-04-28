# `.\transformers\models\roberta_prelayernorm\convert_roberta_prelayernorm_original_pytorch_checkpoint_to_pytorch.py`

```
# 导入必要的库和模块
import argparse

import torch
from huggingface_hub import hf_hub_download

from transformers import AutoTokenizer, RobertaPreLayerNormConfig, RobertaPreLayerNormForMaskedLM
from transformers.utils import logging


# 设置日志记录级别为 info
logging.set_verbosity_info()
# 获取 logger 对象
logger = logging.get_logger(__name__)


# 定义函数，将 RoBERTa-PreLayerNorm 模型检查点转换为 PyTorch 格式
def convert_roberta_prelayernorm_checkpoint_to_pytorch(checkpoint_repo: str, pytorch_dump_folder_path: str):
    """
    Copy/paste/tweak roberta_prelayernorm's weights to our BERT structure.
    """
    # 转换配置
    config = RobertaPreLayerNormConfig.from_pretrained(
        checkpoint_repo, architectures=["RobertaPreLayerNormForMaskedLM"]
    )

    # 转换状态字典
    original_state_dict = torch.load(hf_hub_download(repo_id=checkpoint_repo, filename="pytorch_model.bin"))
    state_dict = {}
    for tensor_key, tensor_value in original_state_dict.items():
        # Transformer 实现给模型一个唯一的名称，而不是覆盖 'roberta'
        if tensor_key.startswith("roberta."):
            tensor_key = "roberta_prelayernorm." + tensor_key[len("roberta.") :]

        # 原始实现包含未使用的权重，从状态字典中删除这些权重
        if tensor_key.endswith(".self.LayerNorm.weight") or tensor_key.endswith(".self.LayerNorm.bias"):
            continue

        state_dict[tensor_key] = tensor_value

    # 根据配置和状态字典创建模型
    model = RobertaPreLayerNormForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=None, config=config, state_dict=state_dict
    )
    # 保存模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path)

    # 转换 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_repo)
    # 保存 tokenizer 到指定路径
    tokenizer.save_pretrained(pytorch_dump_folder_path)


# 程序入口，解析命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument(
        "--checkpoint-repo",
        default=None,
        type=str,
        required=True,
        help="Path the official PyTorch dump, e.g. 'andreasmadsen/efficient_mlm_m0.40'.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数执行转换
    convert_roberta_prelayernorm_checkpoint_to_pytorch(args.checkpoint_repo, args.pytorch_dump_folder_path)
```