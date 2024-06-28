# `.\models\prophetnet\convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`

```
# coding=utf-8
# 定义脚本编码格式为 UTF-8

# 版权声明及许可证信息，使用 Apache License 2.0
# 更多信息可访问 http://www.apache.org/licenses/LICENSE-2.0

"""Convert ProphetNet checkpoint."""


import argparse  # 导入用于处理命令行参数的模块 argparse

from torch import nn  # 导入 PyTorch 的神经网络模块 nn

# 导入旧版本的 ProphetNet 和 XLMProphetNet 模型定义，对应分支 `save_old_prophetnet_model_structure`
# 原始的 prophetnet_checkpoints 存储在 `patrickvonplaten/..._old` 中
from transformers_old.modeling_prophetnet import (
    ProphetNetForConditionalGeneration as ProphetNetForConditionalGenerationOld,
)
from transformers_old.modeling_xlm_prophetnet import (
    XLMProphetNetForConditionalGeneration as XLMProphetNetForConditionalGenerationOld,
)

# 导入新版本的 ProphetNet 和 XLMProphetNet 模型定义以及日志记录模块
from transformers import ProphetNetForConditionalGeneration, XLMProphetNetForConditionalGeneration, logging


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器
logging.set_verbosity_info()  # 设置日志记录级别为 info


def convert_prophetnet_checkpoint_to_pytorch(prophetnet_checkpoint_path: str, pytorch_dump_folder_path: str):
    """
    Copy/paste/tweak prohpetnet's weights to our prophetnet structure.
    将 ProphetNet 的权重复制/粘贴/调整到我们的 ProphetNet 结构中。
    """
    # 根据路径中是否包含 "xprophetnet" 来判断使用哪个版本的模型进行加载和转换
    if "xprophetnet" in prophetnet_checkpoint_path:
        prophet_old = XLMProphetNetForConditionalGenerationOld.from_pretrained(prophetnet_checkpoint_path)
        # 加载旧版 XLMProphetNet 模型，并输出加载信息
        prophet, loading_info = XLMProphetNetForConditionalGeneration.from_pretrained(
            prophetnet_checkpoint_path, output_loading_info=True
        )
    else:
        prophet_old = ProphetNetForConditionalGenerationOld.from_pretrained(prophetnet_checkpoint_path)
        # 加载旧版 ProphetNet 模型，并输出加载信息
        prophet, loading_info = ProphetNetForConditionalGeneration.from_pretrained(
            prophetnet_checkpoint_path, output_loading_info=True
        )

    # 定义需要特殊处理的关键字列表
    special_keys = ["key_proj", "value_proj", "query_proj"]

    # 定义模型中权重名称的映射关系，用于转换模型结构
    mapping = {
        "self_attn": "ngram_self_attn",
        "cross_attn": "encoder_attn",
        "cross_attn_layer_norm": "encoder_attn_layer_norm",
        "feed_forward_layer_norm": "final_layer_norm",
        "feed_forward": "",
        "intermediate": "fc1",
        "output": "fc2",
        "key_proj": "k_proj",
        "query_proj": "q_proj",
        "value_proj": "v_proj",
        "word_embeddings": "embed_tokens",
        "embeddings_layer_norm": "emb_layer_norm",
        "relative_pos_embeddings": "relative_linear",
        "ngram_embeddings": "ngram_input_embed",
        "position_embeddings": "embed_positions",
    }

    print(f"Saving model to {pytorch_dump_folder_path}")  # 打印保存模型的目录路径
    prophet.save_pretrained(pytorch_dump_folder_path)  # 保存转换后的 PyTorch 模型


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    # 添加必需的参数定义
    parser.add_argument(
        "--prophetnet_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    # 添加必需的参数定义
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析命令行参数，将其存储到args变量中
    args = parser.parse_args()
    # 调用函数将ProphetNet的检查点文件转换为PyTorch模型文件
    convert_prophetnet_checkpoint_to_pytorch(args.prophetnet_checkpoint_path, args.pytorch_dump_folder_path)
```