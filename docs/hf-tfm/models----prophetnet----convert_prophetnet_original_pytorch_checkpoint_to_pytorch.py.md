# `.\transformers\models\prophetnet\convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，引用了 HuggingFace 公司的团队
# 根据 Apache 许可证版本 2.0 进行许可
# 可以在符合许可证的情况下使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“现状”提供软件
# 没有任何担保或条件，无论是明示还是暗示
# 请查看许可证以获取特定语言的权限和限制
"""Convert ProphetNet checkpoint."""

import argparse  # 导入解析命令行参数的库

from torch import nn  # 导入 PyTorch 的神经网络模块

# transformers_old 应该对应`save_old_prophetnet_model_structure`分支
# 原始的 prophetnet_checkpoints 分别保存在 `patrickvonplaten/..._old` 下
from transformers_old.modeling_prophetnet import (
    ProphetNetForConditionalGeneration as ProphetNetForConditionalGenerationOld,
)
from transformers_old.modeling_xlm_prophetnet import (
    XLMProphetNetForConditionalGeneration as XLMProphetNetForConditionalGenerationOld,
)

from transformers import ProphetNetForConditionalGeneration, XLMProphetNetForConditionalGeneration, logging

logger = logging.get_logger(__name__)  # 获取日志记录器
logging.set_verbosity_info()  # 设置日志的详细程度为 info

def convert_prophetnet_checkpoint_to_pytorch(prophetnet_checkpoint_path: str, pytorch_dump_folder_path: str):
    """
    Copy/paste/tweak prohpetnet's weights to our prophetnet structure.
    """
    # 如果 prophetnet_checkpoint_path 中包含 "xprophetnet" 字符串 
    if "xprophetnet" in prophetnet_checkpoint_path:
        # 从预训练的 prophetnet_checkpoint_path 中加载旧的 XLMProphetNetForConditionalGeneration 模型
        prophet_old = XLMProphetNetForConditionalGenerationOld.from_pretrained(prophetnet_checkpoint_path)
        
        # 从预训练的 prophetnet_checkpoint_path 中加载新的 XLMProphetNetForConditionalGeneration 模型，并返回加载信息
        prophet, loading_info = XLMProphetNetForConditionalGeneration.from_pretrained(
            prophetnet_checkpoint_path, output_loading_info=True
        )
    else:
        # 从预训练的 prophetnet_checkpoint_path 中加载旧的 ProphetNetForConditionalGeneration 模型
        prophet_old = ProphetNetForConditionalGenerationOld.from_pretrained(prophetnet_checkpoint_path)

        # 从预训练的 prophetnet_checkpoint_path 中加载新的 ProphetNetForConditionalGeneration 模型，并返回加载信息
        prophet, loading_info = ProphetNetForConditionalGeneration.from_pretrained(
            prophetnet_checkpoint_path, output_loading_info=True
        )

    # 定义一些需要特殊处理的键值对
    special_keys = ["key_proj", "value_proj", "query_proj"]

    # 映射旧模型���参数到新模型的参数
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

    # 打印信息，保存模型至指定路径
    print(f"Saving model to {pytorch_dump_folder_path}")
    prophet.save_pretrained(pytorch_dump_folder_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器
    # 添加必需的参数
    parser.add_argument(
        "--prophetnet_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    # 添加必需的参数
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将ProphetNet模型检查点转换为PyTorch模型
    convert_prophetnet_checkpoint_to_pytorch(args.prophetnet_checkpoint_path, args.pytorch_dump_folder_path)
```