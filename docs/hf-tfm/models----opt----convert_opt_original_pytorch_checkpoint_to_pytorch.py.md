# `.\models\opt\convert_opt_original_pytorch_checkpoint_to_pytorch.py`

```
# 导入必要的库和模块
import argparse     # 导入命令行参数解析模块
from pathlib import Path   # 导入处理路径的模块

import torch    # 导入PyTorch库

# 从transformers库中导入OPTConfig和OPTModel类
from transformers import OPTConfig, OPTModel
# 从transformers.utils中导入logging模块
from transformers.utils import logging

# 设置日志级别为INFO
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 加载检查点文件的函数，checkpoint_path应该以'model.pt'结尾
def load_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    # 使用CPU加载检查点文件中的状态字典
    sd = torch.load(checkpoint_path, map_location="cpu")
    # 如果状态字典中包含"model"键，则仅使用"model"键对应的值
    if "model" in sd.keys():
        sd = torch.load(checkpoint_path, map_location="cpu")["model"]

    # 删除不必要的权重
    keys_to_delete = [
        "decoder.version",
        "decoder.output_projection.weight",
    ]
    for key in keys_to_delete:
        # 如果状态字典中包含需要删除的键，则删除该键
        if key in sd:
            sd.pop(key)

    # 将需要重命名的键从旧名称映射到新名称
    keys_to_rename = {
        "decoder.project_in_dim.weight": "decoder.project_in.weight",
        "decoder.project_out_dim.weight": "decoder.project_out.weight",
        "decoder.layer_norm.weight": "decoder.final_layer_norm.weight",
        "decoder.layer_norm.bias": "decoder.final_layer_norm.bias",
    }
    for old_key, new_key in keys_to_rename.items():
        # 如果状态字典中包含需要重命名的旧键，则将其重命名为新键
        if old_key in sd:
            sd[new_key] = sd.pop(old_key)

    # 获取状态字典中的所有键列表
    keys = list(sd.keys())
    for key in keys:
        # 如果键名中包含".qkv_proj."，则进行以下操作
        if ".qkv_proj." in key:
            value = sd[key]
            # 将QKV权重分割为独立的Q、K、V
            q_name = key.replace(".qkv_proj.", ".q_proj.")
            k_name = key.replace(".qkv_proj.", ".k_proj.")
            v_name = key.replace(".qkv_proj.", ".v_proj.")
            depth = value.shape[0]
            assert depth % 3 == 0  # 断言深度能被3整除
            # 将value在dim=0维度上按照depth//3进行分割为Q、K、V
            k, v, q = torch.split(value, depth // 3, dim=0)

            sd[q_name] = q   # 将分割后的Q赋值给新的Q键名
            sd[k_name] = k   # 将分割后的K赋值给新的K键名
            sd[v_name] = v   # 将分割后的V赋值给新的V键名
            del sd[key]      # 删除原始的QKV键名

    return sd


# 使用torch.no_grad()装饰器定义函数，不会计算梯度
@torch.no_grad()
def convert_opt_checkpoint(checkpoint_path, pytorch_dump_folder_path, config=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    # 加载检查点文件中的状态字典
    state_dict = load_checkpoint(checkpoint_path)

    # 如果提供了config参数，则从预训练配置文件中加载OPTConfig对象
    if config is not None:
        config = OPTConfig.from_pretrained(config)
    else:
        config = OPTConfig()  # 否则创建一个空的OPTConfig对象

    # 创建一个OPTModel对象，并设置为半精度(half())和评估模式(eval())
    model = OPTModel(config).half().eval()
    # 使用给定的状态字典加载模型的状态
    model.load_state_dict(state_dict)

    # 检查结果
    # 如果指定路径不存在，则创建目录，用于保存 PyTorch 模型
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 将模型保存到指定路径下，以便后续使用
    model.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # 必选参数
    parser.add_argument(
        "--fairseq_path",
        type=str,
        help=(
            "path to fairseq checkpoint in correct format. You can find all checkpoints in the correct format here:"
            " https://huggingface.co/models?other=opt_metasq"
        ),
    )
    # 添加一个必选参数，指定fairseq模型的路径，必须是字符串类型，并提供帮助文本

    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加一个可选参数，指定输出PyTorch模型的路径，默认值为None，类型为字符串，并提供帮助文本

    parser.add_argument("--hf_config", default=None, type=str, help="Define HF config.")
    # 添加一个可选参数，指定HF配置的路径，默认值为None，类型为字符串，并提供帮助文本

    args = parser.parse_args()
    # 解析命令行参数并存储到args对象中

    convert_opt_checkpoint(args.fairseq_path, args.pytorch_dump_folder_path, config=args.hf_config)
    # 调用convert_opt_checkpoint函数，传入fairseq路径、PyTorch模型输出路径和HF配置路径作为参数
```