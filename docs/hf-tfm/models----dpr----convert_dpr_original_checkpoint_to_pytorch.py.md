# `.\models\dpr\convert_dpr_original_checkpoint_to_pytorch.py`

```
# 版权声明
# 版权所有，2020年HuggingFace团队。
#
# 根据Apache许可证2.0版（“许可证”）许可;
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或经协议书面同意，否则将基于"现状"基础分发软件，
# 没有任何形式的担保或条件，无论是明示的还是暗示的。
# 请参阅许可证，了解特定语言管理权限和许可的限制。
#
# 导入需要的库
import argparse
import collections
from pathlib import Path
import torch
from torch.serialization import default_restore_location
from transformers import BertConfig, DPRConfig, DPRContextEncoder, DPRQuestionEncoder, DPRReader


# 定义类：CheckpointState
CheckpointState = collections.namedtuple(
    "CheckpointState", ["model_dict", "optimizer_dict", "scheduler_dict", "offset", "epoch", "encoder_params"]
)


# 定义函数：load_states_from_checkpoint，用于从检查点文件中加载状态
def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    print(f"Reading saved model from {model_file}")
    # 使用 Torch 加载模型文件，并将其映射到 CPU 上
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    return CheckpointState(**state_dict)


# 定义类：DPRState
class DPRState:
    # 初始化方法，接收源文件路径参数
    def __init__(self, src_file: Path):
        self.src_file = src_file

    # 加载 DPR 模型的抽象方法
    def load_dpr_model(self):
        raise NotImplementedError

    # 从类型创建 DPRState 实例的静态方法
    @staticmethod
    def from_type(comp_type: str, *args, **kwargs) -> "DPRState":
        if comp_type.startswith("c"):
            return DPRContextEncoderState(*args, **kwargs)
        if comp_type.startswith("q"):
            return DPRQuestionEncoderState(*args, **kwargs)
        if comp_type.startswith("r"):
            return DPRReaderState(*args, **kwargs)
        else:
            raise ValueError("Component type must be either 'ctx_encoder', 'question_encoder' or 'reader.")


# 定义类：DPRContextEncoderState，继承自DPRState
class DPRContextEncoderState(DPRState):
    # 加载 DPR 模型的方法
    def load_dpr_model(self):
        # 创建 DPRContextEncoder 实例，使用BertConfig初始化
        model = DPRContextEncoder(DPRConfig(**BertConfig.get_config_dict("bert-base-uncased")[0]))
        print(f"Loading DPR biencoder from {self.src_file}")
        # 从检查点文件中加载状态
        saved_state = load_states_from_checkpoint(self.src_file)
        encoder, prefix = model.ctx_encoder, "ctx_model."
        # 修复来自https://github.com/huggingface/transformers/commit/614fef1691edb806de976756d4948ecbcd0c0ca3的更改
        state_dict = {"bert_model.embeddings.position_ids": model.ctx_encoder.bert_model.embeddings.position_ids}
        # 遍历加载的模型字典，进行必要的更改
        for key, value in saved_state.model_dict.items():
            if key.startswith(prefix):
                key = key[len(prefix) :]
                if not key.startswith("encode_proj."):
                    key = "bert_model." + key
                state_dict[key] = value
        # 加载模型的状态字典
        encoder.load_state_dict(state_dict)
        return model


class DPRQuestionEncoderState(DPRState):
# 这里应该是其他类的定义，但是给出的示例代码不足以完整呈现这部分的逻辑，故省略
    # 加载 DPR 模型
    def load_dpr_model(self):
        # 使用 DPR 配置创建 DPR 问题编码器模型
        model = DPRQuestionEncoder(DPRConfig(**BertConfig.get_config_dict("bert-base-uncased")[0]))
        # 打印加载 DPR 双向编码器的来源文件路径
        print(f"Loading DPR biencoder from {self.src_file}")
        # 从检查点中加载保存的状态
        saved_state = load_states_from_checkpoint(self.src_file)
        # 获取问题编码器和前缀
        encoder, prefix = model.question_encoder, "question_model."
        # 修复自 https://github.com/huggingface/transformers/commit/614fef1691edb806de976756d4948ecbcd0c0ca3 的更改
        # 初始化状态字典，用于存储加载的模型状态
        state_dict = {"bert_model.embeddings.position_ids": model.question_encoder.bert_model.embeddings.position_ids}
        # 遍历保存的状态模型字典
        for key, value in saved_state.model_dict.items():
            # 如果键以前缀开头
            if key.startswith(prefix):
                # 去除前缀
                key = key[len(prefix) :]
                # 如果不是以 "encode_proj." 开头，则添加 "bert_model." 前缀
                if not key.startswith("encode_proj."):
                    key = "bert_model." + key
                # 将键值对添加到状态字典中
                state_dict[key] = value
        # 加载状态字典到编码器模型中
        encoder.load_state_dict(state_dict)
        # 返回加载的模型
        return model
class DPRReaderState(DPRState):
    # DPRReaderState 类继承自 DPRState 类
    def load_dpr_model(self):
        # 加载 DPRReader 模型
        model = DPRReader(DPRConfig(**BertConfig.get_config_dict("bert-base-uncased")[0]))
        # 打印加载 DPR reader 模型的来源文件路径
        print(f"Loading DPR reader from {self.src_file}")
        # 从检查点文件中加载保存的模型状态
        saved_state = load_states_from_checkpoint(self.src_file)
        
        # 修正自 https://github.com/huggingface/transformers/commit/614fef1691edb806de976756d4948ecbcd0c0ca3 的更改
        # 设置状态字典，将特定的模型参数路径映射到新模型中
        state_dict = {
            "encoder.bert_model.embeddings.position_ids": model.span_predictor.encoder.bert_model.embeddings.position_ids
        }
        for key, value in saved_state.model_dict.items():
            if key.startswith("encoder.") and not key.startswith("encoder.encode_proj"):
                # 更新模型参数的键名以适应新模型
                key = "encoder.bert_model." + key[len("encoder.") :]
            state_dict[key] = value
        
        # 加载模型的状态字典
        model.span_predictor.load_state_dict(state_dict)
        # 返回加载后的模型
        return model


def convert(comp_type: str, src_file: Path, dest_dir: Path):
    # 将目标目录转换为 Path 对象
    dest_dir = Path(dest_dir)
    # 如果目标目录不存在，则创建目录
    dest_dir.mkdir(exist_ok=True)
    
    # 根据组件类型和源文件创建 DPRState 实例
    dpr_state = DPRState.from_type(comp_type, src_file=src_file)
    # 加载 DPR 模型
    model = dpr_state.load_dpr_model()
    # 将模型保存到指定的目标目录
    model.save_pretrained(dest_dir)
    # 从保存的目录重新加载模型，以确保转换正确
    model.from_pretrained(dest_dir)  # sanity check


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必需的参数
    parser.add_argument(
        "--type", type=str, help="Type of the component to convert: 'ctx_encoder', 'question_encoder' or 'reader'."
    )
    parser.add_argument(
        "--src",
        type=str,
        help=(
            "Path to the dpr checkpoint file. They can be downloaded from the official DPR repo"
            " https://github.com/facebookresearch/DPR. Note that in the official repo, both encoders are stored in the"
            " 'retriever' checkpoints."
        ),
    )
    parser.add_argument("--dest", type=str, default=None, help="Path to the output PyTorch model directory.")
    args = parser.parse_args()

    # 解析源文件路径
    src_file = Path(args.src)
    # 确定目标目录的名称
    dest_dir = f"converted-{src_file.name}" if args.dest is None else args.dest
    # 将目标目录转换为 Path 对象
    dest_dir = Path(dest_dir)
    # 断言源文件存在
    assert src_file.exists()
    # 断言已指定 DPR 模型的组件类型
    assert (
        args.type is not None
    ), "Please specify the component type of the DPR model to convert: 'ctx_encoder', 'question_encoder' or 'reader'."
    # 执行模型转换
    convert(args.type, src_file, dest_dir)
```