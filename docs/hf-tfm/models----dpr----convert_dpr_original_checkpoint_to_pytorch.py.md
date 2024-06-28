# `.\models\dpr\convert_dpr_original_checkpoint_to_pytorch.py`

```py
# 导入必要的库
import argparse  # 用于解析命令行参数
import collections  # 提供额外的数据结构
from pathlib import Path  # 处理路径相关操作

import torch  # PyTorch深度学习库
from torch.serialization import default_restore_location  # 默认的模型加载位置

# 导入transformers库中的相关组件
from transformers import BertConfig, DPRConfig, DPRContextEncoder, DPRQuestionEncoder, DPRReader

# 定义一个命名元组CheckpointState，用于保存模型加载状态
CheckpointState = collections.namedtuple(
    "CheckpointState", ["model_dict", "optimizer_dict", "scheduler_dict", "offset", "epoch", "encoder_params"]
)

# 从文件中加载模型状态
def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    # 打印正在读取的模型文件信息
    print(f"Reading saved model from {model_file}")
    # 使用torch.load加载模型文件，并映射到CPU上
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    # 返回包含模型状态的CheckpointState命名元组
    return CheckpointState(**state_dict)


# 定义一个DPRState类，用于处理DPR相关的状态
class DPRState:
    def __init__(self, src_file: Path):
        self.src_file = src_file  # 初始化源文件路径

    def load_dpr_model(self):
        raise NotImplementedError  # 抽象方法，子类需实现具体功能

    # 静态方法，根据组件类型创建对应的DPRState子类实例
    @staticmethod
    def from_type(comp_type: str, *args, **kwargs) -> "DPRState":
        # 根据组件类型选择相应的子类实例化
        if comp_type.startswith("c"):
            return DPRContextEncoderState(*args, **kwargs)
        if comp_type.startswith("q"):
            return DPRQuestionEncoderState(*args, **kwargs)
        if comp_type.startswith("r"):
            return DPRReaderState(*args, **kwargs)
        else:
            raise ValueError("Component type must be either 'ctx_encoder', 'question_encoder' or 'reader'.")


# DPRContextEncoderState类继承自DPRState，处理上下文编码器相关状态
class DPRContextEncoderState(DPRState):
    def load_dpr_model(self):
        # 创建DPRContextEncoder模型实例，基于指定的配置
        model = DPRContextEncoder(DPRConfig(**BertConfig.get_config_dict("google-bert/bert-base-uncased")[0]))
        # 打印正在加载的DPR双编码器信息
        print(f"Loading DPR biencoder from {self.src_file}")
        # 从检查点文件中加载保存的模型状态
        saved_state = load_states_from_checkpoint(self.src_file)
        encoder, prefix = model.ctx_encoder, "ctx_model."

        # 修复自GitHub提交中的更改，更新模型状态字典
        state_dict = {"bert_model.embeddings.position_ids": model.ctx_encoder.bert_model.embeddings.position_ids}
        for key, value in saved_state.model_dict.items():
            if key.startswith(prefix):
                key = key[len(prefix) :]
                if not key.startswith("encode_proj."):
                    key = "bert_model." + key
                state_dict[key] = value
        
        # 加载更新后的状态字典到编码器模型中
        encoder.load_state_dict(state_dict)
        return model


class DPRQuestionEncoderState(DPRState):
    # 加载 DPR 模型
    def load_dpr_model(self):
        # 创建 DPRQuestionEncoder 对象，使用指定的 DPRConfig 和 BertConfig
        model = DPRQuestionEncoder(DPRConfig(**BertConfig.get_config_dict("google-bert/bert-base-uncased")[0]))
        # 打印正在加载的 DPR biencoder 的来源文件路径
        print(f"Loading DPR biencoder from {self.src_file}")
        # 从指定的文件中加载模型状态
        saved_state = load_states_from_checkpoint(self.src_file)
        # 获取模型的 encoder 部分和前缀字符串
        encoder, prefix = model.question_encoder, "question_model."
        # 修复来自特定提交的更改，更新状态字典以适应新版本的模型
        state_dict = {"bert_model.embeddings.position_ids": model.question_encoder.bert_model.embeddings.position_ids}
        # 遍历加载的模型状态字典的每个键值对
        for key, value in saved_state.model_dict.items():
            # 如果键以指定前缀开头
            if key.startswith(prefix):
                # 去掉前缀
                key = key[len(prefix) :]
                # 如果不是以 "encode_proj." 开头的键，则加上 "bert_model." 前缀
                if not key.startswith("encode_proj."):
                    key = "bert_model." + key
                # 更新状态字典
                state_dict[key] = value
        # 使用更新后的状态字典加载 encoder 的状态
        encoder.load_state_dict(state_dict)
        # 返回加载后的模型对象
        return model
class DPRReaderState(DPRState):
    # 继承自 DPRState 类的 DPRReaderState 类

    def load_dpr_model(self):
        # 加载 DPR 模型
        model = DPRReader(DPRConfig(**BertConfig.get_config_dict("google-bert/bert-base-uncased")[0]))
        # 打印加载的 DPR 读取器模型的信息，显示源文件路径
        print(f"Loading DPR reader from {self.src_file}")
        # 从检查点文件加载保存的模型状态
        saved_state = load_states_from_checkpoint(self.src_file)
        
        # 修复自 https://github.com/huggingface/transformers/commit/614fef1691edb806de976756d4948ecbcd0c0ca3 的更改
        # 准备状态字典，映射加载的模型状态到正确的位置
        state_dict = {
            "encoder.bert_model.embeddings.position_ids": model.span_predictor.encoder.bert_model.embeddings.position_ids
        }
        
        # 遍历保存的模型字典的键值对
        for key, value in saved_state.model_dict.items():
            # 如果键以 "encoder." 开头但不以 "encoder.encode_proj" 开头，修正为 "encoder.bert_model."
            if key.startswith("encoder.") and not key.startswith("encoder.encode_proj"):
                key = "encoder.bert_model." + key[len("encoder.") :]
            # 将修正后的键值对加入到状态字典中
            state_dict[key] = value
        
        # 加载状态字典到 DPR 读取器模型的 span_predictor 部分
        model.span_predictor.load_state_dict(state_dict)
        # 返回加载完成的 DPR 读取器模型
        return model


def convert(comp_type: str, src_file: Path, dest_dir: Path):
    # 转换函数，将指定类型的 DPR 模型检查点文件转换为 PyTorch 模型

    # 确保输出目录存在或创建它
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)

    # 根据组件类型和源文件路径创建 DPRState 对象
    dpr_state = DPRState.from_type(comp_type, src_file=src_file)
    # 加载 DPR 模型
    model = dpr_state.load_dpr_model()
    # 将模型保存到指定的输出目录
    model.save_pretrained(dest_dir)
    # 从输出目录重新加载模型进行验证
    model.from_pretrained(dest_dir)  # sanity check


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必需参数
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

    # 获取源文件路径和输出目录路径
    src_file = Path(args.src)
    dest_dir = f"converted-{src_file.name}" if args.dest is None else args.dest
    dest_dir = Path(dest_dir)
    
    # 确保指定的源文件存在
    assert src_file.exists()
    # 确保指定了 DPR 模型的组件类型
    assert (
        args.type is not None
    ), "Please specify the component type of the DPR model to convert: 'ctx_encoder', 'question_encoder' or 'reader'."
    
    # 执行转换操作，将指定类型的 DPR 模型转换为 PyTorch 模型
    convert(args.type, src_file, dest_dir)
```