# `.\models\fnet\convert_fnet_original_flax_checkpoint_to_pytorch.py`

```py
    # 导入所需的库和模块
import argparse  # 用于解析命令行参数

import torch  # 导入PyTorch库
from flax.training.checkpoints import restore_checkpoint  # 从Flax库中导入恢复检查点的函数

from transformers import FNetConfig, FNetForPreTraining  # 导入FNet模型的配置和预训练类
from transformers.utils import logging  # 导入日志记录工具

# 设置日志输出级别为信息
logging.set_verbosity_info()

def convert_flax_checkpoint_to_pytorch(flax_checkpoint_path, fnet_config_file, save_path):
    # 使用FNetConfig类从JSON文件加载FNet的配置
    config = FNetConfig.from_json_file(fnet_config_file)
    print(f"Building PyTorch model from configuration: {config}")
    # 根据配置初始化FNetForPreTraining模型
    fnet_pretraining_model = FNetForPreTraining(config)

    # 从Flax的检查点中恢复模型参数
    checkpoint_dict = restore_checkpoint(flax_checkpoint_path, None)
    pretrained_model_params = checkpoint_dict["target"]

    # 初始化新的状态字典，用于存储PyTorch模型的参数
    state_dict = fnet_pretraining_model.state_dict()

    # 处理嵌入层的参数转换

    # 位置编码
    position_ids = state_dict["fnet.embeddings.position_ids"]
    new_state_dict = {"fnet.embeddings.position_ids": position_ids}

    # 单词嵌入
    new_state_dict["fnet.embeddings.word_embeddings.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["word"]["embedding"]
    )

    # 位置嵌入
    new_state_dict["fnet.embeddings.position_embeddings.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["position"]["embedding"][0]
    )

    # 类型嵌入
    new_state_dict["fnet.embeddings.token_type_embeddings.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["type"]["embedding"]
    )

    # 投影层的权重和偏置
    new_state_dict["fnet.embeddings.projection.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["hidden_mapping_in"]["kernel"]
    ).T
    new_state_dict["fnet.embeddings.projection.bias"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["hidden_mapping_in"]["bias"]
    )

    # LayerNorm层的权重和偏置
    new_state_dict["fnet.embeddings.LayerNorm.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["layer_norm"]["scale"]
    )
    new_state_dict["fnet.embeddings.LayerNorm.bias"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["layer_norm"]["bias"]
    )

    # 处理编码器层的参数转换
    # 对每个隐藏层进行循环，从预训练模型参数中加载相关的权重和偏置

    # 加载当前层的 Fourier 输出的 LayerNorm 权重
    new_state_dict[f"fnet.encoder.layer.{layer}.fourier.output.LayerNorm.weight"] = torch.tensor(
        pretrained_model_params["encoder"][f"encoder_{layer}"]["mixing_layer_norm"]["scale"]
    )
    # 加载当前层的 Fourier 输出的 LayerNorm 偏置
    new_state_dict[f"fnet.encoder.layer.{layer}.fourier.output.LayerNorm.bias"] = torch.tensor(
        pretrained_model_params["encoder"][f"encoder_{layer}"]["mixing_layer_norm"]["bias"]
    )

    # 加载当前层的 intermediate dense 层权重，并转置
    new_state_dict[f"fnet.encoder.layer.{layer}.intermediate.dense.weight"] = torch.tensor(
        pretrained_model_params["encoder"][f"feed_forward_{layer}"]["intermediate"]["kernel"]
    ).T
    # 加载当前层的 intermediate dense 层偏置
    new_state_dict[f"fnet.encoder.layer.{layer}.intermediate.dense.bias"] = torch.tensor(
        pretrained_model_params["encoder"][f"feed_forward_{layer}"]["intermediate"]["bias"]
    )

    # 加载当前层的 output dense 层权重，并转置
    new_state_dict[f"fnet.encoder.layer.{layer}.output.dense.weight"] = torch.tensor(
        pretrained_model_params["encoder"][f"feed_forward_{layer}"]["output"]["kernel"]
    ).T
    # 加载当前层的 output dense 层偏置
    new_state_dict[f"fnet.encoder.layer.{layer}.output.dense.bias"] = torch.tensor(
        pretrained_model_params["encoder"][f"feed_forward_{layer}"]["output"]["bias"]
    )

    # 加载当前层的 output LayerNorm 权重
    new_state_dict[f"fnet.encoder.layer.{layer}.output.LayerNorm.weight"] = torch.tensor(
        pretrained_model_params["encoder"][f"encoder_{layer}"]["output_layer_norm"]["scale"]
    )
    # 加载当前层的 output LayerNorm 偏置
    new_state_dict[f"fnet.encoder.layer.{layer}.output.LayerNorm.bias"] = torch.tensor(
        pretrained_model_params["encoder"][f"encoder_{layer}"]["output_layer_norm"]["bias"]
    )

    # 加载池化层的 dense 权重，并转置
    new_state_dict["fnet.pooler.dense.weight"] = torch.tensor(pretrained_model_params["encoder"]["pooler"]["kernel"]).T
    # 加载池化层的 dense 偏置
    new_state_dict["fnet.pooler.dense.bias"] = torch.tensor(pretrained_model_params["encoder"]["pooler"]["bias"])

    # 加载预测层的 transform dense 权重，并转置
    new_state_dict["cls.predictions.transform.dense.weight"] = torch.tensor(
        pretrained_model_params["predictions_dense"]["kernel"]
    ).T
    # 加载预测层的 transform dense 偏置
    new_state_dict["cls.predictions.transform.dense.bias"] = torch.tensor(
        pretrained_model_params["predictions_dense"]["bias"]
    )
    # 加载预测层的 transform LayerNorm 权重
    new_state_dict["cls.predictions.transform.LayerNorm.weight"] = torch.tensor(
        pretrained_model_params["predictions_layer_norm"]["scale"]
    )
    # 加载预测层的 transform LayerNorm 偏置
    new_state_dict["cls.predictions.transform.LayerNorm.bias"] = torch.tensor(
        pretrained_model_params["predictions_layer_norm"]["bias"]
    )
    
    # 加载预测层的 decoder 权重
    new_state_dict["cls.predictions.decoder.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["word"]["embedding"]
    )
    # 加载预测层的 decoder 偏置
    new_state_dict["cls.predictions.decoder.bias"] = torch.tensor(
        pretrained_model_params["predictions_output"]["output_bias"]
    )
    # 加载预测层的 bias
    new_state_dict["cls.predictions.bias"] = torch.tensor(pretrained_model_params["predictions_output"]["output_bias"])

    # Seq Relationship Layers
    # 使用预训练模型参数中的输出核和偏置，创建新的张量并赋给新状态字典的键"cls.seq_relationship.weight"
    new_state_dict["cls.seq_relationship.weight"] = torch.tensor(
        pretrained_model_params["classification"]["output_kernel"]
    )
    # 使用预训练模型参数中的输出偏置，创建新的张量并赋给新状态字典的键"cls.seq_relationship.bias"
    new_state_dict["cls.seq_relationship.bias"] = torch.tensor(
        pretrained_model_params["classification"]["output_bias"]
    )

    # 加载新状态字典到预训练模型中
    fnet_pretraining_model.load_state_dict(new_state_dict)

    # 打印信息，指示正在将预训练模型保存到指定路径
    print(f"Saving pretrained model to {save_path}")

    # 将预训练模型保存到指定路径
    fnet_pretraining_model.save_pretrained(save_path)
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行

    parser = argparse.ArgumentParser()
    # 创建命令行参数解析器对象

    # Required parameters
    parser.add_argument(
        "--flax_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    # 添加必需的命令行参数 --flax_checkpoint_path，指定 TensorFlow 检查点路径

    parser.add_argument(
        "--fnet_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained FNet model. \n"
            "This specifies the model architecture."
        ),
    )
    # 添加必需的命令行参数 --fnet_config_file，指定预训练 FNet 模型的配置 JSON 文件路径
    # 该文件用于指定模型的架构

    parser.add_argument("--save_path", default=None, type=str, required=True, help="Path to the output model.")
    # 添加必需的命令行参数 --save_path，指定输出模型的路径

    args = parser.parse_args()
    # 解析命令行参数并将其存储在 args 变量中

    convert_flax_checkpoint_to_pytorch(args.flax_checkpoint_path, args.fnet_config_file, args.save_path)
    # 调用 convert_flax_checkpoint_to_pytorch 函数，传递命令行参数中指定的路径信息
```