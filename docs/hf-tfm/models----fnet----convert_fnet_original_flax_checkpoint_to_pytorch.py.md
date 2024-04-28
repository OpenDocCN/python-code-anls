# `.\models\fnet\convert_fnet_original_flax_checkpoint_to_pytorch.py`

```
# 导入所需的库和模块
import argparse
import torch
from flax.training.checkpoints import restore_checkpoint
from transformers import FNetConfig, FNetForPreTraining
from transformers.utils import logging

# 设置日志输出级别
logging.set_verbosity_info()

# 将 Flax 的检查点转换为 PyTorch 模型
def convert_flax_checkpoint_to_pytorch(flax_checkpoint_path, fnet_config_file, save_path):
    # 根据配置文件创建 FNetConfig 对象
    config = FNetConfig.from_json_file(fnet_config_file)
    print(f"Building PyTorch model from configuration: {config}")
    # 使用配置对象创建 FNetForPreTraining 模型
    fnet_pretraining_model = FNetForPreTraining(config)

    # 恢复 Flax 检查点
    checkpoint_dict = restore_checkpoint(flax_checkpoint_path, None)
    # 获取预训练模型参数
    pretrained_model_params = checkpoint_dict["target"]

    # 嵌入层
    state_dict = fnet_pretraining_model.state_dict()
    
    # 位置 ID
    position_ids = state_dict["fnet.embeddings.position_ids"]
    new_state_dict = {"fnet.embeddings.position_ids": position_ids}
    
    # 嵌入层权重
    new_state_dict["fnet.embeddings.word_embeddings.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["word"]["embedding"]
    )
    new_state_dict["fnet.embeddings.position_embeddings.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["position"]["embedding"][0]
    )
    new_state_dict["fnet.embeddings.token_type_embeddings.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["type"]["embedding"]
    )
    new_state_dict["fnet.embeddings.projection.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["hidden_mapping_in"]["kernel"]
    ).T
    new_state_dict["fnet.embeddings.projection.bias"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["hidden_mapping_in"]["bias"]
    )
    new_state_dict["fnet.embeddings.LayerNorm.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["layer_norm"]["scale"]
    )
    new_state_dict["fnet.embeddings.LayerNorm.bias"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["layer_norm"]["bias"]
    )

    # 编码器层
    # 遍历隐藏层的数量
    for layer in range(config.num_hidden_layers):
        # 设置新状态字典中的权重为预训练模型的混合层归一化权重
        new_state_dict[f"fnet.encoder.layer.{layer}.fourier.output.LayerNorm.weight"] = torch.tensor(
            pretrained_model_params["encoder"][f"encoder_{layer}"]["mixing_layer_norm"]["scale"]
        )
        # 设置新状态字典中的偏置为预训练模型的混合层归一化偏置
        new_state_dict[f"fnet.encoder.layer.{layer}.fourier.output.LayerNorm.bias"] = torch.tensor(
            pretrained_model_params["encoder"][f"encoder_{layer}"]["mixing_layer_norm"]["bias"]
        )

        # 设置新状态字典中的权重为预训练模型的中间层的转置核
        new_state_dict[f"fnet.encoder.layer.{layer}.intermediate.dense.weight"] = torch.tensor(
            pretrained_model_params["encoder"][f"feed_forward_{layer}"]["intermediate"]["kernel"]
        ).T
        # 设置新状态字典中的偏置为预训练模型的中间层的偏置
        new_state_dict[f"fnet.encoder.layer.{layer}.intermediate.dense.bias"] = torch.tensor(
            pretrained_model_params["encoder"][f"feed_forward_{layer}"]["intermediate"]["bias"]
        )

        # 设置新状态字典中的权重为预训练模型的输出层的转置核
        new_state_dict[f"fnet.encoder.layer.{layer}.output.dense.weight"] = torch.tensor(
            pretrained_model_params["encoder"][f"feed_forward_{layer}"]["output"]["kernel"]
        ).T
        # 设置新状态字典中的偏置为预训练模型的输出层的偏置
        new_state_dict[f"fnet.encoder.layer.{layer}.output.dense.bias"] = torch.tensor(
            pretrained_model_params["encoder"][f"feed_forward_{layer}"]["output"]["bias"]
        )

        # 设置新状态字典中的权重为预训练模型的输出层归一化权重
        new_state_dict[f"fnet.encoder.layer.{layer}.output.LayerNorm.weight"] = torch.tensor(
            pretrained_model_params["encoder"][f"encoder_{layer}"]["output_layer_norm"]["scale"]
        )
        # 设置新状态字典中的偏置为预训练模型的输出层归一化偏置
        new_state_dict[f"fnet.encoder.layer.{layer}.output.LayerNorm.bias"] = torch.tensor(
            pretrained_model_params["encoder"][f"encoder_{layer}"]["output_layer_norm"]["bias"]
        )

    # Pooler Layers
    # 设置新状态字典中的权重为预训练模型的池化层的转置核
    new_state_dict["fnet.pooler.dense.weight"] = torch.tensor(pretrained_model_params["encoder"]["pooler"]["kernel"]).T
    # 设置新状态字典中的偏置为预训练模型的池化层的偏置
    new_state_dict["fnet.pooler.dense.bias"] = torch.tensor(pretrained_model_params["encoder"]["pooler"]["bias"])

    # Masked LM Layers
    # 设置新状态字典中的权重为预训练模型的预测层的转置核
    new_state_dict["cls.predictions.transform.dense.weight"] = torch.tensor(
        pretrained_model_params["predictions_dense"]["kernel"]
    ).T
    # 设置新状态字典中的偏置为预训练模型的预测层的偏置
    new_state_dict["cls.predictions.transform.dense.bias"] = torch.tensor(
        pretrained_model_params["predictions_dense"]["bias"]
    )
    # 设置新状态字典中的权重为预训练模型的预测层归一化权重
    new_state_dict["cls.predictions.transform.LayerNorm.weight"] = torch.tensor(
        pretrained_model_params["predictions_layer_norm"]["scale"]
    )
    # 设置新状态字典中的偏置为预训练模型的预测层归一化偏置
    new_state_dict["cls.predictions.transform.LayerNorm.bias"] = torch.tensor(
        pretrained_model_params["predictions_layer_norm"]["bias"]
    )
    # 设置新状态字典中的权重为预训练模型的词嵌入权重
    new_state_dict["cls.predictions.decoder.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["word"]["embedding"]
    )
    # 设置新状态字典中的偏置为预训练模型的输出偏置
    new_state_dict["cls.predictions.decoder.bias"] = torch.tensor(
        pretrained_model_params["predictions_output"]["output_bias"]
    )
    # 设置新状态字典中的偏置为预训练模型的输出偏置
    new_state_dict["cls.predictions.bias"] = torch.tensor(pretrained_model_params["predictions_output"]["output_bias"])

    # Seq Relationship Layers
    # 将预训练模型参数中的分类任务的输出层权重加载到新的状态字典中
    new_state_dict["cls.seq_relationship.weight"] = torch.tensor(
        pretrained_model_params["classification"]["output_kernel"]
    )
    # 将预训练模型参数中的分类任务的输出层偏置加载到新的状态字典中
    new_state_dict["cls.seq_relationship.bias"] = torch.tensor(
        pretrained_model_params["classification"]["output_bias"]
    )

    # 加载新的状态字典到 Fine-Tuning 网络的模型中
    fnet_pretraining_model.load_state_dict(new_state_dict)

    # 保存 Fine-Tuning 后的预训练模型
    print(f"Saving pretrained model to {save_path}")
    fnet_pretraining_model.save_pretrained(save_path)
# 如果当前脚本被直接执行而不是被导入，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必须的参数
    parser.add_argument(
        "--flax_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
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
    parser.add_argument("--save_path", default=None, type=str, required=True, help="Path to the output model.")
    # 解析命令行参数并存储到args对象
    args = parser.parse_args()
    # 调用函数，将Flax的checkpoint转换为PyTorch的模型
    convert_flax_checkpoint_to_pytorch(args.flax_checkpoint_path, args.fnet_config_file, args.save_path)
```