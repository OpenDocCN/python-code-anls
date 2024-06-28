# `.\convert_tf_hub_seq_to_seq_bert_to_pytorch.py`

```py
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Seq2Seq TF Hub checkpoint."""


import argparse  # 导入解析命令行参数的模块

from . import (  # 从当前包中导入以下模块
    BertConfig,  # 导入BertConfig类
    BertGenerationConfig,  # 导入BertGenerationConfig类
    BertGenerationDecoder,  # 导入BertGenerationDecoder类
    BertGenerationEncoder,  # 导入BertGenerationEncoder类
    load_tf_weights_in_bert_generation,  # 导入加载TF权重函数
    logging,  # 导入日志模块
)


logging.set_verbosity_info()  # 设置日志输出级别为info


def convert_tf_checkpoint_to_pytorch(tf_hub_path, pytorch_dump_path, is_encoder_named_decoder, vocab_size, is_encoder):
    # Initialise PyTorch model
    bert_config = BertConfig.from_pretrained(  # 从预训练配置中创建BertConfig对象
        "google-bert/bert-large-cased",  # 预训练模型名称
        vocab_size=vocab_size,  # 词汇表大小
        max_position_embeddings=512,  # 最大位置嵌入长度
        is_decoder=True,  # 设置为解码器模式
        add_cross_attention=True,  # 添加交叉注意力机制
    )
    bert_config_dict = bert_config.to_dict()  # 将BertConfig对象转换为字典形式
    del bert_config_dict["type_vocab_size"]  # 删除字典中的"type_vocab_size"键
    config = BertGenerationConfig(**bert_config_dict)  # 使用BertGenerationConfig类和字典初始化config对象
    if is_encoder:
        model = BertGenerationEncoder(config)  # 如果是编码器，创建BertGenerationEncoder模型
    else:
        model = BertGenerationDecoder(config)  # 如果不是编码器，创建BertGenerationDecoder模型
    print(f"Building PyTorch model from configuration: {config}")  # 打印构建PyTorch模型的配置信息

    # Load weights from tf checkpoint
    load_tf_weights_in_bert_generation(  # 载入TF检查点中的权重到模型中
        model,
        tf_hub_path,  # TensorFlow Hub检查点路径
        model_class="bert",  # 模型类别为BERT
        is_encoder_named_decoder=is_encoder_named_decoder,  # 是否将解码器命名为编码器
        is_encoder=is_encoder,  # 是否是编码器模型
    )

    # Save pytorch-model
    print(f"Save PyTorch model and config to {pytorch_dump_path}")  # 打印保存PyTorch模型和配置的路径
    model.save_pretrained(pytorch_dump_path)  # 保存预训练模型到指定路径


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器

    # Required parameters
    parser.add_argument(
        "--tf_hub_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )  # 添加tf_hub_path参数，必需，指定TensorFlow检查点路径
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )  # 添加pytorch_dump_path参数，必需，指定输出PyTorch模型路径
    parser.add_argument(
        "--is_encoder_named_decoder",
        action="store_true",
        help="If decoder has to be renamed to encoder in PyTorch model.",
    )  # 添加is_encoder_named_decoder参数，如果需要在PyTorch模型中将解码器命名为编码器
    parser.add_argument("--is_encoder", action="store_true", help="If model is an encoder.")  # 添加is_encoder参数，如果模型是编码器
    parser.add_argument("--vocab_size", default=50358, type=int, help="Vocab size of model")  # 添加vocab_size参数，默认为50358，模型的词汇表大小
    args = parser.parse_args()  # 解析命令行参数
    convert_tf_checkpoint_to_pytorch(  # 调用转换函数，将TF检查点转换为PyTorch模型
        args.tf_hub_path,
        args.pytorch_dump_path,
        args.is_encoder_named_decoder,
        args.vocab_size,
        is_encoder=args.is_encoder,
    )
```