# `.\models\bert\convert_bert_token_dropping_original_tf2_checkpoint_to_pytorch.py`

```py
# 打印加载基于给定配置文件的模型信息
print(f"Loading model based on config from {config_path}...")
# 从指定路径加载BERT配置信息
config = BertConfig.from_json_file(config_path)
# 使用加载的配置信息创建一个BertForMaskedLM模型实例
model = BertForMaskedLM(config)

# 以下是待继续完善的部分，涉及模型的各个层次
    # 遍历每个隐藏层的索引，从0到config.num_hidden_layers-1
    for layer_index in range(0, config.num_hidden_layers):
        # 获取当前层的BertLayer对象
        layer: BertLayer = model.bert.encoder.layer[layer_index]

        # Self-attention部分
        # 获取当前层的self-attention模块
        self_attn: BertSelfAttention = layer.attention.self

        # 设置self-attention中的query权重数据
        self_attn.query.weight.data = get_encoder_attention_layer_array(
            layer_index, "_query_dense/kernel", self_attn.query.weight.data.shape
        )
        # 设置self-attention中的query偏置数据
        self_attn.query.bias.data = get_encoder_attention_layer_array(
            layer_index, "_query_dense/bias", self_attn.query.bias.data.shape
        )
        # 设置self-attention中的key权重数据
        self_attn.key.weight.data = get_encoder_attention_layer_array(
            layer_index, "_key_dense/kernel", self_attn.key.weight.data.shape
        )
        # 设置self-attention中的key偏置数据
        self_attn.key.bias.data = get_encoder_attention_layer_array(
            layer_index, "_key_dense/bias", self_attn.key.bias.data.shape
        )
        # 设置self-attention中的value权重数据
        self_attn.value.weight.data = get_encoder_attention_layer_array(
            layer_index, "_value_dense/kernel", self_attn.value.weight.data.shape
        )
        # 设置self-attention中的value偏置数据
        self_attn.value.bias.data = get_encoder_attention_layer_array(
            layer_index, "_value_dense/bias", self_attn.value.bias.data.shape
        )

        # Self-attention输出部分
        # 获取self-attention输出层对象
        self_output: BertSelfOutput = layer.attention.output

        # 设置self-attention输出层中dense层的权重数据
        self_output.dense.weight.data = get_encoder_attention_layer_array(
            layer_index, "_output_dense/kernel", self_output.dense.weight.data.shape
        )
        # 设置self-attention输出层中dense层的偏置数据
        self_output.dense.bias.data = get_encoder_attention_layer_array(
            layer_index, "_output_dense/bias", self_output.dense.bias.data.shape
        )

        # 设置self-attention输出层中LayerNorm的权重数据
        self_output.LayerNorm.weight.data = get_encoder_layer_array(layer_index, "_attention_layer_norm/gamma")
        # 设置self-attention输出层中LayerNorm的偏置数据
        self_output.LayerNorm.bias.data = get_encoder_layer_array(layer_index, "_attention_layer_norm/beta")

        # Intermediate部分
        # 获取当前层的Intermediate对象
        intermediate: BertIntermediate = layer.intermediate

        # 设置Intermediate层中dense层的权重数据
        intermediate.dense.weight.data = get_encoder_layer_array(layer_index, "_intermediate_dense/kernel")
        # 设置Intermediate层中dense层的偏置数据
        intermediate.dense.bias.data = get_encoder_layer_array(layer_index, "_intermediate_dense/bias")

        # Output部分
        # 获取当前层的Output对象
        bert_output: BertOutput = layer.output

        # 设置Output层中dense层的权重数据
        bert_output.dense.weight.data = get_encoder_layer_array(layer_index, "_output_dense/kernel")
        # 设置Output层中dense层的偏置数据
        bert_output.dense.bias.data = get_encoder_layer_array(layer_index, "_output_dense/bias")

        # 设置Output层中LayerNorm的权重数据
        bert_output.LayerNorm.weight.data = get_encoder_layer_array(layer_index, "_output_layer_norm/gamma")
        # 设置Output层中LayerNorm的偏置数据
        bert_output.LayerNorm.bias.data = get_encoder_layer_array(layer_index, "_output_layer_norm/beta")

    # Embeddings部分
    # 设置BERT模型的位置嵌入权重数据
    model.bert.embeddings.position_embeddings.weight.data = get_encoder_array("_position_embedding_layer/embeddings")
    # 设置BERT模型的token类型嵌入权重数据
    model.bert.embeddings.token_type_embeddings.weight.data = get_encoder_array("_type_embedding_layer/embeddings")
    # 设置BERT模型的嵌入层LayerNorm的权重数据
    model.bert.embeddings.LayerNorm.weight.data = get_encoder_array("_embedding_norm_layer/gamma")
    # 设置BERT模型的嵌入层LayerNorm的偏置数据为从文件中获取的编码器数组
    model.bert.embeddings.LayerNorm.bias.data = get_encoder_array("_embedding_norm_layer/beta")

    # LM头部
    lm_head = model.cls.predictions.transform

    # 设置LM头部中dense层的权重数据为从文件中获取的masked LM数组
    lm_head.dense.weight.data = get_masked_lm_array("dense/kernel")
    # 设置LM头部中dense层的偏置数据为从文件中获取的masked LM数组
    lm_head.dense.bias.data = get_masked_lm_array("dense/bias")

    # 设置LM头部中LayerNorm层的权重数据为从文件中获取的masked LM数组
    lm_head.LayerNorm.weight.data = get_masked_lm_array("layer_norm/gamma")
    # 设置LM头部中LayerNorm层的偏置数据为从文件中获取的masked LM数组
    lm_head.LayerNorm.bias.data = get_masked_lm_array("layer_norm/beta")

    # 设置BERT模型的嵌入层中词嵌入权重数据为从文件中获取的masked LM数组
    model.bert.embeddings.word_embeddings.weight.data = get_masked_lm_array("embedding_table")

    # 设置BERT模型的池化层为一个新的BertPooler对象，根据配置信息
    model.bert.pooler = BertPooler(config=config)
    # 设置BERT模型的池化层dense层的权重数据为从文件中获取的编码器数组
    model.bert.pooler.dense.weight.data: BertPooler = get_encoder_array("_pooler_layer/kernel")
    # 设置BERT模型的池化层dense层的偏置数据为从文件中获取的编码器数组
    model.bert.pooler.dense.bias.data: BertPooler = get_encoder_array("_pooler_layer/bias")

    # 导出最终的模型到指定的PyTorch保存路径
    model.save_pretrained(pytorch_dump_path)

    # 集成测试 - 应该能够无错误加载 ;)
    # 从指定的PyTorch保存路径加载一个新的BertForMaskedLM模型
    new_model = BertForMaskedLM.from_pretrained(pytorch_dump_path)
    # 打印新模型的评估结果
    print(new_model.eval())

    # 打印信息：模型转换成功完成！
    print("Model conversion was done successfully!")
if __name__ == "__main__":
    # 当该模块被直接运行时执行以下代码
    parser = argparse.ArgumentParser()
    # 创建参数解析器对象
    parser.add_argument(
        "--tf_checkpoint_path", type=str, required=True, help="Path to the TensorFlow Token Dropping checkpoint path."
    )
    # 添加命令行参数：TensorFlow Token Dropping 检查点的路径
    parser.add_argument(
        "--bert_config_file",
        type=str,
        required=True,
        help="The config json file corresponding to the BERT model. This specifies the model architecture.",
    )
    # 添加命令行参数：BERT 模型配置文件的路径，指定了模型的架构
    parser.add_argument(
        "--pytorch_dump_path",
        type=str,
        required=True,
        help="Path to the output PyTorch model.",
    )
    # 添加命令行参数：PyTorch 模型输出路径
    args = parser.parse_args()
    # 解析命令行参数，并将其存储在 args 对象中
    convert_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)
    # 调用函数 convert_checkpoint_to_pytorch，传入解析得到的参数
```