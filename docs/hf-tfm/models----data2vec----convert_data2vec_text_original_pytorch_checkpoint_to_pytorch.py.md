# `.\models\data2vec\convert_data2vec_text_original_pytorch_checkpoint_to_pytorch.py`

```
# 指明代码以 UTF-8 编码格式编写
# 版权声明
# 遵循 Apache 协议，引入包、模块和类
# 导入 argparse 模块用于解析命令行参数
# 导入 os 和 pathlib 模块用于处理文件路径
# 导入 fairseq 模块
# 导入 torch 模块
# 导入 fairseq 的 TransformerSentenceEncoderLayer 类
# 导入 packaging 的 version 类
# 导入 transformers 中的不同类和函数
# 导入 logging 模块
# 检查 fairseq 版本是否大于等于 0.9.0
# 设置日志输出级别
# 获取 logger 对象
# 定义一个示例文本
# 定义转换数据2vec检查点到PyTorch的函数
    ＃将data2vec模型的权重复制/粘贴/调整到我们的BERT结构中
# 分割检查点路径，获取检查点目录和文件名
# 加载预训练的data2vec模型
# 将data2vec模型设为评估模式
# 获取data2vec模型的encoder和sentence_encoder
# 创建Data2VecTextConfig对象，并设置相关参数
# （这里可能有一个错误，data2vec.model应该是data2vec_model）
# 输出BERT的配置信息
    # 根据需要选择模型类型，并进行评估
    model = Data2VecTextForSequenceClassification(config) if classification_head else Data2VecTextForMaskedLM(config)
    model.eval()
    
    # 复制所有的权重
    # 嵌入层权重
    model.data2vec_text.embeddings.word_embeddings.weight = data2vec_sent_encoder.embed_tokens.weight  # 复制词嵌入权重
    model.data2vec_text.embeddings.position_embeddings.weight = data2vec_sent_encoder.embed_positions.weight  # 复制位置嵌入权重
    model.data2vec_text.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.data2vec_text.embeddings.token_type_embeddings.weight
    )  # 将标记类型嵌入权重清零，因为data2vec不使用它们
    model.data2vec_text.embeddings.LayerNorm.weight = data2vec_sent_encoder.layernorm_embedding.weight  # 复制归一化层权重
    model.data2vec_text.embeddings.LayerNorm.bias = data2vec_sent_encoder.layernorm_embedding.bias  # 复制归一化层偏置
    
    if classification_head:
        # 分类器权重
        model.classifier.dense.weight = data2vec.model.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = data2vec.model.classification_heads["mnli"].dense.bias
        model.classifier.out_proj.weight = data2vec.model.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = data2vec.model.classification_heads["mnli"].out_proj.bias
    else:
        # 语言模型头权重
        model.lm_head.dense.weight = data2vec_model.encoder.lm_head.dense.weight
        model.lm_head.dense.bias = data2vec_model.encoder.lm_head.dense.bias
        model.lm_head.layer_norm.weight = data2vec_model.encoder.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = data2vec_model.encoder.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = data2vec_model.encoder.lm_head.weight
        model.lm_head.decoder.bias = data2vec_model.encoder.lm_head.bias
    
    # 检查是否得到相同的结果
    input_ids: torch.Tensor = data2vec.encode(SAMPLE_TEXT).unsqueeze(0)  # 大小为1的批量输入
    
    our_output = model(input_ids)[0]  # 我们的模型输出
    if classification_head:
        their_output = data2vec.model.classification_heads["mnli"](data2vec.extract_features(input_ids))  # 对比模型的输出
    else:
        their_output = data2vec_model(input_ids)[0]
    print(our_output.shape, their_output.shape)
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # 最大的绝对差值
    success = torch.allclose(our_output, their_output, atol=1e-3)  # 查看两个模型输出的张量是否相同
    print("Do both models output the same tensors?", "🔥" if success else "💩")  # 打印两个模型输出的张量是否相同
    if not success:
        raise Exception("Something went wRoNg")  # 如果不相同，抛出异常
    
    pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)  # 创建存储路径
    print(f"Saving model to {pytorch_dump_folder_path}")  # 保存模型并打印保存路径
    model.save_pretrained(pytorch_dump_folder_path)  # 保存模型
# 如果该脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    # 解析传递给脚本的参数
    args = parser.parse_args()
    # 将参数传递给转换函数，执行数据转换
    convert_data2vec_checkpoint_to_pytorch(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.classification_head
    )
```