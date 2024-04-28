# `.\models\deprecated\bort\convert_bort_original_gluonnlp_checkpoint_to_pytorch.py`

```py
    # 设置编码格式为 UTF-8
    # 版权声明及许可信息
    # 导入必要的库
    # 导入 argparse 模块用于命令行参数解析
    # 导入 os 模块用于与操作系统交互
    # 导入 gluonnlp 库，用于自然语言处理任务
    # 导入 mxnet 库，用于深度学习任务
    # 导入 numpy 库，用于数值计算
    # 导入 torch 库，用于深度学习任务
    # 从 gluonnlp 库中导入一些基础函数和类
    # 从 gluonnlp.model.bert 模块中导入 BERTEncoder 类
    # 从 gluonnlp.model.utils 模块中导入 _load_vocab 函数
    # 从 gluonnlp.vocab 模块中导入 Vocab 类
    # 从 packaging 库中导入 version 类
    # 从 torch 库中导入 nn 模块
    # 从 transformers 库中导入 BertConfig、BertForMaskedLM、BertModel 和 RobertaTokenizer 类
    # 从 transformers.models.bert.modeling_bert 模块中导入一些 BERT 模型相关的类
    # 从 transformers.utils 模块中导入 logging 函数

    # 检查 gluonnlp 库的版本是否为 0.8.3，否则抛出异常
    if version.parse(nlp.__version__) != version.parse("0.8.3"):
        raise Exception("requires gluonnlp == 0.8.3")

    # 检查 mxnet 库的版本是否为 1.5.0，否则抛出异常
    if version.parse(mx.__version__) != version.parse("1.5.0"):
        raise Exception("requires mxnet == 1.5.0")

    # 设置日志记录的详细程度为信息级别
    logging.set_verbosity_info()
    # 获取日志记录器
    logger = logging.get_logger(__name__)

    # 定义一个示例文本
    SAMPLE_TEXT = "The Nymphenburg Palace is a beautiful palace in Munich!"

    # 定义一个函数，用于将原始的 Bort 检查点转换为 PyTorch 格式
    def convert_bort_checkpoint_to_pytorch(bort_checkpoint_path: str, pytorch_dump_folder_path: str):
        """
        Convert the original Bort checkpoint (based on MXNET and Gluonnlp) to our BERT structure-
        """

        # 原始 Bort 模型的配置参数
        bort_4_8_768_1024_hparams = {
            "attention_cell": "multi_head",
            "num_layers": 4,
            "units": 1024,
            "hidden_size": 768,
            "max_length": 512,
            "num_heads": 8,
            "scaled": True,
            "dropout": 0.1,
            "use_residual": True,
            "embed_size": 1024,
            "embed_dropout": 0.1,
            "word_embed": None,
            "layer_norm_eps": 1e-5,
            "token_type_vocab_size": 2,
        }

        # 预定义参数为原始 Bort 模型的配置参数
        predefined_args = bort_4_8_768_1024_hparams

        # 构建原始的 Bort 模型
        # 参考官方 BERT 实现，详情请见：
        # https://github.com/alexa/bort/blob/master/bort/bort.py
    # 创建一个 BERTEncoder 对象，由预定义的参数进行设置
    encoder = BERTEncoder(
        attention_cell=predefined_args["attention_cell"],
        num_layers=predefined_args["num_layers"],
        units=predefined_args["units"],
        hidden_size=predefined_args["hidden_size"],
        max_length=predefined_args["max_length"],
        num_heads=predefined_args["num_heads"],
        scaled=predefined_args["scaled"],
        dropout=predefined_args["dropout"],
        output_attention=False,
        output_all_encodings=False,
        use_residual=predefined_args["use_residual"],
        activation=predefined_args.get("activation", "gelu"),
        layer_norm_eps=predefined_args.get("layer_norm_eps", None),
    )

    # 指定词汇表的名称
    vocab_name = "openwebtext_ccnews_stories_books_cased"

    # 设定 Gluonnlp 的词汇表的下载路径
    gluon_cache_dir = os.path.join(get_home_dir(), "models")
    # 从指定名称和下载路径加载词汇表
    bort_vocab = _load_vocab(vocab_name, None, gluon_cache_dir, cls=Vocab)

    # 创建一个 BERTModel 对象，设置其参数
    original_bort = nlp.model.BERTModel(
        encoder,
        len(bort_vocab),
        units=predefined_args["units"],
        embed_size=predefined_args["embed_size"],
        embed_dropout=predefined_args["embed_dropout"],
        word_embed=predefined_args["word_embed"],
        use_pooler=False,
        use_token_type_embed=False,
        token_type_vocab_size=predefined_args["token_type_vocab_size"],
        use_classifier=False,
        use_decoder=False,
    )

    # 加载 BERT 模型的参数
    original_bort.load_parameters(bort_checkpoint_path, cast_dtype=True, ignore_extra=True)
    # 获取 BERT 模型的参数
    params = original_bort._collect_params_with_prefix()

    # 创建一个字典，包含了 HF BORT 的配置参数
    hf_bort_config_json = {
        "architectures": ["BertForMaskedLM"],
        "attention_probs_dropout_prob": predefined_args["dropout"],
        "hidden_act": "gelu",
        "hidden_dropout_prob": predefined_args["dropout"],
        "hidden_size": predefined_args["embed_size"],
        "initializer_range": 0.02,
        "intermediate_size": predefined_args["hidden_size"],
        "layer_norm_eps": predefined_args["layer_norm_eps"],
        "max_position_embeddings": predefined_args["max_length"],
        "model_type": "bort",
        "num_attention_heads": predefined_args["num_heads"],
        "num_hidden_layers": predefined_args["num_layers"],
        "pad_token_id": 1,  # 2 = BERT, 1 = RoBERTa
        "type_vocab_size": 1,  # 2 = BERT, 1 = RoBERTa
        "vocab_size": len(bort_vocab),
    }

    # 从字典创建一个 HF BertConfig 对象
    hf_bort_config = BertConfig.from_dict(hf_bort_config_json)
    # 创建一个 HF BertForMaskedLM 模型对象
    hf_bort_model = BertForMaskedLM(hf_bort_config)
    # 设置为评估模式
    hf_bort_model.eval()

    # 创建参数映射表，将 Gluonnlp 参数映射到 Transformers 参数
    # * 表示层索引
    #
    # | Gluon Parameter                                                | Transformers Parameter
    # | -------------------------------------------------------------- | ----------------------
    # 将下列参数从 MXNET 形式转换为 PyTorch 的形式:
    # | `encoder.layer_norm.beta`                                      | `bert.embeddings.LayerNorm.bias`
    # | `encoder.layer_norm.gamma`                                     | `bert.embeddings.LayerNorm.weight`
    # | `encoder.position_weight`                                      | `bert.embeddings.position_embeddings.weight`
    # | `word_embed.0.weight`                                          | `bert.embeddings.word_embeddings.weight`
    # | `encoder.transformer_cells.*.attention_cell.proj_key.bias`     | `bert.encoder.layer.*.attention.self.key.bias`
    # | `encoder.transformer_cells.*.attention_cell.proj_key.weight`   | `bert.encoder.layer.*.attention.self.key.weight`
    # | `encoder.transformer_cells.*.attention_cell.proj_query.bias`   | `bert.encoder.layer.*.attention.self.query.bias`
    # | `encoder.transformer_cells.*.attention_cell.proj_query.weight` | `bert.encoder.layer.*.attention.self.query.weight`
    # | `encoder.transformer_cells.*.attention_cell.proj_value.bias`   | `bert.encoder.layer.*.attention.self.value.bias`
    # | `encoder.transformer_cells.*.attention_cell.proj_value.weight` | `bert.encoder.layer.*.attention.self.value.weight`
    # | `encoder.transformer_cells.*.ffn.ffn_2.bias`                   | `bert.encoder.layer.*.attention.output.dense.bias`
    # | `encoder.transformer_cells.*.ffn.ffn_2.weight`                 | `bert.encoder.layer.*.attention.output.dense.weight`
    # | `encoder.transformer_cells.*.layer_norm.beta`                  | `bert.encoder.layer.*.attention.output.LayerNorm.bias`
    # | `encoder.transformer_cells.*.layer_norm.gamma`                 | `bert.encoder.layer.*.attention.output.LayerNorm.weight`
    # | `encoder.transformer_cells.*.ffn.ffn_1.bias`                   | `bert.encoder.layer.*.intermediate.dense.bias`
    # | `encoder.transformer_cells.*.ffn.ffn_1.weight`                 | `bert.encoder.layer.*.intermediate.dense.weight`
    # | `encoder.transformer_cells.*.ffn.layer_norm.beta`              | `bert.encoder.layer.*.output.LayerNorm.bias`
    # | `encoder.transformer_cells.*.ffn.layer_norm.gamma`             | `bert.encoder.layer.*.output.LayerNorm.weight`
    # | `encoder.transformer_cells.*.proj.bias`                        | `bert.encoder.layer.*.output.dense.bias`
    # | `encoder.transformer_cells.*.proj.weight`                      | `bert.encoder.layer.*.output.dense.weight`

    # 定义一个将 MXNET 数组转换为 PyTorch 参数的辅助函数
    def to_torch(mx_array) -> nn.Parameter:
        # 将 MXNET 数组转换为 PyTorch 的 FloatTensor，并封装为参数形式
        return nn.Parameter(torch.FloatTensor(mx_array.data().asnumpy()))

    # 检查参数的形状并将新的 HF 参数映射回去
    # 定义函数用于检查和映射参数
    def check_and_map_params(hf_param, gluon_param):
        # 获取参数的形状
        shape_hf = hf_param.shape

        # 将 gluon_param 转换为 torch 张量
        gluon_param = to_torch(params[gluon_param])
        shape_gluon = gluon_param.shape

        # 断言参数形状是否相同
        assert (
            shape_hf == shape_gluon
        ), f"The gluon parameter {gluon_param} has shape {shape_gluon}, but expects shape {shape_hf} for Transformers"

        # 返回 gluon 参数
        return gluon_param

    # 更新模型的 word_embeddings 权重
    hf_bort_model.bert.embeddings.word_embeddings.weight = check_and_map_params(
        hf_bort_model.bert.embeddings.word_embeddings.weight, "word_embed.0.weight"
    )
    # 更新模型的 position_embeddings 权重
    hf_bort_model.bert.embeddings.position_embeddings.weight = check_and_map_params(
        hf_bort_model.bert.embeddings.position_embeddings.weight, "encoder.position_weight"
    )
    # 更新模型的 LayerNorm 偏置
    hf_bort_model.bert.embeddings.LayerNorm.bias = check_and_map_params(
        hf_bort_model.bert.embeddings.LayerNorm.bias, "encoder.layer_norm.beta"
    )
    # 更新模型的 LayerNorm 权重
    hf_bort_model.bert.embeddings.LayerNorm.weight = check_and_map_params(
        hf_bort_model.bert.embeddings.LayerNorm.weight, "encoder.layer_norm.gamma"
    )

    # 根据 RoBERTa 转换脚本的灵感，将它们置零（Bort 没有使用它们）
    # 将 token_type_embeddings 权重置零
    hf_bort_model.bert.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        hf_bort_model.bert.embeddings.token_type_embeddings.weight.data
    )

    # 转换模型为半精度，以节省空间和能耗
    hf_bort_model.half()

    # 比较两个模型的输出
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # 编码示例文本
    input_ids = tokenizer.encode_plus(SAMPLE_TEXT)["input_ids"]

    # 获取 gluon 模型的输出
    gluon_input_ids = mx.nd.array([input_ids])
    output_gluon = original_bort(inputs=gluon_input_ids, token_types=[])

    # 获取 Transformer 模型的输出（保存并重新加载模型）
    hf_bort_model.save_pretrained(pytorch_dump_folder_path)
    hf_bort_model = BertModel.from_pretrained(pytorch_dump_folder_path)
    hf_bort_model.eval()

    input_ids = tokenizer.encode_plus(SAMPLE_TEXT, return_tensors="pt")
    output_hf = hf_bort_model(**input_ids)[0]

    # 将 gluon_layer 转换为 numpy 数组
    gluon_layer = output_gluon[0].asnumpy()
    # 将 hf_layer 转换为 numpy 数组
    hf_layer = output_hf[0].detach().numpy()

    # 计算最大的绝对差异
    max_absolute_diff = np.max(np.abs(hf_layer - gluon_layer)).item()
    # 判断是否所有元素的绝对差异均小于给定的绝对差值
    success = np.allclose(gluon_layer, hf_layer, atol=1e-3)

    if success:
        # 输出信息，表示两个模型输出相同的张量
        print("✔️ Both model do output the same tensors")
    else:
        # 输出信息，表示两个模型输出不同的张量
        print("❌ Both model do **NOT** output the same tensors")
        print("Absolute difference is:", max_absolute_diff)
# 如果当前脚本被直接执行，而非被作为模块导入，那么执行以下代码
if __name__ == "__main__":
    # 创建解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--bort_checkpoint_path", default=None, type=str, required=True, help="Path the official Bort params file."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析参数
    args = parser.parse_args()
    # 调用函数，将 Bort 检查点文件转换为 PyTorch 模型，传入解析后的参数
    convert_bort_checkpoint_to_pytorch(args.bort_checkpoint_path, args.pytorch_dump_folder_path)
```