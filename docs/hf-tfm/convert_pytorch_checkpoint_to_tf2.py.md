# `.\transformers\convert_pytorch_checkpoint_to_tf2.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" 将 pytorch 检查点转换为 TensorFlow"""

# 导入必要的库
import argparse
import os

# 从模块中导入相关内容
from . import (
    ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BART_PRETRAINED_MODEL_ARCHIVE_LIST,
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
    DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
    DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
    ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
    LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    T5_PRETRAINED_CONFIG_ARCHIVE_MAP,
    TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    AlbertConfig,
    BartConfig,
    BertConfig,
    CamembertConfig,
    CTRLConfig,
    DistilBertConfig,
    DPRConfig,
    ElectraConfig,
    FlaubertConfig,
    GPT2Config,
    LayoutLMConfig,
    LxmertConfig,
    OpenAIGPTConfig,
    RobertaConfig,
    T5Config,
    TFAlbertForPreTraining,
    TFBartForConditionalGeneration,
    TFBartForSequenceClassification,
    TFBertForPreTraining,
    TFBertForQuestionAnswering,
    TFBertForSequenceClassification,
    TFCamembertForMaskedLM,
    TFCTRLLMHeadModel,
    TFDistilBertForMaskedLM,
    TFDistilBertForQuestionAnswering,
    TFDPRContextEncoder,
    TFDPRQuestionEncoder,
    TFDPRReader,
    TFElectraForPreTraining,
    TFFlaubertWithLMHeadModel,
    TFGPT2LMHeadModel,
    TFLayoutLMForMaskedLM,
    TFLxmertForPreTraining,
    TFLxmertVisualFeatureEncoder,
    TFOpenAIGPTLMHeadModel,
    TFRobertaForCausalLM,
    TFRobertaForMaskedLM,
    TFRobertaForSequenceClassification,
    TFT5ForConditionalGeneration,
    TFTransfoXLLMHeadModel,
    TFWav2Vec2Model,
    TFXLMRobertaForMaskedLM,
    TFXLMWithLMHeadModel,
    TFXLNetLMHeadModel,
    TransfoXLConfig,
    Wav2Vec2Config,
    Wav2Vec2Model,
    XLMConfig,
    XLMRobertaConfig,
    XLNetConfig,
    is_torch_available,
    load_pytorch_checkpoint_in_tf2_model,
)
# 从utils模块中导入CONFIG_NAME, WEIGHTS_NAME, cached_file, logging
from .utils import CONFIG_NAME, WEIGHTS_NAME, cached_file, logging

# 如果torch可用，则导入必要的库和模型类
if is_torch_available():
    import numpy as np
    import torch

    from . import (
        AlbertForPreTraining,
        BartForConditionalGeneration,
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        CamembertForMaskedLM,
        CTRLLMHeadModel,
        DistilBertForMaskedLM,
        DistilBertForQuestionAnswering,
        DPRContextEncoder,
        DPRQuestionEncoder,
        DPRReader,
        ElectraForPreTraining,
        FlaubertWithLMHeadModel,
        GPT2LMHeadModel,
        LayoutLMForMaskedLM,
        LxmertForPreTraining,
        LxmertVisualFeatureEncoder,
        OpenAIGPTLMHeadModel,
        RobertaForMaskedLM,
        RobertaForSequenceClassification,
        T5ForConditionalGeneration,
        TransfoXLLMHeadModel,
        XLMRobertaForMaskedLM,
        XLMWithLMHeadModel,
        XLNetLMHeadModel,
    )
    # 导入pytorch_utils中的is_torch_greater_or_equal_than_1_13函数

# 设置日志级别为info
logging.set_verbosity_info()

# 定义模型类字典MODEL_CLASSES
MODEL_CLASSES = {
    "bart": (
        BartConfig,
        TFBartForConditionalGeneration,
        TFBartForSequenceClassification,
        BartForConditionalGeneration,
        BART_PRETRAINED_MODEL_ARCHIVE_LIST,
    ),
    "bert": (
        BertConfig,
        TFBertForPreTraining,
        BertForPreTraining,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    # 其他模型类的定义
}
    "transfo-xl": (
        TransfoXLConfig,  # TransfoXL 模型的配置类
        TFTransfoXLLMHeadModel,  # 基于 TensorFlow 的 TransfoXL 语言模型
        TransfoXLLMHeadModel,  # TransfoXL 语言模型
        TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型的配置映射
    ),
    "openai-gpt": (
        OpenAIGPTConfig,  # OpenAI GPT 模型的配置类
        TFOpenAIGPTLMHeadModel,  # 基于 TensorFlow 的 OpenAI GPT 语言模型
        OpenAIGPTLMHeadModel,  # OpenAI GPT 语言模型
        OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型的配置映射
    ),
    "roberta": (
        RobertaConfig,  # RoBERTa 模型的配置类
        TFRobertaForCausalLM,  # 基于 TensorFlow 的 RoBERTa 有因果关系的语言模型
        TFRobertaForMaskedLM,  # 基于 TensorFlow 的 RoBERTa 掩码语言模型
        RobertaForMaskedLM,  # RoBERTa 掩码语言模型
        ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型的配置映射
    ),
    "layoutlm": (
        LayoutLMConfig,  # LayoutLM 模型的配置类
        TFLayoutLMForMaskedLM,  # 基于 TensorFlow 的 LayoutLM 掩码语言模型
        LayoutLMForMaskedLM,  # LayoutLM 掩码语言模型
        LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,  # 预训练模型的存档列表
    ),
    "roberta-large-mnli": (
        RobertaConfig,  # RoBERTa 模型的配置类
        TFRobertaForSequenceClassification,  # 基于 TensorFlow 的 RoBERTa 序列分类模型
        RobertaForSequenceClassification,  # RoBERTa 序列分类模型
        ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型的配置映射
    ),
    "camembert": (
        CamembertConfig,  # Camembert 模型的配置类
        TFCamembertForMaskedLM,  # 基于 TensorFlow 的 Camembert 掩码语言模型
        CamembertForMaskedLM,  # Camembert 掩码语言模型
        CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型的配置映射
    ),
    "flaubert": (
        FlaubertConfig,  # Flaubert 模型的配置类
        TFFlaubertWithLMHeadModel,  # 基于 TensorFlow 的 Flaubert 带有语言模型头的模型
        FlaubertWithLMHeadModel,  # Flaubert 带有语言模型头的模型
        FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型的配置映射
    ),
    "distilbert": (
        DistilBertConfig,  # DistilBERT 模型的配置类
        TFDistilBertForMaskedLM,  # 基于 TensorFlow 的 DistilBERT 掩码语言模型
        DistilBertForMaskedLM,  # DistilBERT 掩码语言模型
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型的配置映射
    ),
    "distilbert-base-distilled-squad": (
        DistilBertConfig,  # DistilBERT 模型的配置类
        TFDistilBertForQuestionAnswering,  # 基于 TensorFlow 的 DistilBERT 问答模型
        DistilBertForQuestionAnswering,  # DistilBERT 问答模型
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型的配置映射
    ),
    "lxmert": (
        LxmertConfig,  # LXMERT 模型的配置类
        TFLxmertForPreTraining,  # 基于 TensorFlow 的 LXMERT 预训练模型
        LxmertForPreTraining,  # LXMERT 预训练模型
        LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型的配置映射
    ),
    "lxmert-visual-feature-encoder": (
        LxmertConfig,  # LXMERT 模型的配置类
        TFLxmertVisualFeatureEncoder,  # 基于 TensorFlow 的 LXMERT 视觉特征编码器
        LxmertVisualFeatureEncoder,  # LXMERT 视觉特征编码器
        LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型的配置映射
    ),
    "ctrl": (
        CTRLConfig,  # CTRL 模型的配置类
        TFCTRLLMHeadModel,  # 基于 TensorFlow 的 CTRL 语言模型
        CTRLLMHeadModel,  # CTRL 语言模型
        CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型的配置映射
    ),
    "albert": (
        AlbertConfig,  # ALBERT 模型的配置类
        TFAlbertForPreTraining,  # 基于 TensorFlow 的 ALBERT 预训练模型
        AlbertForPreTraining,  # ALBERT 预训练模型
        ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型的配置映射
    ),
    "t5": (
        T5Config,  # T5 模型的配置类
        TFT5ForConditionalGeneration,  # 基于 TensorFlow 的 T5 条件生成模型
        T5ForConditionalGeneration,  # T5 条件生成模型
        T5_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型的配置映射
    ),
    "electra": (
        ElectraConfig,  # Electra 模型的配置类
        TFElectraForPreTraining,  # 基于 TensorFlow 的 Electra 预训练模型
        ElectraForPreTraining,  # Electra 预训练模型
        ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型的配置映射
    ),
    "wav2vec2": (
        Wav2Vec2Config,  # Wav2Vec2 模型的配置类
        TFWav2Vec2Model,  # 基于 TensorFlow 的 Wav2Vec2 模型
        Wav2Vec2Model,  # Wav2Vec2 模型
        WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练模型的配置映射
    ),
}

# 将 PyTorch 检查点转换为 TensorFlow 检查点
def convert_pt_checkpoint_to_tf(
    model_type, pytorch_checkpoint_path, config_file, tf_dump_path, compare_with_pt_model=False, use_cached_models=True
):
    # 如果模型类型不在 MODEL_CLASSES 中，则引发 ValueError 异常
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unrecognized model type, should be one of {list(MODEL_CLASSES.keys())}.")

    config_class, model_class, pt_model_class, aws_config_map = MODEL_CLASSES[model_type]

    # 初始化 TF 模型
    if config_file in aws_config_map:
        # 如果配置文件在 aws_config_map 中，则从缓存中获取或下载配置文件
        config_file = cached_file(config_file, CONFIG_NAME, force_download=not use_cached_models)
    config = config_class.from_json_file(config_file)
    config.output_hidden_states = True
    config.output_attentions = True
    print(f"Building TensorFlow model from configuration: {config}")
    tf_model = model_class(config)

    # 从 tf 检查点加载权重
    if pytorch_checkpoint_path in aws_config_map.keys():
        pytorch_checkpoint_path = cached_file(
            pytorch_checkpoint_path, WEIGHTS_NAME, force_download=not use_cached_models
        )
    # 在 tf2 模型中加载 PyTorch 检查点
    tf_model = load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path)

    if compare_with_pt_model:
        tfo = tf_model(tf_model.dummy_inputs, training=False)  # 构建网络

        state_dict = torch.load(
            pytorch_checkpoint_path,
            map_location="cpu",
            weights_only=is_torch_greater_or_equal_than_1_13,
        )
        pt_model = pt_model_class.from_pretrained(
            pretrained_model_name_or_path=None, config=config, state_dict=state_dict
        )

        with torch.no_grad():
            pto = pt_model(**pt_model.dummy_inputs)

        np_pt = pto[0].numpy()
        np_tf = tfo[0].numpy()
        diff = np.amax(np.abs(np_pt - np_tf))
        print(f"Max absolute difference between models outputs {diff}")
        assert diff <= 2e-2, f"Error, model absolute difference is >2e-2: {diff}"

    # 保存 PyTorch 模型
    print(f"Save TensorFlow model to {tf_dump_path}")
    tf_model.save_weights(tf_dump_path, save_format="h5")


# 将所有 PyTorch 检查点转换为 TensorFlow 检查点
def convert_all_pt_checkpoints_to_tf(
    args_model_type,
    tf_dump_path,
    model_shortcut_names_or_path=None,
    config_shortcut_names_or_path=None,
    compare_with_pt_model=False,
    use_cached_models=False,
    remove_cached_files=False,
    only_convert_finetuned_models=False,
):
    if args_model_type is None:
        model_types = list(MODEL_CLASSES.keys())
    else:
        model_types = [args_model_type]
    # 遍历模型类型列表，同时获取索引和模型类型
    for j, model_type in enumerate(model_types, start=1):
        # 打印分隔线
        print("=" * 100)
        # 打印当前转换的模型类型信息
        print(f" Converting model type {j}/{len(model_types)}: {model_type}")
        # 打印分隔线
        print("=" * 100)
        # 如果模型类型不在MODEL_CLASSES中，抛出异常
        if model_type not in MODEL_CLASSES:
            raise ValueError(f"Unrecognized model type {model_type}, should be one of {list(MODEL_CLASSES.keys())}.")

        # 从MODEL_CLASSES中获取模型类型对应的类和映射信息
        config_class, model_class, pt_model_class, aws_model_maps, aws_config_map = MODEL_CLASSES[model_type]

        # 如果未提供模型快捷名称或路径，则使用AWS模型映射的键列表
        if model_shortcut_names_or_path is None:
            model_shortcut_names_or_path = list(aws_model_maps.keys())
        # 如果未提供配置快捷名称或路径，则使用模型快捷名称或路径列表
        if config_shortcut_names_or_path is None:
            config_shortcut_names_or_path = model_shortcut_names_or_path

        # 遍历模型快捷名称或路径列表，同时获取索引和模型/配置快捷名称或路径
        for i, (model_shortcut_name, config_shortcut_name) in enumerate(
            zip(model_shortcut_names_or_path, config_shortcut_names_or_path), start=1
        ):
            # 打印分隔线
            print("-" * 100)
            # 如果模型快捷名称中包含指定字符串或只转换精调模型，则跳过当前迭代
            if "-squad" in model_shortcut_name or "-mrpc" in model_shortcut_name or "-mnli" in model_shortcut_name:
                if not only_convert_finetuned_models:
                    # 打印跳过精调模型的信息
                    print(f"    Skipping finetuned checkpoint {model_shortcut_name}")
                    # 跳过当前迭代
                    continue
                # 将模型类型设置为模型快捷名称，用于后续处理
                model_type = model_shortcut_name
            # 如果只转换精调模型且模型不是精调模型，则跳过当前迭代
            elif only_convert_finetuned_models:
                # 打印跳过非精调模型的信息
                print(f"    Skipping not finetuned checkpoint {model_shortcut_name}")
                # 跳过当前迭代
                continue
            # 打印转换当前模型的信息
            print(
                f"    Converting checkpoint {i}/{len(aws_config_map)}: {model_shortcut_name} - model_type {model_type}"
            )
            # 打印分隔线
            print("-" * 100)

            # 如果配置快捷名称在AWS配置映射中，则使用缓存文件获取配置文件
            if config_shortcut_name in aws_config_map:
                config_file = cached_file(config_shortcut_name, CONFIG_NAME, force_download=not use_cached_models)
            else:
                config_file = config_shortcut_name

            # 如果模型快捷名称在AWS模型映射中，则使用缓存文件获取模型文件
            if model_shortcut_name in aws_model_maps:
                model_file = cached_file(model_shortcut_name, WEIGHTS_NAME, force_download=not use_cached_models)
            else:
                model_file = model_shortcut_name

            # 如果模型快捷名称对应的文件存在，则将模型快捷名称设置为"converted_model"
            if os.path.isfile(model_shortcut_name):
                model_shortcut_name = "converted_model"

            # 转换PyTorch的检查点到TensorFlow的检查点
            convert_pt_checkpoint_to_tf(
                model_type=model_type,
                pytorch_checkpoint_path=model_file,
                config_file=config_file,
                tf_dump_path=os.path.join(tf_dump_path, model_shortcut_name + "-tf_model.h5"),
                compare_with_pt_model=compare_with_pt_model,
            )
            # 如果需要移除缓存文件，则删除配置文件和模型文件
            if remove_cached_files:
                os.remove(config_file)
                os.remove(model_file)
# 如果这个脚本作为主程序执行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--tf_dump_path", default=None, type=str, required=True, help="Path to the output Tensorflow dump file."
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        help=(
            f"Model type selected in the list of {list(MODEL_CLASSES.keys())}. If not given, will download and "
            "convert all the models from AWS."
        ),
    )
    parser.add_argument(
        "--pytorch_checkpoint_path",
        default=None,
        type=str,
        help=(
            "Path to the PyTorch checkpoint path or shortcut name to download from AWS. "
            "If not given, will download and convert all the checkpoints from AWS."
        ),
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        help=(
            "The config json file corresponding to the pre-trained model. \n"
            "This specifies the model architecture. If not given and "
            "--pytorch_checkpoint_path is not given or is a shortcut name "
            "use the configuration associated to the shortcut name on the AWS"
        ),
    )
    parser.add_argument(
        "--compare_with_pt_model", action="store_true", help="Compare Tensorflow and PyTorch model predictions."
    )
    parser.add_argument(
        "--use_cached_models",
        action="store_true",
        help="Use cached models if possible instead of updating to latest checkpoint versions.",
    )
    parser.add_argument(
        "--remove_cached_files",
        action="store_true",
        help="Remove pytorch models after conversion (save memory when converting in batches).",
    )
    parser.add_argument("--only_convert_finetuned_models", action="store_true", help="Only convert finetuned models.")
    # 解析命令行参数
    args = parser.parse_args()

    # 如果指定了 PyTorch 检查点路径
    if args.pytorch_checkpoint_path is not None:
        # 调用函数将 PyTorch 检查点转换为 TensorFlow 格式
        convert_pt_checkpoint_to_tf(args.model_type.lower(),
                                    args.pytorch_checkpoint_path,
                                    args.config_file if args.config_file is not None else args.pytorch_checkpoint_path,
                                    args.tf_dump_path,
                                    compare_with_pt_model=args.compare_with_pt_model,
                                    use_cached_models=args.use_cached_models)
    # 否则，如果未指定 PyTorch 检查点路径
    else:
```  
    # 调用函数将所有 PyTorch 检查点转换为 TensorFlow 格式
    convert_all_pt_checkpoints_to_tf(
        # 如果指定了模型类型，则将其转换为小写；否则为 None
        args.model_type.lower() if args.model_type is not None else None,
        # TensorFlow 转换后的路径
        args.tf_dump_path,
        # 模型的快捷名称或路径列表，如果指定了 PyTorch 检查点路径，则为该路径；否则为 None
        model_shortcut_names_or_path=[args.pytorch_checkpoint_path]
        if args.pytorch_checkpoint_path is not None
        else None,
        # 配置文件的快捷名称或路径列表，如果指定了配置文件路径，则为该路径；否则为 None
        config_shortcut_names_or_path=[args.config_file] if args.config_file is not None else None,
        # 是否与 PyTorch 模型进行比较
        compare_with_pt_model=args.compare_with_pt_model,
        # 是否使用缓存的模型
        use_cached_models=args.use_cached_models,
        # 是否删除缓存文件
        remove_cached_files=args.remove_cached_files,
        # 是否仅转换微调过的模型
        only_convert_finetuned_models=args.only_convert_finetuned_models,
    )
```