# `.\convert_pytorch_checkpoint_to_tf2.py`

```py
# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Convert pytorch checkpoints to TensorFlow"""


import argparse  # 导入解析命令行参数的模块
import os  # 导入操作系统功能的模块

from . import (  # 导入当前包中的模块和符号
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
# 从当前包的utils模块中导入所需的符号：CONFIG_NAME, WEIGHTS_NAME, cached_file, logging
from .utils import CONFIG_NAME, WEIGHTS_NAME, cached_file, logging

# 如果torch可用，导入必要的模块：numpy和torch
if is_torch_available():
    import numpy as np
    import torch

    # 从当前包中导入多个模型类
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

    # 从pytorch_utils模块中导入is_torch_greater_or_equal_than_1_13函数
    from .pytorch_utils import is_torch_greater_or_equal_than_1_13

# 设置日志记录的详细程度为INFO级别
logging.set_verbosity_info()

# 定义模型类的映射字典，键为模型名称，值为元组，包含相应模型的配置类、TF/PyTorch模型类、预训练模型类以及预训练模型的存档列表
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
    "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad": (
        BertConfig,
        TFBertForQuestionAnswering,
        BertForQuestionAnswering,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "google-bert/bert-large-cased-whole-word-masking-finetuned-squad": (
        BertConfig,
        TFBertForQuestionAnswering,
        BertForQuestionAnswering,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "google-bert/bert-base-cased-finetuned-mrpc": (
        BertConfig,
        TFBertForSequenceClassification,
        BertForSequenceClassification,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "dpr": (
        DPRConfig,
        TFDPRQuestionEncoder,
        TFDPRContextEncoder,
        TFDPRReader,
        DPRQuestionEncoder,
        DPRContextEncoder,
        DPRReader,
        DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
    ),
    "openai-community/gpt2": (
        GPT2Config,
        TFGPT2LMHeadModel,
        GPT2LMHeadModel,
        GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "xlnet": (
        XLNetConfig,
        TFXLNetLMHeadModel,
        XLNetLMHeadModel,
        XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "xlm": (
        XLMConfig,
        TFXLMWithLMHeadModel,
        XLMWithLMHeadModel,
        XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "xlm-roberta": (
        XLMRobertaConfig,                           # XLMRoberta 模型的配置类
        TFXLMRobertaForMaskedLM,                    # 用于 TensorFlow 的 XLMRoberta 语言模型（MLM）
        XLMRobertaForMaskedLM,                      # XLMRoberta 语言模型（MLM）
        XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,  # XLMRoberta 预训练模型配置文件的映射
    ),
    "transfo-xl": (
        TransfoXLConfig,                            # TransfoXL 模型的配置类
        TFTransfoXLLMHeadModel,                     # 用于 TensorFlow 的 TransfoXL 语言模型头部
        TransfoXLLMHeadModel,                       # TransfoXL 语言模型头部
        TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,   # TransfoXL 预训练模型配置文件的映射
    ),
    "openai-community/openai-gpt": (
        OpenAIGPTConfig,                            # OpenAI GPT 模型的配置类
        TFOpenAIGPTLMHeadModel,                     # 用于 TensorFlow 的 OpenAI GPT 语言模型头部
        OpenAIGPTLMHeadModel,                       # OpenAI GPT 语言模型头部
        OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,   # OpenAI GPT 预训练模型配置文件的映射
    ),
    "roberta": (
        RobertaConfig,                              # Roberta 模型的配置类
        TFRobertaForCausalLM,                       # 用于 TensorFlow 的 Roberta 因果语言模型
        TFRobertaForMaskedLM,                       # 用于 TensorFlow 的 Roberta 语言模型（MLM）
        RobertaForMaskedLM,                         # Roberta 语言模型（MLM）
        ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,       # Roberta 预训练模型配置文件的映射
    ),
    "layoutlm": (
        LayoutLMConfig,                             # LayoutLM 模型的配置类
        TFLayoutLMForMaskedLM,                      # 用于 TensorFlow 的 LayoutLM 语言模型（MLM）
        LayoutLMForMaskedLM,                        # LayoutLM 语言模型（MLM）
        LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,     # LayoutLM 预训练模型的存档列表
    ),
    "FacebookAI/roberta-large-mnli": (
        RobertaConfig,                              # Roberta 模型的配置类
        TFRobertaForSequenceClassification,          # 用于 TensorFlow 的 Roberta 序列分类模型
        RobertaForSequenceClassification,           # Roberta 序列分类模型
        ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,       # Roberta 预训练模型配置文件的映射
    ),
    "camembert": (
        CamembertConfig,                            # Camembert 模型的配置类
        TFCamembertForMaskedLM,                     # 用于 TensorFlow 的 Camembert 语言模型（MLM）
        CamembertForMaskedLM,                       # Camembert 语言模型（MLM）
        CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,     # Camembert 预训练模型配置文件的映射
    ),
    "flaubert": (
        FlaubertConfig,                             # Flaubert 模型的配置类
        TFFlaubertWithLMHeadModel,                  # 用于 TensorFlow 的 Flaubert 语言模型头部
        FlaubertWithLMHeadModel,                    # Flaubert 语言模型头部
        FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,      # Flaubert 预训练模型配置文件的映射
    ),
    "distilbert": (
        DistilBertConfig,                           # DistilBERT 模型的配置类
        TFDistilBertForMaskedLM,                    # 用于 TensorFlow 的 DistilBERT 语言模型（MLM）
        DistilBertForMaskedLM,                      # DistilBERT 语言模型（MLM）
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,    # DistilBERT 预训练模型配置文件的映射
    ),
    "distilbert-base-distilled-squad": (
        DistilBertConfig,                           # DistilBERT 模型的配置类
        TFDistilBertForQuestionAnswering,           # 用于 TensorFlow 的 DistilBERT 问答模型
        DistilBertForQuestionAnswering,             # DistilBERT 问答模型
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,    # DistilBERT 预训练模型配置文件的映射
    ),
    "lxmert": (
        LxmertConfig,                               # LXMERT 模型的配置类
        TFLxmertForPreTraining,                     # 用于 TensorFlow 的 LXMERT 预训练模型
        LxmertForPreTraining,                       # LXMERT 预训练模型
        LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,        # LXMERT 预训练模型配置文件的映射
    ),
    "lxmert-visual-feature-encoder": (
        LxmertConfig,                               # LXMERT 模型的配置类
        TFLxmertVisualFeatureEncoder,               # 用于 TensorFlow 的 LXMERT 视觉特征编码器
        LxmertVisualFeatureEncoder,                 # LXMERT 视觉特征编码器
        LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,        # LXMERT 预训练模型配置文件的映射
    ),
    "Salesforce/ctrl": (
        CTRLConfig,                                 # CTRL 模型的配置类
        TFCTRLLMHeadModel,                          # 用于 TensorFlow 的 CTRL 语言模型头部
        CTRLLMHeadModel,                            # CTRL 语言模型头部
        CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP,          # CTRL 预训练模型配置文件的映射
    ),
    "albert": (
        AlbertConfig,                               # ALBERT 模型的配置类
        TFAlbertForPreTraining,                     # 用于 TensorFlow 的 ALBERT 预训练模型
        AlbertForPreTraining,                       # ALBERT 预训练模型
        ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,        # ALBERT 预训练模型配置文件的映射
    ),
    "t5": (
        T5Config,                                   # T5 模型的配置类
        TFT5ForConditionalGeneration,               # 用于 TensorFlow 的 T5 条件生成模型
        T5ForConditionalGeneration,                 # T5 条件生成模型
        T5_PRETRAINED_CONFIG_ARCHIVE_MAP,            # T5 预训练模型配置文件的映射
    ),
    "electra": (
        ElectraConfig,                              # Electra 模型的配置类
        TFElectraForPreTraining,                    # 用于 TensorFlow 的 Electra 预训练模型
        ElectraForPreTraining,                      # Electra 预训练模型
        ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP,       # Electra 预训练模型配置文件的映射
    ),
    "wav2vec2": (
        Wav2Vec2Config,                             # Wav2Vec2 模型的配置类
        TFWav2Vec2Model,                            # 用于 TensorFlow 的 Wav2Vec2 模型
        Wav2Vec2Model,                              # Wav2Vec2 模型
        WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP,   # Wav2Vec2 预训练模型配置文件的映射
    ),
}

# 将 PyTorch 检查点转换为 TensorFlow 格式
def convert_pt_checkpoint_to_tf(
    model_type, pytorch_checkpoint_path, config_file, tf_dump_path, compare_with_pt_model=False, use_cached_models=True
):
    # 检查模型类型是否在已知的模型类别中
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unrecognized model type, should be one of {list(MODEL_CLASSES.keys())}.")

    # 根据模型类型获取相应的类别信息
    config_class, model_class, pt_model_class, aws_config_map = MODEL_CLASSES[model_type]

    # 初始化 TensorFlow 模型
    if config_file in aws_config_map:
        # 如果配置文件在 AWS 配置映射中，可能需要缓存或下载配置文件
        config_file = cached_file(config_file, CONFIG_NAME, force_download=not use_cached_models)
    config = config_class.from_json_file(config_file)
    config.output_hidden_states = True
    config.output_attentions = True
    print(f"Building TensorFlow model from configuration: {config}")
    tf_model = model_class(config)

    # 从 TensorFlow 检查点加载权重
    if pytorch_checkpoint_path in aws_config_map.keys():
        # 如果 PyTorch 检查点路径在 AWS 配置映射中，可能需要缓存或下载检查点文件
        pytorch_checkpoint_path = cached_file(
            pytorch_checkpoint_path, WEIGHTS_NAME, force_download=not use_cached_models
        )
    # 将 PyTorch 检查点加载到 TensorFlow 模型中
    tf_model = load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path)

    # 如果需要与 PyTorch 模型进行比较
    if compare_with_pt_model:
        # 构建 TensorFlow 模型以获取网络结构
        tfo = tf_model(tf_model.dummy_inputs, training=False)

        # 根据 Torch 版本选择权重仅参数
        weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
        # 从 PyTorch 检查点加载状态字典
        state_dict = torch.load(
            pytorch_checkpoint_path,
            map_location="cpu",
            **weights_only_kwarg,
        )
        # 使用 PyTorch 模型类从预训练模型名称或路径加载模型
        pt_model = pt_model_class.from_pretrained(
            pretrained_model_name_or_path=None, config=config, state_dict=state_dict
        )

        # 通过禁用梯度计算运行 PyTorch 模型
        with torch.no_grad():
            pto = pt_model(**pt_model.dummy_inputs)

        # 转换为 NumPy 数组并计算模型输出的最大绝对差异
        np_pt = pto[0].numpy()
        np_tf = tfo[0].numpy()
        diff = np.amax(np.abs(np_pt - np_tf))
        print(f"Max absolute difference between models outputs {diff}")
        # 断言模型的最大绝对差异是否在可接受范围内
        assert diff <= 2e-2, f"Error, model absolute difference is >2e-2: {diff}"

    # 保存 PyTorch 模型的权重
    print(f"Save TensorFlow model to {tf_dump_path}")
    tf_model.save_weights(tf_dump_path, save_format="h5")


# 将所有 PyTorch 检查点转换为 TensorFlow 格式
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
    # 如果未提供模型类型参数，则使用所有已知模型类型
    if args_model_type is None:
        model_types = list(MODEL_CLASSES.keys())
    else:
        model_types = [args_model_type]
    # 对于每个模型类型，在循环中进行迭代，使用 enumerate 函数生成索引和元素
    for j, model_type in enumerate(model_types, start=1):
        # 打印分隔线，用于区分不同模型类型的输出
        print("=" * 100)
        # 打印当前转换的模型类型信息，包括总数和当前处理的序号
        print(f" Converting model type {j}/{len(model_types)}: {model_type}")
        # 打印分隔线，用于区分不同模型类型的输出
        print("=" * 100)
        
        # 如果当前模型类型不在预定义的模型类别中，则抛出数值错误异常
        if model_type not in MODEL_CLASSES:
            raise ValueError(f"Unrecognized model type {model_type}, should be one of {list(MODEL_CLASSES.keys())}.")

        # 从预定义的模型映射中获取配置类、模型类、PyTorch模型类以及AWS相关映射信息
        config_class, model_class, pt_model_class, aws_model_maps, aws_config_map = MODEL_CLASSES[model_type]

        # 如果未提供模型路径或名称，则使用AWS模型映射中的名称列表作为默认值
        if model_shortcut_names_or_path is None:
            model_shortcut_names_or_path = list(aws_model_maps.keys())
        # 如果未提供配置路径或名称，则使用模型快捷名称列表作为默认值
        if config_shortcut_names_or_path is None:
            config_shortcut_names_or_path = model_shortcut_names_or_path

        # 对于每个模型快捷名称和配置快捷名称的组合，使用 zip 函数生成索引和元素
        for i, (model_shortcut_name, config_shortcut_name) in enumerate(
            zip(model_shortcut_names_or_path, config_shortcut_names_or_path), start=1
        ):
            # 打印分隔线，用于区分不同模型转换过程的输出
            print("-" * 100)
            
            # 如果模型快捷名称中包含特定字符串（如"-squad"、"-mrpc"、"-mnli"），并且不是仅转换微调模型，则跳过当前模型的转换
            if "-squad" in model_shortcut_name or "-mrpc" in model_shortcut_name or "-mnli" in model_shortcut_name:
                if not only_convert_finetuned_models:
                    # 打印跳过信息，指示未转换的微调模型
                    print(f"    Skipping finetuned checkpoint {model_shortcut_name}")
                    continue
                # 将模型类型设为当前模型的名称，用于后续转换过程
                model_type = model_shortcut_name
            elif only_convert_finetuned_models:
                # 如果仅转换微调模型选项为真，则跳过非微调模型的转换
                print(f"    Skipping not finetuned checkpoint {model_shortcut_name}")
                continue
            
            # 打印当前转换的检查点信息，包括总数和当前处理的序号，以及模型快捷名称和模型类型
            print(f"    Converting checkpoint {i}/{len(aws_config_map)}: {model_shortcut_name} - model_type {model_type}")
            # 打印分隔线，用于区分不同模型转换过程的输出
            print("-" * 100)

            # 如果配置快捷名称存在于AWS配置映射中，则根据配置快捷名称下载配置文件
            if config_shortcut_name in aws_config_map:
                config_file = cached_file(config_shortcut_name, CONFIG_NAME, force_download=not use_cached_models)
            else:
                # 否则，将配置快捷名称作为配置文件名
                config_file = config_shortcut_name

            # 如果模型快捷名称存在于AWS模型映射中，则根据模型快捷名称下载模型权重文件
            if model_shortcut_name in aws_model_maps:
                model_file = cached_file(model_shortcut_name, WEIGHTS_NAME, force_download=not use_cached_models)
            else:
                # 否则，将模型快捷名称作为模型文件名
                model_file = model_shortcut_name

            # 如果模型快捷名称对应的文件已存在，则将模型快捷名称设为"converted_model"
            if os.path.isfile(model_shortcut_name):
                model_shortcut_name = "converted_model"

            # 调用转换函数，将PyTorch模型检查点转换为TensorFlow模型
            convert_pt_checkpoint_to_tf(
                model_type=model_type,
                pytorch_checkpoint_path=model_file,
                config_file=config_file,
                tf_dump_path=os.path.join(tf_dump_path, model_shortcut_name + "-tf_model.h5"),
                compare_with_pt_model=compare_with_pt_model,
            )
            # 如果设定移除缓存文件选项为真，则删除配置文件和模型文件
            if remove_cached_files:
                os.remove(config_file)
                os.remove(model_file)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 创建参数解析器

    # 必选参数
    parser.add_argument(
        "--tf_dump_path", default=None, type=str, required=True, help="Path to the output Tensorflow dump file."
    )
    # 输出TensorFlow转储文件的路径

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        help=(
            f"Model type selected in the list of {list(MODEL_CLASSES.keys())}. If not given, will download and "
            "convert all the models from AWS."
        ),
    )
    # 模型类型，可以选择预定义的模型类别或者从AWS下载转换所有模型

    parser.add_argument(
        "--pytorch_checkpoint_path",
        default=None,
        type=str,
        help=(
            "Path to the PyTorch checkpoint path or shortcut name to download from AWS. "
            "If not given, will download and convert all the checkpoints from AWS."
        ),
    )
    # PyTorch检查点文件路径或者从AWS下载的快捷名称

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
    # 预训练模型对应的配置文件，用于指定模型架构

    parser.add_argument(
        "--compare_with_pt_model", action="store_true", help="Compare Tensorflow and PyTorch model predictions."
    )
    # 比较TensorFlow和PyTorch模型预测结果

    parser.add_argument(
        "--use_cached_models",
        action="store_true",
        help="Use cached models if possible instead of updating to latest checkpoint versions.",
    )
    # 如果可能的话使用缓存的模型，而不是更新到最新的检查点版本

    parser.add_argument(
        "--remove_cached_files",
        action="store_true",
        help="Remove pytorch models after conversion (save memory when converting in batches).",
    )
    # 在转换完成后删除PyTorch模型文件，以节省内存（批量转换时）

    parser.add_argument("--only_convert_finetuned_models", action="store_true", help="Only convert finetuned models.")
    # 只转换微调过的模型

    args = parser.parse_args()

    # if args.pytorch_checkpoint_path is not None:
    #     convert_pt_checkpoint_to_tf(args.model_type.lower(),
    #                                 args.pytorch_checkpoint_path,
    #                                 args.config_file if args.config_file is not None else args.pytorch_checkpoint_path,
    #                                 args.tf_dump_path,
    #                                 compare_with_pt_model=args.compare_with_pt_model,
    #                                 use_cached_models=args.use_cached_models)
    # else:
    # 转换所有的 PyTorch 检查点到 TensorFlow 格式
    convert_all_pt_checkpoints_to_tf(
        # 将模型类型参数转换为小写，如果未提供则为 None
        args.model_type.lower() if args.model_type is not None else None,
        # TensorFlow 转换后的输出路径
        args.tf_dump_path,
        # 模型的快捷方式名称或路径的列表，如果提供了 PyTorch 检查点路径则作为单个元素传递，否则为 None
        model_shortcut_names_or_path=[args.pytorch_checkpoint_path] if args.pytorch_checkpoint_path is not None else None,
        # 配置文件的快捷方式名称或路径的列表，如果提供了配置文件路径则作为单个元素传递，否则为 None
        config_shortcut_names_or_path=[args.config_file] if args.config_file is not None else None,
        # 是否与 PyTorch 模型进行比较
        compare_with_pt_model=args.compare_with_pt_model,
        # 是否使用缓存的模型（如果可用）
        use_cached_models=args.use_cached_models,
        # 是否删除缓存的文件
        remove_cached_files=args.remove_cached_files,
        # 是否仅转换微调过的模型
        only_convert_finetuned_models=args.only_convert_finetuned_models,
    )
```