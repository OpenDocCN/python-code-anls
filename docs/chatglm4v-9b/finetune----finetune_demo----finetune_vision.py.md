# `.\chatglm4-finetune\finetune_demo\finetune_vision.py`

```py
# -*- coding: utf-8 -*-  # 指定文件编码为 UTF-8
import os  # 导入操作系统功能模块
import jieba  # 导入中文分词库
import dataclasses as dc  # 导入数据类模块并重命名为 dc
import functools  # 导入高阶函数模块
from collections.abc import Callable, Mapping, Sequence  # 导入集合相关的类型
from pathlib import Path  # 导入路径处理模块
from typing import Annotated, Any, Union  # 导入类型提示相关模块
import numpy as np  # 导入 NumPy 数组处理库
import ruamel.yaml as yaml  # 导入 YAML 处理库
import torch  # 导入 PyTorch 深度学习框架
import typer  # 导入命令行界面库
from datasets import Dataset, Split  # 从 datasets 导入数据集和分割
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # 导入 BLEU 分数计算功能
from peft import PeftConfig, get_peft_config, get_peft_model  # 导入 PEFT 配置相关模块
from rouge_chinese import Rouge  # 导入中文 ROUGE 评估工具
from torch import nn  # 导入 PyTorch 神经网络模块
from transformers import (  # 从 transformers 库导入各种模型和工具
    AutoModelForCausalLM,  # 自动加载因果语言模型
    AutoTokenizer,  # 自动加载分词器
    EvalPrediction,  # 导入评估预测结果的类
    GenerationConfig,  # 导入生成配置类
    PreTrainedTokenizer,  # 导入预训练分词器
    Seq2SeqTrainingArguments,  # 导入序列到序列训练参数类
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq  # 导入序列到序列数据整理类并重命名
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer  # 导入序列到序列训练器并重命名
from datasets import load_dataset, DatasetDict, NamedSplit  # 导入数据集加载和字典功能
from typing import Optional  # 导入可选类型提示
from PIL import Image  # 导入图像处理库

app = typer.Typer(pretty_exceptions_show_locals=False)  # 创建 Typer 应用，禁用本地异常显示
img = Image.new('L', (224, 224), 0).convert('RGB')  # 创建一个 224x224 的黑色灰度图像并转换为 RGB

class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):  # 定义数据整理类，继承自 _DataCollatorForSeq2Seq
    def __call__(self, features, return_tensors=None):  # 定义调用方法，处理输入特征
        output_ids = ([feature['output_ids'] for feature in features] if 'output_ids' in features[0].keys() else None)  # 提取输出 ID
        if output_ids is not None:  # 检查输出 ID 是否存在
            max_output_length = max(len(out) for out in output_ids)  # 获取最大输出长度
            if self.pad_to_multiple_of is not None:  # 如果需要填充到特定倍数
                max_output_length = (  # 计算填充后的最大输出长度
                        (
                                max_output_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
            for feature in features:  # 遍历特征进行填充
                remainder = [self.tokenizer.pad_token_id] * (  # 创建填充列表
                        max_output_length - len(feature['output_ids'])
                )
                if isinstance(feature['output_ids'], list):  # 检查输出 ID 类型
                    feature['output_ids'] = feature['output_ids'] + remainder  # 列表形式直接拼接
                else:  # 否则使用 NumPy 进行拼接
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)  # 转换为 int64 类型
        return super().__call__(features, return_tensors)  # 调用父类的方法返回结果


class Seq2SeqTrainer(_Seq2SeqTrainer):  # 定义序列到序列训练器类，继承自 _Seq2SeqTrainer
    # Not Support for apex  # 说明不支持 apex
    def training_step(self, model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:  # 定义训练步骤

        model.train()  # 将模型设置为训练模式
        inputs = self._prepare_inputs(inputs)  # 准备输入数据

        with self.compute_loss_context_manager():  # 计算损失的上下文管理器
            loss = self.compute_loss(model, inputs)  # 计算模型的损失

        if self.args.n_gpu > 1:  # 检查是否使用多 GPU
            loss = loss.mean()  # 如果是，取平均损失
        self.accelerator.backward(loss)  # 反向传播损失
        detached_loss = loss.detach() / self.args.gradient_accumulation_steps  # 分离损失并进行梯度累积
        del inputs  # 删除输入数据以释放内存
        torch.cuda.empty_cache()  # 清空 CUDA 缓存
        return detached_loss  # 返回分离后的损失

    def prediction_step(  # 定义预测步骤
            self,
            model: nn.Module,  # 输入模型
            inputs: dict,  # 输入字典
            prediction_loss_only: bool,  # 是否仅计算预测损失
            ignore_keys=None,  # 可选的忽略键
            **gen_kwargs,  # 其他生成参数
    # 返回一个包含可选浮点数和两个可选张量的元组
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    
        # 禁用梯度计算，以减少内存使用和提高速度
        with torch.no_grad():
            # 如果设置为使用生成进行预测，则提取输出 ID
            if self.args.predict_with_generate:
                output_ids = inputs.pop('output_ids', None)
            # 调用父类的预测步骤方法，计算损失和生成的标记
            loss, generated_tokens, labels = super().prediction_step(
                model=model,  # 传入模型
                inputs=inputs,  # 传入输入数据
                prediction_loss_only=prediction_loss_only,  # 是否仅计算损失
                ignore_keys=ignore_keys,  # 忽略的键
                **gen_kwargs  # 其他生成参数
            )
    
            # 如果生成的标记不为空，则裁剪标记以移除输入部分
            if generated_tokens is not None:
                generated_tokens = generated_tokens[:, inputs["input_ids"].size()[1]:]
    
            # 如果设置为使用生成进行预测，则将标签设置为输出 ID
            if self.args.predict_with_generate:
                labels = output_ids
    
            # 删除输入数据和输出 ID，以释放内存
            del inputs, output_ids
            # 清空 CUDA 缓存以释放显存
            torch.cuda.empty_cache()
    
        # 返回损失、生成的标记和标签
        return loss, generated_tokens, labels
# 使用 dataclass 装饰器定义数据配置类
@dc.dataclass
class DataConfig(object):
    # 训练文件的可选路径
    train_file: Optional[str] = None
    # 验证文件的可选路径
    val_file: Optional[str] = None
    # 测试文件的可选路径
    test_file: Optional[str] = None
    # 处理数据时使用的进程数量的可选值
    num_proc: Optional[int] = None

    # 定义一个只读属性，用于获取训练文件的后缀
    @property
    def data_format(self) -> str:
        # 返回训练文件的文件扩展名
        return Path(self.train_file).suffix

    # 定义一个只读属性，用于获取数据文件的字典
    @property
    def data_files(self) -> dict[NamedSplit, str]:
        # 生成包含数据集划分与对应文件路径的字典
        return {
            split: data_file
            for split, data_file in zip(
                # 列出数据集的划分类型
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                # 列出对应的文件路径
                [self.train_file, self.val_file, self.test_file],
            )
            # 仅包含文件路径不为 None 的条目
            if data_file is not None
        }


# 使用 dataclass 装饰器定义微调配置类
@dc.dataclass
class FinetuningConfig(object):
    # 数据配置的实例
    data_config: DataConfig

    # 最大输入长度
    max_input_length: int
    # 最大输出长度
    max_output_length: int
    # 是否合并数据的标志
    combine: bool
    # 是否冻结某些参数的标志
    freezeV: bool

    # 训练参数的实例，使用默认工厂函数初始化
    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir='./output')
    )
    # 可选的 Peft 配置
    peft_config: Optional[PeftConfig] = None

    # 类的初始化后处理函数
    def __post_init__(self):
        # 如果不进行评估或验证文件为空，则禁用评估
        if not self.training_args.do_eval or self.data_config.val_file is None:
            self.training_args.do_eval = False
            # 设置评估策略为不评估
            self.training_args.evaluation_strategy = 'no'
            # 将验证文件设置为 None
            self.data_config.val_file = None
        else:
            # 设置评估批次大小，如果未定义则使用训练批次大小
            self.training_args.per_device_eval_batch_size = (
                    self.training_args.per_device_eval_batch_size
                    or self.training_args.per_device_train_batch_size
            )

    # 从字典创建 FinetuningConfig 实例的类方法
    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        # 从字典中获取训练参数
        training_args = kwargs.get('training_args', None)
        # 如果训练参数存在且不是 Seq2SeqTrainingArguments 类型
        if training_args is not None and not isinstance(
                training_args, Seq2SeqTrainingArguments
        ):
            # 获取生成配置
            gen_config = training_args.get('generation_config')
            # 如果生成配置不是 GenerationConfig 类型，则进行转换
            if not isinstance(gen_config, GenerationConfig):
                training_args['generation_config'] = GenerationConfig(
                    **gen_config
                )
            # 将训练参数转换为 Seq2SeqTrainingArguments 实例
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        # 从字典中获取数据配置
        data_config = kwargs.get('data_config')
        # 如果数据配置不是 DataConfig 类型，则进行转换
        if not isinstance(data_config, DataConfig):
            kwargs['data_config'] = DataConfig(**data_config)

        # 从字典中获取 Peft 配置
        peft_config = kwargs.get('peft_config', None)
        # 如果 Peft 配置存在且不是 PeftConfig 类型，则进行转换
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs['peft_config'] = get_peft_config(config_dict=peft_config)
        # 创建 FinetuningConfig 实例并返回
        return cls(**kwargs)

    # 从文件创建 FinetuningConfig 实例的类方法
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        # 将路径转换为 Path 对象
        path = Path(path)
        # 创建 YAML 解析器，使用安全模式
        parser = yaml.YAML(typ='safe', pure=True)
        # 设置解析器的缩进格式
        parser.indent(mapping=2, offset=2, sequence=4)
        # 设置默认的流样式为非流式
        parser.default_flow_style = False
        # 解析 YAML 文件并加载内容
        kwargs = parser.load(path)
        # 从解析后的字典中创建 FinetuningConfig 实例并返回
        return cls.from_dict(**kwargs)


# 定义一个加载数据集的私有函数
def _load_datasets(
        # 数据目录路径
        data_dir: str,
        # 数据格式
        data_format: str,
        # 数据文件字典
        data_files: dict[NamedSplit, str],
        # 进程数量的可选值
        num_proc: Optional[int],
) -> DatasetDict:
    # 检查数据格式是否为 JSON Lines 格式
        if data_format == '.jsonl':
            # 加载数据集，指定数据目录和文件，未划分子集，使用指定进程数
            dataset_dct = load_dataset(
                data_dir,
                data_files=data_files,
                split=None,
                num_proc=num_proc,
            )
        else:
            # 如果数据格式不是支持的格式，则引发未实现错误
            raise NotImplementedError(f"Cannot load dataset in the '{data_format}' format.")
        # 返回加载的数据集字典
        return dataset_dct
# 数据管理器类，用于处理数据集相关操作
class DataManager(object):
    # 初始化方法，接收数据目录和数据配置作为参数
    def __init__(self, data_dir: str, data_config: DataConfig):
        # 从数据配置中获取进程数量
        self._num_proc = data_config.num_proc

        # 加载数据集，并存储为字典
        self._dataset_dct = _load_datasets(
            data_dir,  # 数据目录
            data_config.data_format,  # 数据格式
            data_config.data_files,  # 数据文件列表
            self._num_proc,  # 进程数量
        )

    # 获取指定划分的数据集，如果不存在则返回 None
    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        return self._dataset_dct.get(split, None)  # 从字典中获取数据集

    # 获取处理过的数据集，支持批处理和原始列删除
    def get_dataset(
            self,
            split: NamedSplit,  # 数据集划分
            process_fn: Callable[[dict[str, Any]], dict[str, Any]],  # 处理函数
            batched: bool = True,  # 是否批处理
            remove_orig_columns: bool = True,  # 是否移除原始列
    ) -> Optional[Dataset]:
        # 获取原始数据集
        orig_dataset = self._get_dataset(split)
        if orig_dataset is None:  # 如果数据集不存在
            return  # 返回 None
        if remove_orig_columns:  # 如果需要移除原始列
            remove_columns = orig_dataset.column_names  # 获取列名
        else:
            remove_columns = None  # 不移除列
        # 对原始数据集应用处理函数并返回结果
        return orig_dataset.map(
            process_fn,  # 处理函数
            batched=batched,  # 是否批处理
            remove_columns=remove_columns,  # 需要移除的列
            num_proc=self._num_proc,  # 进程数量
            # 默认的 orig_dataset.map 参数，可以调整为更小
            # https://github.com/THUDM/GLM-4/issues/277
            writer_batch_size=1000,  # 写入时的批处理大小
            batch_size=1000,  # 处理时的批处理大小
        )


# 处理批次数据的函数
def process_batch(
        batch: Mapping[str, Sequence],  # 输入批次数据
        tokenizer: PreTrainedTokenizer,  # 预训练的分词器
        max_input_length: int,  # 最大输入长度
        max_output_length: int,  # 最大输出长度
        combine: bool,  # 是否合并
) -> dict[str, list]:  # 返回处理后的字典
    # 获取批次中的消息
    batched_conv = batch['messages']
    # 初始化各类批处理列表
    batched_input_ids = []  # 输入 ID 列表
    batched_attention_mask = []  # 注意力掩码列表
    batched_position_ids = []  # 位置 ID 列表
    batched_labels = []  # 标签列表
    batched_images = []  # 图像列表

    # 计算最大长度
    max_length = max_input_length + max_output_length
    # 遍历每个批次的对话
        for conv in batched_conv:
            # 初始化输入 ID 列表
            input_ids = [151331, 151333]
            # 初始化注意力掩码列表
            attention_mask = [1, 1]
            # 创建位置 ID 列表
            position_ids = list(range(len(input_ids)))
            # 初始化损失掩码列表
            loss_masks = [False, False]
            # 初始化图像列表
            images = []
            
            # 检查对话的第一个元素是否有图像
            if conv[0].get('image'):
                # 打开图像并转换为 RGB 模式
                conv[0]['image'] = Image.open(conv[0]['image']).convert('RGB')
            else:
                # 如果没有图像，则使用默认图像
                conv[0]['image'] = img
    
            # 遍历对话中的每条消息
            for message in conv:
                # 设置损失掩码值，基于消息的角色判断
                loss_mask_val = False if message['role'] in ('system', 'user', 'observation') else True
                # 应用聊天模板，对消息进行标记化
                new_input_ids_all = tokenizer.apply_chat_template(
                    [message],
                    tokenize=True,
                    return_dict=True,
                    padding=True
                )
                # 提取新输入 ID，去掉特殊标记
                new_input_ids = new_input_ids_all['input_ids'][0][2:]
                # 提取新注意力掩码，去掉特殊标记
                new_attention_mask = new_input_ids_all['attention_mask'][0][2:]
                # 创建新的位置 ID 列表
                new_position_ids = list(range(position_ids[-1] + 1, position_ids[-1] + 1 + len(new_input_ids)))
                # 如果消息有图像，则添加到图像列表
                if message.get('image'):  # 仅处理一张图像
                    images.append(new_input_ids_all['images'])
    
                # 创建新的损失掩码
                new_loss_masks = [loss_mask_val] * len(new_input_ids)
                # 更新输入 ID 列表
                input_ids += new_input_ids
                # 更新注意力掩码列表
                attention_mask += new_attention_mask
                # 更新位置 ID 列表
                position_ids += new_position_ids
                # 更新损失掩码列表
                loss_masks += new_loss_masks
    
            # 添加结束标记到输入 ID
            input_ids.append(151336)  # EOS
            # 添加结束标记的注意力掩码
            attention_mask.append(1)
            # 更新位置 ID 列表以包含结束标记
            position_ids.append(len(position_ids))
            # 添加结束标记的损失掩码
            loss_masks.append(False)
    
            # 初始化标签列表
            labels = []
            # 遍历输入 ID 和损失掩码，生成标签
            for input_id, mask in zip(input_ids, loss_masks):
                if mask:
                    # 如果掩码为真，则将输入 ID 添加到标签
                    labels.append(input_id)
                else:
                    # 否则添加 -100 表示忽略
                    labels.append(-100)
    
            # 添加批处理输入 ID 到列表，限制长度
            batched_input_ids.append(input_ids[:max_length])
            # 添加批处理注意力掩码到列表，限制长度
            batched_attention_mask.append(attention_mask[:max_length])
            # 添加批处理位置 ID 到列表，限制长度
            batched_position_ids.append(position_ids[:max_length])
            # 添加批处理标签到列表，限制长度
            batched_labels.append(labels[:max_length])
            # 添加第一张图像到批处理图像列表
            batched_images.append(images[0][0])
    
        # 删除临时变量以释放内存
        del batched_conv, conv, input_ids, attention_mask, position_ids, loss_masks, message, new_input_ids, new_loss_masks, labels, input_id, mask
        # 清空 GPU 缓存以释放内存
        torch.cuda.empty_cache()
    
        # 返回结果字典，包含所有批处理数据
        return {
            'input_ids': batched_input_ids,
            'attention_mask': batched_attention_mask,
            'position_ids': batched_position_ids,
            'labels': batched_labels,
            'images': batched_images
        }
# 处理批量评估的函数，接受批量数据、分词器、输入输出长度等参数，返回处理结果字典
def process_batch_eval(
        batch: Mapping[str, Sequence],  # 批量输入，包含消息的映射
        tokenizer: PreTrainedTokenizer,  # 预训练的分词器
        max_input_length: int,  # 最大输入长度限制
        max_output_length: int,  # 最大输出长度限制
        combine: bool,  # 是否合并处理标志
) -> dict[str, list]:  # 返回字典，键为字符串，值为列表
    # 从批量数据中提取消息部分
    batched_conv = batch['messages']
    # 初始化各类存储列表
    batched_input_ids = []  # 存储输入 ID 列表
    batched_attention_mask = []  # 存储注意力掩码列表
    batched_position_ids = []  # 存储位置 ID 列表
    batched_output_ids = []  # 存储输出 ID 列表
    batched_images = []  # 存储图像列表

    # 遍历每个对话
    for conv in batched_conv:
        # 如果对话包含图像，则打开并转换为 RGB 格式
        if conv[0].get('image'):
            image = Image.open(conv[0]['image']).convert('RGB')
        else:
            # 如果没有图像，使用默认图像
            image = img   
        
        # 将图像存回对话数据中
        conv[0]['image'] = image
        # 应用聊天模板分词，并返回分词结果
        new_input_ids_all = tokenizer.apply_chat_template(
            conv,
            tokenize=True,  # 是否分词
            return_dict=True,  # 返回字典格式
            padding=True  # 是否进行填充
        )

        # 提取分词后的输入 ID
        input_ids = new_input_ids_all['input_ids'][0]
        # 提取注意力掩码
        attention_mask = new_input_ids_all['attention_mask'][0]
        # 生成位置 ID 列表
        position_ids = list(range(len(input_ids)))

        # 初始化对话部分列表
        dialogue_parts = [0]
        # 遍历输入 ID，寻找对话分隔符
        for idx, token_id in enumerate(input_ids):
            if token_id == 151337:  # 特定标识符表示对话分隔
                dialogue_parts.append(idx + 1)

        # 如果没有对话部分或最后一部分未结束，添加结束位置
        if not dialogue_parts or dialogue_parts[-1] != len(input_ids):
            dialogue_parts.append(len(input_ids))

            # 将对话拆分为多个对话段
        for end_idx in range(1, len(dialogue_parts)):
            # 获取当前对话段的输入
            input_segment = input_ids[:dialogue_parts[end_idx]]
            # 获取当前对话段的注意力掩码
            attention_segment = attention_mask[:dialogue_parts[end_idx]]
            # 获取当前对话段的位置 ID
            position_segment = position_ids[:dialogue_parts[end_idx]]
            # 获取当前对话段的输出，添加结束符
            output_segment = input_ids[dialogue_parts[end_idx - 1]:dialogue_parts[end_idx]]
            output_segment.append(151336)  # 添加结束标识符

            # 将处理结果添加到批量列表中
            batched_input_ids.append(input_segment[:max_input_length])  # 限制输入长度
            batched_attention_mask.append(attention_segment[:max_input_length])  # 限制注意力掩码长度
            batched_position_ids.append(position_segment[:max_input_length])  # 限制位置 ID 长度
            batched_output_ids.append(output_segment[:max_output_length])  # 限制输出长度
            batched_images.append(new_input_ids_all['images'][0])  # 添加图像

    # 清理不再使用的变量以释放内存
    del batched_conv, input_ids, attention_mask, position_ids, new_input_ids_all, output_segment
    # 清空 CUDA 缓存以释放 GPU 内存
    torch.cuda.empty_cache()

    # 返回处理后的结果字典
    return {
        'input_ids': batched_input_ids,  # 输入 ID 列表
        'attention_mask': batched_attention_mask,  # 注意力掩码列表
        'position_ids': batched_position_ids,  # 位置 ID 列表
        'output_ids': batched_output_ids,  # 输出 ID 列表
        'images': batched_images  # 图像列表
    }


# 加载分词器和模型的函数，接受模型目录和可选的 PEFT 配置
def load_tokenizer_and_model(
        model_dir: str,  # 模型目录
        peft_config: Optional[PeftConfig] = None,  # 可选的 PEFT 配置
):
    # 从预训练模型目录加载分词器，信任远程代码
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # 如果提供了 PEFT 配置，则加载模型
    if peft_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,  # 模型目录
            trust_remote_code=True,  # 信任远程代码
            empty_init=False,  # 不进行空初始化
            use_cache=False,  # 禁用缓存
            torch_dtype=torch.bfloat16  # 使用 BFloat 16 数据类型
        )
        # 应用 PEFT 模型配置
        model = get_peft_model(model, peft_config)
        # 打印可训练参数
        model.print_trainable_parameters()
    # 如果前面的条件不满足，执行以下代码
        else:
            # 从指定的模型目录加载预训练的因果语言模型，允许使用远程代码
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,                          # 模型目录路径
                trust_remote_code=True,            # 信任远程代码
                empty_init=False,                  # 不使用空初始化
                use_cache=False,                   # 不使用缓存
                torch_dtype=torch.bfloat16         # 使用 bfloat16 数据类型
            )
        # 返回分词器和加载的模型
        return tokenizer, model
# 定义一个计算评估指标的函数，接收评估预测和分词器
def compute_metrics(eval_preds: EvalPrediction, tokenizer):
    # 解包评估预测，获取预测ID和标签ID
    batched_pred_ids, batched_label_ids = eval_preds
    # 初始化一个字典来存储各种指标的分数
    metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
    # 遍历每一组预测ID和标签ID
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        # 使用分词器解码预测ID为文本，并去除首尾空白
        pred_txt = tokenizer.decode(pred_ids).strip()
        # 使用分词器解码标签ID为文本，并去除首尾空白
        label_txt = tokenizer.decode(label_ids).strip()
        # 对预测文本进行分词，生成token列表
        pred_tokens = list(jieba.cut(pred_txt))
        # 对标签文本进行分词，生成token列表
        label_tokens = list(jieba.cut(label_txt))
        # 创建Rouge评分对象
        rouge = Rouge()
        # 计算Rouge分数，得到各项评分
        scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
        # 遍历评分结果，保存F值到指标字典中
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v['f'] * 100, 4))
        # 计算Bleu-4分数并保存到指标字典中
        metrics_dct['bleu-4'].append(
            sentence_bleu([label_tokens], pred_tokens, smoothing_function=SmoothingFunction().method3))
    # 返回每个指标的平均值
    return {k: np.mean(v) for k, v in metrics_dct.items()}


# 定义主命令行函数，接收多个参数
@app.command()
def main(
        # 数据目录参数
        data_dir: Annotated[str, typer.Argument(help='')],
        # 模型目录参数，包含模型配置的路径或ID
        model_dir: Annotated[
            str,
            typer.Argument(
                help='A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file.'
            ),
        ],
        # 配置文件路径参数
        config_file: Annotated[str, typer.Argument(help='')],
        # 自动恢复检查点的参数，默认值为空字符串
        auto_resume_from_checkpoint: str = typer.Argument(
            default='',
            help='If entered as yes, automatically use the latest save checkpoint. If it is a numerical example 12 15, use the corresponding save checkpoint. If the input is no, restart training'
        ),
):
    # 从配置文件加载微调配置
    ft_config = FinetuningConfig.from_file(config_file)
    # 加载分词器和模型
    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)
    
    # 如果配置中冻结视觉参数，则不更新这些参数
    if ft_config.freezeV:
        for param in model.transformer.vision.parameters():
            param.requires_grad = False
    # 创建数据管理器，负责加载数据
    data_manager = DataManager(data_dir, ft_config.data_config)

    # 获取训练数据集，进行批处理
    train_dataset = data_manager.get_dataset(
        Split.TRAIN,
        functools.partial(
            process_batch,
            combine=ft_config.combine, # 目前未使用的组合参数
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    # 打印训练数据集的信息
    print('train_dataset:', train_dataset)

    # 获取验证数据集，进行批处理
    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            combine=ft_config.combine,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,

        ),
        batched=True,
    )

    # 如果验证数据集存在，则打印其信息
    if val_dataset is not None:
        print('val_dataset:', val_dataset)
    # 获取测试数据集，使用数据管理器，并对每个批次应用评估处理
        test_dataset = data_manager.get_dataset(
            Split.TEST,  # 指定数据集的拆分类型为测试集
            functools.partial(  # 使用偏函数来固定参数
                process_batch_eval,  # 处理每个批次的评估函数
                combine=ft_config.combine,  # 传入组合参数
                tokenizer=tokenizer,  # 传入分词器
                max_input_length=ft_config.max_input_length,  # 最大输入长度
                max_output_length=ft_config.max_output_length,  # 最大输出长度
            ),
            batched=True,  # 指定数据集为批处理模式
        )
        # 如果测试数据集不为空，则打印其内容
        if test_dataset is not None:
            print('test_dataset:', test_dataset)
    
        # 启用梯度检查点功能以节省内存
        model.gradient_checkpointing_enable()
        # 允许输入张量计算梯度
        model.enable_input_require_grads()
        
        # 设置生成配置中的填充标记ID
        ft_config.training_args.generation_config.pad_token_id = (
            151329  # 填充标记的ID
        )
        # 设置生成配置中的结束标记ID列表
        ft_config.training_args.generation_config.eos_token_id = [
            151329, 151336, 151338  # 结束标记的ID列表
        ]
    
        # 创建序列到序列训练器实例
        trainer = Seq2SeqTrainer(
            model=model,  # 指定使用的模型
            args=ft_config.training_args,  # 传入训练参数
            data_collator=DataCollatorForSeq2Seq(  # 数据整理器，用于处理输入数据
                tokenizer=tokenizer,  # 传入分词器
                padding='longest',  # 使用最长序列进行填充
                return_tensors='pt',  # 返回PyTorch张量
            ),
            train_dataset=train_dataset,  # 传入训练数据集
            eval_dataset=val_dataset,  # 传入评估数据集
            compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),  # 计算指标的偏函数
        )
    
        # 检查是否需要从检查点恢复训练
        if auto_resume_from_checkpoint.upper() == "" or auto_resume_from_checkpoint is None:
            trainer.train()  # 如果没有指定，直接开始训练
        else:
            output_dir = ft_config.training_args.output_dir  # 获取输出目录
            dirlist = os.listdir(output_dir)  # 列出输出目录中的文件
            checkpoint_sn = 0  # 初始化检查点序号
            # 遍历文件列表，查找有效的检查点
            for checkpoint_str in dirlist:
                if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find("tmp") == -1:  # 检查文件名
                    checkpoint = int(checkpoint_str.replace("checkpoint-", ""))  # 提取检查点序号
                    if checkpoint > checkpoint_sn:  # 更新最大检查点序号
                        checkpoint_sn = checkpoint
            # 如果指定了要恢复的检查点
            if auto_resume_from_checkpoint.upper() == "YES":
                if checkpoint_sn > 0:  # 确保存在有效检查点
                    model.gradient_checkpointing_enable()  # 启用梯度检查点
                    model.enable_input_require_grads()  # 允许计算梯度
                    checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))  # 构造检查点路径
                    print("resume checkpoint from checkpoint-" + str(checkpoint_sn))  # 打印恢复信息
                    trainer.train(resume_from_checkpoint=checkpoint_directory)  # 从指定检查点恢复训练
                else:
                    trainer.train()  # 没有有效检查点，直接训练
            else:
                # 如果指定的恢复参数是数字
                if auto_resume_from_checkpoint.isdigit():
                    if int(auto_resume_from_checkpoint) > 0:  # 检查指定序号有效性
                        checkpoint_sn = int(auto_resume_from_checkpoint)  # 更新检查点序号
                        model.gradient_checkpointing_enable()  # 启用梯度检查点
                        model.enable_input_require_grads()  # 允许计算梯度
                        checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))  # 构造检查点路径
                        print("resume checkpoint from checkpoint-" + str(checkpoint_sn))  # 打印恢复信息
                        trainer.train(resume_from_checkpoint=checkpoint_directory)  # 从指定检查点恢复训练
                else:
                    # 如果指定的检查点无效，打印错误信息
                    print(auto_resume_from_checkpoint,
                          "The specified checkpoint sn(" + auto_resume_from_checkpoint + ") has not been saved. Please search for the correct checkpoint in the model output directory")
    # 检查测试数据集是否不为空
        if test_dataset is not None:
            # 如果测试数据集存在，则进行预测
            trainer.predict(test_dataset)
# 如果当前脚本是主程序入口
if __name__ == '__main__':
    # 调用应用程序函数
    app()
```