# `.\chatglm4-finetune\finetune_demo\finetune.py`

```
# -*- coding: utf-8 -*-  # 指定文件编码为 UTF-8
import os  # 导入操作系统相关的模块
import jieba  # 导入中文分词库
import dataclasses as dc  # 导入数据类模块
import functools  # 导入用于高阶函数的工具
from collections.abc import Callable, Mapping, Sequence  # 导入集合相关的抽象基类
from pathlib import Path  # 导入处理路径的模块
from typing import Annotated, Any, Union  # 导入类型注解
import numpy as np  # 导入 NumPy 库
import ruamel.yaml as yaml  # 导入 YAML 处理库
import torch  # 导入 PyTorch 库
import typer  # 导入命令行界面库
from datasets import Dataset, Split  # 从 datasets 导入数据集相关类
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # 导入 BLEU 分数计算工具
from peft import PeftConfig, get_peft_config, get_peft_model  # 导入 PEFT 配置及模型获取函数
from rouge_chinese import Rouge  # 导入中文 ROUGE 评测工具
from torch import nn  # 导入 PyTorch 的神经网络模块
from transformers import (  # 导入变换器相关类和函数
    AutoModelForCausalLM,  # 自动加载因果语言模型
    AutoTokenizer,  # 自动加载分词器
    EvalPrediction,  # 导入评估预测结果的类
    GenerationConfig,  # 导入生成配置
    PreTrainedTokenizer,  # 导入预训练分词器
    Seq2SeqTrainingArguments,  # 导入序列到序列训练参数类
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq  # 导入序列到序列的数据整理器并重命名
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer  # 导入序列到序列训练器并重命名
from datasets import load_dataset, DatasetDict, NamedSplit  # 导入数据集加载和字典类
from typing import Optional  # 导入可选类型注解

app = typer.Typer(pretty_exceptions_show_locals=False)  # 创建命令行应用，禁用本地异常显示


class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):  # 定义数据整理器类
    def __call__(self, features, return_tensors=None):  # 重载调用方法，接受特征和返回张量选项
        output_ids = ([feature['output_ids'] for feature in features] if 'output_ids' in features[0].keys() else None)  # 获取输出 ID
        if output_ids is not None:  # 如果存在输出 ID
            max_output_length = max(len(out) for out in output_ids)  # 计算最大输出长度
            if self.pad_to_multiple_of is not None:  # 如果需要填充到特定倍数
                max_output_length = (  # 计算新的最大输出长度
                        (
                                max_output_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
            for feature in features:  # 遍历特征
                remainder = [self.tokenizer.pad_token_id] * (  # 计算填充所需的剩余部分
                        max_output_length - len(feature['output_ids'])
                )
                if isinstance(feature['output_ids'], list):  # 如果输出 ID 是列表
                    feature['output_ids'] = feature['output_ids'] + remainder  # 追加填充
                else:  # 否则
                    feature['output_ids'] = np.concatenate(  # 将输出 ID 和填充合并
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)  # 转换为整型数组
        return super().__call__(features, return_tensors)  # 调用父类方法返回结果


class Seq2SeqTrainer(_Seq2SeqTrainer):  # 定义序列到序列训练器类
    # Not Support for apex  # 不支持 apex

    def training_step(self, model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:  # 定义训练步骤方法
        model.train()  # 设置模型为训练模式
        inputs = self._prepare_inputs(inputs)  # 准备输入数据

        with self.compute_loss_context_manager():  # 使用计算损失的上下文管理器
            loss = self.compute_loss(model, inputs)  # 计算损失

        if self.args.n_gpu > 1:  # 如果使用多个 GPU
            loss = loss.mean()  # 对损失进行平均
        self.accelerator.backward(loss)  # 反向传播损失
        detached_loss = loss.detach() / self.args.gradient_accumulation_steps  # 分离损失并归一化
        del inputs  # 删除输入以释放内存
        torch.cuda.empty_cache()  # 清空 CUDA 缓存
        return detached_loss  # 返回处理后的损失

    def prediction_step(  # 定义预测步骤方法
            self,
            model: nn.Module,  # 模型
            inputs: dict[str, Any],  # 输入数据
            prediction_loss_only: bool,  # 是否仅返回预测损失
            ignore_keys=None,  # 要忽略的键
            **gen_kwargs,  # 其他生成参数
    # 定义函数返回值类型为包含可选浮点数和两个可选的 Torch 张量的元组
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    
        # 禁用梯度计算，以节省内存和提高速度
        with torch.no_grad():  # Ensure no gradient computation
            # 如果设置为生成预测，则从输入中移除输出 ID
            if self.args.predict_with_generate:
                output_ids = inputs.pop('output_ids')
            # 从输入中获取输入 ID
            input_ids = inputs['input_ids']
    
            # 调用父类的方法执行预测步骤，获取损失、生成的标记和标签
            loss, generated_tokens, labels = super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
            )
    
            # 截取生成的标记，去掉输入 ID 部分
            generated_tokens = generated_tokens[:, input_ids.size()[1]:]
            # 将标签设置为输出 ID
            labels = output_ids
    
            # 删除输入、输入 ID 和输出 ID，以释放内存
            del inputs, input_ids, output_ids
            # 清空 CUDA 缓存，避免内存溢出
            torch.cuda.empty_cache()
    
        # 返回损失、生成的标记和标签
        return loss, generated_tokens, labels
# 使用数据类装饰器定义一个数据配置类
@dc.dataclass
class DataConfig(object):
    # 训练数据文件路径，可选，默认为 None
    train_file: Optional[str] = None
    # 验证数据文件路径，可选，默认为 None
    val_file: Optional[str] = None
    # 测试数据文件路径，可选，默认为 None
    test_file: Optional[str] = None
    # 处理数据的进程数量，可选，默认为 None
    num_proc: Optional[int] = None

    # 定义一个属性，用于获取训练文件的格式后缀
    @property
    def data_format(self) -> str:
        # 返回训练文件路径的后缀
        return Path(self.train_file).suffix

    # 定义一个属性，用于获取文件路径的字典
    @property
    def data_files(self) -> dict[NamedSplit, str]:
        # 返回一个字典，包含各个数据分割及其对应的文件路径
        return {
            split: data_file
            # zip 函数将分割类型与文件路径配对
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            )
            # 仅包含非 None 的文件路径
            if data_file is not None
        }


# 使用数据类装饰器定义一个微调配置类
@dc.dataclass
class FinetuningConfig(object):
    # 关联的数据配置
    data_config: DataConfig

    # 最大输入长度
    max_input_length: int
    # 最大输出长度
    max_output_length: int
    # 是否合并数据
    combine: bool
    # 是否冻结 V
    freezeV: bool

    # 定义训练参数，使用默认工厂函数生成对象
    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir='./output')
    )
    # 可选的 PEFT 配置
    peft_config: Optional[PeftConfig] = None

    # 后初始化方法，调整训练参数
    def __post_init__(self):
        # 如果不进行评估或验证文件为 None
        if not self.training_args.do_eval or self.data_config.val_file is None:
            # 设置不进行评估
            self.training_args.do_eval = False
            # 评估策略设置为 'no'
            self.training_args.evaluation_strategy = 'no'
            # 清空验证文件路径
            self.data_config.val_file = None
        else:
            # 设置评估批次大小
            self.training_args.per_device_eval_batch_size = (
                    self.training_args.per_device_eval_batch_size
                    or self.training_args.per_device_train_batch_size
            )

    # 从字典创建类的类方法
    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        # 获取训练参数
        training_args = kwargs.get('training_args', None)
        # 如果训练参数存在且不是 Seq2SeqTrainingArguments 类型
        if training_args is not None and not isinstance(
                training_args, Seq2SeqTrainingArguments
        ):
            # 获取生成配置
            gen_config = training_args.get('generation_config')
            # 如果生成配置不是 GenerationConfig 类型
            if not isinstance(gen_config, GenerationConfig):
                # 创建生成配置并赋值
                training_args['generation_config'] = GenerationConfig(
                    **gen_config
                )
            # 更新训练参数为 Seq2SeqTrainingArguments 类型
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        # 获取数据配置
        data_config = kwargs.get('data_config')
        # 如果数据配置不是 DataConfig 类型
        if not isinstance(data_config, DataConfig):
            # 更新为 DataConfig 类型
            kwargs['data_config'] = DataConfig(**data_config)

        # 获取 PEFT 配置
        peft_config = kwargs.get('peft_config', None)
        # 如果 PEFT 配置存在且不是 PeftConfig 类型
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            # 获取 PEFT 配置并赋值
            kwargs['peft_config'] = get_peft_config(config_dict=peft_config)
        # 返回新的类实例
        return cls(**kwargs)

    # 从文件创建类的类方法
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        # 将路径转换为 Path 对象
        path = Path(path)
        # 创建 YAML 解析器
        parser = yaml.YAML(typ='safe', pure=True)
        # 设置缩进格式
        parser.indent(mapping=2, offset=2, sequence=4)
        # 设置默认的流样式为 False
        parser.default_flow_style = False
        # 从文件加载内容
        kwargs = parser.load(path)
        # 从字典创建类实例并返回
        return cls.from_dict(**kwargs)


# 定义一个加载数据集的函数
def _load_datasets(
        # 数据目录
        data_dir: str,
        # 数据格式
        data_format: str,
        # 数据文件字典
        data_files: dict[NamedSplit, str],
        # 进程数量
        num_proc: Optional[int],
) -> DatasetDict:
    # 检查数据格式是否为 '.jsonl'
    if data_format == '.jsonl':
        # 加载数据集，指定数据目录、数据文件、拆分方式和并行处理进程数
        dataset_dct = load_dataset(
            data_dir,  # 数据存储目录
            data_files=data_files,  # 要加载的数据文件列表
            split=None,  # 不指定拆分，加载全部数据
            num_proc=num_proc,  # 指定并行处理的进程数
        )
    else:
        # 如果数据格式不被支持，抛出未实现错误并提示格式
        raise NotImplementedError(f"Cannot load dataset in the '{data_format}' format.")
    # 返回加载的数据集字典
    return dataset_dct
# 数据管理类，用于管理数据集
class DataManager(object):
    # 初始化方法，接受数据目录和数据配置对象
    def __init__(self, data_dir: str, data_config: DataConfig):
        # 从数据配置中获取并存储处理进程的数量
        self._num_proc = data_config.num_proc

        # 加载数据集并存储为字典，键为数据集分割，值为数据集
        self._dataset_dct = _load_datasets(
            data_dir,
            data_config.data_format,
            data_config.data_files,
            self._num_proc,
        )

    # 根据分割获取对应的数据集
    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        # 从数据集字典中获取指定分割的数据集，若不存在则返回 None
        return self._dataset_dct.get(split, None)

    # 获取指定分割的数据集，并处理数据
    def get_dataset(
            self,
            split: NamedSplit,
            process_fn: Callable[[dict[str, Any]], dict[str, Any]],
            batched: bool = True,
            remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        # 获取原始数据集
        orig_dataset = self._get_dataset(split)
        # 若原始数据集不存在，则返回 None
        if orig_dataset is None:
            return

        # 根据标志决定是否移除原始列
        if remove_orig_columns:
            remove_columns = orig_dataset.column_names
        else:
            remove_columns = None
        # 调用 map 方法处理数据集并返回结果
        return orig_dataset.map(
            process_fn,
            batched=batched,
            remove_columns=remove_columns,
            num_proc=self._num_proc,
        )


# 处理消息函数
def process_message(message):
    # 如果消息中包含工具且角色为系统，则处理工具参数
    if 'tools' in message and message['role'] == 'system':
        for tool in message['tools']:
            # 获取工具的参数属性
            parameters = tool['function']['parameters']['properties']
            # 过滤掉参数值为 None 的属性
            tool['function']['parameters']['properties'] = \
                {k: v for k, v in parameters.items() if
                 v is not None}
    # 如果消息中包含工具，但角色不是系统，则删除工具
    elif 'tools' in message:
        del message['tools']
    # 返回处理后的消息
    return message


# 处理批次消息的函数
def process_batch(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
        combine: bool,
) -> dict[str, list]:
    # 从批次中提取消息
    batched_conv = batch['messages']
    # 初始化存储输入 ID 的列表
    batched_input_ids = []
    # 初始化存储标签的列表
    batched_labels = []
    # 遍历分批的对话
        for conv in batched_conv:
            # 初始化输入 ID 列表
            input_ids = [151331, 151333]
            # 初始化损失掩码列表
            loss_masks = [False, False]
            # 如果需要合并对话
            if combine:
                # 应用聊天模板将对话转换为新的输入 ID
                new_input_ids = tokenizer.apply_chat_template(conv, tokenize=True, return_dict=False)
                # 更新输入 ID 列表
                input_ids = new_input_ids
                # 创建新的损失掩码，所有元素初始为 False
                loss_masks = [False] * len(input_ids)
                # 找到最后一个助手的索引
                last_assistant_index = len(input_ids) - input_ids[::-1].index(151337) - 1
                # 为最后助手之后的输入设置掩码为 True
                for j in range(last_assistant_index + 1, len(input_ids)):
                    loss_masks[j] = True
            else:
                # 如果不合并，则处理每条消息
                for message in conv:
                    # 处理消息，提取有效信息
                    message = process_message(message)
                    # 确定损失掩码的值，根据角色决定
                    loss_mask_val = False if message['role'] in ('system', 'user', 'observation') else True
                    # 应用聊天模板并更新输入 ID 列表，跳过前两个元素
                    new_input_ids = tokenizer.apply_chat_template([message], tokenize=True, return_dict=False)[2:]
                    # 将新的输入 ID 添加到输入 ID 列表
                    input_ids += new_input_ids
                    # 根据新的输入 ID 更新损失掩码
                    loss_masks += [loss_mask_val] * len(new_input_ids)
    
            # 在输入 ID 列表末尾添加结束符
            input_ids.append(151336)  # EOS for chat
            # 在损失掩码列表前添加一个 False
            loss_masks = [False, *loss_masks]
            # 初始化标签列表
            labels = []
            # 根据输入 ID 和损失掩码生成标签
            for input_id, mask in zip(input_ids, loss_masks):
                if mask:
                    labels.append(input_id)  # 如果掩码为 True，添加输入 ID
                else:
                    labels.append(-100)  # 否则添加 -100 作为无效标签
            # 计算最大长度
            max_length = max_input_length + max_output_length + 1
            # 将处理后的输入 ID 和标签添加到批次列表中，限制长度
            batched_input_ids.append(input_ids[:max_length])
            batched_labels.append(labels[:max_length])
    
        # 删除不再使用的变量以释放内存
        del batched_conv, conv, input_ids, loss_masks, new_input_ids, labels
        # 清空 CUDA 缓存以释放显存
        torch.cuda.empty_cache()
    
        # 返回输入 ID 和标签的字典
        return {'input_ids': batched_input_ids, 'labels': batched_labels}
# 处理批量评估的函数，返回输入和输出 ID 的字典
def process_batch_eval(
        batch: Mapping[str, Sequence],  # 输入批次，包含消息的映射
        tokenizer: PreTrainedTokenizer,  # 预训练的分词器
        max_input_length: int,  # 输入的最大长度
        max_output_length: int,  # 输出的最大长度
        combine: bool,  # 是否组合消息
) -> dict[str, list]:  # 返回类型为包含输入和输出 ID 的字典
    # 从批次中提取对话消息
    batched_conv = batch['messages']
    # 存储处理后的输入 ID 的列表
    batched_input_ids = []
    # 存储处理后的输出 ID 的列表
    batched_output_ids = []

    # 遍历每个对话
    for conv in batched_conv:
        if combine:  # 如果选择组合模式
            # 应用聊天模板对对话进行编码
            new_input_ids = tokenizer.apply_chat_template(conv, tokenize=True, return_dict=False)
            # 将新的输入 ID 赋值给输入 ID
            input_ids = new_input_ids
            # 获取最后一个助手消息的索引
            last_assistant_index = len(input_ids) - input_ids[::-1].index(151337) - 1
            # 分割输出提示和输出 ID
            output_prompt, output_ids = (
                input_ids[:1],  # 取第一个输入 ID 作为输出提示
                input_ids[last_assistant_index:],  # 取从助手消息开始的输出 ID
            )
            output_ids.append(151336)  # 添加结束符
            # 将处理后的输入 ID 添加到列表中，限制长度
            batched_input_ids.append(
                input_ids[:max_input_length] + output_prompt[:1]
            )
            # 将处理后的输出 ID 添加到列表中，限制长度
            batched_output_ids.append(output_ids[:max_output_length])
        else:  # 如果选择不组合模式
            input_ids = [151331, 151333]  # 初始化输入 ID
            # 遍历对话中的每个消息
            for message in conv:
                if len(input_ids) >= max_input_length:  # 如果输入长度超过最大限制
                    break  # 跳出循环
                else:
                    # 处理当前消息
                    message = process_message(message)
                    # 应用聊天模板对消息进行编码
                    new_input_ids = tokenizer.apply_chat_template([message], tokenize=True, return_dict=False)[2:]
                    if message['role'] == 'assistant':  # 如果消息来自助手
                        output_prompt, output_ids = (
                            new_input_ids[:1],  # 取第一个新的输入 ID 作为输出提示
                            new_input_ids[1:],  # 取剩余的输入 ID 作为输出 ID
                        )
                        output_ids.append(151336)  # 添加结束符
                        # 将处理后的输入 ID 添加到列表中，限制长度
                        batched_input_ids.append(
                            input_ids[:max_input_length] + output_prompt[:1]
                        )
                        # 将处理后的输出 ID 添加到列表中，限制长度
                        batched_output_ids.append(output_ids[:max_output_length])
                    # 更新输入 ID
                    input_ids += new_input_ids

    # 删除不再需要的变量以释放内存
    del batched_conv, conv, input_ids, new_input_ids, output_prompt, output_ids
    # 清空 GPU 缓存
    torch.cuda.empty_cache()

    # 返回包含输入和输出 ID 的字典
    return {'input_ids': batched_input_ids, 'output_ids': batched_output_ids}


# 加载分词器和模型的函数
def load_tokenizer_and_model(
        model_dir: str,  # 模型目录
        peft_config: Optional[PeftConfig] = None,  # 可选的配置
):
    # 从指定目录加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if peft_config is not None:  # 如果提供了 PEFT 配置
        # 从指定目录加载因果语言模型
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            empty_init=False,
            use_cache=False,
            torch_dtype=torch.bfloat16  # 必须使用 BFloat 16
        )
        # 应用 PEFT 模型配置
        model = get_peft_model(model, peft_config)
        # 打印可训练参数
        model.print_trainable_parameters()
    else:  # 如果没有 PEFT 配置
        # 从指定目录加载因果语言模型
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            empty_init=False,
            use_cache=False,
            torch_dtype=torch.bfloat16
        )
    # 返回分词器和模型
    return tokenizer, model


# 计算指标的函数
def compute_metrics(eval_preds: EvalPrediction, tokenizer):  
    # 解包评估预测和标签 ID
    batched_pred_ids, batched_label_ids = eval_preds
    # 初始化一个字典，用于存储不同评估指标的分数
        metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
        # 遍历批次中的预测 ID 和标签 ID
        for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
            # 将预测 ID 解码为文本，并去除首尾空格
            pred_txt = tokenizer.decode(pred_ids).strip()
            # 将标签 ID 解码为文本，并去除首尾空格
            label_txt = tokenizer.decode(label_ids).strip()
            # 使用结巴分词对预测文本进行分词
            pred_tokens = list(jieba.cut(pred_txt))
            # 使用结巴分词对标签文本进行分词
            label_tokens = list(jieba.cut(label_txt))
            # 创建 Rouge 评分的实例
            rouge = Rouge()
            # 计算 Rouge 分数，获取得分字典
            scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
            # 遍历第一个得分字典的键值对
            for k, v in scores[0].items():
                # 将 F 值乘以 100 后四舍五入，存储到对应的指标列表中
                metrics_dct[k].append(round(v['f'] * 100, 4))
            # 计算 BLEU-4 分数，并存储到字典中
            metrics_dct['bleu-4'].append(
                sentence_bleu([label_tokens], pred_tokens, smoothing_function=SmoothingFunction().method3))
        # 返回每个指标的平均分数字典
        return {k: np.mean(v) for k, v in metrics_dct.items()}
# 定义命令行工具的主入口函数
@app.command()
def main(
        # 指定数据目录，帮助信息为空
        data_dir: Annotated[str, typer.Argument(help='')],
        # 指定模型目录或模型配置文件路径，并提供帮助信息
        model_dir: Annotated[
            str,
            typer.Argument(
                help='A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file.'
            ),
        ],
        # 指定配置文件路径，帮助信息为空
        config_file: Annotated[str, typer.Argument(help='')],
        # 自动恢复训练的检查点选项，默认值为空字符串
        auto_resume_from_checkpoint: str = typer.Argument(
            default='',
            help='If entered as yes, automatically use the latest save checkpoint. If it is a numerical example 12 15, use the corresponding save checkpoint. If the input is no, restart training'
        ),
):
    # 从配置文件加载微调配置
    ft_config = FinetuningConfig.from_file(config_file)
    # 加载分词器和模型，传入微调配置
    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)
    # 创建数据管理对象，传入数据目录和数据配置
    data_manager = DataManager(data_dir, ft_config.data_config)

    # 获取训练数据集，处理批次数据
    train_dataset = data_manager.get_dataset(
        Split.TRAIN,
        functools.partial(
            process_batch,
            tokenizer=tokenizer,
            combine=ft_config.combine,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    # 打印训练数据集
    print('train_dataset:', train_dataset)
    # 获取验证数据集，处理批次数据
    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            combine=ft_config.combine,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    # 如果验证数据集不为空，则打印
    if val_dataset is not None:
        print('val_dataset:', val_dataset)
    # 获取测试数据集，处理批次数据
    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            combine=ft_config.combine,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    # 如果测试数据集不为空，则打印
    if test_dataset is not None:
        print('test_dataset:', test_dataset)

    # 启用模型的梯度检查点
    model.gradient_checkpointing_enable()
    # 启用模型输入的梯度计算
    model.enable_input_require_grads()
    
    # 设置生成配置的填充标记ID
    ft_config.training_args.generation_config.pad_token_id = (
        151329
    )
    # 设置生成配置的结束标记ID
    ft_config.training_args.generation_config.eos_token_id = [
        151329, 151336, 151338
    ]

    # 初始化序列到序列训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        # 设置数据整理器
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # 设置计算指标的函数
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )

    # 如果未选择自动恢复检查点，则开始训练
    if auto_resume_from_checkpoint.upper() == "" or auto_resume_from_checkpoint is None:
        trainer.train()
    else:  # 如果不是首次训练，则执行以下逻辑
        output_dir = ft_config.training_args.output_dir  # 获取输出目录路径
        dirlist = os.listdir(output_dir)  # 列出输出目录下的所有文件和文件夹
        checkpoint_sn = 0  # 初始化检查点序号为 0
        for checkpoint_str in dirlist:  # 遍历输出目录中的每个项
            if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find("tmp") == -1:  # 检查项是否包含 "eckpoint" 且不包含 "tmp"
                checkpoint = int(checkpoint_str.replace("checkpoint-", ""))  # 提取数字部分作为检查点编号
                if checkpoint > checkpoint_sn:  # 如果当前检查点编号大于已记录的最大值
                    checkpoint_sn = checkpoint  # 更新最大检查点编号
        if auto_resume_from_checkpoint.upper() == "YES":  # 如果设置为自动从检查点恢复
            if checkpoint_sn > 0:  # 如果找到有效的检查点编号
                model.gradient_checkpointing_enable()  # 启用模型的梯度检查点功能
                model.enable_input_require_grads()  # 启用输入的梯度计算
                checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))  # 构建检查点目录的完整路径
                print("resume checkpoint from checkpoint-" + str(checkpoint_sn))  # 输出正在恢复的检查点信息
                trainer.train(resume_from_checkpoint=checkpoint_directory)  # 从指定检查点恢复训练
            else:  # 如果没有找到有效的检查点
                trainer.train()  # 开始新的训练
        else:  # 如果不自动恢复检查点
            if auto_resume_from_checkpoint.isdigit():  # 如果指定的恢复检查点是数字
                if int(auto_resume_from_checkpoint) > 0:  # 检查点编号大于 0
                    checkpoint_sn = int(auto_resume_from_checkpoint)  # 设置检查点编号
                    model.gradient_checkpointing_enable()  # 启用模型的梯度检查点功能
                    model.enable_input_require_grads()  # 启用输入的梯度计算
                    checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))  # 构建检查点目录的完整路径
                    print("resume checkpoint from checkpoint-" + str(checkpoint_sn))  # 输出正在恢复的检查点信息
                    trainer.train(resume_from_checkpoint=checkpoint_directory)  # 从指定检查点恢复训练
            else:  # 如果指定的恢复检查点不是有效数字
                print(auto_resume_from_checkpoint,  # 输出自动恢复检查点的信息
                      "The specified checkpoint sn(" + auto_resume_from_checkpoint + ") has not been saved. Please search for the correct checkpoint in the model output directory")  # 提示用户指定的检查点不存在

    if test_dataset is not None:  # 如果测试数据集不为空
        trainer.predict(test_dataset)  # 使用训练器对测试数据集进行预测
# 检查当前模块是否是主程序
if __name__ == '__main__':
    # 调用应用程序的主函数
    app()
```