# `.\pipelines\table_question_answering.py`

```
# 导入必要的库和模块

import collections  # 导入 collections 模块，用于高效操作集合数据类型
import types  # 导入 types 模块，用于操作 Python 中的类型信息

import numpy as np  # 导入 numpy 库，并简写为 np，用于科学计算

# 导入相对路径下的模块和函数
from ..utils import (
    add_end_docstrings,  # 导入 add_end_docstrings 函数，用于处理文档字符串的附加信息
    is_tf_available,  # 导入 is_tf_available 函数，用于检查 TensorFlow 是否可用
    is_torch_available,  # 导入 is_torch_available 函数，用于检查 PyTorch 是否可用
    requires_backends,  # 导入 requires_backends 装饰器，用于声明依赖的后端库
)

# 导入本地模块
from .base import (
    ArgumentHandler,  # 从 base 模块中导入 ArgumentHandler 类，处理参数相关逻辑
    Dataset,  # 从 base 模块中导入 Dataset 类，处理数据集相关逻辑
    Pipeline,  # 从 base 模块中导入 Pipeline 类，处理数据处理流程相关逻辑
    PipelineException,  # 从 base 模块中导入 PipelineException 类，处理数据处理流程中的异常
    build_pipeline_init_args,  # 从 base 模块中导入 build_pipeline_init_args 函数，用于构建流程初始化参数
)

# 如果 Torch 可用，则导入相关模块和函数
if is_torch_available():
    import torch  # 导入 torch 库，用于深度学习模型构建和训练

    from ..models.auto.modeling_auto import (
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,  # 导入模型名称映射字典，用于序列到序列生成模型
        MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES,  # 导入模型名称映射字典，用于表格问答模型
    )

# 如果 TensorFlow 可用，则导入相关模块和函数
if is_tf_available():
    import tensorflow as tf  # 导入 tensorflow 库，用于构建和训练深度学习模型

    from ..models.auto.modeling_tf_auto import (
        TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,  # 导入 TensorFlow 模型名称映射字典，用于序列到序列生成模型
        TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES,  # 导入 TensorFlow 模型名称映射字典，用于表格问答模型
    )


class TableQuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    处理 TableQuestionAnsweringPipeline 的参数
    """
    def __call__(self, table=None, query=None, **kwargs):
        # 定义一个特殊方法，用于调用对象实例，接收参数 table 和 query
        # 返回 tqa_pipeline_inputs 的结构如下：
        # [
        #   {"table": pd.DataFrame, "query": List[str]},
        #   ...,
        #   {"table": pd.DataFrame, "query" : List[str]}
        # ]
        
        # 要求导入 pandas 库
        requires_backends(self, "pandas")
        import pandas as pd

        # 如果 table 参数为 None，则抛出数值错误异常
        if table is None:
            raise ValueError("Keyword argument `table` cannot be None.")
        
        # 如果 query 参数为 None
        elif query is None:
            # 如果 table 是一个字典，且包含 "query" 和 "table" 键
            if isinstance(table, dict) and table.get("query") is not None and table.get("table") is not None:
                # 创建包含单个元素的 tqa_pipeline_inputs 列表
                tqa_pipeline_inputs = [table]
            
            # 如果 table 是一个列表且长度大于 0
            elif isinstance(table, list) and len(table) > 0:
                # 如果 table 中的所有元素都是字典
                if not all(isinstance(d, dict) for d in table):
                    raise ValueError(
                        f"Keyword argument `table` should be a list of dict, but is {(type(d) for d in table)}"
                    )

                # 如果第一个字典包含 "query" 和 "table" 键
                if table[0].get("query") is not None and table[0].get("table") is not None:
                    # 使用整个 table 列表作为 tqa_pipeline_inputs
                    tqa_pipeline_inputs = table
                else:
                    raise ValueError(
                        "If keyword argument `table` is a list of dictionaries, each dictionary should have a `table`"
                        f" and `query` key, but only dictionary has keys {table[0].keys()} `table` and `query` keys."
                    )
            
            # 如果 Dataset 不为 None，且 table 是 Dataset 类型或生成器类型
            elif Dataset is not None and isinstance(table, Dataset) or isinstance(table, types.GeneratorType):
                return table
            
            # 其他情况抛出数值错误异常
            else:
                raise ValueError(
                    "Invalid input. Keyword argument `table` should be either of type `dict` or `list`, but "
                    f"is {type(table)})"
                )
        
        # 如果 query 参数不为 None
        else:
            # 创建包含单个元素的 tqa_pipeline_inputs 列表，其中包含传入的 table 和 query
            tqa_pipeline_inputs = [{"table": table, "query": query}]

        # 对于 tqa_pipeline_inputs 中的每个元素
        for tqa_pipeline_input in tqa_pipeline_inputs:
            # 如果 table 不是 pd.DataFrame 类型
            if not isinstance(tqa_pipeline_input["table"], pd.DataFrame):
                # 如果 table 为 None，则抛出数值错误异常
                if tqa_pipeline_input["table"] is None:
                    raise ValueError("Table cannot be None.")

                # 将非 DataFrame 类型的 table 转换为 pd.DataFrame 类型
                tqa_pipeline_input["table"] = pd.DataFrame(tqa_pipeline_input["table"])

        # 返回处理后的 tqa_pipeline_inputs 列表
        return tqa_pipeline_inputs
# 使用装饰器为 TableQuestionAnsweringPipeline 添加文档字符串，文档内容由 build_pipeline_init_args(has_tokenizer=True) 函数生成
@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
# 定义 TableQuestionAnsweringPipeline 类，继承自 Pipeline 类
class TableQuestionAnsweringPipeline(Pipeline):
    """
    Table Question Answering pipeline using a `ModelForTableQuestionAnswering`. This pipeline is only available in
    PyTorch.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> oracle = pipeline(model="google/tapas-base-finetuned-wtq")
    >>> table = {
    ...     "Repository": ["Transformers", "Datasets", "Tokenizers"],
    ...     "Stars": ["36542", "4512", "3934"],
    ...     "Contributors": ["651", "77", "34"],
    ...     "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
    ... }
    >>> oracle(query="How many stars does the transformers repository have?", table=table)
    {'answer': 'AVERAGE > 36542', 'coordinates': [(0, 1)], 'cells': ['36542'], 'aggregator': 'AVERAGE'}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This tabular question answering pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"table-question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a tabular question answering task.
    See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=table-question-answering).
    """

    # 默认输入参数名设定为 "table,query"
    default_input_names = "table,query"

    # 初始化方法，接受一个参数解析器 args_parser，默认为 TableQuestionAnsweringArgumentHandler 的实例
    def __init__(self, args_parser=TableQuestionAnsweringArgumentHandler(), *args, **kwargs):
        # 调用父类 Pipeline 的初始化方法
        super().__init__(*args, **kwargs)
        # 将参数解析器存储在 _args_parser 属性中
        self._args_parser = args_parser

        # 根据框架类型选择模型映射，更新映射表
        if self.framework == "tf":
            mapping = TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES.copy()
            mapping.update(TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES)
        else:
            mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES.copy()
            mapping.update(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES)
        # 检查模型类型是否符合要求
        self.check_model_type(mapping)

        # 根据模型配置检查是否支持聚合操作，并存储相关信息
        self.aggregate = bool(getattr(self.model.config, "aggregation_labels", None)) and bool(
            getattr(self.model.config, "num_aggregation_labels", None)
        )
        self.type = "tapas" if hasattr(self.model.config, "aggregation_labels") else None

    # 批量推断方法，接受任意输入并调用模型进行推断
    def batch_inference(self, **inputs):
        return self.model(**inputs)

    # 内部方法，用于清理和规范化输入参数，返回预处理和前向传递所需的参数字典
    def _sanitize_parameters(self, sequential=None, padding=None, truncation=None, **kwargs):
        preprocess_params = {}
        if padding is not None:
            preprocess_params["padding"] = padding
        if truncation is not None:
            preprocess_params["truncation"] = truncation

        forward_params = {}
        if sequential is not None:
            forward_params["sequential"] = sequential

        # 返回预处理参数、前向传递参数和空字典
        return preprocess_params, forward_params, {}
    # 预处理函数，接受输入并进行预处理，返回模型所需的输入格式
    def preprocess(self, pipeline_input, sequential=None, padding=True, truncation=None):
        # 如果未指定截断方式，则根据模型类型设置默认截断方式
        if truncation is None:
            if self.type == "tapas":
                truncation = "drop_rows_to_fit"
            else:
                truncation = "do_not_truncate"

        # 从pipeline_input中获取数据表和查询语句
        table, query = pipeline_input["table"], pipeline_input["query"]
        # 如果数据表为空，则抛出数值错误
        if table.empty:
            raise ValueError("table is empty")
        # 如果查询为空或空字符串，则抛出数值错误
        if query is None or query == "":
            raise ValueError("query is empty")
        
        # 使用tokenizer对数据表和查询进行标记化处理，返回模型所需的输入格式
        inputs = self.tokenizer(table, query, return_tensors=self.framework, truncation=truncation, padding=padding)
        inputs["table"] = table  # 将数据表添加到模型输入中
        return inputs  # 返回预处理后的输入数据格式

    # 私有方法，模型前向传播函数，接受模型输入并返回模型输出
    def _forward(self, model_inputs, sequential=False, **generate_kwargs):
        # 从模型输入中弹出数据表
        table = model_inputs.pop("table")

        # 根据模型类型选择不同的推断方式
        if self.type == "tapas":
            if sequential:
                outputs = self.sequential_inference(**model_inputs)  # 顺序推断方式
            else:
                outputs = self.batch_inference(**model_inputs)  # 批量推断方式
        else:
            outputs = self.model.generate(**model_inputs, **generate_kwargs)  # 使用模型生成方法进行推断
        
        # 组装模型输出，包括模型输入、数据表和模型输出结果
        model_outputs = {"model_inputs": model_inputs, "table": table, "outputs": outputs}
        return model_outputs  # 返回模型输出结果
    # 定义一个方法用于后处理模型输出
    def postprocess(self, model_outputs):
        # 从模型输出中获取输入数据、表格和输出结果
        inputs = model_outputs["model_inputs"]
        table = model_outputs["table"]
        outputs = model_outputs["outputs"]
        
        # 如果模型类型是 "tapas"
        if self.type == "tapas":
            # 如果需要聚合结果
            if self.aggregate:
                # 从输出中获取 logits 和 logits_agg
                logits, logits_agg = outputs[:2]
                # 使用 tokenizer 将 logits 和 logits_agg 转换为预测结果
                predictions = self.tokenizer.convert_logits_to_predictions(inputs, logits, logits_agg)
                # 分别获取答案坐标批次和聚合器预测结果
                answer_coordinates_batch, agg_predictions = predictions
                # 根据预测结果和配置获取聚合器标签
                aggregators = {i: self.model.config.aggregation_labels[pred] for i, pred in enumerate(agg_predictions)}

                # 获取没有聚合标签的索引
                no_agg_label_index = self.model.config.no_aggregation_label_index
                # 创建聚合器前缀字典，仅包含非空聚合结果的索引
                aggregators_prefix = {
                    i: aggregators[i] + " > " for i, pred in enumerate(agg_predictions) if pred != no_agg_label_index
                }
            else:
                # 如果不需要聚合结果，从输出中获取 logits
                logits = outputs[0]
                # 使用 tokenizer 将 logits 转换为预测结果
                predictions = self.tokenizer.convert_logits_to_predictions(inputs, logits)
                # 获取答案坐标批次
                answer_coordinates_batch = predictions[0]
                # 初始化空的聚合器和聚合器前缀字典
                aggregators = {}
                aggregators_prefix = {}

            # 初始化空的答案列表
            answers = []
            # 遍历答案坐标批次中的索引和坐标
            for index, coordinates in enumerate(answer_coordinates_batch):
                # 根据坐标从表格中获取单元格内容
                cells = [table.iat[coordinate] for coordinate in coordinates]
                # 获取当前索引对应的聚合器和聚合器前缀
                aggregator = aggregators.get(index, "")
                aggregator_prefix = aggregators_prefix.get(index, "")
                # 创建答案对象，包括单元格内容、坐标和可能的聚合器信息
                answer = {
                    "answer": aggregator_prefix + ", ".join(cells),
                    "coordinates": coordinates,
                    "cells": [table.iat[coordinate] for coordinate in coordinates],
                }
                # 如果存在聚合器，则将其添加到答案对象中
                if aggregator:
                    answer["aggregator"] = aggregator

                # 将答案对象添加到答案列表中
                answers.append(answer)

            # 如果答案列表为空，则抛出流水线异常
            if len(answers) == 0:
                raise PipelineException("Empty answer")
        else:
            # 如果模型类型不是 "tapas"，则直接将输出结果解码为答案列表
            answers = [{"answer": answer} for answer in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]

        # 返回答案列表，如果列表长度大于 1 则返回所有答案，否则返回第一个答案
        return answers if len(answers) > 1 else answers[0]
```