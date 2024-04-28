# `.\transformers\pipelines\table_question_answering.py`

```
# 导入必要的库
import collections
import types

import numpy as np

# 导入自定义工具函数
from ..utils import (
    add_end_docstrings,
    is_tensorflow_probability_available,
    is_tf_available,
    is_torch_available,
    requires_backends,
)
# 导入基类和常量
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, Dataset, Pipeline, PipelineException

# 如果是使用 PyTorch，导入 PyTorch 库和相关模型
if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import (
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES,
    )

# 如果同时使用 TensorFlow 和 TensorFlow Probability，导入相应库和模型
if is_tf_available() and is_tensorflow_probability_available():
    import tensorflow as tf
    import tensorflow_probability as tfp

    from ..models.auto.modeling_tf_auto import (
        TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
        TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES,
    )

# 表格问答管道参数处理器，继承自参数处理器基类
class TableQuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    Handles arguments for the TableQuestionAnsweringPipeline
    """
    # 定义一个方法，可以接受参数table和query以及其他关键字参数
    def __call__(self, table=None, query=None, **kwargs):
        # 返回一个包含"table"和"query"的tqa_pipeline_inputs的列表形式
        # [
        #   {"table": pd.DataFrame, "query": List[str]},
        #   ...,
        #   {"table": pd.DataFrame, "query" : List[str]}
        # ]
        
        # 检查是否存在pandas后端
        requires_backends(self, "pandas")
        # 导入pandas库
        import pandas as pd

        if table is None:
            raise ValueError("Keyword argument `table` cannot be None.")
        elif query is None:
            # 若table是一个字典，包含"query"和"table"两个键
            if isinstance(table, dict) and table.get("query") is not None and table.get("table") is not None:
                tqa_pipeline_inputs = [table]
            # 若table是一个列表，且包含多个字典
            elif isinstance(table, list) and len(table) > 0:
                # 检查列表中所有元素是否为字典类型
                if not all(isinstance(d, dict) for d in table):
                    raise ValueError(
                        f"Keyword argument `table` should be a list of dict, but is {(type(d) for d in table)}"
                    )

                # 如果第一个字典中包含"query"和"table"键
                if table[0].get("query") is not None and table[0].get("table") is not None:
                    tqa_pipeline_inputs = table
                else:
                    raise ValueError(
                        "If keyword argument `table` is a list of dictionaries, each dictionary should have a `table`"
                        f" and `query` key, but only dictionary has keys {table[0].keys()} `table` and `query` keys."
                    )
            # 如果table是Dataset类型或者GeneratorType类型
            elif Dataset is not None and isinstance(table, Dataset) or isinstance(table, types.GeneratorType):
                return table
            else:
                raise ValueError(
                    "Invalid input. Keyword argument `table` should be either of type `dict` or `list`, but "
                    f"is {type(table)})"
                )
        else:
            # 如果有table和query参数，则转换成字典形式
            tqa_pipeline_inputs = [{"table": table, "query": query}]

        # 对每个tqa_pipeline_input的字典元素中的"table"值进行检查，如果不是pd.DataFrame类型，则转换成pd.DataFrame
        for tqa_pipeline_input in tqa_pipeline_inputs:
            if not isinstance(tqa_pipeline_input["table"], pd.DataFrame):
                if tqa_pipeline_input["table"] is None:
                    raise ValueError("Table cannot be None.")

                tqa_pipeline_input["table"] = pd.DataFrame(tqa_pipeline_input["table"])

        # 返回tqa_pipeline_inputs
        return tqa_pipeline_inputs
# 使用装饰器为 TableQuestionAnsweringPipeline 添加尾部文档字符串，文档字符串包含 PIPELINE_INIT_ARGS 的信息
@add_end_docstrings(PIPELINE_INIT_ARGS)
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

    # 定义默认输入名称
    default_input_names = "table,query"

    # 初始化函数，接受 args_parser 参数，默认为 TableQuestionAnsweringArgumentHandler 实例
    def __init__(self, args_parser=TableQuestionAnsweringArgumentHandler(), *args, **kwargs):
        # 调用父类 Pipeline 的初始化方法
        super().__init__(*args, **kwargs)
        # 将传入的 args_parser 赋值给实例属性 _args_parser
        self._args_parser = args_parser

        # 如果框架是 TensorFlow
        if self.framework == "tf":
            # 复制 TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES 到 mapping
            mapping = TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES.copy()
            # 将 TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES 添加到 mapping 中
            mapping.update(TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES)
        else:
            # 复制 MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES 到 mapping
            mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES.copy()
            # 将 MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES 添加到 mapping 中
            mapping.update(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES)
        # 检查模型类型
        self.check_model_type(mapping)

        # 检查是否存在聚合标签和聚合标签数量
        self.aggregate = bool(getattr(self.model.config, "aggregation_labels", None)) and bool(
            getattr(self.model.config, "num_aggregation_labels", None)
        )
        # 如果模型配置中存在聚合标签，则类型为 "tapas"
        self.type = "tapas" if hasattr(self.model.config, "aggregation_labels") else None

    # 批量推理方法
    def batch_inference(self, **inputs):
        # 调用模型进行推理
        return self.model(**inputs)

    # 清理参数方法
    def _sanitize_parameters(self, sequential=None, padding=None, truncation=None, **kwargs):
        # 预处理参数字典
        preprocess_params = {}
        # 如果 padding 不为空，则设置预处理参数中的 padding
        if padding is not None:
            preprocess_params["padding"] = padding
        # 如果 truncation 不为空，则设置预处理参数中的 truncation
        if truncation is not None:
            preprocess_params["truncation"] = truncation

        # 前向传递参数字典
        forward_params = {}
        # 如果 sequential 不为空，则设置前向传递参数中的 sequential
        if sequential is not None:
            forward_params["sequential"] = sequential
        # 返回预处理参数、前向传递参数和空字典
        return preprocess_params, forward_params, {}
    # 对输入数据进行预处理，包括序列化、填充和截断
    def preprocess(self, pipeline_input, sequential=None, padding=True, truncation=None):
        # 如果没有指定截断方式，则根据模型类型进行设定
        if truncation is None:
            if self.type == "tapas":
                truncation = "drop_rows_to_fit"
            else:
                truncation = "do_not_truncate"

        # 从pipeline_input中获取table和query
        table, query = pipeline_input["table"], pipeline_input["query"]
        # 如果table为空则抛出数值错误
        if table.empty:
            raise ValueError("table is empty")
        # 如果query为空或者为None则抛出数值错误
        if query is None or query == "":
            raise ValueError("query is empty")
        # 使用tokenizer对table和query进行编码，并根据参数进行tensor返回、截断和填充
        inputs = self.tokenizer(table, query, return_tensors=self.framework, truncation=truncation, padding=padding)
        # 将table添加到inputs中
        inputs["table"] = table
        # 返回处理后的输入数据
        return inputs

    # 模型的前向传播
    def _forward(self, model_inputs, sequential=False):
        # 将table从model_inputs中弹出
        table = model_inputs.pop("table")

        # 如果模型类型为tapas
        if self.type == "tapas":
            # 如果sequential为True，则使用sequential_inference进行推理，否则使用batch_inference
            if sequential:
                outputs = self.sequential_inference(**model_inputs)
            else:
                outputs = self.batch_inference(**model_inputs)
        else:
            # 使用model.generate进行生成
            outputs = self.model.generate(**model_inputs)
        # 构建模型输出结构
        model_outputs = {"model_inputs": model_inputs, "table": table, "outputs": outputs}
        # 返回模型输出
        return model_outputs
    # 对模型输出进行后处理，提取模型输入、表格、输出
    def postprocess(self, model_outputs):
        # 获取模型输入、表格、输出
        inputs = model_outputs["model_inputs"]
        table = model_outputs["table"]
        outputs = model_outputs["outputs"]
        # 如果模型类型是 "tapas"
        if self.type == "tapas":
            # 如果需要聚合
            if self.aggregate:
                # 获取 logits 和 logits_agg
                logits, logits_agg = outputs[:2]
                # 根据模型预测的 logits 和 logits_agg 转换成预测结果
                predictions = self.tokenizer.convert_logits_to_predictions(inputs, logits, logits_agg)
                answer_coordinates_batch, agg_predictions = predictions
                # 根据聚合预测结果找到对应的聚合器
                aggregators = {i: self.model.config.aggregation_labels[pred] for i, pred in enumerate(agg_predictions)}

                # 获取没有聚合的 label 索引
                no_agg_label_index = self.model.config.no_aggregation_label_index
                # 根据聚合结果找到对应的聚合器前缀
                aggregators_prefix = {
                    i: aggregators[i] + " > " for i, pred in enumerate(agg_predictions) if pred != no_agg_label_index
                }
            else:
                # 如果不需要聚合，只获取 logits
                logits = outputs[0]
                # 根据模型预测的 logits 转换成预测结果
                predictions = self.tokenizer.convert_logits_to_predictions(inputs, logits)
                answer_coordinates_batch = predictions[0]
                # 初始化聚合器和聚合器前缀
                aggregators = {}
                aggregators_prefix = {}
            # 初始化答案列表
            answers = []
            # 遍历每一个答案的坐标
            for index, coordinates in enumerate(answer_coordinates_batch):
                # 根据坐标从表格中获取单元格内容
                cells = [table.iat[coordinate] for coordinate in coordinates]
                # 获取对应的聚合器和聚合器前缀
                aggregator = aggregators.get(index, "")
                aggregator_prefix = aggregators_prefix.get(index, "")
                # 构建答案对象
                answer = {
                    "answer": aggregator_prefix + ", ".join(cells),
                    "coordinates": coordinates,
                    "cells": [table.iat[coordinate] for coordinate in coordinates],
                }
                # 如果存在聚合器，则添加到答案对象中
                if aggregator:
                    answer["aggregator"] = aggregator

                answers.append(answer)
            # 如果答案列表为空，抛出异常
            if len(answer) == 0:
                raise PipelineException("Empty answer")
        else:
            # 如果模型类型不是 "tapas"，直接将输出转换成答案
            answers = [{"answer": answer} for answer in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]

        # 返回答案列表
        return answers if len(answers) > 1 else answers[0]
```