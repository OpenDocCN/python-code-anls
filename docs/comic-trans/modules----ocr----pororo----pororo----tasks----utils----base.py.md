# `.\comic-translate\modules\ocr\pororo\pororo\tasks\utils\base.py`

```py
import re
import unicodedata
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Mapping, Optional, Union

@dataclass
class TaskConfig:
    task: str
    lang: str
    n_model: str

class PororoTaskBase:
    r"""Task base class that implements basic functions for prediction"""

    def __init__(self, config: TaskConfig):
        self.config = config

    @property
    def n_model(self):
        return self.config.n_model

    @property
    def lang(self):
        return self.config.lang

    @abstractmethod
    def predict(
        self,
        text: Union[str, List[str]],
        **kwargs,
    ):
        raise NotImplementedError("`predict()` function is not implemented properly!")

    def __call__(self):
        raise NotImplementedError("`call()` function is not implemented properly!")

    def __repr__(self):
        return f"[TASK]: {self.config.task.upper()}\n[LANG]: {self.config.lang.upper()}\n[MODEL]: {self.config.n_model}"

    def _normalize(self, text: str):
        """Unicode normalization and whitespace removal (often needed for contexts)"""
        # 对文本进行Unicode规范化和去除空白字符的操作
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

class PororoFactoryBase(object):
    r"""This is a factory base class that construct task-specific module"""

    def __init__(
        self,
        task: str,
        lang: str,
        model: Optional[str] = None,
    ):
        # 获取可用的语言列表和模型列表
        self._available_langs = self.get_available_langs()
        self._available_models = self.get_available_models()

        # 构建模型到语言的映射关系
        self._model2lang = {
            v: k for k, vs in self._available_models.items() for v in vs
        }

        # 设置默认语言为支持的第一个语言
        assert (lang in self._available_langs), f"Following langs are supported for this task: {self._available_langs}"

        if lang is None:
            lang = self._available_langs[0]

        # 如果用户定义了模型，则根据模型选择语言
        if model is not None:
            lang = self._model2lang[model]

        # 设置默认模型
        if model is None:
            model = self.get_default_model(lang)

        # 确保选择的模型在所选语言的支持范围内
        # yapf: disable
        assert (model in self._available_models[lang]), f"{model} is NOT supported for {lang}"
        # yapf: enable

        # 创建任务配置对象
        self.config = TaskConfig(task, lang, model)

    @abstractmethod
    def get_available_langs(self) -> List[str]:
        raise NotImplementedError("`get_available_langs()` is not implemented properly!")

    @abstractmethod
    def get_available_models(self) -> Mapping[str, List[str]]:
        raise NotImplementedError("`get_available_models()` is not implemented properly!")

    @abstractmethod
    def get_default_model(self, lang: str) -> str:
        return self._available_models[lang][0]

    @classmethod
    # 定义一个类方法 `load`，用于加载 PororoTaskBase 的子类
    def load(cls) -> PororoTaskBase:
        # 抛出 NotImplementedError 异常，提示模型加载函数未正确实现
        raise NotImplementedError(
            "Model load function is not implemented properly!")
# PororoSimpleBase 类，继承自 PororoTaskBase，提供了一个简单任务的基础包装类
class PororoSimpleBase(PororoTaskBase):
    r"""Simple task base wrapper class"""

    # 重写了 __call__ 方法，允许对象实例像函数一样被调用，传入文本参数 text 和关键字参数 kwargs
    def __call__(self, text: str, **kwargs):
        # 调用 predict 方法进行预测，返回预测结果
        return self.predict(text, **kwargs)


# PororoBiencoderBase 类，继承自 PororoTaskBase，提供了一个双编码器任务的基础包装类
class PororoBiencoderBase(PororoTaskBase):
    r"""Bi-Encoder base wrapper class"""

    # 重写了 __call__ 方法，允许对象实例像函数一样被调用，传入句子 sent_a 和 sent_b，以及关键字参数 kwargs
    def __call__(
        self,
        sent_a: str,
        sent_b: Union[str, List[str]],
        **kwargs,
    ):
        # 检查 sent_a 必须为字符串类型
        assert isinstance(sent_a, str), "sent_a should be string type"
        # 检查 sent_b 必须为字符串或字符串列表类型
        assert isinstance(sent_b, str) or isinstance(
            sent_b, list), "sent_b should be string or list of string type"

        # 对 sent_a 进行规范化处理
        sent_a = self._normalize(sent_a)

        # 对于 "找相似句子" 任务
        if isinstance(sent_b, list):
            # 如果 sent_b 是列表，则对列表中的每个元素进行规范化处理
            sent_b = [self._normalize(t) for t in sent_b]
        else:
            # 否则，对 sent_b 进行规范化处理
            sent_b = self._normalize(sent_b)

        # 调用 predict 方法进行预测，传入处理后的 sent_a 和 sent_b，以及其他关键字参数 kwargs，返回预测结果
        return self.predict(sent_a, sent_b, **kwargs)


# PororoGenerationBase 类，继承自 PororoTaskBase，提供了一个生成任务的基础包装类，使用各种生成技巧
class PororoGenerationBase(PororoTaskBase):
    r"""Generation task wrapper class using various generation tricks"""

    # 重写了 __call__ 方法，允许对象实例像函数一样被调用，传入文本参数 text 和多个生成参数
    def __call__(
        self,
        text: str,
        beam: int = 5,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = -1,
        no_repeat_ngram_size: int = 4,
        len_penalty: float = 1.0,
        **kwargs,
    ):
        # 检查 text 必须为字符串类型
        assert isinstance(text, str), "Input text should be string type"

        # 调用 predict 方法进行预测，传入文本 text 和生成参数，返回生成的结果
        return self.predict(
            text,
            beam=beam,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            len_penalty=len_penalty,
            **kwargs,
        )


# PororoTaskGenerationBase 类，继承自 PororoTaskBase，提供了一个仅使用 beam search 的生成任务基础包装类
class PororoTaskGenerationBase(PororoTaskBase):
    r"""Generation task wrapper class using only beam search"""

    # 重写了 __call__ 方法，允许对象实例像函数一样被调用，传入文本参数 text 和生成参数 beam，以及其他关键字参数 kwargs
    def __call__(self, text: str, beam: int = 1, **kwargs):
        # 检查 text 必须为字符串类型
        assert isinstance(text, str), "Input text should be string type"

        # 对文本 text 进行规范化处理
        text = self._normalize(text)

        # 调用 predict 方法进行预测，传入处理后的文本 text 和生成参数 beam，以及其他关键字参数 kwargs，返回生成的结果
        return self.predict(text, beam=beam, **kwargs)
```