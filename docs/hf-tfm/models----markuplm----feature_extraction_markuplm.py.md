# `.\models\markuplm\feature_extraction_markuplm.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""
Feature extractor class for MarkupLM.
"""

import html

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...utils import is_bs4_available, logging, requires_backends

# 检查是否安装了 BeautifulSoup 库，若安装则导入
if is_bs4_available():
    import bs4
    from bs4 import BeautifulSoup

# 获取模块的日志记录器
logger = logging.get_logger(__name__)

class MarkupLMFeatureExtractor(FeatureExtractionMixin):
    r"""
    Constructs a MarkupLM feature extractor. This can be used to get a list of nodes and corresponding xpaths from HTML
    strings.

    This feature extractor inherits from [`~feature_extraction_utils.PreTrainedFeatureExtractor`] which contains most
    of the main methods. Users should refer to this superclass for more information regarding those methods.

    """

    def __init__(self, **kwargs):
        # 要求后端依赖检查，确保 BeautifulSoup 库存在
        requires_backends(self, ["bs4"])
        super().__init__(**kwargs)

    def xpath_soup(self, element):
        # 初始化 xpath 标签和下标列表
        xpath_tags = []
        xpath_subscripts = []
        child = element if element.name else element.parent
        # 遍历元素的所有父级节点，获取路径信息
        for parent in child.parents:  # type: bs4.element.Tag
            siblings = parent.find_all(child.name, recursive=False)
            xpath_tags.append(child.name)
            # 计算当前节点在其兄弟节点中的位置索引
            xpath_subscripts.append(
                0 if 1 == len(siblings) else next(i for i, s in enumerate(siblings, 1) if s is child)
            )
            child = parent
        # 由于路径是从叶子节点到根节点的顺序，需要反转列表顺序以得到正确的 XPath 路径
        xpath_tags.reverse()
        xpath_subscripts.reverse()
        return xpath_tags, xpath_subscripts
    # 定义一个方法，从给定的 HTML 字符串中提取文本和相关信息
    def get_three_from_single(self, html_string):
        # 使用 BeautifulSoup 解析 HTML 字符串
        html_code = BeautifulSoup(html_string, "html.parser")

        # 初始化存储所有文本的列表和两个空的序列列表
        all_doc_strings = []  # 存储所有文本内容的列表
        string2xtag_seq = []  # 存储每个文本的 XPath 标签序列
        string2xsubs_seq = []  # 存储每个文本的 XPath 下标序列

        # 遍历 HTML 文档的所有节点
        for element in html_code.descendants:
            # 如果当前节点是可导航字符串类型
            if isinstance(element, bs4.element.NavigableString):
                # 如果当前节点的父节点不是标签节点，则跳过
                if type(element.parent) != bs4.element.Tag:
                    continue

                # 解码并去除文本内容的空白字符和转义序列
                text_in_this_tag = html.unescape(element).strip()
                # 如果处理后的文本为空，则跳过
                if not text_in_this_tag:
                    continue

                # 将处理后的文本添加到文本列表中
                all_doc_strings.append(text_in_this_tag)

                # 调用 xpath_soup 方法获取当前节点的 XPath 标签和下标序列
                xpath_tags, xpath_subscripts = self.xpath_soup(element)
                # 将 XPath 标签序列和下标序列添加到对应的列表中
                string2xtag_seq.append(xpath_tags)
                string2xsubs_seq.append(xpath_subscripts)

        # 检查文本列表和 XPath 序列列表的长度是否一致，若不一致则抛出 ValueError
        if len(all_doc_strings) != len(string2xtag_seq):
            raise ValueError("Number of doc strings and xtags does not correspond")
        if len(all_doc_strings) != len(string2xsubs_seq):
            raise ValueError("Number of doc strings and xsubs does not correspond")

        # 返回三个列表：所有文本内容、每个文本的 XPath 标签序列、每个文本的 XPath 下标序列
        return all_doc_strings, string2xtag_seq, string2xsubs_seq

    # 定义一个方法，根据给定的 XPath 标签序列和下标序列构造 XPath 表达式
    def construct_xpath(self, xpath_tags, xpath_subscripts):
        # 初始化空的 XPath 字符串
        xpath = ""
        # 遍历 XPath 标签序列和下标序列，构造 XPath 表达式
        for tagname, subs in zip(xpath_tags, xpath_subscripts):
            xpath += f"/{tagname}"  # 添加标签名到 XPath 中
            if subs != 0:
                xpath += f"[{subs}]"  # 如果下标不为 0，则添加下标到 XPath 中
        # 返回构造好的 XPath 表达式
        return xpath
    def __call__(self, html_strings) -> BatchFeature:
        """
        Main method to prepare for the model one or several HTML strings.

        Args:
            html_strings (`str`, `List[str]`):
                The HTML string or batch of HTML strings from which to extract nodes and corresponding xpaths.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **nodes** -- Nodes.
            - **xpaths** -- Corresponding xpaths.

        Examples:

        ```
        >>> from transformers import MarkupLMFeatureExtractor

        >>> page_name_1 = "page1.html"
        >>> page_name_2 = "page2.html"
        >>> page_name_3 = "page3.html"

        >>> with open(page_name_1) as f:
        ...     single_html_string = f.read()

        >>> feature_extractor = MarkupLMFeatureExtractor()

        >>> # single example
        >>> encoding = feature_extractor(single_html_string)
        >>> print(encoding.keys())
        >>> # dict_keys(['nodes', 'xpaths'])

        >>> # batched example

        >>> multi_html_strings = []

        >>> with open(page_name_2) as f:
        ...     multi_html_strings.append(f.read())
        >>> with open(page_name_3) as f:
        ...     multi_html_strings.append(f.read())

        >>> encoding = feature_extractor(multi_html_strings)
        >>> print(encoding.keys())
        >>> # dict_keys(['nodes', 'xpaths'])
        ```"""

        # Input type checking for clearer error
        # 检查输入类型以提供更清晰的错误信息
        valid_strings = False

        # Check that strings has a valid type
        # 检查字符串的类型是否有效
        if isinstance(html_strings, str):
            valid_strings = True
        elif isinstance(html_strings, (list, tuple)):
            if len(html_strings) == 0 or isinstance(html_strings[0], str):
                valid_strings = True

        if not valid_strings:
            raise ValueError(
                "HTML strings must of type `str`, `List[str]` (batch of examples), "
                f"but is of type {type(html_strings)}."
            )

        is_batched = bool(isinstance(html_strings, (list, tuple)) and (isinstance(html_strings[0], str)))

        if not is_batched:
            html_strings = [html_strings]

        # Get nodes + xpaths
        # 获取节点和对应的 XPath
        nodes = []
        xpaths = []
        for html_string in html_strings:
            # Extract nodes and related sequences
            # 提取节点和相关的序列
            all_doc_strings, string2xtag_seq, string2xsubs_seq = self.get_three_from_single(html_string)
            nodes.append(all_doc_strings)
            xpath_strings = []
            for node, tag_list, sub_list in zip(all_doc_strings, string2xtag_seq, string2xsubs_seq):
                # Construct XPath string from tag and sub lists
                # 从标签列表和子列表构建 XPath 字符串
                xpath_string = self.construct_xpath(tag_list, sub_list)
                xpath_strings.append(xpath_string)
            xpaths.append(xpath_strings)

        # return as Dict
        # 作为字典返回
        data = {"nodes": nodes, "xpaths": xpaths}
        encoded_inputs = BatchFeature(data=data, tensor_type=None)

        return encoded_inputs
```