# `.\transformers\models\markuplm\feature_extraction_markuplm.py`

```py
# 设置文件编码为 UTF-8

# 版权声明，使用 Apache 许可证版本 2.0
# 详情请参阅：http://www.apache.org/licenses/LICENSE-2.0
"""
MarkupLM 的特征提取器类。
"""

# 导入必要的模块
import html

# 导入特征提取相关的模块和工具函数
from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
# 导入判断是否可用 BeautifulSoup 模块的函数和工具函数
from ...utils import is_bs4_available, logging, requires_backends

# 如果 BeautifulSoup 可用，则导入它
if is_bs4_available():
    import bs4
    from bs4 import BeautifulSoup

# 获取日志记录器
logger = logging.get_logger(__name__)


# 定义 MarkupLMFeatureExtractor 类，继承自 FeatureExtractionMixin 类
class MarkupLMFeatureExtractor(FeatureExtractionMixin):
    r"""
    构建一个 MarkupLM 特征提取器。用于从 HTML 字符串中获取节点列表和对应的 XPath。

    此特征提取器继承自 [`~feature_extraction_utils.PreTrainedFeatureExtractor`]，其中包含大多数主要方法。
    用户应参考此超类以获取有关这些方法的更多信息。

    """

    # 初始化函数
    def __init__(self, **kwargs):
        # 检查依赖是否满足，需要 BeautifulSoup 模块
        requires_backends(self, ["bs4"])
        # 调用父类的初始化函数
        super().__init__(**kwargs)

    # 定义一个函数，用于生成节点的 XPath
    def xpath_soup(self, element):
        # 初始化 XPath 的标签列表和下标列表
        xpath_tags = []
        xpath_subscripts = []
        # 获取子节点或当前节点
        child = element if element.name else element.parent
        # 遍历子节点或当前节点的所有父节点
        for parent in child.parents:  # type: bs4.element.Tag
            # 获取当前节点的同级节点
            siblings = parent.find_all(child.name, recursive=False)
            # 将当前节点的标签名添加到 XPath 标签列表中
            xpath_tags.append(child.name)
            # 计算当前节点在同级节点中的下标
            xpath_subscripts.append(
                # 如果同级节点数量为 1，则下标为 0；否则找到当前节点在同级节点中的索引位置
                0 if 1 == len(siblings) else next(i for i, s in enumerate(siblings, 1) if s is child)
            )
            # 将父节点设为当前节点，继续向上遍历
            child = parent
        # 翻转 XPath 标签列表和下标列表
        xpath_tags.reverse()
        xpath_subscripts.reverse()
        # 返回 XPath 标签列表和下标列表
        return xpath_tags, xpath_subscripts
```  
    # 从给定的 HTML 字符串中获取三个不同的信息
    def get_three_from_single(self, html_string):
        # 使用 BeautifulSoup 解析 HTML 字符串
        html_code = BeautifulSoup(html_string, "html.parser")

        # 用于存储所有文档字符串的列表
        all_doc_strings = []
        # 用于存储每个文档字符串对应的xpath标签序列的列表
        string2xtag_seq = []
        # 用于存储每个文档字符串对应的xpath子标签序列的列表
        string2xsubs_seq = []

        # 遍历HTML代码的所有子孙节点
        for element in html_code.descendants:
            # 如果该节点是可导航字符串
            if isinstance(element, bs4.element.NavigableString):
                # 如果该节点的父节点不是HTML标签
                if type(element.parent) != bs4.element.Tag:
                    continue

                # 对该节点的文本进行解码和去除首尾空格处理
                text_in_this_tag = html.unescape(element).strip()
                # 如果文本为空，则继续下一次循环
                if not text_in_this_tag:
                    continue

                # 将文本添加到文档字符串列表中
                all_doc_strings.append(text_in_this_tag)

                # 调用xpath_soup函数获取该节点对应的xpath标签序列和子标签序列
                xpath_tags, xpath_subscripts = self.xpath_soup(element)
                # 将xpath标签序列添加到对应列表中
                string2xtag_seq.append(xpath_tags)
                # 将子标签序列添加到对应列表中
                string2xsubs_seq.append(xpath_subscripts)

        # 如果文档字符串的数量和xpath标签的数量不相等，则抛出异常
        if len(all_doc_strings) != len(string2xtag_seq):
            raise ValueError("Number of doc strings and xtags does not correspond")
        # 如果文档字符串的数量和子标签的数量不相等，则抛出异常
        if len(all_doc_strings) != len(string2xsubs_seq):
            raise ValueError("Number of doc strings and xsubs does not correspond")

        # 返回所有文档字符串、xpath标签序列和子标签序列
        return all_doc_strings, string2xtag_seq, string2xsubs_seq

    # 构建xpath表达式
    def construct_xpath(self, xpath_tags, xpath_subscripts):
        # 初始化xpath为空字符串
        xpath = ""
        # 遍历xpath标签序列和子标签序列
        for tagname, subs in zip(xpath_tags, xpath_subscripts):
            # 将标签和子标签拼接成完整的xpath表达式
            xpath += f"/{tagname}"
            if subs != 0:
                xpath += f"[{subs}]"
        # 返回构建好的xpath表达式
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

        ```py
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
        valid_strings = False

        # Check that strings has a valid type
        if isinstance(html_strings, str):
            valid_strings = True
        elif isinstance(html_strings, (list, tuple)):
            if len(html_strings) == 0 or isinstance(html_strings[0], str):
                valid_strings = True

        if not valid_strings:
            # 抛出值错误，指示 HTML 字符串的类型无效
            raise ValueError(
                "HTML strings must of type `str`, `List[str]` (batch of examples), "
                f"but is of type {type(html_strings)}."
            )

        is_batched = bool(isinstance(html_strings, (list, tuple)) and (isinstance(html_strings[0], str)))

        if not is_batched:
            # 如果不是批处理，将输入转换为包含单个字符串的列表
            html_strings = [html_strings]

        # Get nodes + xpaths
        nodes = []
        xpaths = []
        for html_string in html_strings:
            # 使用给定的 HTML 字符串从单个页面获取三元组
            all_doc_strings, string2xtag_seq, string2xsubs_seq = self.get_three_from_single(html_string)
            nodes.append(all_doc_strings)  # 将文档字符串添加到节点列表中
            xpath_strings = []
            for node, tag_list, sub_list in zip(all_doc_strings, string2xtag_seq, string2xsubs_seq):
                # 构造节点的 XPath 表达式
                xpath_string = self.construct_xpath(tag_list, sub_list)
                xpath_strings.append(xpath_string)
            xpaths.append(xpath_strings)  # 将所有节点的 XPath 字符串列表添加到 xpaths 中

        # 将节点和对应的 XPath 字符串作为字典数据返回
        data = {"nodes": nodes, "xpaths": xpaths}
        encoded_inputs = BatchFeature(data=data, tensor_type=None)  # 将数据包装成 BatchFeature 类型

        return encoded_inputs  # 返回编码后的输入数据
```