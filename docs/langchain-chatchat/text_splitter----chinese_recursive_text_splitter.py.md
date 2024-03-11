# `.\Langchain-Chatchat\text_splitter\chinese_recursive_text_splitter.py`

```
import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

# 导入必要的模块

logger = logging.getLogger(__name__)

# 获取当前模块的日志记录器

def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # 从末尾使用正则表达式分割文本
    # 现在我们有了分隔符，对文本进行分割
    if separator:
        if keep_separator:
            # 使用正则表达式拆分文本，保留分隔符
            # 模式中的括号将分隔符保留在结果中
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # 如果拆分结果的长度为奇数，则添加最后一个元素
            # splits = [_splits[0]] + splits
        else:
            # 使用正则表达式拆分文本，不保留分隔符
            splits = re.split(separator, text)
    else:
        # 如果没有分隔符，则将文本拆分为单个字符
        splits = list(text)
    return [s for s in splits if s != ""]

# 定义一个函数，根据正则表达式从文本末尾分割文本

class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        # 初始化中文递归文本分割器
        super().__init__(keep_separator=keep_separator, **kwargs)
        # 调用父类的初始化方法
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s"
        ]
        # 设置默认的分隔符列表
        self._is_separator_regex = is_separator_regex

# 定义一个中文递归文本分割器类，继承自递归字符文本分割器类
    # 定义一个方法，用于将文本按照指定分隔符进行分割并返回分块
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        # 初始化最终结果列表
        final_chunks = []
        # 获取要使用的分隔符
        separator = separators[-1]
        new_separators = []
        # 遍历分隔符列表，找到第一个在文本中出现的分隔符
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        # 使用正则表达式从文本末尾分割文本
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # 递归地合并文本，将较长的文本进行分割
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        # 去除空白行并返回最终结果列表
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip()!=""]
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建一个中文递归文本拆分器对象，设置参数为保留分隔符、使用正则表达式作为分隔符、每个块的大小为50、块之间不重叠
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=50,
        chunk_overlap=0
    )
    # 定义一个包含文本的列表
    ls = [
        """中国对外贸易形势报告（75页）。前 10 个月，一般贸易进出口 19.5 万亿元，增长 25.1%， 比整体进出口增速高出 2.9 个百分点，占进出口总额的 61.7%，较去年同期提升 1.6 个百分点。其中，一般贸易出口 10.6 万亿元，增长 25.3%，占出口总额的 60.9%，提升 1.5 个百分点；进口8.9万亿元，增长24.9%，占进口总额的62.7%， 提升 1.8 个百分点。加工贸易进出口 6.8 万亿元，增长 11.8%， 占进出口总额的 21.5%，减少 2.0 个百分点。其中，出口增 长 10.4%，占出口总额的 24.3%，减少 2.6 个百分点；进口增 长 14.2%，占进口总额的 18.0%，减少 1.2 个百分点。此外， 以保税物流方式进出口 3.96 万亿元，增长 27.9%。其中，出 口 1.47 万亿元，增长 38.9%；进口 2.49 万亿元，增长 22.2%。前三季度，中国服务贸易继续保持快速增长态势。服务 进出口总额 37834.3 亿元，增长 11.6%；其中服务出口 17820.9 亿元，增长 27.3%；进口 20013.4 亿元，增长 0.5%，进口增 速实现了疫情以来的首次转正。服务出口增幅大于进口 26.8 个百分点，带动服务贸易逆差下降 62.9%至 2192.5 亿元。服 务贸易结构持续优化，知识密集型服务进出口 16917.7 亿元， 增长 13.3%，占服务进出口总额的比重达到 44.7%，提升 0.7 个百分点。 二、中国对外贸易发展环境分析和展望 全球疫情起伏反复，经济复苏分化加剧，大宗商品价格 上涨、能源紧缺、运力紧张及发达经济体政策调整外溢等风 险交织叠加。同时也要看到，我国经济长期向好的趋势没有 改变，外贸企业韧性和活力不断增强，新业态新模式加快发 展，创新转型步伐提速。产业链供应链面临挑战。美欧等加快出台制造业回迁计 划，加速产业链供应链本土布局，跨国公司调整产业链供应 链，全球双链面临新一轮重构，区域化、近岸化、本土化、 短链化趋势凸显。疫苗供应不足，制造业“缺芯”、物流受限、 运价高企，全球产业链供应链面临压力。 全球通胀持续高位运行。能源价格上涨加大主要经济体 的通胀压力，增加全球经济复苏的不确定性。世界银行今年 10 月发布《大宗商品市场展望》指出，能源价格在 2021 年 大涨逾 80%，并且仍将在 2022 年小幅上涨。IMF 指出，全 球通胀上行风险加剧，通胀前景存在巨大不确定性。""",
        ]
    # 遍历文本列表，获取索引和文本内容
    for inum, text in enumerate(ls):
        # 打印索引
        print(inum)
        # 使用文本拆分器拆分文本，返回拆分后的块列表
        chunks = text_splitter.split_text(text)
        # 遍历拆分后的块列表
        for chunk in chunks:
            # 打印每个块
            print(chunk)
```