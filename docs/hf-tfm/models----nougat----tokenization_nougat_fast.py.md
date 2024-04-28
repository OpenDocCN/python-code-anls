# `.\transformers\models\nougat\tokenization_nougat_fast.py`

```
# 设置文件编码为 UTF-8
# 版权声明及许可证信息
"""
Fast tokenizer class for Nougat.
"""

# 导入模块
import re  # 导入正则表达式模块
from functools import partial  # 导入 partial 函数
from multiprocessing import Pool  # 导入进程池类
from typing import List, Union  # 导入类型提示相关的模块

import numpy as np  # 导入 NumPy 库

# 导入 Hugging Face 的 tokenizer 相关模块
from transformers.tokenization_utils_base import INIT_TOKENIZER_DOCSTRING  # 导入初始化 tokenizer 的文档字符串
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast  # 导入快速 tokenizer 基类
from transformers.utils import add_end_docstrings  # 导入添加文档字符串的辅助函数

# 导入自定义的模块
from ...utils import is_levenshtein_available, is_nltk_available, logging, requires_backends  # 导入一些自定义的工具函数

# 如果 Levenshtein 可用，则导入 ratio 函数
if is_levenshtein_available():
    from Levenshtein import ratio

# 如果 NLTK 可用，则导入 NLTK 库
if is_nltk_available():
    import nltk  # 导入自然语言工具包 NLTK

# 获取日志记录器
logger = logging.get_logger(__name__)

# 将 tokenizer 的初始化文档字符串添加额外内容
INIT_TOKENIZER_DOCSTRING += """
        tokenizer_object ([`tokenizers.Tokenizer`]):
            A [`tokenizers.Tokenizer`] object from 🤗 tokenizers to instantiate from. See [Using tokenizers from 🤗
            tokenizers](../fast_tokenizers) for more information.
        tokenizer_file ([`str`]):
            A path to a local JSON file representing a previously serialized [`tokenizers.Tokenizer`] object from 🤗
            tokenizers.
"""

# 预训练 tokenizer 文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "tokenizer_file": {
        "facebook/nougat-base": "https://huggingface.co/facebook/nougat-base/tokenizer/blob/main/tokenizer.json",
    },
}

# tokenizer 文件名称
VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}

# 预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/nougat-base": 3584}


def markdown_compatible(text: str) -> str:
    """
    Make text compatible with Markdown formatting.

    This function makes various text formatting adjustments to make it compatible with Markdown.

    Args:
        text (`str`):
            The input text to be made Markdown-compatible.

    Returns:
        `str`: The Markdown-compatible text.
    """
    # 方程式标签
    # 将以 (十进制) [some text] 开头的行替换为 \[[some text] \tag{decimal}\]
    text = re.sub(r"^\(([\d.]+[a-zA-Z]?)\) \\\[(.+?)\\\]$", r"\[\2 \\tag{\1}\]", text, flags=re.M)
    # 将以 \[some text\] (decimal)  开头的行替换为 \[[some text] \tag{decimal}\]
    text = re.sub(r"^\\\[(.+?)\\\] \(([\d.]+[a-zA-Z]?)\)$", r"\[\1 \\tag{\2}\]", text, flags=re.M)
    # 将以 \[some text\] (digits) \[another text\]  开头的行替换为 \[[some text] \tag{digits}\] [another text].
    # 使用正则表达式替换文本中符合特定格式的字符串，将其格式化为特定的标记形式
    text = re.sub(
        r"^\\\[(.+?)\\\] \(([\d.]+[a-zA-Z]?)\) (\\\[.+?\\\])$",
        r"\[\1 \\tag{\2}\] \3",
        text,
        flags=re.M,
    )
    # 替换文本中的转义字符 ". " 为普通的句号和空格
    text = text.replace(r"\. ", ". ")
    # 替换文本中的加粗格式化字符串为 LaTeX 的数学粗体格式
    text = text.replace(r"\bm{", r"\mathbf{").replace(r"{\\bm ", r"\mathbf{")
    # 使用正则表达式替换文本中符合特定格式的字符串，将其格式化为 Markdown 链接形式
    text = re.sub(r"\\mbox{ ?\\boldmath\$(.*?)\$}", r"\\mathbf{\1}", text)
    # 使用正则表达式替换文本中的 URL 字符串为 Markdown 可点击的链接格式
    text = re.sub(
        r"((?:http|ftp|https):\/\/(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-]))",
        r"[\1](\1)",
        text,
    )
    # 使用正则表达式替换文本中的算法块格式，将其格式化为 Markdown 代码块形式
    text = re.sub(r"```\s*(.+?)\s*```", r"```\n\1\n```", text, flags=re.S)

    return text
# 定义函数，用于规范化类似列表项的文本行
def normalize_list_like_lines(generation):
    """
    Normalize lines in the given text that resemble list items. The function looks for lines that start optionally with
    '-' or '*', possibly followed by Roman numerals or digits indicating nesting levels. The function reformats such
    lines to make them more structured.

    Args:
        generation (str): The input text containing lines that need to be normalized.

    Returns:
        str: The input text with the list-like lines normalized.

    Note:
        The function uses regular expressions to identify and reformat the list-like lines. The patterns capture
        optional bullet points, nesting levels indicated by numerals, and the actual list item content. The
        normalization adjusts the bullet point style and nesting levels based on the captured patterns.
    """
    import re  # 导入正则表达式模块
    # 匹配以-或*开头且不跟随-或*的行，后面可能跟着表示嵌套级别的罗马数字或数字
    # 再加上用\d或罗马数字表示的该行可选附加编号，然后通过re.finditer调用
    pattern = r"(?:^)(-|\*)?(?!-|\*) ?((?:\d|[ixv])+ )?.+? (-|\*) (((?:\d|[ixv])+)\.(\d|[ixv]) )?.*(?:$)"

    # 逆向遍历找到的匹配项列表
    for match in reversed(list(re.finditer(pattern, generation, flags=re.I | re.M))):
        start, stop = match.span()  # 获取匹配项的起始和终止索引
        delim = match.group(3) + " "  # 获取匹配项第三个分组中的分隔符
        splits = match.group(0).split(delim)  # 使用分隔符分割匹配项内容
        replacement = ""

        if match.group(1) is not None:
            splits = splits[1:]  # 如果第一个分组不为空，移除第一个元素
            delim1 = match.group(1) + " "  # 获取第一个分组中的分隔符
        else:
            delim1 = ""
            continue  # 跳过错误匹配

        pre, post = generation[:start], generation[stop:]  # 将输入文本分成前部分和后部分

        for i, item in enumerate(splits):
            level = 0  # 初始化嵌套级别为0
            potential_numeral, _, rest = item.strip().partition(" ")  # 使用空格分割匹配项的可能编号
            if not rest:  # 如果没有剩余内容
                continue
            # 根据检测到的编号推断当前嵌套级别
            if re.match(r"^[\dixv]+((?:\.[\dixv])?)+$", potential_numeral, flags=re.I | re.M):
                level = potential_numeral.count(".")

            replacement += (
                ("\n" if i > 0 else "") + ("\t" * level) + (delim if i > 0 or start == 0 else delim1) + item.strip()
            )

        if post == "":
            post = "\n"  # 如果后部分为空，则设置为换行符

        generation = pre + replacement + post  # 更新文本内容

    return generation  # 返回处理后的文本内容


# 定义函数，查找下一个标点符号的索引
def find_next_punctuation(text: str, start_idx=0):
    """
    Find the index of the next punctuation mark.

    Args:
        text (`str`):
            String to examine
        start_idx (`int`, *optional*)
            Index where to start
    """
    # 遍历文本，查找下一个标点符号的索引
    for i in range(start_idx, len(text)):
        if text[i] in [".", "?", "!", "\n"]:
            return i  # 如果找到标点符号，则返回索引

    return None  # 没有找到标点符号，返回None


# 定义函数，尝试在输入字符串中截断重复的片段
def truncate_repetitions(text: str, min_len: int = 30) -> str:
    """
    Attempt to truncate repeating segments in the input string.
    This function looks for the longest repeating substring at the end of the input string and truncates it to appear
    only once. To be considered for removal, repetitions need to be continuous.

    Args:
        text (`str`):
            The input raw prediction to be truncated.
        min_len (int):
            The minimum length of the repeating segment.

    Returns:
        `str`: The input string with repeated segments truncated.
    """

    # 将输入的字符串转换为小写
    text_lower = text.lower()
    # 获取文本的长度
    text_length = len(text_lower)

    if text_length < 2 * min_len:
        return text

    # 尝试找到尾部连续重复的长度
    max_repetition_length = None
    for repetition_length in range(min_len, int(text_length / 2)):
        # 检查末尾是否存在重复
        same = True
        for i in range(0, repetition_length):
            if text_lower[text_length - repetition_length - i - 1] != text_lower[text_length - i - 1]:
                same = False
                break

        if same:
            max_repetition_length = repetition_length

    if max_repetition_length is None:
        return text

    # 获取尾部重复的子串
    lcs = text_lower[-max_repetition_length:]

    # 移除除了最后一次重复之外的所有重复
    substituted_text = text
    substituted_text_lower = text_lower
    while substituted_text_lower.endswith(lcs):
        substituted_text = substituted_text[:-max_repetition_length]
        substituted_text_lower = substituted_text_lower[:-max_repetition_length]

    # 这是含有重复的尾部
    repeating_tail = text_lower[len(substituted_text_lower):]

    # 添加直到下一个标点符号，并确保最后一句不重复
    substituted_text_lower_out = substituted_text_lower
    while True:
        sentence_end = find_next_punctuation(text_lower, len(substituted_text_lower_out))
        sentence_start = find_next_punctuation(text_lower[::-1], len(substituted_text_lower_out))
        if sentence_end and sentence_start:
            sentence = text_lower[sentence_start:sentence_end]
            substituted_text_lower_out = text_lower[:sentence_end + 1]
            if sentence in repeating_tail:
                break
        else:
            break

    text_out = text[:len(substituted_text_lower_out)]

    return text_out
def remove_numbers(lines):
    # 定义内部函数_clean，用于移除数字和下划线，然后去除首尾空格
    def _clean(s):
        return re.sub(r"(?:[\d_]|\*\*)", "", s).strip()

    # 如果输入是字符串，则直接使用_clean函数处理后返回
    if isinstance(lines, str):
        return _clean(lines)
    # 否则，对每行应用_clean函数后存储到out列表中再返回
    out = []
    for l in lines:
        out.append(_clean(l))
    return out


def get_slices(lines, clean_lines):
    """
    Get slices of text based on specific criteria within the lines.

    This function identifies and returns slices of text from the input lines based on certain conditions.

    These conditions were chosen by the Nougat authors:
    - The slice is less than 200 characters long.
    - The slice is more than 3 characters long.
    - The slice does not start with "[MISSING_PAGE".
    - The slice is either the same as the next slice or the ratio of the two in terms of Levensthein distance is
      greater than 0.9.

    Args:
        lines (`List[str]`):
            The list of lines containing the text.
        clean_lines (`List[str]`):
            A cleaned version of the text (without numbers).

    Returns:
        `List[tuple]`: A list of tuples representing the start and end indices of text slices.
    """
    # 初始化一个全零数组用于记录是否已处理过某行
    indices = np.zeros(len(lines))
    # 遍历每行文本，直到倒数第二行
    for i in range(len(lines) - 1):
        j = i + 1
        # 向后找到第一个非空行的索引
        while not clean_lines[j] and j < len(lines) - 1:
            j += 1
        # 判断是否符合条件，并标记该片段已处理过
        if (
            len(clean_lines[i]) < 200
            and len(clean_lines[i]) > 3
            and len(clean_lines[j]) < 200
            and len(clean_lines[j]) > 3
            and not clean_lines[i].startswith("[MISSING_PAGE")
            and (clean_lines[i] == clean_lines[j] or ratio(clean_lines[i], clean_lines[j]) > 0.9)
        ):
            indices[i:j] = 1
    # 获取已标记的索引位置
    ids = np.where(indices)[0]
    slices = []
    # 如果没有符合条件的索引位置，直接返回空列表
    if len(ids) == 0:
        return slices
    j0 = 0
    # 向后遍历索引位置差，找出满足条件的片段，组成起始和结束索引的元组，并存储到slices列表中
    for j, x in enumerate(np.diff(ids) > 3):
        if x:
            slices.append((ids[j0], ids[j] + 2))
            j0 = j + 1
    # 处理最后一个片段，将其加入slices列表
    slices.append((ids[j0], ids[-1] + 2))
    # 返回长度大于15的片段列表
    return [sli for sli in slices if sli[1] - sli[0] > 15]


def remove_slice_from_lines(lines, clean_text, slice) -> str:
    """
    Remove a slice of text from the lines based on specific criteria.

    This function identifies a slice of text within the lines and removes it based on certain conditions.

    Args:
        lines (list of str): The list of lines containing the text.
        clean_text (list of str): A cleaned version of the text (without numbers).
        slice (tuple): A tuple representing the start and end indices of the slice to be removed.

    Returns:
        str: The removed slice of text as a single string.
    """
    # 获取要移除的片段的起始行文本
    base = clean_text[slice[0]]
    # 将slice转换为可变列表
    section = list(slice)
    # 设置标志用于检查向前行文本的起始行
    check_start_flag = False
    # 向前遍历，最多5行
    # 从给定的文本片段中找到参考文献部分的起始和结束位置进行切割
    for line_idx in range(max(0, slice[0] - 1), max(0, slice[0] - 5), -1):
        # 如果当前行为空，则跳过
        if not lines[line_idx]:
            continue
        # 如果当前行是"## References"，则将参考文献部分的起始位置设为当前行的索引，并跳出循环
        if lines[line_idx] == "## References":
            section[0] = line_idx
            break
        # 如果当前行和基准比例小于0.9，则设定参考文献部分的起始位置，并检查前一行可能的参考文献信息
        elif ratio(base, remove_numbers(lines[line_idx])) < 0.9:
            section[0] = line_idx + 1
            potential_ref = remove_numbers(lines[max(0, line_idx - 1)].partition("* [")[-1])
            if len(potential_ref) >= 0.75 * len(base) and ratio(base, potential_ref) < 0.9:
                section[0] = line_idx
            check_start_flag = True
            break
    # 向前查找，最多查找5行
    for line_idx in range(min(len(lines), slice[1]), min(len(lines), slice[1] + 5)):
        # 如果当前行和基准比例小于0.9，则将参考文献部分的结束位置设为当前行的索引，并跳出循环
        if ratio(base, remove_numbers(lines[line_idx])) < 0.9:
            section[1] = line_idx
            break
    # 如果文本行数小于等于参考文献部分的结束位置，则将结束位置设为文本行数减1
    if len(lines) <= section[1]:
        section[1] = len(lines) - 1
    # 获取需要删除的文本内容
    to_delete = "\n".join(lines[section[0] : section[1] + 1])
    # 截断下一页的内容
    itera, iterb = enumerate(lines[section[1] - 1]), enumerate(lines[section[1]])
    while True:
        try:
            (ia, a) = next(itera)
            while a.isnumeric():
                (ia, a) = next(itera)
            (ib, b) = next(iterb)
            while b.isnumeric():
                (ib, b) = next(iterb)
            if a != b:
                break
        except StopIteration:
            break
    # 如果需要检查起始标志并且to_delete中包含"* ["，则进行处理
    if check_start_flag and "* [" in to_delete:
        to_delete = "* [" + to_delete.partition("* [")[-1]
    # 尝试获取需要删除的文本与当前文本行长度之差delta，然后截断相应长度的文本内容
    try:
        delta = len(lines[section[1]]) - ib - 1
        if delta > 0:
            to_delete = to_delete[:-delta]
    except UnboundLocalError:
        pass
    # 返回需要删除的文本内容，并去除首尾空白字符
    return to_delete.strip()
# 导入需要的库和模块
@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class NougatTokenizerFast(PreTrainedTokenizerFast):
    """
    Nougat 的快速分词器（由 HuggingFace tokenizers 库支持）。

    这个分词器继承自 `PreTrainedTokenizerFast`，其中包含大多数主要的方法。用户应该参考这个超类以了解有关这些方法的更多信息。
    这个类主要添加了 Nougat 特定的方法，用于对生成的文本进行后处理。

    Args:
        vocab_file (`str`, *optional*):
            包含实例化分词器所需词汇表的 [SentencePiece](https://github.com/google/sentencepiece) 文件（通常具有 .model 扩展名）。
        tokenizer_file (`str`, *optional*):
            包含加载分词器所需内容的 [tokenizers](https://github.com/huggingface/tokenizers) 文件（通常具有 .json 扩展名）。

        clean_up_tokenization_spaces (`str`, *optional*，默认为 `False`):
            是否在解码后清除空格，清除包括去除额外的空格等潜在连在一起的词。

        unk_token (`str`, *optional*，默认为 `"<unk>"`):
            未知的标记。词汇表中不存在的标记无法转换为 ID，并会替换为此标记。

        bos_token (`str`, *optional*，默认为 `"<s>"`):
            在预训练过程中用于表示序列开始的标记。可用作序列分类器的标记。

        eos_token (`str`, *optional*，默认为 `"</s>"`):
            用于表示序列结束的标记。

        pad_token (`str`, *optional*，默认为 `"<pad>"`):
            用于填充的标记，例如在批处理不等长度的序列时使用。

    """

    # 定义类别名字、预训练词汇表文件名字、以及词汇表大小等常量
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    # 初始化方法，用于创建 NougatTokenizerFast 实例
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        **kwargs,
    ):
        # 调用超类 PreTrainedTokenizerFast 的初始化方法，传入相应的参数
        super().__init__(
            vocab_file=vocab_file, 
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )
        # 将参数 vocab_file 赋值给实例属性 vocab_file
        self.vocab_file = vocab_file
    def remove_hallucinated_references(self, text: str) -> str:
        """
        Remove hallucinated or missing references from the text.

        This function identifies and removes references that are marked as missing or hallucinated from the input text.

        Args:
            text (`str`):
                The input text containing references.

        Returns:
            `str`: The text with hallucinated references removed.
        """
        # 将输入文本按行分割
        lines = text.split("\n")
        # 如果行数为0，则返回空字符串
        if len(lines) == 0:
            return ""
        # 删除行中的数字
        clean_lines = remove_numbers(lines)
        # 获取干净行和原始行的切片
        slices = get_slices(lines, clean_lines)
        # 初始化记录需要删除的部分列表
        to_delete = []
        # 遍历切片，将需要删除的部分添加到to_delete中
        for slice in slices:
            to_delete.append(remove_slice_from_lines(lines, clean_lines, slice))
        # 反向遍历to_delete列表，将需要删除的部分替换为指定标记
        for to_delete in reversed(to_delete):
            text = text.replace(to_delete, "\n\n[MISSING_PAGE_POST]\n\n")
        # 通过正则表达式替换匹配的内容
        text = re.sub(
            r"## References\n+\[MISSING_PAGE_POST(:\d+)?\]",
            "\n\n[MISSING_PAGE_POST\\1]",
            text,
        )
        # 返回处理后的文本
        return text

    def correct_tables(self, generation: str) -> str:
        """
        Takes a generated string and fixes tables/tabulars to make them match the markdown format needed.

        Args:
            generation (str): The generated text to be postprocessed.

        Returns:
            str: The postprocessed text.

        Example:

        ```python
        correct_tables("\\begin{table} \\begin{tabular}{l l} & \\ \\end{tabular} \\end{table}")
        "\\begin{table}\n\\begin{tabular}{l l} & \\ \\end{tabular}\n\\end{table}"
        ```
        """
        # 移除明显错误的表
        for l in generation.split("\n"):
            if l.count("\\begin{tabular}") > 15 or l.count("\\multicolumn") > 60 or l.count("&") > 400:
                generation = generation.replace(l, "")
        # 修正空白
        generation = generation.replace("\\begin{table} \\begin{tabular}", "\\begin{table}\n\\begin{tabular}")
        generation = generation.replace("\\end{tabular} \\end{table}", "\\end{tabular}\n\\end{table}")
        generation = generation.replace("\\end{table} Tab", "\\end{table}\nTab")
        generation = re.sub(r"(^.+)\\begin{tab", r"\1\n\\begin{tab", generation, flags=re.M)
        # 移除左对齐的空LaTeX tabular块
        generation = generation.replace(r"\begin{tabular}{l l}  & \\ \end{tabular}", "")
        # 移除只有2个换行符的tabulars
        generation = generation.replace("\\begin{tabular}{}\n\n\\end{tabular}", "")
        # 返回处理后的文本
        return generation

    def post_process_generation(
        self,
        generation: Union[str, List[str]],
        fix_markdown: bool = True,
        num_workers: int = None,
    # 定义一个函数，用于后处理生成的文本或文本列表
    def postprocess_generation(self, generation: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Postprocess a generated text or a list of generated texts.
    
        This function can be used to perform postprocessing on generated text, such as fixing Markdown formatting.
    
        Postprocessing is quite slow so it is recommended to use multiprocessing to speed up the process.
    
        Args:
            generation (Union[str, List[str]]):
                The generated text or a list of generated texts.
            fix_markdown (`bool`, *optional*, defaults to `True`):
                Whether to perform Markdown formatting fixes.
            num_workers (`int`, *optional*):
                Optional number of workers to pass to leverage multiprocessing (postprocessing several texts in
                parallel).
    
        Returns:
            Union[str, List[str]]: The postprocessed text or list of postprocessed texts.
        """
        # 检查是否已导入所需的后端
        requires_backends(self, ["nltk", "levenshtein"])
    
        # 如果生成物是一个文本列表
        if isinstance(generation, list):
            # 如果指定并且是整数类型，则使用多进程处理
            if num_workers is not None and isinstance(num_workers, int):
                # 使用指定数量的进程来后处理生成的文本列表
                with Pool(num_workers) as p:
                    return p.map(partial(self.post_process_single, fix_markdown=fix_markdown), generation)
            else:
                # 否则对生成的每个文本进行后处理
                return [self.post_process_single(s, fix_markdown=fix_markdown) for s in generation]
        else:
            # 如果生成物是单个文本，则直接对其进行后处理
            return self.post_process_single(generation, fix_markdown=fix_markdown)
```