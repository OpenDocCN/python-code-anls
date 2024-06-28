# `.\models\nougat\tokenization_nougat_fast.py`

```
# 设置文件编码为 UTF-8
# 版权声明，声明此代码的版权归 HuggingFace Inc. 团队所有，使用 Apache License 2.0 授权
# 根据 Apache License 2.0 许可证，除非符合许可证的规定，否则不得使用此文件
# 获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
"""
Nougat 的快速分词器类。
"""
# 导入正则表达式模块
import re
# 导入 functools 模块的 partial 函数
from functools import partial
# 导入 multiprocessing 模块的 Pool 类
from multiprocessing import Pool
# 导入 List 和 Union 类型提示
from typing import List, Union

# 导入 numpy 库，并用 np 作为别名
import numpy as np

# 从 transformers 库中导入相关的函数和类
# INIT_TOKENIZER_DOCSTRING 是来自 tokenization_utils_base 模块的一个文档字符串
from transformers.tokenization_utils_base import INIT_TOKENIZER_DOCSTRING
# 导入 PreTrainedTokenizerFast 类
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
# 导入 add_end_docstrings 函数
from transformers.utils import add_end_docstrings

# 导入本地工具函数和模块
# is_levenshtein_available 和 is_nltk_available 是本地工具函数
from ...utils import is_levenshtein_available, is_nltk_available, logging, requires_backends

# 如果 Levenshtein 可用，则从 Levenshtein 库导入 ratio 函数
if is_levenshtein_available():
    from Levenshtein import ratio

# 如果 NLTK 可用，则导入 nltk 库
if is_nltk_available():
    import nltk

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 在 INIT_TOKENIZER_DOCSTRING 后添加额外的文档字符串说明
INIT_TOKENIZER_DOCSTRING += """
        tokenizer_object ([`tokenizers.Tokenizer`]):
            A [`tokenizers.Tokenizer`] object from 🤗 tokenizers to instantiate from. See [Using tokenizers from 🤗
            tokenizers](../fast_tokenizers) for more information.
        tokenizer_file ([`str`]):
            A path to a local JSON file representing a previously serialized [`tokenizers.Tokenizer`] object from 🤗
            tokenizers.
"""

# 预训练词汇文件映射，指定了预训练模型的 tokenizer_file 的 URL
PRETRAINED_VOCAB_FILES_MAP = {
    "tokenizer_file": {
        "facebook/nougat-base": "https://huggingface.co/facebook/nougat-base/tokenizer/blob/main/tokenizer.json",
    },
}

# 指定词汇文件的名称
VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}

# 指定预训练位置编码的尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/nougat-base": 3584}


def markdown_compatible(text: str) -> str:
    """
    使文本兼容 Markdown 格式。

    此函数对文本进行各种格式调整，以使其与 Markdown 兼容。

    Args:
        text (`str`):
            要使其兼容 Markdown 的输入文本。

    Returns:
        `str`: 兼容 Markdown 的文本。
    """
    # 等式标签
    # 用 \[some text\] 样式的模式替换以十进制数字开头的行，将其转换为 \[[some text] \tag{decimal}\]。
    text = re.sub(r"^\(([\d.]+[a-zA-Z]?)\) \\\[(.+?)\\\]$", r"\[\2 \\tag{\1}\]", text, flags=re.M)
    # 用 \[some text\] 样式的模式替换以十进制数字结尾的行，将其转换为 \[[some text] \tag{decimal}\]。
    text = re.sub(r"^\\\[(.+?)\\\] \(([\d.]+[a-zA-Z]?)\)$", r"\[\1 \\tag{\2}\]", text, flags=re.M)
    # 用 \[some text\] 样式的模式替换以数字开头，以 \[another text\] 结尾的行，将其转换为 \[[some text] \tag{digits}\] [another text].
    # 使用正则表达式替换文本中符合特定格式的字符串，将其转换为特定格式的标记
    text = re.sub(
        r"^\\\[(.+?)\\\] \(([\d.]+[a-zA-Z]?)\) (\\\[.+?\\\])$",
        r"\[\1 \\tag{\2}\] \3",
        text,
        flags=re.M,
    )
    # 替换文本中的特定字符串，去除反斜杠和句点之间的空格
    text = text.replace(r"\. ", ". ")
    # 将文本中的特定粗体格式符号替换为 LaTeX 中的数学粗体格式符号
    text = text.replace(r"\bm{", r"\mathbf{").replace(r"{\\bm ", r"\mathbf{")
    # 使用正则表达式替换文本中特定格式的字符串，将其转换为数学粗体格式
    text = re.sub(r"\\mbox{ ?\\boldmath\$(.*?)\$}", r"\\mathbf{\1}", text)
    # 将文本中的 URL 格式化为 Markdown 可点击的链接格式
    text = re.sub(
        r"((?:http|ftp|https):\/\/(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-]))",
        r"[\1](\1)",
        text,
    )
    # 使用正则表达式重新格式化文本中的算法块，确保其在 Markdown 中显示为合适的代码块格式
    text = re.sub(r"```\s*(.+?)\s*```", r"```\n\1\n```", text, flags=re.S)

    # 返回处理后的文本结果
    return text
# 将文本中类似列表项的行规范化。函数查找以'-'或'*'开头的行，可能后跟表示嵌套级别的罗马数字或数字。
# 函数重新格式化这些行，使其结构更加清晰。
def normalize_list_like_lines(generation):
    # 匹配以'-'或'*'开头的行，后面不是'-'或'*'（列表），可能后跟数字\d或罗马数字，然后捕获此行的可选附加编号，
    # 传递给re.finditer使用的正则表达式模式。
    pattern = r"(?:^)(-|\*)?(?!-|\*) ?((?:\d|[ixv])+ )?.+? (-|\*) (((?:\d|[ixv])+)\.(\d|[ixv]) )?.*(?:$)"

    # 以逆序遍历生成的匹配对象列表
    for match in reversed(list(re.finditer(pattern, generation, flags=re.I | re.M))):
        start, stop = match.span()
        delim = match.group(3) + " "
        splits = match.group(0).split(delim)
        replacement = ""

        # 如果第一个捕获组匹配不是None，则忽略第一个条目并继续
        if match.group(1) is not None:
            splits = splits[1:]
            delim1 = match.group(1) + " "
        else:
            delim1 = ""
            continue  # 跳过错误的正则表达式结果

        pre, post = generation[:start], generation[stop:]

        # 处理分割的条目并生成替换文本
        for i, item in enumerate(splits):
            level = 0
            potential_numeral, _, rest = item.strip().partition(" ")
            if not rest:
                continue
            # 根据检测到的编号推断当前嵌套级别
            if re.match(r"^[\dixv]+((?:\.[\dixv])?)+$", potential_numeral, flags=re.I | re.M):
                level = potential_numeral.count(".")

            replacement += (
                ("\n" if i > 0 else "") + ("\t" * level) + (delim if i > 0 or start == 0 else delim1) + item.strip()
            )

        if post == "":
            post = "\n"

        generation = pre + replacement + post

    return generation


# 找到文本中下一个标点符号的索引
def find_next_punctuation(text: str, start_idx=0):
    """
    Find the index of the next punctuation mark.

    Args:
        text (`str`):
            String to examine
        start_idx (`int`, *optional*)
            Index where to start
    """
    for i in range(start_idx, len(text)):
        if text[i] in [".", "?", "!", "\n"]:
            return i

    return None


# 尝试截断输入字符串中的重复部分
def truncate_repetitions(text: str, min_len: int = 30) -> str:
    """
    Attempt to truncate repeating segments in the input string.

    Args:
        text (str): The input text to process.
        min_len (int, optional): The minimum length of repeating segments to truncate.

    Returns:
        str: The processed text with repeated segments truncated.
    """
    # 将输入文本转换为小写
    text_lower = text.lower()
    # 获取输入文本的长度
    text_length = len(text_lower)

    # 如果输入文本长度小于最小重复段长度的两倍，直接返回原始文本
    if text_length < 2 * min_len:
        return text

    # 尝试查找尾部重复的最大长度
    max_repetition_length = None
    for repetition_length in range(min_len, int(text_length / 2)):
        # 检查尾部是否有重复
        same = True
        for i in range(0, repetition_length):
            if text_lower[text_length - repetition_length - i - 1] != text_lower[text_length - i - 1]:
                same = False
                break

        if same:
            max_repetition_length = repetition_length

    # 如果没有找到重复的部分，返回原始文本
    if max_repetition_length is None:
        return text

    # 获取最长重复子串
    lcs = text_lower[-max_repetition_length:]

    # 移除除最后一个重复外的所有重复
    substituted_text = text
    substituted_text_lower = text_lower
    while substituted_text_lower.endswith(lcs):
        substituted_text = substituted_text[:-max_repetition_length]
        substituted_text_lower = substituted_text_lower[:-max_repetition_length]

    # 获取包含重复的尾部内容
    repeating_tail = text_lower[len(substituted_text_lower):]

    # 从文本开头添加内容，直到下一个标点，并确保最后一句不重复
    substituted_text_lower_out = substituted_text_lower
    while True:
        # 找到下一个标点符号的位置
        sentence_end = find_next_punctuation(text_lower, len(substituted_text_lower_out))
        sentence_start = find_next_punctuation(text_lower[::-1], len(substituted_text_lower_out))
        if sentence_end and sentence_start:
            # 提取当前句子
            sentence = text_lower[sentence_start:sentence_end]
            # 更新输出的文本为当前位置前的内容
            substituted_text_lower_out = text_lower[:sentence_end + 1]
            # 如果当前句子在重复的尾部出现，结束循环
            if sentence in repeating_tail:
                break
        else:
            break

    # 获取最终输出的文本
    text_out = text[:len(substituted_text_lower_out)]

    return text_out
def remove_numbers(lines):
    def _clean(s):
        return re.sub(r"(?:[\d_]|\*\*)", "", s).strip()  # 移除字符串中的数字和特殊字符

    if isinstance(lines, str):
        return _clean(lines)  # 如果输入是字符串，则直接清理并返回该字符串
    out = []
    for l in lines:
        out.append(_clean(l))  # 对列表中的每个字符串进行清理，并加入到输出列表中
    return out  # 返回清理后的字符串列表


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
    indices = np.zeros(len(lines))  # 创建一个与输入行数相同长度的零数组
    for i in range(len(lines) - 1):
        j = i + 1
        while not clean_lines[j] and j < len(lines) - 1:
            j += 1  # 跳过空行或清理后为空的行，直到找到非空行或清理后不为空的行
        if (
            len(clean_lines[i]) < 200
            and len(clean_lines[i]) > 3
            and len(clean_lines[j]) < 200
            and len(clean_lines[j]) > 3
            and not clean_lines[i].startswith("[MISSING_PAGE")
            and (clean_lines[i] == clean_lines[j] or ratio(clean_lines[i], clean_lines[j]) > 0.9)
        ):
            indices[i:j] = 1  # 根据条件设置索引数组中的标记为1
    ids = np.where(indices)[0]  # 获取所有标记为1的索引位置
    slices = []
    if len(ids) == 0:
        return slices  # 如果没有找到符合条件的索引，直接返回空列表
    j0 = 0
    for j, x in enumerate(np.diff(ids) > 3):
        if x:
            slices.append((ids[j0], ids[j] + 2))  # 将符合条件的片段起始和结束索引添加到slices列表中
            j0 = j + 1
    slices.append((ids[j0], ids[-1] + 2))  # 添加最后一个符合条件的片段起始和结束索引
    return [sli for sli in slices if sli[1] - sli[0] > 15]  # 返回长度大于15的片段列表


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
    base = clean_text[slice[0]]  # 获取要删除的片段的起始位置的清理后文本
    section = list(slice)  # 将要删除的片段转换为列表形式
    check_start_flag = False  # 初始化检查开始标志为False
    # backwards pass, at most 5 lines
    # 循环遍历文本的指定范围，最多向前查找5行
    for line_idx in range(max(0, slice[0] - 1), max(0, slice[0] - 5), -1):
        # 如果当前行为空，则跳过
        if not lines[line_idx]:
            continue
        # 如果当前行是"## References"，则确定该段落起始位置为当前行
        if lines[line_idx] == "## References":
            section[0] = line_idx
            break
        # 否则，比较当前行与基准字符串的相似度，如果低于阈值0.9
        elif ratio(base, remove_numbers(lines[line_idx])) < 0.9:
            # 将段落起始位置设为当前行的下一行
            section[0] = line_idx + 1
            # 获取潜在引用的内容（当前行的上一行以"* ["开头的部分）
            potential_ref = remove_numbers(lines[max(0, line_idx - 1)].partition("* [")[-1])
            # 如果潜在引用的长度大于基准字符串长度的0.75，并且与基准字符串的相似度低于0.9
            if len(potential_ref) >= 0.75 * len(base) and ratio(base, potential_ref) < 0.9:
                # 将段落起始位置设为当前行
                section[0] = line_idx
            # 设置检查起始标志为真
            check_start_flag = True
            break
    
    # forward pass，最多向后查找5行
    for line_idx in range(min(len(lines), slice[1]), min(len(lines), slice[1] + 5)):
        # 如果当前行与基准字符串的相似度低于阈值0.9
        if ratio(base, remove_numbers(lines[line_idx])) < 0.9:
            # 确定段落结束位置为当前行
            section[1] = line_idx
            break
    
    # 如果文本行数小于等于段落结束位置，将段落结束位置设为文本行数减1
    if len(lines) <= section[1]:
        section[1] = len(lines) - 1
    
    # 获取待删除的文本段落，从section[0]到section[1]行（包括）
    to_delete = "\n".join(lines[section[0] : section[1] + 1])
    
    # 截取下一页内容
    itera, iterb = enumerate(lines[section[1] - 1]), enumerate(lines[section[1]])
    while True:
        try:
            (ia, a) = next(itera)
            # 跳过数字行
            while a.isnumeric():
                (ia, a) = next(itera)
            (ib, b) = next(iterb)
            # 跳过数字行
            while b.isnumeric():
                (ib, b) = next(iterb)
            # 如果遇到不相同的字符，则停止截取
            if a != b:
                break
        except StopIteration:
            break
    
    # 如果检查起始标志为真且待删除的文本包含"* ["，则保留"* ["开头的内容
    if check_start_flag and "* [" in to_delete:
        to_delete = "* [" + to_delete.partition("* [")[-1]
    
    try:
        # 计算截取的尾部内容长度
        delta = len(lines[section[1]]) - ib - 1
        # 如果长度大于0，则从to_delete中删除相应长度的尾部内容
        if delta > 0:
            to_delete = to_delete[:-delta]
    except UnboundLocalError:
        # 忽略未绑定局部变量的异常
        pass
    
    # 返回经过处理的待删除文本段落（去除首尾空白字符）
    return to_delete.strip()
# 使用装饰器添加文档字符串，文档字符串内容为INIT_TOKENIZER_DOCSTRING变量的值
@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
# 定义一个名为NougatTokenizerFast的类，它继承自PreTrainedTokenizerFast类
class NougatTokenizerFast(PreTrainedTokenizerFast):
    """
    Fast tokenizer for Nougat (backed by HuggingFace tokenizers library).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods. This class mainly adds Nougat-specific
    methods for postprocessing the generated text.

    Args:
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        tokenizer_file (`str`, *optional*):
            [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.

        clean_up_tokenization_spaces (`str`, *optional*, defaults to `False`):
            Wether to cleanup spaces after decoding, cleanup consists in removing potential artifacts like extra
            spaces.

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.

        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    # 类属性：定义词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 类属性：定义预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 类属性：定义最大模型输入尺寸列表
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 类属性：定义模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 类属性：定义慢速分词器类别，此处为None
    slow_tokenizer_class = None

    # 初始化方法
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
        # 调用父类PreTrainedTokenizerFast的初始化方法，传入各种参数
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
        # 设置实例属性self.vocab_file为传入的vocab_file参数值
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
        # 将文本按行分割成列表
        lines = text.split("\n")
        # 如果行数为0，返回空字符串
        if len(lines) == 0:
            return ""
        # 从文本中移除数字
        clean_lines = remove_numbers(lines)
        # 获取干净行和原始行的切片
        slices = get_slices(lines, clean_lines)
        # 存储待删除的切片
        to_delete = []
        # 遍历每个切片
        for slice in slices:
            # 从原始行和干净行中移除切片，返回要删除的部分
            to_delete.append(remove_slice_from_lines(lines, clean_lines, slice))
        # 反向遍历待删除的部分
        for to_delete in reversed(to_delete):
            # 用指定标记替换要删除的部分，用于标记丢失页码
            text = text.replace(to_delete, "\n\n[MISSING_PAGE_POST]\n\n")
        # 使用正则表达式替换文本中的特定格式的引用标记
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
        # 移除明显错误的表格
        for l in generation.split("\n"):
            if l.count("\\begin{tabular}") > 15 or l.count("\\multicolumn") > 60 or l.count("&") > 400:
                generation = generation.replace(l, "")
        # 修正空白和格式
        generation = generation.replace("\\begin{table} \\begin{tabular}", "\\begin{table}\n\\begin{tabular}")
        generation = generation.replace("\\end{tabular} \\end{table}", "\\end{tabular}\n\\end{table}")
        generation = generation.replace("\\end{table} Tab", "\\end{table}\nTab")
        # 在每个表格前加入新行，以确保正确的格式
        generation = re.sub(r"(^.+)\\begin{tab", r"\1\n\\begin{tab", generation, flags=re.M)
        # 移除左对齐的空 LaTeX 表格块
        generation = generation.replace(r"\begin{tabular}{l l}  & \\ \end{tabular}", "")
        # 移除只有两个换行符的表格
        generation = generation.replace("\\begin{tabular}{}\n\n\\end{tabular}", "")
        # 返回处理后的文本
        return generation
    ) -> Union[str, List[str]]:
        """
        Postprocess a generated text or a list of generated texts.

        This function can be used to perform postprocessing on generated text, such as fixing Markdown formatting.

        Postprocessing is quite slow so it is recommended to use multiprocessing to speed up the process.

        Args:
            generation (Union[str, List[str]]):
                The generated text or a list of generated texts.
            fix_markdown (`bool`, *optional*, defaults to `True`):
                Whether to perform Markdown formatting fixes.
            num_workers (`int`, *optional`):
                Optional number of workers to pass to leverage multiprocessing (postprocessing several texts in
                parallel).

        Returns:
            Union[str, List[str]]: The postprocessed text or list of postprocessed texts.
        """
        # 确保依赖的后端模块已经加载
        requires_backends(self, ["nltk", "levenshtein"])

        # 如果生成物是一个列表
        if isinstance(generation, list):
            # 如果指定了并行处理的 worker 数量
            if num_workers is not None and isinstance(num_workers, int):
                # 使用 multiprocessing.Pool 创建一个进程池
                with Pool(num_workers) as p:
                    # 使用进程池的 map 函数并行处理生成的文本列表
                    return p.map(partial(self.post_process_single, fix_markdown=fix_markdown), generation)
            else:
                # 否则，串行处理每个生成的文本
                return [self.post_process_single(s, fix_markdown=fix_markdown) for s in generation]
        else:
            # 如果生成物是单个字符串，则直接调用单文本处理函数
            return self.post_process_single(generation, fix_markdown=fix_markdown)
```