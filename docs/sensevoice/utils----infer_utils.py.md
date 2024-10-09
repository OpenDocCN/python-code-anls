# `.\SenseVoiceSmall-src\utils\infer_utils.py`

```
# -*- encoding: utf-8 -*-  # 指定文件编码为 UTF-8，以支持中文字符

import functools  # 导入 functools 模块以使用高阶函数
import logging  # 导入 logging 模块以进行日志记录
from pathlib import Path  # 从 pathlib 导入 Path 类以处理文件路径
from typing import Any, Dict, Iterable, List, NamedTuple, Set, Tuple, Union  # 导入类型提示相关的类

import re  # 导入 re 模块以进行正则表达式操作
import numpy as np  # 导入 numpy 并重命名为 np，以进行数值计算
import yaml  # 导入 yaml 模块以处理 YAML 文件

try:  # 尝试导入 onnxruntime 相关的类和函数
    from onnxruntime import (
        GraphOptimizationLevel,  # 导入图优化级别
        InferenceSession,  # 导入推理会话类
        SessionOptions,  # 导入会话选项类
        get_available_providers,  # 导入获取可用提供者的函数
        get_device,  # 导入获取设备的函数
    )
except:  # 如果导入失败，打印安装提示
    print("please pip3 install onnxruntime")
import jieba  # 导入 jieba 模块用于中文分词
import warnings  # 导入 warnings 模块以处理警告

root_dir = Path(__file__).resolve().parent  # 获取当前文件的根目录路径

logger_initialized = {}  # 初始化一个空字典以存储日志器状态

def pad_list(xs, pad_value, max_len=None):  # 定义一个函数用于填充列表
    n_batch = len(xs)  # 获取输入列表的批次大小
    if max_len is None:  # 如果最大长度未指定
        max_len = max(x.size(0) for x in xs)  # 设置为输入列表中最大张量的大小
    # pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)  # 创建一个填充的张量（已注释）
    # numpy format  # 指示接下来的操作使用 numpy 格式
    pad = (np.zeros((n_batch, max_len)) + pad_value).astype(np.int32)  # 创建填充数组并转换为 int32 类型
    for i in range(n_batch):  # 遍历每个输入张量
        pad[i, : xs[i].shape[0]] = xs[i]  # 将输入张量的内容填充到对应位置

    return pad  # 返回填充后的数组


"""
def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):  # 定义一个用于创建填充掩码的函数
    if length_dim == 0:  # 如果长度维度为0，抛出异常
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):  # 如果 lengths 不是列表
        lengths = lengths.tolist()  # 转换为列表
    bs = int(len(lengths))  # 获取批次大小
    if maxlen is None:  # 如果最大长度未指定
        if xs is None:  # 如果没有提供 xs
            maxlen = int(max(lengths))  # 设置为 lengths 的最大值
        else:  # 如果提供了 xs
            maxlen = xs.size(length_dim)  # 设置为 xs 的长度
    else:  # 如果提供了 maxlen
        assert xs is None  # 确保 xs 为 None
        assert maxlen >= int(max(lengths))  # 确保 maxlen 大于或等于 lengths 的最大值

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)  # 创建从0到maxlen的序列
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)  # 扩展序列维度以适应批次
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)  # 扩展长度以匹配批次
    mask = seq_range_expand >= seq_length_expand  # 创建填充掩码

    if xs is not None:  # 如果提供了 xs
        assert xs.size(0) == bs, (xs.size(0), bs)  # 确保 xs 的批次大小匹配

        if length_dim < 0:  # 如果长度维度为负数
            length_dim = xs.dim() + length_dim  # 转换为正数维度
        # ind = (:, None, ..., None, :, , None, ..., None)  # 指示如何创建索引（已注释）
        ind = tuple(  # 创建索引元组
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)  # 扩展掩码以匹配 xs 的形状并移动到相同设备
    return mask  # 返回填充掩码
"""


class TokenIDConverter:  # 定义一个用于转换 token 和 ID 的类
    def __init__(  # 初始化方法
        self,
        token_list: Union[List, str],  # 接受 token 列表或字符串
    ):

        self.token_list = token_list  # 保存 token 列表
        self.unk_symbol = token_list[-1]  # 获取未知符号（列表最后一个 token）
        self.token2id = {v: i for i, v in enumerate(self.token_list)}  # 创建 token 到 ID 的映射
        self.unk_id = self.token2id[self.unk_symbol]  # 获取未知符号的 ID

    def get_num_vocabulary_size(self) -> int:  # 定义一个方法以获取词汇表大小
        return len(self.token_list)  # 返回 token 列表的长度

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:  # 定义一个方法将 ID 转换为 tokens
        if isinstance(integers, np.ndarray) and integers.ndim != 1:  # 检查输入是否为 1 维数组
            raise TokenIDConverterError(f"Must be 1 dim ndarray, but got {integers.ndim}")  # 抛出异常
        return [self.token_list[i] for i in integers]  # 根据索引返回对应的 tokens

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:  # 定义一个方法将 tokens 转换为 ID
        return [self.token2id.get(i, self.unk_id) for i in tokens]  # 返回 tokens 对应的 ID 列表


class CharTokenizer:  # 定义一个字符分词器类
    # 初始化方法，用于创建类的实例
        def __init__(
            # 接受符号值，可以是路径、字符串或字符串迭代器
            self,
            symbol_value: Union[Path, str, Iterable[str]] = None,
            # 空格符号的表示，默认为"<space>"
            space_symbol: str = "<space>",
            # 是否移除非语言符号的标志，默认为False
            remove_non_linguistic_symbols: bool = False,
        ):
    
            # 设置实例的空格符号
            self.space_symbol = space_symbol
            # 加载符号并赋值给实例变量
            self.non_linguistic_symbols = self.load_symbols(symbol_value)
            # 设置是否移除非语言符号的标志
            self.remove_non_linguistic_symbols = remove_non_linguistic_symbols
    
        # 静态方法，加载符号，返回一个符号集合
        @staticmethod
        def load_symbols(value: Union[Path, str, Iterable[str]] = None) -> Set:
            # 如果没有传入值，返回一个空集合
            if value is None:
                return set()
    
            # 如果值是字符串迭代器，则将其转换为集合
            if isinstance(value, Iterable[str]):
                return set(value)
    
            # 将值转换为路径对象
            file_path = Path(value)
            # 检查文件路径是否存在，不存在则记录警告并返回空集合
            if not file_path.exists():
                logging.warning("%s doesn't exist.", file_path)
                return set()
    
            # 打开文件并读取每一行，去除行尾空白后返回集合
            with file_path.open("r", encoding="utf-8") as f:
                return set(line.rstrip() for line in f)
    
        # 将文本转换为标记列表的方法
        def text2tokens(self, line: Union[str, list]) -> List[str]:
            tokens = []  # 初始化标记列表
            # 当行不为空时，循环处理
            while len(line) != 0:
                # 遍历非语言符号
                for w in self.non_linguistic_symbols:
                    # 如果行以非语言符号开头
                    if line.startswith(w):
                        # 如果不移除非语言符号，将其添加到标记列表
                        if not self.remove_non_linguistic_symbols:
                            tokens.append(line[: len(w)])
                        # 去除已匹配的符号部分
                        line = line[len(w) :]
                        break
                else:
                    # 获取行的第一个字符
                    t = line[0]
                    # 如果是空格，替换为定义的空格符号
                    if t == " ":
                        t = "<space>"
                    # 将字符添加到标记列表
                    tokens.append(t)
                    # 去除已处理的字符
                    line = line[1:]
            # 返回标记列表
            return tokens
    
        # 将标记转换回文本的方法
        def tokens2text(self, tokens: Iterable[str]) -> str:
            # 将空格符号转换为实际空格
            tokens = [t if t != self.space_symbol else " " for t in tokens]
            # 连接标记并返回字符串
            return "".join(tokens)
    
        # 返回类实例的字符串表示
        def __repr__(self):
            return (
                f"{self.__class__.__name__}("
                # 返回空格符号的字符串表示
                f'space_symbol="{self.space_symbol}"'
                # 返回非语言符号的字符串表示
                f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
                f")"
            )
# 定义一个名为 Hypothesis 的命名元组，表示假设数据类型
class Hypothesis(NamedTuple):
    """Hypothesis data type."""

    # 用于存储序列的 NumPy 数组
    yseq: np.ndarray
    # 假设的评分，默认为 0，可以是 float 或 NumPy 数组
    score: Union[float, np.ndarray] = 0
    # 存储其他评分的字典，键为字符串，值可以是 float 或 NumPy 数组
    scores: Dict[str, Union[float, np.ndarray]] = dict()
    # 存储状态信息的字典，键为字符串，值为任意类型
    states: Dict[str, Any] = dict()

    # 将数据转换为 JSON 友好的字典
    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            # 将 yseq 转换为列表
            yseq=self.yseq.tolist(),
            # 将 score 转换为浮点数
            score=float(self.score),
            # 遍历 scores 字典，转换每个值为浮点数
            scores={k: float(v) for k, v in self.scores.items()},
        )._asdict()  # 返回字典表示

# 定义自定义异常类，用于表示 TokenID 转换错误
class TokenIDConverterError(Exception):
    pass

# 定义自定义异常类，用于表示 ONNX 运行时错误
class ONNXRuntimeError(Exception):
    pass

# 定义 OrtInferSession 类，用于处理 ONNX 模型推理
class OrtInferSession:
    # 初始化方法，设置模型文件和设备参数
    def __init__(self, model_file, device_id=-1, intra_op_num_threads=4):
        # 将设备 ID 转换为字符串
        device_id = str(device_id)
        # 创建会话选项对象
        sess_opt = SessionOptions()
        # 设置操作线程数
        sess_opt.intra_op_num_threads = intra_op_num_threads
        # 设置日志严重级别
        sess_opt.log_severity_level = 4
        # 禁用 CPU 内存区域
        sess_opt.enable_cpu_mem_arena = False
        # 设置图优化级别
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # CUDA 执行提供者名称
        cuda_ep = "CUDAExecutionProvider"
        # CUDA 提供者选项的字典
        cuda_provider_options = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "true",
        }
        # CPU 执行提供者名称
        cpu_ep = "CPUExecutionProvider"
        # CPU 提供者选项的字典
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        # 执行提供者列表
        EP_list = []
        # 如果使用 GPU 并且可用，添加 CUDA 提供者
        if device_id != "-1" and get_device() == "GPU" and cuda_ep in get_available_providers():
            EP_list = [(cuda_ep, cuda_provider_options)]
        # 添加 CPU 提供者
        EP_list.append((cpu_ep, cpu_provider_options))

        # 验证模型文件
        self._verify_model(model_file)
        # 创建推理会话
        self.session = InferenceSession(model_file, sess_options=sess_opt, providers=EP_list)

        # 如果 CUDA 提供者不可用，发出警告
        if device_id != "-1" and cuda_ep not in self.session.get_providers():
            warnings.warn(
                f"{cuda_ep} is not avaiable for current env, the inference part is automatically shifted to be executed under {cpu_ep}.\n"
                "Please ensure the installed onnxruntime-gpu version matches your cuda and cudnn version, "
                "you can check their relations from the offical web site: "
                "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
                RuntimeWarning,
            )

    # 定义调用方法，执行模型推理
    def __call__(self, input_content: List[Union[np.ndarray, np.ndarray]]) -> np.ndarray:
        # 创建输入字典，键为输入名称，值为对应的内容
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            # 运行推理并返回结果
            return self.session.run(self.get_output_names(), input_dict)
        except Exception as e:
            # 捕获异常并抛出自定义运行时错误
            raise ONNXRuntimeError("ONNXRuntime inferece failed.") from e

    # 获取模型输入名称的方法
    def get_input_names(
        self,
    ):
        # 返回输入名称的列表
        return [v.name for v in self.session.get_inputs()]

    # 获取模型输出名称的方法
    def get_output_names(
        self,
    ):
        # 返回输出名称的列表
        return [v.name for v in self.session.get_outputs()]
    # 定义一个获取角色列表的方法，默认键为 "character"
    def get_character_list(self, key: str = "character"):
        # 根据指定的键获取相应的元数据，并按行分割成列表返回
        return self.meta_dict[key].splitlines()
    
    # 定义一个检查指定键是否存在的方法，默认键为 "character"
    def have_key(self, key: str = "character") -> bool:
        # 获取模型元数据的自定义元数据映射
        self.meta_dict = self.session.get_modelmeta().custom_metadata_map
        # 如果指定的键在元数据字典中，则返回 True
        if key in self.meta_dict.keys():
            return True
        # 否则返回 False
        return False
    
    # 定义一个静态方法，用于验证模型路径的有效性
    @staticmethod
    def _verify_model(model_path):
        # 将传入的模型路径转换为 Path 对象
        model_path = Path(model_path)
        # 如果模型路径不存在，则抛出文件未找到异常
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        # 如果模型路径不是文件，则抛出文件存在异常
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")
# 将输入单词列表拆分为小句，限制每个小句的单词数量
def split_to_mini_sentence(words: list, word_limit: int = 20):
    # 确保单词限制大于 1
    assert word_limit > 1
    # 如果单词数量小于等于限制，则返回原列表作为一个子列表
    if len(words) <= word_limit:
        return [words]
    # 初始化存储小句的列表
    sentences = []
    # 获取单词列表的长度
    length = len(words)
    # 计算每个小句的基本长度
    sentence_len = length // word_limit
    # 根据计算的长度拆分单词列表
    for i in range(sentence_len):
        sentences.append(words[i * word_limit : (i + 1) * word_limit])
    # 如果有剩余单词，添加到最后一个小句
    if length % word_limit > 0:
        sentences.append(words[sentence_len * word_limit :])
    # 返回拆分后的小句列表
    return sentences


# 将文本按中英文混合的方式拆分成单词
def code_mix_split_words(text: str):
    # 初始化单词列表
    words = []
    # 根据空格拆分文本
    segs = text.split()
    # 遍历每个拆分出的部分
    for seg in segs:
        # 当前分段没有空格
        current_word = ""
        # 遍历每个字符
        for c in seg:
            if len(c.encode()) == 1:
                # 这是一个 ASCII 字符
                current_word += c
            else:
                # 这是一个中文字符
                if len(current_word) > 0:
                    # 如果当前有构建的单词，添加到单词列表
                    words.append(current_word)
                    current_word = ""
                # 添加中文字符到单词列表
                words.append(c)
        # 如果最后还有构建的单词，添加到单词列表
        if len(current_word) > 0:
            words.append(current_word)
    # 返回拆分后的单词列表
    return words


# 检查文本是否为全英文
def isEnglish(text: str):
    # 使用正则表达式匹配全英文文本
    if re.search("^[a-zA-Z']+$", text):
        return True
    else:
        return False


# 将中文和英文混合的单词列表连接为一行字符串
def join_chinese_and_english(input_list):
    # 初始化行字符串
    line = ""
    # 遍历输入列表的每个标记
    for token in input_list:
        if isEnglish(token):
            # 如果是英文，前面加空格再添加到行字符串
            line = line + " " + token
        else:
            # 如果是中文，直接添加到行字符串
            line = line + token

    # 去掉行字符串前后的空格
    line = line.strip()
    # 返回结果字符串
    return line


# 使用 jieba 进行中英文混合的分词
def code_mix_split_words_jieba(seg_dict_file: str):
    # 加载用户词典
    jieba.load_userdict(seg_dict_file)

    # 内部函数用于处理文本
    def _fn(text: str):
        # 将文本按空格拆分
        input_list = text.split()
        # 初始化所有标记的列表
        token_list_all = []
        # 初始化语言标记列表
        langauge_list = []
        # 临时存储标记的列表
        token_list_tmp = []
        # 初始化语言标志
        language_flag = None
        # 遍历输入列表的每个标记
        for token in input_list:
            # 如果当前是英文且前一个是中文，保存当前临时列表
            if isEnglish(token) and language_flag == "Chinese":
                token_list_all.append(token_list_tmp)
                langauge_list.append("Chinese")
                token_list_tmp = []
            # 如果当前是中文且前一个是英文，保存当前临时列表
            elif not isEnglish(token) and language_flag == "English":
                token_list_all.append(token_list_tmp)
                langauge_list.append("English")
                token_list_tmp = []

            # 将当前标记添加到临时列表
            token_list_tmp.append(token)

            # 更新语言标志
            if isEnglish(token):
                language_flag = "English"
            else:
                language_flag = "Chinese"

        # 如果临时列表不为空，保存最后的列表
        if token_list_tmp:
            token_list_all.append(token_list_tmp)
            langauge_list.append(language_flag)

        # 初始化结果列表
        result_list = []
        # 遍历每个标记列表和对应语言标志
        for token_list_tmp, language_flag in zip(token_list_all, langauge_list):
            if language_flag == "English":
                # 如果是英文，直接扩展到结果列表
                result_list.extend(token_list_tmp)
            else:
                # 如果是中文，进行分词并扩展到结果列表
                seg_list = jieba.cut(join_chinese_and_english(token_list_tmp), HMM=False)
                result_list.extend(seg_list)

        # 返回最终结果列表
        return result_list

    # 返回内部处理函数
    return _fn


# 读取 YAML 文件并返回其内容
def read_yaml(yaml_path: Union[str, Path]) -> Dict:
    # 检查指定路径是否存在
    if not Path(yaml_path).exists():
        # 如果不存在，抛出文件不存在错误
        raise FileExistsError(f"The {yaml_path} does not exist.")
    # 使用上下文管理器打开指定的 YAML 文件，以二进制模式读取
        with open(str(yaml_path), "rb") as f:
            # 加载文件内容为 Python 对象，使用 YAML 的加载器
            data = yaml.load(f, Loader=yaml.Loader)
        # 返回加载的 Python 对象
        return data
@functools.lru_cache()  # 使用 LRU 缓存装饰器来缓存 logger 实例，提高性能
def get_logger(name="funasr_onnx"):  # 定义获取 logger 的函数，默认名称为 "funasr_onnx"
    """Initialize and get a logger by name.  # 函数文档，描述功能
    If the logger has not been initialized, this method will initialize the  # 如果 logger 尚未初始化，则进行初始化
    logger by adding one or two handlers, otherwise the initialized logger will  # 添加一个或多个处理器，否则直接返回已初始化的 logger
    be directly returned. During initialization, a StreamHandler will always be  # 初始化时始终添加 StreamHandler
    added.
    Args:
        name (str): Logger name.  # 参数说明，logger 名称
    Returns:
        logging.Logger: The expected logger.  # 返回值说明，返回期望的 logger 实例
    """
    logger = logging.getLogger(name)  # 获取指定名称的 logger 实例
    if name in logger_initialized:  # 检查 logger 是否已初始化
        return logger  # 如果已初始化，直接返回该 logger

    for logger_name in logger_initialized:  # 遍历已初始化的 logger 名称
        if name.startswith(logger_name):  # 检查当前名称是否以已初始化名称开头
            return logger  # 如果是，直接返回该 logger

    formatter = logging.Formatter(  # 创建一个格式化器，定义日志输出格式
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S"  # 日志格式及日期格式
    )

    sh = logging.StreamHandler()  # 创建一个流处理器，用于输出日志到控制台
    sh.setFormatter(formatter)  # 设置流处理器的格式化器
    logger.addHandler(sh)  # 将流处理器添加到 logger
    logger_initialized[name] = True  # 将当前 logger 名称标记为已初始化
    logger.propagate = False  # 禁止 logger 向上级传播日志
    logging.basicConfig(level=logging.ERROR)  # 配置全局日志级别为 ERROR
    return logger  # 返回初始化后的 logger
```