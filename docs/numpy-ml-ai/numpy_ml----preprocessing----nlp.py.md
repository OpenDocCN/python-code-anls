# `numpy-ml\numpy_ml\preprocessing\nlp.py`

```
# 导入必要的库和模块
import re
import heapq
import os.path as op
from collections import Counter, OrderedDict, defaultdict
import numpy as np

# 定义英文停用词列表，来源于"Glasgow Information Retrieval Group"
_STOP_WORDS = set(
    ).split(" "),
)

# 定义用于匹配单词的正则表达式，用于分词
_WORD_REGEX = re.compile(r"(?u)\b\w\w+\b")  # sklearn默认
_WORD_REGEX_W_PUNC = re.compile(r"(?u)\w+|[^a-zA-Z0-9\s]")
_WORD_REGEX_W_PUNC_AND_WHITESPACE = re.compile(r"(?u)s?\w+\s?|\s?[^a-zA-Z0-9\s]\s?")

# 定义用于匹配标点符号的正则表达式
_PUNC_BYTE_REGEX = re.compile(
    r"(33|34|35|36|37|38|39|40|41|42|43|44|45|"
    r"46|47|58|59|60|61|62|63|64|91|92|93|94|"
    r"95|96|123|124|125|126)",
)
# 定义标点符号
_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
# 创建用于去除标点符号的转换表
_PUNC_TABLE = str.maketrans("", "", _PUNCTUATION)

# 定义函数，返回指定长度的n-gram序列
def ngrams(sequence, N):
    """Return all `N`-grams of the elements in `sequence`"""
    assert N >= 1
    return list(zip(*[sequence[i:] for i in range(N)]))

# 定义函数，将字符串按空格分词，可选择是否转为小写、过滤停用词和标点符号
def tokenize_whitespace(
    line, lowercase=True, filter_stopwords=True, filter_punctuation=True, **kwargs,
):
    """
    Split a string at any whitespace characters, optionally removing
    punctuation and stop-words in the process.
    """
    line = line.lower() if lowercase else line
    words = line.split()
    line = [strip_punctuation(w) for w in words] if filter_punctuation else line
    return remove_stop_words(words) if filter_stopwords else words

# 定义函数，将字符串按单词分词，可选择是否转为小写、过滤停用词和标点符号
def tokenize_words(
    line, lowercase=True, filter_stopwords=True, filter_punctuation=True, **kwargs,
):
    """
    Split a string into individual words, optionally removing punctuation and
    stop-words in the process.
    """
    REGEX = _WORD_REGEX if filter_punctuation else _WORD_REGEX_W_PUNC
    words = REGEX.findall(line.lower() if lowercase else line)
    return remove_stop_words(words) if filter_stopwords else words

# 定义函数，将字符串按字节分词
def tokenize_words_bytes(
    line,
    # 设置是否将文本转换为小写
    lowercase=True,
    # 设置是否过滤停用词
    filter_stopwords=True,
    # 设置是否过滤标点符号
    filter_punctuation=True,
    # 设置文本编码格式为 UTF-8
    encoding="utf-8",
    # **kwargs 表示接受任意数量的关键字参数，这些参数会被传递给函数的其他部分进行处理
    **kwargs,
# 将字符串拆分为单词，并在此过程中选择性地删除标点符号和停用词。将每个单词转换为字节列表。
def tokenize_words(
    line,
    lowercase=lowercase,
    filter_stopwords=filter_stopwords,
    filter_punctuation=filter_punctuation,
    **kwargs,
):
    # 对单词进行分词处理，根据参数选择是否转换为小写、过滤停用词和标点符号
    words = tokenize_words(
        line,
        lowercase=lowercase,
        filter_stopwords=filter_stopwords,
        filter_punctuation=filter_punctuation,
        **kwargs,
    )
    # 将单词转换为字节列表，每个字节用空格分隔
    words = [" ".join([str(i) for i in w.encode(encoding)]) for w in words]
    # 返回字节列表
    return words


# 将字符串中的字符转换为字节集合。每个字节用0到255之间的整数表示。
def tokenize_bytes_raw(line, encoding="utf-8", splitter=None, **kwargs):
    # 将字符串中的字符编码为字节，每个字节用空格分隔
    byte_str = [" ".join([str(i) for i in line.encode(encoding)])
    # 如果指定了分隔符为标点符号，则在编码为字节之前在标点符号处进行分割
    if splitter == "punctuation":
        byte_str = _PUNC_BYTE_REGEX.sub(r"-\1-", byte_str[0]).split("-")
    return byte_str


# 将字节（表示为0到255之间的整数）解码为指定编码的字符。
def bytes_to_chars(byte_list, encoding="utf-8"):
    # 将字节列表中的整数转换为十六进制字符串
    hex_array = [hex(a).replace("0x", "") for a in byte_list]
    # 将十六进制字符串连接起来，并在需要时在前面补0
    hex_array = " ".join([h if len(h) > 1 else f"0{h}" for h in hex_array])
    # 将十六进制字符串转换为字节数组，再根据指定编码解码为字符
    return bytearray.fromhex(hex_array).decode(encoding)


# 将字符串中的字符转换为小写，并根据参数选择是否过滤标点符号。
def tokenize_chars(line, lowercase=True, filter_punctuation=True, **kwargs):
    # 将字符串拆分为单个字符，可选择在此过程中删除标点符号和停用词
    """
    # 如果需要转换为小写，则将字符串转换为小写
    line = line.lower() if lowercase else line
    # 如果需要过滤标点符号，则调用函数去除标点符号
    line = strip_punctuation(line) if filter_punctuation else line
    # 使用正则表达式将连续多个空格替换为一个空格，并去除首尾空格，然后将结果转换为字符列表
    chars = list(re.sub(" {2,}", " ", line).strip())
    # 返回字符列表
    return chars
# 从单词字符串列表中移除停用词
def remove_stop_words(words):
    """Remove stop words from a list of word strings"""
    # 返回不在停用词列表中的单词
    return [w for w in words if w.lower() not in _STOP_WORDS]


# 从字符串中移除标点符号
def strip_punctuation(line):
    """Remove punctuation from a string"""
    # 使用_PUNC_TABLE来移除字符串中的标点符号，并去除首尾空格
    return line.translate(_PUNC_TABLE).strip()


#######################################################################
#                          Byte-Pair Encoder                          #
#######################################################################


# 定义一个Byte-Pair编码器类
class BytePairEncoder(object):
    def __init__(self, max_merges=3000, encoding="utf-8"):
        """
        A byte-pair encoder for sub-word embeddings.

        Notes
        -----
        Byte-pair encoding [1][2] is a compression algorithm that iteratively
        replaces the most frequently ocurring byte pairs in a set of documents
        with a new, single token. It has gained popularity as a preprocessing
        step for many NLP tasks due to its simplicity and expressiveness: using
        a base coebook of just 256 unique tokens (bytes), any string can be
        encoded.

        References
        ----------
        .. [1] Gage, P. (1994). A new algorithm for data compression. *C
           Users Journal, 12(2)*, 23–38.
        .. [2] Sennrich, R., Haddow, B., & Birch, A. (2015). Neural machine
           translation of rare words with subword units, *Proceedings of the
           54th Annual Meeting of the Association for Computational
           Linguistics,* 1715-1725.

        Parameters
        ----------
        max_merges : int
            The maximum number of byte pair merges to perform during the
            :meth:`fit` operation. Default is 3000.
        encoding : str
            The encoding scheme for the documents used to train the encoder.
            Default is `'utf-8'`.
        """
        # 初始化参数字典
        self.parameters = {
            "max_merges": max_merges,
            "encoding": encoding,
        }

        # 初始化字节到标记和标记到字节的有序字典。字节以十进制表示为0到255之间的整数。
        # 在255之前，标记和字节表示之间存在一对一的对应关系。
        self.byte2token = OrderedDict({i: i for i in range(256)})
        self.token2byte = OrderedDict({v: k for k, v in self.byte2token.items()})
    # 在给定语料库上训练一个字节对编码表
    def fit(self, corpus_fps, encoding="utf-8"):
        """
        Train a byte pair codebook on a set of documents.

        Parameters
        ----------
        corpus_fps : str or list of strs
            The filepath / list of filepaths for the document(s) to be used to
            learn the byte pair codebook.
        encoding : str
            The text encoding for documents. Common entries are either 'utf-8'
            (no header byte), or 'utf-8-sig' (header byte). Default is
            'utf-8'.
        """
        # 创建一个词汇表对象，用于存储字节对编码表
        vocab = (
            Vocabulary(
                lowercase=False,
                min_count=None,
                max_tokens=None,
                filter_stopwords=False,
                filter_punctuation=False,
                tokenizer="bytes",
            )
            # 在给定语料库上拟合词汇表
            .fit(corpus_fps, encoding=encoding)
            # 获取词汇表中的计数信息
            .counts
        )

        # 迭代地合并跨文档中最常见的字节二元组
        for _ in range(self.parameters["max_merges"]):
            # 获取词汇表中的字节二元组计数信息
            pair_counts = self._get_counts(vocab)
            # 找到出现次数最多的字节二元组
            most_common_bigram = max(pair_counts, key=pair_counts.get)
            # 合并最常见的字节二元组到词汇表中
            vocab = self._merge(most_common_bigram, vocab)

        # 初始化一个空集合，用于存储字节标记
        token_bytes = set()
        # 遍历词汇表中的键
        for k in vocab.keys():
            # 将键按空格分割，筛选包含"-"的字节标记
            token_bytes = token_bytes.union([w for w in k.split(" ") if "-" in w])

        # 遍历字节标记集合
        for i, t in enumerate(token_bytes):
            # 将字节标记转换为元组形式
            byte_tuple = tuple(int(j) for j in t.split("-"))
            # 将字节标记映射到对应的标记索引
            self.token2byte[256 + i] = byte_tuple
            # 将字节标记索引映射到对应的字节标记
            self.byte2token[byte_tuple] = 256 + i

        # 返回当前对象
        return self

    # 获取词汇表中的字节二元组计数信息
    def _get_counts(self, vocab):
        """Collect bigram counts for the tokens in vocab"""
        # 初始化一个默认字典，用于存储字节二元组计数
        pair_counts = defaultdict(int)
        # 遍历词汇表中的单词和计数信息
        for word, count in vocab.items():
            # 生成单词的二元组
            pairs = ngrams(word.split(" "), 2)
            # 遍历单词的二元组
            for p in pairs:
                # 更新字节二元组计数信息
                pair_counts[p] += count
        # 返回字节二元组计数信息
        return pair_counts
    # 将给定的二元组替换为单个标记，并相应更新词汇表
    def _merge(self, bigram, vocab):
        v_out = {}
        # 转义二元组中的单词，用于正则表达式匹配
        bg = re.escape(" ".join(bigram))
        # 创建匹配二元组的正则表达式
        bigram_regex = re.compile(r"(?<!\S)" + bg + r"(?!\S)")
        # 遍历词汇表中的单词
        for word in vocab.keys():
            # 将匹配到的二元组替换为连接符"-"
            w_out = bigram_regex.sub("-".join(bigram), word)
            v_out[w_out] = vocab[word]
        return v_out

    # 将文本中的单词转换为其字节对编码的标记ID
    def transform(self, text):
        """
        Transform the words in `text` into their byte pair encoded token IDs.

        Parameters
        ----------
        text: str or list of `N` strings
            The list of strings to encode

        Returns
        -------
        codes : list of `N` lists
            A list of byte pair token IDs for each of the `N` strings in
            `text`.

        Examples
        --------
        >>> B = BytePairEncoder(max_merges=100).fit("./example.txt")
        >>> encoded_tokens = B.transform("Hello! How are you 😁 ?")
        >>> encoded_tokens
        [[72, 879, 474, ...]]
        """
        # 如果输入是字符串，则转换为列表
        if isinstance(text, str):
            text = [text]
        # 对文本中的每个字符串进行转换
        return [self._transform(string) for string in text]
    # 将单个文本字符串转换为字节对 ID 列表
    def _transform(self, text):
        # 获取参数配置
        P = self.parameters
        # 将文本字符串转换为原始字节流
        _bytes = tokenize_bytes_raw(text, encoding=P["encoding"])

        # 初始化编码结果列表
        encoded = []
        # 遍历每个字节对
        for w in _bytes:
            l, r = 0, len(w)
            # 将字节对转换为整数列表
            w = [int(i) for i in w.split(" ")]

            # 循环处理字节对
            while l < len(w):
                candidate = tuple(w[l:r])

                # 如果候选字节对长度大于1且在词汇表中
                if len(candidate) > 1 and candidate in self.byte2token:
                    # 将候选字节对的 ID 添加到编码结果列表中
                    encoded.append(self.byte2token[candidate])
                    l, r = r, len(w)
                # 如果候选字节对长度为1
                elif len(candidate) == 1:
                    # 将候选字节的 ID 添加到编码结果列表中
                    encoded.append(candidate[0])
                    l, r = r, len(w)
                else:
                    # 如果候选字节对不在词汇表中，则减小上下文窗口大小并重试
                    r -= 1
        # 返回编码结果列表
        return encoded
    def inverse_transform(self, codes):
        """
        Transform an encoded sequence of byte pair codeword IDs back into
        human-readable text.

        Parameters
        ----------
        codes : list of `N` lists
            A list of `N` lists. Each sublist is a collection of integer
            byte-pair token IDs representing a particular text string.

        Returns
        -------
        text: list of `N` strings
            The decoded strings corresponding to the `N` sublists in `codes`.

        Examples
        --------
        >>> B = BytePairEncoder(max_merges=100).fit("./example.txt")
        >>> encoded_tokens = B.transform("Hello! How are you 😁 ?")
        >>> encoded_tokens
        [[72, 879, 474, ...]]
        >>> B.inverse_transform(encoded_tokens)
        ["Hello! How are you 😁 ?"]
        """
        # 如果输入的codes是一个整数，将其转换为包含一个列表的形式
        if isinstance(codes[0], int):
            codes = [codes]

        decoded = []
        P = self.parameters

        # 遍历codes中的每个列表
        for code in codes:
            # 将每个token转换为对应的字节
            _bytes = [self.token2byte[t] if t > 255 else [t] for t in code]
            # 将字节列表展开为一维列表
            _bytes = [b for blist in _bytes for b in blist]
            # 将字节转换为字符并添加到decoded列表中
            decoded.append(bytes_to_chars(_bytes, encoding=P["encoding"]))
        return decoded

    @property
    def codebook(self):
        """
        A list of the learned byte pair codewords, decoded into human-readable
        format
        """
        # 返回学习到的字节对编码的人类可读形式
        return [
            self.inverse_transform(t)[0]
            for t in self.byte2token.keys()
            if isinstance(t, tuple)
        ]

    @property
    def tokens(self):
        """A list of the byte pair codeword IDs"""
        # 返回字节对编码的ID列表
        return list(self.token2byte.keys())
# 定义节点类，用于构建哈夫曼树
class Node(object):
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.left = None
        self.right = None

    # 重载大于运算符
    def __gt__(self, other):
        """Greater than"""
        if not isinstance(other, Node):
            return -1
        return self.val > other.val

    # 重载大于等于运算符
    def __ge__(self, other):
        """Greater than or equal to"""
        if not isinstance(other, Node):
            return -1
        return self.val >= other.val

    # 重载小于运算符
    def __lt__(self, other):
        """Less than"""
        if not isinstance(other, Node):
            return -1
        return self.val < other.val

    # 重载小于等于运算符
    def __le__(self, other):
        """Less than or equal to"""
        if not isinstance(other, Node):
            return -1
        return self.val <= other.val

# 定义哈夫曼编码器类
class HuffmanEncoder(object):
    # 为文本中的标记构建一个哈夫曼树，并计算每个标记的二进制编码。

    # 在哈夫曼编码中，出现频率更高的标记通常使用较少的位表示。哈夫曼编码产生了所有方法中对单独编码标记的最小期望码字长度。

    # 哈夫曼编码对应于通过二叉树的路径，其中1表示“向右移动”，0表示“向左移动”。与标准二叉树相反，哈夫曼树是自底向上构建的。构造始于初始化一个最小堆优先队列，其中包含语料库中的每个标记，优先级对应于标记频率。在每一步中，语料库中最不频繁的两个标记被移除，并成为一个父伪标记的子节点，其“频率”是其子节点频率的总和。将这个新的父伪标记添加到优先队列中，并递归重复这个过程，直到没有标记剩余。

    # 参数
    # text: 字符串列表或Vocabulary类的实例
    #     标记化的文本或用于构建哈夫曼编码的预训练Vocabulary对象。
    
    def fit(self, text):
        # 构建哈夫曼树
        self._build_tree(text)
        # 生成编码
        self._generate_codes()
    def transform(self, text):
        """
        Transform the words in `text` into their Huffman-code representations.

        Parameters
        ----------
        text: list of `N` strings
            The list of words to encode

        Returns
        -------
        codes : list of `N` binary strings
            The encoded words in `text`
        """
        # 如果输入的是字符串，则转换为包含该字符串的列表
        if isinstance(text, str):
            text = [text]
        # 遍历文本中的每个单词
        for token in set(text):
            # 如果单词不在 Huffman 树中，则抛出警告并跳过
            if token not in self._item2code:
                raise Warning("Token '{}' not in Huffman tree. Skipping".format(token))
        # 返回每个单词的 Huffman 编码
        return [self._item2code.get(t, None) for t in text]

    def inverse_transform(self, codes):
        """
        Transform an encoded sequence of bit-strings back into words.

        Parameters
        ----------
        codes : list of `N` binary strings
            A list of encoded bit-strings, represented as strings.

        Returns
        -------
        text: list of `N` strings
            The decoded text.
        """
        # 如果输入的是字符串，则转换为包含该字符串的列表
        if isinstance(codes, str):
            codes = [codes]
        # 遍历编码序列中的每个编码
        for code in set(codes):
            # 如果编码不在 Huffman 树中，则抛出警告并跳过
            if code not in self._code2item:
                raise Warning("Code '{}' not in Huffman tree. Skipping".format(code))
        # 返回每个编码对应的单词
        return [self._code2item.get(c, None) for c in codes]

    @property
    def tokens(self):
        """A list the unique tokens in `text`"""
        # 返回 Huffman 树中的所有唯一单词
        return list(self._item2code.keys())

    @property
    def codes(self):
        """A list with the Huffman code for each unique token in `text`"""
        # 返回 Huffman 树中每个唯一单词的 Huffman 编码
        return list(self._code2item.keys())

    def _counter(self, text):
        counts = {}
        # 统计文本中每个单词的出现次数
        for item in text:
            counts[item] = counts.get(item, 0) + 1
        return counts
    # 构建哈夫曼树
    def _build_tree(self, text):
        """Construct Huffman Tree"""
        # 初始化优先队列
        PQ = []

        # 如果输入是 Vocabulary 对象，则使用其 counts 属性
        if isinstance(text, Vocabulary):
            counts = text.counts
        else:
            # 否则使用 _counter 方法计算频率
            counts = self._counter(text)

        # 将每个字符及其频率作为节点加入优先队列
        for (k, c) in counts.items():
            PQ.append(Node(k, c))

        # 创建一个优先队列，优先级为频率
        heapq.heapify(PQ)

        # 构建哈夫曼树
        while len(PQ) > 1:
            node1 = heapq.heappop(PQ)  # 弹出频率最小的节点
            node2 = heapq.heappop(PQ)  # 弹出频率第二小的节点

            parent = Node(None, node1.val + node2.val)
            parent.left = node1
            parent.right = node2

            heapq.heappush(PQ, parent)

        self._root = heapq.heappop(PQ)

    # 生成编码
    def _generate_codes(self):
        current_code = ""
        self._item2code = {}
        self._code2item = {}
        self._build_code(self._root, current_code)

    # 递归构建编码
    def _build_code(self, root, current_code):
        if root is None:
            return

        if root.key is not None:
            # 将叶子节点的字符与编码对应存储
            self._item2code[root.key] = current_code
            self._code2item[current_code] = root.key
            return

        # 0 = 向左移动，1 = 向右移动
        self._build_code(root.left, current_code + "0")
        self._build_code(root.right, current_code + "1")
# 定义 Token 类，用于表示一个单词的计数和内容
class Token:
    def __init__(self, word):
        # 初始化单词计数为 0
        self.count = 0
        # 初始化单词内容
        self.word = word

    def __repr__(self):
        """A string representation of the token"""
        # 返回 Token 对象的字符串表示，包括单词内容和计数
        return "Token(word='{}', count={})".format(self.word, self.count)


# 定义 TFIDFEncoder 类，用于计算 TF-IDF 编码
class TFIDFEncoder:
    def __init__(
        self,
        vocab=None,
        lowercase=True,
        min_count=0,
        smooth_idf=True,
        max_tokens=None,
        input_type="files",
        filter_stopwords=True,
        filter_punctuation=True,
        tokenizer="words",
    ):
        # 初始化 TFIDFEncoder 对象的各种参数

    # 定义内部方法 _encode_document，用于对文档进行编码
    def _encode_document(
        self, doc, word2idx, idx2word, tokens, doc_count, bol_ix, eol_ix,
    ):
        """Perform tokenization and compute token counts for a single document"""
        # 获取超参数
        H = self.hyperparameters
        # 是否转换为小写
        lowercase = H["lowercase"]
        # 是否过滤停用词
        filter_stop = H["filter_stopwords"]
        # 是否过滤标点符号
        filter_punc = H["filter_punctuation"]

        # 如果输入类型为文件
        if H["input_type"] == "files":
            # 打开文件并读取内容
            with open(doc, "r", encoding=H["encoding"]) as handle:
                doc = handle.read()

        # 定义不同类型的分词器
        tokenizer_dict = {
            "words": tokenize_words,
            "characters": tokenize_chars,
            "whitespace": tokenize_whitespace,
            "bytes": tokenize_bytes_raw,
        }
        # 根据超参数选择相应的分词器
        tokenizer = tokenizer_dict[H["tokenizer"]]

        # 初始化单词数量
        n_words = 0
        # 将文档按行分割
        lines = doc.split("\n")
        # 遍历每一行
        for line in lines:
            # 对每一行进行分词
            words = tokenizer(
                line,
                lowercase=lowercase,
                filter_stopwords=filter_stop,
                filter_punctuation=filter_punc,
                encoding=H["encoding"],
            )
            # 过滤词汇表中不存在的词
            words = self._filter_vocab(words)
            # 更新单词数量
            n_words += len(words)

            # 遍历每个词
            for ww in words:
                # 如果词不在 word2idx 中，则添加
                if ww not in word2idx:
                    word2idx[ww] = len(tokens)
                    idx2word[len(tokens)] = ww
                    tokens.append(Token(ww))

                # 获取词的索引
                t_idx = word2idx[ww]
                # 更新词频
                tokens[t_idx].count += 1
                # 更新文档中词的出现次数
                doc_count[t_idx] = doc_count.get(t_idx, 0) + 1

            # 在每行开头和结尾添加 <bol> 和 <eol> 标签
            tokens[bol_ix].count += 1
            tokens[eol_ix].count += 1

            doc_count[bol_ix] = doc_count.get(bol_ix, 0) + 1
            doc_count[eol_ix] = doc_count.get(eol_ix, 0) + 1
        # 返回单词到索引的映射、索引到单词的映射、单词列表、文档中单词出现次数
        return word2idx, idx2word, tokens, doc_count
    # 保留前 N 个最频繁出现的词汇
    def _keep_top_n_tokens(self):
        # 获取最大词汇数
        N = self.hyperparameters["max_tokens"]
        # 初始化词汇计数、词汇到索引、索引到词汇的字典
        doc_counts, word2idx, idx2word = {}, {}, {}
        # 根据词汇出现次数排序词汇列表
        tokens = sorted(self._tokens, key=lambda x: x.count, reverse=True)

        # 重新索引前 N 个词汇...
        unk_ix = None
        for idx, tt in enumerate(tokens[:N]):
            word2idx[tt.word] = idx
            idx2word[idx] = tt.word

            # 如果 <unk> 不在前 N 个词汇中，将其添加进去，替换第 N 个最频繁出现的词汇，并相应调整 <unk> 的计数...
            if tt.word == "<unk>":
                unk_ix = idx

        # ... 最后，将所有被删除的词汇重新编码为 "<unk>"
        for tt in tokens[N:]:
            tokens[unk_ix].count += tt.count

        # ... 最后，重新为每个文档重新索引词汇计数
        for d_ix in self.term_freq.keys():
            doc_counts[d_ix] = {}
            for old_ix, d_count in self.term_freq[d_ix].items():
                word = self.idx2token[old_ix]
                new_ix = word2idx.get(word, unk_ix)
                doc_counts[d_ix][new_ix] = doc_counts[d_ix].get(new_ix, 0) + d_count

        # 更新词汇列表、词汇到索引、索引到词汇的字典以及文档词频
        self._tokens = tokens[:N]
        self.token2idx = word2idx
        self.idx2token = idx2word
        self.term_freq = doc_counts

        # 断言词汇列表长度不超过 N
        assert len(self._tokens) <= N
    def _drop_low_freq_tokens(self):
        """
        替换所有出现次数少于 `min_count` 的标记为 `<unk>` 标记。
        """
        H = self.hyperparameters
        # 获取 `<unk>` 标记的索引
        unk_token = self._tokens[self.token2idx["<unk>"]]
        # 获取 `<eol>` 标记的索引
        eol_token = self._tokens[self.token2idx["<eol>"]]
        # 获取 `<bol>` 标记的索引
        bol_token = self._tokens[self.token2idx["<bol>"]]
        # 初始化特殊标记列表
        tokens = [unk_token, eol_token, bol_token]

        # 初始化 `<unk>` 标记的索引
        unk_idx = 0
        # 初始化特殊标记到索引的映射
        word2idx = {"<unk>": 0, "<eol>": 1, "<bol>": 2}
        # 初始化索引到特殊标记的映射
        idx2word = {0: "<unk>", 1: "<eol>", 2: "<bol>"}
        # 初始化特殊标记集合
        special = {"<eol>", "<bol>", "<unk>"}

        # 遍历所有标记
        for tt in self._tokens:
            # 如果标记不是特殊标记
            if tt.word not in special:
                # 如果标记出现次数小于 `min_count`
                if tt.count < H["min_count"]:
                    # 将出现次数加到 `<unk>` 标记上
                    tokens[unk_idx].count += tt.count
                else:
                    # 更新标记到索引的映射
                    word2idx[tt.word] = len(tokens)
                    # 更新索引到标记的映射
                    idx2word[len(tokens)] = tt.word
                    # 添加标记到列表中
                    tokens.append(tt)

        # 重新索引文档计数
        doc_counts = {}
        for d_idx in self.term_freq.keys():
            doc_counts[d_idx] = {}
            for old_idx, d_count in self.term_freq[d_idx].items():
                word = self.idx2token[old_idx]
                new_idx = word2idx.get(word, unk_idx)
                doc_counts[d_idx][new_idx] = doc_counts[d_idx].get(new_idx, 0) + d_count

        # 更新标记列表
        self._tokens = tokens
        # 更新标记到索引的映射
        self.token2idx = word2idx
        # 更新索引到标记的映射
        self.idx2token = idx2word
        # 更新文档计数
        self.term_freq = doc_counts
    # 对 tokens 进行排序，按字母顺序排序并重新编码
    def _sort_tokens(self):
        # 初始化索引
        ix = 0
        # 初始化 token 到索引和索引到 token 的字典
        token2idx, idx2token, = (
            {},
            {},
        )
        # 特殊 token 列表
        special = ["<eol>", "<bol>", "<unk>"]
        # 对 token2idx 字典中的键进行排序
        words = sorted(self.token2idx.keys())
        # 初始化 term_freq 字典
        term_freq = {d: {} for d in self.term_freq.keys()}

        # 遍历排序后的 tokens
        for w in words:
            # 如果当前 token 不在特殊 token 列表中
            if w not in special:
                # 获取当前 token 的旧索引
                old_ix = self.token2idx[w]
                # 更新 token2idx 和 idx2token 字典
                token2idx[w], idx2token[ix] = ix, w
                # 更新 term_freq 字典
                for d in self.term_freq.keys():
                    if old_ix in self.term_freq[d]:
                        count = self.term_freq[d][old_ix]
                        term_freq[d][ix] = count
                ix += 1

        # 处理特殊 token
        for w in special:
            token2idx[w] = len(token2idx)
            idx2token[len(idx2token)] = w

        # 更新对象的 token2idx、idx2token、term_freq 和 vocab_counts 属性
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.term_freq = term_freq
        self.vocab_counts = Counter({t.word: t.count for t in self._tokens})
    def _calc_idf(self):
        """
        计算语料库中每个标记的（平滑的）逆文档频率。

        对于一个单词标记 `w`，IDF 简单地定义为

            IDF(w) = log ( |D| / |{ d in D: w in d }| ) + 1

        其中 D 是语料库中所有文档的集合，

            D = {d1, d2, ..., dD}

        如果 `smooth_idf` 为 True，我们对包含给定单词的文档数量进行加法平滑处理，相当于假设存在第 D+1 个文档，其中包含语料库中的每个单词：

            SmoothedIDF(w) = log ( |D| + 1 / [1 + |{ d in D: w in d }|] ) + 1
        """
        inv_doc_freq = {}
        smooth_idf = self.hyperparameters["smooth_idf"]
        tf, doc_idxs = self.term_freq, self._idx2doc.keys()

        D = len(self._idx2doc) + int(smooth_idf)
        for word, w_ix in self.token2idx.items():
            d_count = int(smooth_idf)
            d_count += np.sum([1 if w_ix in tf[d_ix] else 0 for d_ix in doc_idxs])
            inv_doc_freq[w_ix] = 1 if d_count == 0 else np.log(D / d_count) + 1
        self.inv_doc_freq = inv_doc_freq
    def transform(self, ignore_special_chars=True):
        """
        生成文本语料库的词频-逆文档频率编码。

        Parameters
        ----------
        ignore_special_chars : bool
            是否从最终的tfidf编码中删除与"<eol>", "<bol>", "<unk>"标记对应的列。默认为True。

        Returns
        -------
        tfidf : numpy array of shape `(D, M [- 3])`
            编码后的语料库，每行对应一个文档，每列对应一个标记ID。如果`ignore_special_chars`为False，则在`idx2token`属性中存储列号与标记之间的映射。否则，映射不准确。
        """
        D, N = len(self._idx2doc), len(self._tokens)
        # 初始化词频矩阵和逆文档频率矩阵
        tf = np.zeros((D, N))
        idf = np.zeros((D, N))

        # 遍历文档索引
        for d_ix in self._idx2doc.keys():
            # 获取文档中的词和词频
            words, counts = zip(*self.term_freq[d_ix].items())
            # 创建文档索引数组
            docs = np.ones(len(words), dtype=int) * d_ix
            # 更新词频矩阵
            tf[docs, words] = counts

        # 获取所有词的排序列表
        words = sorted(self.idx2token.keys())
        # 根据词的逆文档频率创建矩阵
        idf = np.tile(np.array([self.inv_doc_freq[w] for w in words]), (D, 1))
        # 计算tfidf矩阵
        tfidf = tf * idf

        # 如果忽略特殊字符
        if ignore_special_chars:
            # 获取特殊字符的索引
            idxs = [
                self.token2idx["<unk>"],
                self.token2idx["<eol>"],
                self.token2idx["<bol>"],
            ]
            # 从tfidf矩阵中删除特殊字符列
            tfidf = np.delete(tfidf, idxs, 1)

        # 返回tfidf矩阵
        return tfidf
# 定义一个名为 Vocabulary 的类
class Vocabulary:
    # 初始化方法，设置类的属性
    def __init__(
        self,
        lowercase=True,  # 是否将单词转换为小写，默认为True
        min_count=None,  # 单词最小出现次数，默认为None
        max_tokens=None,  # 最大单词数量，默认为None
        filter_stopwords=True,  # 是否过滤停用词，默认为True
        filter_punctuation=True,  # 是否过滤标点符号，默认为True
        tokenizer="words",  # 分词器类型，默认为"words"
    ):
        """
        用于编译和编码文本语料库中唯一标记的对象。

        参数
        ----------
        lowercase : bool
            是否在标记化之前将每个字符串转换为小写。
            默认为 True。
        min_count : int
            标记必须出现的最小次数才能包含在词汇表中。
            如果为 `None`，则在词汇表中包含来自 `corpus_fp` 的所有标记。
            默认为 None。
        max_tokens : int
            仅将出现次数超过 `min_count` 的前 `max_tokens` 个最常见标记添加到词汇表中。
            如果为 None，则添加所有出现次数超过 `min_count` 的标记。
            默认为 None。
        filter_stopwords : bool
            是否在对语料库中的单词进行编码之前删除停用词。
            默认为 True。
        filter_punctuation : bool
            是否在对语料库中的单词进行编码之前删除标点符号。
            默认为 True。
        tokenizer : {'whitespace', 'words', 'characters', 'bytes'}
            在将字符串映射到标记时要遵循的策略。 
            `'whitespace'` 标记化器在空格字符处拆分字符串。
            `'words'` 标记化器使用“单词”正则表达式拆分字符串。
            `'characters'` 标记化器将字符串拆分为单个字符。
            `'bytes'` 标记化器将字符串拆分为一组单个字节。
        """
        self.hyperparameters = {
            "id": "Vocabulary",
            "encoding": None,
            "corpus_fps": None,
            "lowercase": lowercase,
            "min_count": min_count,
            "max_tokens": max_tokens,
            "filter_stopwords": filter_stopwords,
            "filter_punctuation": filter_punctuation,
            "tokenizer": tokenizer,
        }

    def __len__(self):
        """返回词汇表中标记的数量"""
        return len(self._tokens)
    # 返回一个迭代器，用于遍历词汇表中的标记
    def __iter__(self):
        return iter(self._tokens)

    # 判断给定的单词是否是词汇表中的一个标记
    def __contains__(self, word):
        return word in self.token2idx

    # 根据键返回词汇表中的标记（如果键是整数）或索引（如果键是字符串）
    def __getitem__(self, key):
        if isinstance(key, str):
            return self._tokens[self.token2idx[key]]
        if isinstance(key, int):
            return self._tokens[key]

    # 返回词汇表中唯一单词标记的数量
    @property
    def n_tokens(self):
        return len(self.token2idx)

    # 返回语料库中单词的总数
    @property
    def n_words(self):
        return sum(self.counts.values())

    # 返回词汇表中唯一单词标记的形状
    @property
    def shape(self):
        return self._tokens.shape

    # 返回语料库中出现频率最高的前n个标记
    def most_common(self, n=5):
        return self.counts.most_common()[:n]

    # 返回在语料库中出现k次的所有标记
    def words_with_count(self, k):
        return [w for w, c in self.counts.items() if c == k]
    def filter(self, words, unk=True):  # noqa: A003
        """
        Filter (or replace) any word in `words` that is not present in
        `Vocabulary`.

        Parameters
        ----------
        words : list of strs
            A list of words to filter
        unk : bool
            Whether to replace any out of vocabulary words in `words` with the
            ``<unk>`` token (True) or skip them entirely (False).  Default is
            True.

        Returns
        -------
        filtered : list of strs
            The list of words filtered against the words in Vocabulary.
        """
        # 如果 unk 为 True，则将不在 Vocabulary 中的单词替换为 "<unk>"，否则跳过
        if unk:
            return [w if w in self else "<unk>" for w in words]
        # 如果 unk 为 False，则只保留在 Vocabulary 中的单词
        return [w for w in words if w in self]

    def words_to_indices(self, words):
        """
        Convert the words in `words` to their token indices. If a word is not
        in the vocabulary, return the index for the ``<unk>`` token

        Parameters
        ----------
        words : list of strs
            A list of words to filter

        Returns
        -------
        indices : list of ints
            The token indices for each word in `words`
        """
        # 获取 "<unk>" 的索引
        unk_ix = self.token2idx["<unk>"]
        # 获取是否转换为小写的设置
        lowercase = self.hyperparameters["lowercase"]
        # 如果需要转换为小写，则将单词列表中的单词转换为小写
        words = [w.lower() for w in words] if lowercase else words
        # 将单词转换为它们在词汇表中的索引，如果不在词汇表中，则返回 "<unk>" 的索引
        return [self.token2idx[w] if w in self else unk_ix for w in words]

    def indices_to_words(self, indices):
        """
        Convert the indices in `indices` to their word values. If an index is
        not in the vocabulary, return the ``<unk>`` token.

        Parameters
        ----------
        indices : list of ints
            The token indices for each word in `words`

        Returns
        -------
        words : list of strs
            The word strings corresponding to each token index in `indices`
        """
        # 设置 "<unk>" 标记
        unk = "<unk>"
        # 将索引转换为对应的单词，如果索引不在词汇表中，则返回 "<unk>"
        return [self.idx2token[i] if i in self.idx2token else unk for i in indices]
    # 保留词汇表中出现频率最高的前 N 个词的索引
    def _keep_top_n_tokens(self):
        # 初始化空字典，用于存储词汇表中词语到索引的映射关系
        word2idx, idx2word = {}, {}
        # 获取最大词汇量 N
        N = self.hyperparameters["max_tokens"]
        # 根据词频对词汇表中的词进行排序
        tokens = sorted(self._tokens, key=lambda x: x.count, reverse=True)

        # 重新索引前 N 个词...
        unk_ix = None
        for idx, tt in enumerate(tokens[:N]):
            # 将词语和对应的索引存入字典中
            word2idx[tt.word] = idx
            idx2word[idx] = tt.word

            # 如果词语是 "<unk>"，记录其索引
            if tt.word == "<unk>":
                unk_ix = idx

        # ... 如果 "<unk>" 不在前 N 个词中，将其添加进去，替换第 N 个最常见的词，并相应调整 "<unk>" 的计数 ...
        if unk_ix is None:
            unk_ix = self.token2idx["<unk>"]
            old_count = tokens[N - 1].count
            tokens[N - 1] = self._tokens[unk_ix]
            tokens[N - 1].count += old_count
            word2idx["<unk>"] = N - 1
            idx2word[N - 1] = "<unk>"

        # ... 将所有被删除的词重新编码为 "<unk>"
        for tt in tokens[N:]:
            tokens[unk_ix].count += tt.count

        # 更新词汇表为前 N 个词
        self._tokens = tokens[:N]
        self.token2idx = word2idx
        self.idx2token = idx2word

        # 断言词汇表长度不超过 N
        assert len(self._tokens) <= N
    def _drop_low_freq_tokens(self):
        """
        Replace all tokens that occur less than `min_count` with the `<unk>`
        token.
        """
        # 获取 `<unk>` token 的索引
        unk_idx = 0
        # 获取 `<unk>`、`<eol>`、`<bol>` token 对应的索引
        unk_token = self._tokens[self.token2idx["<unk>"]]
        eol_token = self._tokens[self.token2idx["<eol>"]]
        bol_token = self._tokens[self.token2idx["<bol>"]]

        # 获取超参数
        H = self.hyperparameters
        # 初始化特殊 token 列表
        tokens = [unk_token, eol_token, bol_token]
        # 初始化特殊 token 到索引的映射
        word2idx = {"<unk>": 0, "<eol>": 1, "<bol>": 2}
        # 初始化索引到特殊 token 的映射
        idx2word = {0: "<unk>", 1: "<eol>", 2: "<bol>"}
        # 特殊 token 集合
        special = {"<eol>", "<bol>", "<unk>"}

        # 遍历所有 token
        for tt in self._tokens:
            # 如果 token 不是特殊 token
            if tt.word not in special:
                # 如果 token 出现次数小于 min_count
                if tt.count < H["min_count"]:
                    # 将出现次数小于 min_count 的 token 替换为 `<unk>` token
                    tokens[unk_idx].count += tt.count
                else:
                    # 更新 token 到索引的映射
                    word2idx[tt.word] = len(tokens)
                    # 更新索引到 token 的映射
                    idx2word[len(tokens)] = tt.word
                    # 添加当前 token 到 tokens 列表中
                    tokens.append(tt)

        # 更新 tokens 列表
        self._tokens = tokens
        # 更新 token 到索引的映射
        self.token2idx = word2idx
        # 更新索引到 token 的映射
        self.idx2token = idx2word
```