# `.\models\flaubert\tokenization_flaubert.py`

```
# 定义一个函数用于将文本转换为 Unicode 格式（如果尚未转换），假设输入是 UTF-8 编码的文本
def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.
    """
    # 定义函数 ensure_text，确保输入的文本 s 是字符串类型，并按指定编码和错误处理方式解码为文本
    def ensure_text(s, encoding="utf-8", errors="strict"):
        # 如果 s 是字节类型，则解码为字符串
        if isinstance(s, bytes):
            return s.decode(encoding, errors)
        # 如果 s 已经是字符串类型，则直接返回
        elif isinstance(s, str):
            return s
        # 如果 s 不是预期的字节或字符串类型，则引发类型错误异常
        else:
            raise TypeError(f"not expecting type '{type(s)}'")

    # 调用 ensure_text 函数，确保输入的 text 是字符串类型，使用 utf-8 编码，忽略解码中的错误
    return ensure_text(text, encoding="utf-8", errors="ignore")
# Copied from transformers.models.xlm.tokenization_xlm.get_pairs
def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    pairs = set()
    prev_char = word[0]  # 获取单词的第一个字符
    for char in word[1:]:  # 迭代单词中除第一个字符外的所有字符
        pairs.add((prev_char, char))  # 将相邻字符组成的元组添加到集合中
        prev_char = char  # 更新前一个字符为当前字符
    return pairs  # 返回字符对的集合


# Copied from transformers.models.xlm.tokenization_xlm.replace_unicode_punct
def replace_unicode_punct(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    """
    text = text.replace("，", ",")  # 替换中文逗号为英文逗号
    text = re.sub(r"。\s*", ". ", text)  # 替换中文句号后的空白为单个空格
    text = text.replace("、", ",")  # 替换中文顿号为英文逗号
    text = text.replace("”", '"')  # 替换中文右双引号为英文右双引号
    text = text.replace("“", '"')  # 替换中文左双引号为英文左双引号
    text = text.replace("∶", ":")  # 替换中文分号为英文冒号
    text = text.replace("：", ":")  # 替换中文冒号为英文冒号
    text = text.replace("？", "?")  # 替换中文问号为英文问号
    text = text.replace("《", '"')  # 替换中文书名号为英文双引号
    text = text.replace("》", '"')  # 替换中文书名号为英文双引号
    text = text.replace("）", ")")  # 替换中文右括号为英文右括号
    text = text.replace("！", "!")  # 替换中文感叹号为英文感叹号
    text = text.replace("（", "(")  # 替换中文左括号为英文左括号
    text = text.replace("；", ";")  # 替换中文分号为英文分号
    text = text.replace("１", "1")  # 替换全角数字为半角数字
    text = text.replace("」", '"')  # 替换中文右引号为英文双引号
    text = text.replace("「", '"')  # 替换中文左引号为英文双引号
    text = text.replace("０", "0")  # 替换全角数字为半角数字
    text = text.replace("３", "3")  # 替换全角数字为半角数字
    text = text.replace("２", "2")  # 替换全角数字为半角数字
    text = text.replace("５", "5")  # 替换全角数字为半角数字
    text = text.replace("６", "6")  # 替换全角数字为半角数字
    text = text.replace("９", "9")  # 替换全角数字为半角数字
    text = text.replace("７", "7")  # 替换全角数字为半角数字
    text = text.replace("８", "8")  # 替换全角数字为半角数字
    text = text.replace("４", "4")  # 替换全角数字为半角数字
    text = re.sub(r"．\s*", ". ", text)  # 替换中文句号后的空白为单个空格
    text = text.replace("～", "~")  # 替换中文波浪号为英文波浪号
    text = text.replace("’", "'")  # 替换中文右单引号为英文右单引号
    text = text.replace("…", "...")  # 替换中文省略号为英文省略号
    text = text.replace("━", "-")  # 替换中文长破折号为英文短破折号
    text = text.replace("〈", "<")  # 替换中文左尖括号为英文左尖括号
    text = text.replace("〉", ">")  # 替换中文右尖括号为英文右尖括号
    text = text.replace("【", "[")  # 替换中文左方括号为英文左方括号
    text = text.replace("】", "]")  # 替换中文右方括号为英文右方括号
    text = text.replace("％", "%")  # 替换全角百分号为半角百分号
    return text  # 返回替换后的文本


# Copied from transformers.models.xlm.tokenization_xlm.remove_non_printing_char
def remove_non_printing_char(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
    """
    output = []  # 创建空列表，用于存储输出的字符
    for char in text:  # 遍历输入文本的每一个字符
        cat = unicodedata.category(char)  # 获取当前字符的 Unicode 分类
        if cat.startswith("C"):  # 如果当前字符是控制字符
            continue  # 跳过当前字符
        output.append(char)  # 将非控制字符添加到输出列表中
    return "".join(output)  # 返回连接后的输出字符串


class FlaubertTokenizer(PreTrainedTokenizer):
    """
    Construct a Flaubert tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

    - Moses preprocessing and tokenization.
    - Normalizing all inputs text.
    - The arguments `special_tokens` and the function `set_special_tokens`, can be used to add additional symbols (like
      "__classify__") to a vocabulary.
    - The argument `do_lowercase` controls lower casing (automatically set for pretrained vocabularies).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    # 定义一个类，继承自BertPreTrainedModel类，用于语言模型的预训练
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Vocabulary file.  # 词汇表文件的路径
        merges_file (`str`):
            Merges file.  # 合并文件的路径
        do_lowercase (`bool`, *optional*, defaults to `False`):
            Controls lower casing.  # 控制是否进行小写处理的布尔值，默认为False
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.  # 未知标记，用于表示词汇表中不存在的词语，默认为"<unk>"
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>
            # 在预训练期间使用的序列开始标记，也可用作序列分类器标记
            <Tip>

            在使用特殊标记构建序列时，这不是用于序列开头的标记。实际使用的标记是`cls_token`。

            </Tip>
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
            # 分隔符标记，在构建来自多个序列的序列时使用，例如用于序列分类或问题回答
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
            # 填充标记，在批处理不同长度序列时使用
        cls_token (`str`, *optional*, defaults to `"</s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
            # 分类器标记，在进行序列分类时使用，整个序列而不是每个标记的分类
        mask_token (`str`, *optional*, defaults to `"<special1>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
            # 用于掩码值的标记，在使用掩码语言建模训练模型时使用，模型会预测这种标记
        additional_special_tokens (`List[str]`, *optional*, defaults to `['<special0>', '<special1>', '<special2>', '<special3>', '<special4>', '<special5>', '<special6>', '<special7>', '<special8>', '<special9>']`):
            List of additional special tokens.
            # 额外特殊标记的列表
        lang2id (`Dict[str, int]`, *optional*):
            Dictionary mapping languages string identifiers to their IDs.
            # 将语言字符串标识符映射到其ID的字典
        id2lang (`Dict[int, str]`, *optional*):
            Dictionary mapping language IDs to their string identifiers.
            # 将语言ID映射到其字符串标识符的字典
    """

    vocab_files_names = VOCAB_FILES_NAMES
    # 从预定义的全局变量中获取词汇表文件名列表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 从预定义的全局变量中获取预训练模型的词汇表文件映射
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 从预定义的全局变量中获取预训练模型的初始化配置
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 从预定义的全局变量中获取预训练模型的最大输入尺寸
    def __init__(
        self,
        vocab_file,
        merges_file,
        do_lowercase=False,
        unk_token="<unk>",
        bos_token="<s>",
        sep_token="</s>",
        pad_token="<pad>",
        cls_token="</s>",
        mask_token="<special1>",
        additional_special_tokens=[
            "<special0>",
            "<special1>",
            "<special2>",
            "<special3>",
            "<special4>",
            "<special5>",
            "<special6>",
            "<special7>",
            "<special8>",
            "<special9>",
        ],
        lang2id=None,
        id2lang=None,
        **kwargs,
    ):
        # 检查是否有传入`do_lowercase_and_remove_accent`，但该参数不会起作用于当前类，始终设置为`False`
        do_lowercase_and_remove_accent = kwargs.pop("do_lowercase_and_remove_accent", None)
        if do_lowercase_and_remove_accent is not None:
            logger.warning(
                "`do_lowercase_and_remove_accent` is passed as a keyword argument, but this won't do anything."
                " `FlaubertTokenizer` will always set it to `False`."
            )
        # 始终将`do_lowercase_and_remove_accent`设置为`False`
        self.do_lowercase_and_remove_accent = False

        # 是否将输入文本转换为小写的标志
        self.do_lowercase = do_lowercase

        # 尝试导入`sacremoses`库，如果导入失败则抛出`ImportError`异常
        try:
            import sacremoses
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use FlaubertTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        # 初始化`sacremoses`模块
        self.sm = sacremoses

        # 缓存`sacremoses.MosesPunctNormalizer`实例
        self.cache_moses_punct_normalizer = {}
        # 缓存`sacremoses.MosesTokenizer`实例
        self.cache_moses_tokenizer = {}
        
        # 需要使用自定义分词器的语言集合
        self.lang_with_custom_tokenizer = {"zh", "th", "ja"}
        
        # 设置语言到ID的映射
        self.lang2id = lang2id
        # 设置ID到语言的映射
        self.id2lang = id2lang
        # 如果`lang2id`和`id2lang`都不为`None`，则断言它们长度相同
        if lang2id is not None and id2lang is not None:
            assert len(lang2id) == len(id2lang)

        # 日语分词器实例
        self.ja_word_tokenizer = None
        # 中文分词器实例
        self.zh_word_tokenizer = None

        # 使用UTF-8编码打开词汇表文件，并将其加载为字典`self.encoder`
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        
        # 构建反向词典`self.decoder`
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # 使用UTF-8编码打开BPE合并文件，并解析为合并操作序列`merges`
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        
        # 构建BPE合并操作的排名字典`self.bpe_ranks`
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        
        # 缓存
        self.cache = {}

        # 调用父类的初始化方法，设置特殊token等参数
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            lang2id=lang2id,
            id2lang=id2lang,
            **kwargs,
        )

    @property
    # 从`transformers.models.xlm.tokenization_xlm.XLMTokenizer.do_lower_case`复制而来
    def do_lower_case(self):
        # 返回是否将输入文本转换为小写的标志
        return self.do_lowercase_and_remove_accent
    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.moses_punct_norm 复制而来
    def moses_punct_norm(self, text, lang):
        # 如果语言不在缓存中，则创建一个新的 MosesPunctNormalizer 对象
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = self.sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        else:
            # 否则从缓存中获取已存在的 MosesPunctNormalizer 对象
            punct_normalizer = self.cache_moses_punct_normalizer[lang]
        # 调用 MosesPunctNormalizer 对象的 normalize 方法进行标点符号的规范化处理
        return punct_normalizer.normalize(text)
    
    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.moses_tokenize 复制而来
    def moses_tokenize(self, text, lang):
        # 如果语言不在缓存中，则创建一个新的 MosesTokenizer 对象
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = self.sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        else:
            # 否则从缓存中获取已存在的 MosesTokenizer 对象
            moses_tokenizer = self.cache_moses_tokenizer[lang]
        # 调用 MosesTokenizer 对象的 tokenize 方法对文本进行分词处理
        return moses_tokenizer.tokenize(text, return_str=False, escape=False)
    
    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.moses_pipeline 复制而来
    def moses_pipeline(self, text, lang):
        # 调用 replace_unicode_punct 方法处理文本中的 Unicode 标点符号
        text = replace_unicode_punct(text)
        # 调用 self.moses_punct_norm 方法对文本进行 Moses 标点符号规范化处理
        text = self.moses_punct_norm(text, lang)
        # 调用 remove_non_printing_char 方法移除文本中的非打印字符
        text = remove_non_printing_char(text)
        # 返回处理后的文本
        return text
    
    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.ja_tokenize 复制而来
    def ja_tokenize(self, text):
        # 如果 self.ja_word_tokenizer 为空，则尝试导入 Mykytea 并创建一个新的 Mykytea 对象
        if self.ja_word_tokenizer is None:
            try:
                import Mykytea
    
                self.ja_word_tokenizer = Mykytea.Mykytea(
                    f"-model {os.path.expanduser('~')}/local/share/kytea/model.bin"
                )
            except (AttributeError, ImportError):
                # 如果导入失败，则记录错误信息并引发异常
                logger.error(
                    "Make sure you install KyTea (https://github.com/neubig/kytea) and it's python wrapper"
                    " (https://github.com/chezou/Mykytea-python) with the following steps"
                )
                logger.error("1. git clone git@github.com:neubig/kytea.git && cd kytea")
                logger.error("2. autoreconf -i")
                logger.error("3. ./configure --prefix=$HOME/local")
                logger.error("4. make && make install")
                logger.error("5. pip install kytea")
                raise
        # 调用 Mykytea 对象的 getWS 方法对文本进行日语分词处理，并返回分词结果列表
        return list(self.ja_word_tokenizer.getWS(text))
    
    @property
    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.vocab_size 复制而来
    def vocab_size(self):
        # 返回 self.encoder 的长度，即词汇表的大小
        return len(self.encoder)
    
    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.get_vocab 复制而来
    def get_vocab(self):
        # 将 self.encoder 和 self.added_tokens_encoder 合并为一个字典，并返回
        return dict(self.encoder, **self.added_tokens_encoder)
    
    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.bpe 复制而来
    # 使用 BPE 算法对输入的 token 进行处理
    def bpe(self, token):
        # 将 token 转换为包含特殊结束符的元组形式
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        
        # 如果 token 已经被缓存，直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        
        # 获取 token 的所有可能 bigram 组合
        pairs = get_pairs(word)

        # 如果没有 bigram 组合，直接返回带结束符的 token
        if not pairs:
            return token + "</w>"

        # 循环处理直到无法再合并为止
        while True:
            # 找到当前权重最小的 bigram
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            
            # 如果该 bigram 不在预定义的权重中，停止循环
            if bigram not in self.bpe_ranks:
                break
            
            # 分解 word 中的 bigram 并重新组合
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            
            # 如果 word 只剩一个元素，停止循环
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        
        # 将 word 转换为字符串形式
        word = " ".join(word)
        
        # 处理特定的结束符情况
        if word == "\n  </w>":
            word = "\n</w>"
        
        # 将处理后的结果缓存起来
        self.cache[token] = word
        
        # 返回处理后的 token
        return word

    # 预处理文本，替换特殊标点符号并标准化 Unicode 格式
    def preprocess_text(self, text):
        text = text.replace("``", '"').replace("''", '"')
        text = convert_to_unicode(text)
        text = unicodedata.normalize("NFC", text)

        # 如果需要转换为小写，执行转换操作
        if self.do_lowercase:
            text = text.lower()

        # 返回预处理后的文本
        return text

    # 将输入文本进行分词处理
    def _tokenize(self, text, bypass_tokenizer=False):
        """
        Tokenize a string given language code using Moses.

        Details of tokenization:

            - [sacremoses](https://github.com/alvations/sacremoses): port of Moses
            - Install with `pip install sacremoses`

        Args:
            - bypass_tokenizer: Allow users to preprocess and tokenize the sentences externally (default = False)
              (bool). If True, we only apply BPE.

        Returns:
            List of tokens.
        """
        # 设定语言为法语
        lang = "fr"
        
        # 如果语言码存在且不在预加载的语言映射中，记录错误日志
        if lang and self.lang2id and lang not in self.lang2id:
            logger.error(
                "Supplied language code not found in lang2id mapping. Please check that your language is supported by"
                " the loaded pretrained model."
            )

        # 根据参数决定是否绕过默认的分词器
        if bypass_tokenizer:
            text = text.split()
        else:
            text = self.preprocess_text(text)  # 预处理文本
            text = self.moses_pipeline(text, lang=lang)  # 使用 Moses 处理流水线
            text = self.moses_tokenize(text, lang=lang)  # 使用 Moses 进行分词

        split_tokens = []
        # 对每个 token 进行 BPE 处理并扩展到 split_tokens 列表中
        for token in text:
            if token:
                split_tokens.extend(list(self.bpe(token).split(" ")))

        # 返回处理后的 token 列表
        return split_tokens
    def _convert_token_to_id(self, token):
        """Converts a token (str) into an id using the vocabulary."""
        # Return the ID corresponding to the token from the encoder dictionary; if token not found, return the ID for unknown token (unk_token).
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) into a token (str) using the vocabulary."""
        # Return the token corresponding to the index from the decoder dictionary; if index not found, return the unknown token (unk_token).
        return self.decoder.get(index, self.unk_token)

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings) into a single string."""
        # Concatenate tokens into a single string, replace "</w>" with space, and strip leading/trailing whitespace.
        out_string = "".join(tokens).replace("</w>", " ").strip()
        return out_string

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens. An XLM sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens added.

        """
        bos = [self.bos_token_id]  # Define the beginning-of-sequence token ID
        sep = [self.sep_token_id]  # Define the separator token ID

        if token_ids_1 is None:
            return bos + token_ids_0 + sep  # Return single sequence with special tokens
        return bos + token_ids_0 + sep + token_ids_1 + sep  # Return pair of sequences with special tokens

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve a mask of 1s and 0s indicating the presence of special tokens in the input sequences.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs corresponding to the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs corresponding to the second sequence.
            already_has_special_tokens (`bool`, *optional*):
                Whether the input lists already include special tokens.

        Returns:
            `List[int]`: List where each element is 1 if the corresponding token is special, otherwise 0.

        """
    # 继承父类的方法，获取特殊标记掩码，当已经存在特殊标记时调用
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    # 从两个传入的序列中创建用于序列对分类任务的类型标记。XLM 序列对标记的格式如下：
    #
    # ```
    # 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
    # | 第一个序列    | 第二个序列 |
    # ```
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLM sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]  # 分隔符标记的 ID
        cls = [self.cls_token_id]  # 类别标记的 ID
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]  # 返回第一个序列部分的标记类型 ID 列表（全为0）

        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.save_vocabulary 复制
    # 将词汇表保存到指定目录下的文件中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建词汇文件的路径，包括可选的前缀和文件名
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        
        # 构建合并文件的路径，包括可选的前缀和文件名
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将编码器（encoder）对象以 JSON 格式写入词汇文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 初始化索引值
        index = 0
        # 将 BPE（Byte Pair Encoding）标记和它们的索引按升序排序后写入合并文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                # 如果索引不连续，记录警告信息
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回保存的词汇文件路径和合并文件路径
        return vocab_file, merge_file

    # 从对象状态中获取数据，用于序列化对象
    # 参考自 transformers.models.xlm.tokenization_xlm.XLMTokenizer.__getstate__
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sm"] = None
        return state

    # 从序列化数据中设置对象状态，用于反序列化对象
    # 参考自 transformers.models.xlm.tokenization_xlm.XLMTokenizer.__setstate__
    def __setstate__(self, d):
        self.__dict__ = d

        # 尝试导入 sacremoses 库，如果失败则抛出 ImportError
        try:
            import sacremoses
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        # 将导入的 sacremoses 赋值给对象属性 self.sm
        self.sm = sacremoses
```