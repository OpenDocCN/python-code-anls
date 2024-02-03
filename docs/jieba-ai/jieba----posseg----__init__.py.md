# `jieba\jieba\posseg\__init__.py`

```
# 导入必要的模块和库
from __future__ import absolute_import, unicode_literals
import pickle
import re
import jieba
from .viterbi import viterbi
from .._compat import *

# 定义文件名常量
PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"
CHAR_STATE_TAB_P = "char_state_tab.p"

# 定义正则表达式模式
re_han_detail = re.compile("([\u4E00-\u9FD5]+)")
re_skip_detail = re.compile("([\.0-9]+|[a-zA-Z0-9]+)")
re_han_internal = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")
re_skip_internal = re.compile("(\r\n|\s)")
re_eng = re.compile("[a-zA-Z0-9]+")
re_num = re.compile("[\.0-9]+")
re_eng1 = re.compile('^[a-zA-Z0-9]$', re.U)

# 加载模型数据
def load_model():
    # 从资源中加载概率数据
    start_p = pickle.load(get_module_res("posseg", PROB_START_P))
    trans_p = pickle.load(get_module_res("posseg", PROB_TRANS_P))
    emit_p = pickle.load(get_module_res("posseg", PROB_EMIT_P))
    state = pickle.load(get_module_res("posseg", CHAR_STATE_TAB_P))
    return state, start_p, trans_p, emit_p

# 根据不同平台加载模型数据
if sys.platform.startswith("java"):
    char_state_tab_P, start_P, trans_P, emit_P = load_model()
else:
    from .char_state_tab import P as char_state_tab_P
    from .prob_start import P as start_P
    from .prob_trans import P as trans_P
    from .prob_emit import P as emit_P

# 定义一个类
class pair(object):

    def __init__(self, word, flag):
        self.word = word
        self.flag = flag

    def __unicode__(self):
        return '%s/%s' % (self.word, self.flag)

    def __repr__(self):
        return 'pair(%r, %r)' % (self.word, self.flag)

    def __str__(self):
        if PY2:
            return self.__unicode__().encode(default_encoding)
        else:
            return self.__unicode__()

    def __iter__(self):
        return iter((self.word, self.flag))

    def __lt__(self, other):
        return self.word < other.word

    def __eq__(self, other):
        return isinstance(other, pair) and self.word == other.word and self.flag == other.flag

    def __hash__(self):
        return hash(self.word)
    # 定义一个方法，用于将对象转换为 Unicode 字符串，然后根据指定的编码格式进行编码
    def encode(self, arg):
        # 调用对象的 __unicode__() 方法将对象转换为 Unicode 字符串，然后根据指定的编码格式进行编码
        return self.__unicode__().encode(arg)
# 定义一个名为 POSTokenizer 的类
class POSTokenizer(object):

    # 初始化方法，接受一个 tokenizer 参数，默认为 None
    def __init__(self, tokenizer=None):
        # 如果未提供 tokenizer 参数，则使用 jieba.Tokenizer() 创建一个分词器
        self.tokenizer = tokenizer or jieba.Tokenizer()
        # 调用 load_word_tag 方法，加载分词器的字典文件
        self.load_word_tag(self.tokenizer.get_dict_file())

    # 定义对象的字符串表示形式
    def __repr__(self):
        # 返回包含 tokenizer 属性值的字符串
        return '<POSTokenizer tokenizer=%r>' % self.tokenizer

    # 定义 __getattr__ 方法，用于获取对象的属性
    def __getattr__(self, name):
        # 如果属性名为 'cut_for_search', 'lcut_for_search', 'tokenize' 中的一个，抛出 NotImplementedError
        if name in ('cut_for_search', 'lcut_for_search', 'tokenize'):
            raise NotImplementedError
        # 否则返回 tokenizer 对应属性的值
        return getattr(self.tokenizer, name)

    # 初始化方法，接受一个 dictionary 参数，默认为 None
    def initialize(self, dictionary=None):
        # 调用分词器的 initialize 方法，传入 dictionary 参数
        self.tokenizer.initialize(dictionary)
        # 再次调用 load_word_tag 方法，重新加载字典文件
        self.load_word_tag(self.tokenizer.get_dict_file())

    # 加载词性标注字典的方法，接受一个文件对象 f 作为参数
    def load_word_tag(self, f):
        # 初始化一个空字典用于存储词性标注
        self.word_tag_tab = {}
        # 获取文件名
        f_name = resolve_filename(f)
        # 遍历文件的每一行
        for lineno, line in enumerate(f, 1):
            try:
                # 去除首尾空白并解码为 utf-8 格式
                line = line.strip().decode("utf-8")
                # 如果行为空则跳过
                if not line:
                    continue
                # 按空格分割行，获取词和词性标注
                word, _, tag = line.split(" ")
                # 将词和对应的词性标注存入字典
                self.word_tag_tab[word] = tag
            except Exception:
                # 如果出现异常则抛出 ValueError 异常
                raise ValueError(
                    'invalid POS dictionary entry in %s at Line %s: %s' % (f_name, lineno, line))
        # 关闭文件
        f.close()

    # 确保用户自定义词典已加载的方法
    def makesure_userdict_loaded(self):
        # 如果分词器的用户自定义词典不为空
        if self.tokenizer.user_word_tag_tab:
            # 更新词性标注字典，将用户自定义词典中的词性标注合并进来
            self.word_tag_tab.update(self.tokenizer.user_word_tag_tab)
            # 清空分词器的用户自定义词典
            self.tokenizer.user_word_tag_tab = {}
    # 对输入的句子进行分词处理，返回概率和词性列表
    def __cut(self, sentence):
        prob, pos_list = viterbi(
            sentence, char_state_tab_P, start_P, trans_P, emit_P)
        begin, nexti = 0, 0

        # 遍历句子中的每个字符及其对应的词性
        for i, char in enumerate(sentence):
            pos = pos_list[i][0]
            # 如果当前字符是词的开头
            if pos == 'B':
                begin = i
            # 如果当前字符是词的结尾
            elif pos == 'E':
                yield pair(sentence[begin:i + 1], pos_list[i][1])
                nexti = i + 1
            # 如果当前字符是单字词
            elif pos == 'S':
                yield pair(char, pos_list[i][1])
                nexti = i + 1
        # 处理剩余未处理的字符
        if nexti < len(sentence):
            yield pair(sentence[nexti:], pos_list[nexti][1])

    # 对句子进行详细的分词处理
    def __cut_detail(self, sentence):
        # 使用正则表达式将句子分成块
        blocks = re_han_detail.split(sentence)
        for blk in blocks:
            # 如果块是汉字
            if re_han_detail.match(blk):
                # 对汉字块进行分词处理
                for word in self.__cut(blk):
                    yield word
            else:
                # 如果块不是汉字
                tmp = re_skip_detail.split(blk)
                for x in tmp:
                    if x:
                        # 根据不同的规则判断块的词性
                        if re_num.match(x):
                            yield pair(x, 'm')
                        elif re_eng.match(x):
                            yield pair(x, 'eng')
                        else:
                            yield pair(x, 'x')

    # 不使用隐马尔可夫模型的分词处理
    def __cut_DAG_NO_HMM(self, sentence):
        # 获取句子的有向无环图(DAG)
        DAG = self.tokenizer.get_DAG(sentence)
        route = {}
        # 计算最优路径
        self.tokenizer.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            # 如果是英文单词
            if re_eng1.match(l_word):
                buf += l_word
                x = y
            else:
                if buf:
                    yield pair(buf, 'eng')
                    buf = ''
                # 根据词典获取词的词性，如果不在词典中则标记为'x'
                yield pair(l_word, self.word_tag_tab.get(l_word, 'x'))
                x = y
        if buf:
            yield pair(buf, 'eng')
            buf = ''
    # 对输入的句子进行分词处理，返回一个有向无环图（DAG）
    def __cut_DAG(self, sentence):
        # 获取句子的有向无环图（DAG）
        DAG = self.tokenizer.get_DAG(sentence)
        # 用于记录最短路径的字典
        route = {}

        # 计算最短路径
        self.tokenizer.calc(sentence, DAG, route)

        # 初始化变量
        x = 0
        buf = ''
        N = len(sentence)
        # 遍历句子
        while x < N:
            # 获取当前词的结束位置
            y = route[x][1] + 1
            # 获取当前词
            l_word = sentence[x:y]
            # 判断当前词是否为单字词
            if y - x == 1:
                buf += l_word
            else:
                # 如果 buf 不为空
                if buf:
                    # 如果 buf 只有一个字
                    if len(buf) == 1:
                        yield pair(buf, self.word_tag_tab.get(buf, 'x'))
                    # 如果 buf 不在词频字典中
                    elif not self.tokenizer.FREQ.get(buf):
                        # 对 buf 进行详细分词处理
                        recognized = self.__cut_detail(buf)
                        # 遍历详细分词结果
                        for t in recognized:
                            yield t
                    else:
                        # 遍历 buf 中的每个字
                        for elem in buf:
                            yield pair(elem, self.word_tag_tab.get(elem, 'x'))
                    buf = ''
                # 返回当前词及其词性
                yield pair(l_word, self.word_tag_tab.get(l_word, 'x'))
            x = y

        # 处理最后一个 buf
        if buf:
            # 如果 buf 只有一个字
            if len(buf) == 1:
                yield pair(buf, self.word_tag_tab.get(buf, 'x'))
            # 如果 buf 不在词频字典中
            elif not self.tokenizer.FREQ.get(buf):
                # 对 buf 进行详细分词处理
                recognized = self.__cut_detail(buf)
                # 遍历详细分词结果
                for t in recognized:
                    yield t
            else:
                # 遍历 buf 中的每个字
                for elem in buf:
                    yield pair(elem, self.word_tag_tab.get(elem, 'x'))
    # 对输入的句子进行分词处理，根据是否使用 HMM 模型来选择不同的分词方法
    def __cut_internal(self, sentence, HMM=True):
        # 确保用户自定义词典已加载
        self.makesure_userdict_loaded()
        # 将句子转换为字符串编码
        sentence = strdecode(sentence)
        # 使用正则表达式将句子分成块
        blocks = re_han_internal.split(sentence)
        # 根据是否使用 HMM 模型选择不同的分词方法
        if HMM:
            cut_blk = self.__cut_DAG
        else:
            cut_blk = self.__cut_DAG_NO_HMM

        # 遍历每个块
        for blk in blocks:
            # 如果块是汉字
            if re_han_internal.match(blk):
                # 对汉字块进行分词
                for word in cut_blk(blk):
                    yield word
            else:
                # 如果块不是汉字
                tmp = re_skip_internal.split(blk)
                # 对非汉字块进行处理
                for x in tmp:
                    if re_skip_internal.match(x):
                        yield pair(x, 'x')
                    else:
                        for xx in x:
                            if re_num.match(xx):
                                yield pair(xx, 'm')
                            elif re_eng.match(x):
                                yield pair(xx, 'eng')
                            else:
                                yield pair(xx, 'x')

    # 对句子进行分词处理，并返回分词结果列表
    def _lcut_internal(self, sentence):
        return list(self.__cut_internal(sentence))

    # 对句子进行分词处理（不使用 HMM 模型），并返回分词结果列表
    def _lcut_internal_no_hmm(self, sentence):
        return list(self.__cut_internal(sentence, False))

    # 对句子进行分词处理，返回生成器
    def cut(self, sentence, HMM=True):
        for w in self.__cut_internal(sentence, HMM=HMM):
            yield w

    # 对句子进行分词处理，返回分词结果列表
    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))
# 默认的 Tokenizer 实例
dt = POSTokenizer(jieba.dt)

# 全局函数
initialize = dt.initialize

# 内部函数，用于分词
def _lcut_internal(s):
    return dt._lcut_internal(s)

# 内部函数，用于分词但不使用 HMM
def _lcut_internal_no_hmm(s):
    return dt._lcut_internal_no_hmm(s)

# 分词函数，支持并行处理
def cut(sentence, HMM=True, use_paddle=False):
    """
    Global `cut` function that supports parallel processing.

    Note that this only works using dt, custom POSTokenizer
    instances are not supported.
    """
    # 检查是否安装了 PaddlePaddle 并且使用 PaddlePaddle 进行分词
    is_paddle_installed = check_paddle_install['is_paddle_installed']
    if use_paddle and is_paddle_installed:
        # 如果句子为空，则在 PaddlePaddle 中会引发核心异常
        if sentence is None or sentence == "" or sentence == u"":
            return
        import jieba.lac_small.predict as predict
        sents, tags = predict.get_result(strdecode(sentence))
        for i, sent in enumerate(sents):
            if sent is None or tags[i] is None:
                continue
            yield pair(sent, tags[i])
        return
    # 使用全局的 Tokenizer 实例 dt 进行分词
    global dt
    if jieba.pool is None:
        for w in dt.cut(sentence, HMM=HMM):
            yield w
    else:
        parts = strdecode(sentence).splitlines(True)
        if HMM:
            result = jieba.pool.map(_lcut_internal, parts)
        else:
            result = jieba.pool.map(_lcut_internal_no_hmm, parts)
        for r in result:
            for w in r:
                yield w

# 对句子进行分词并返回列表
def lcut(sentence, HMM=True, use_paddle=False):
    if use_paddle:
        return list(cut(sentence, use_paddle=True))
    return list(cut(sentence, HMM))
```