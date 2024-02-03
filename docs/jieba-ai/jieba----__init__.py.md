# `jieba\jieba\__init__.py`

```
# 导入必要的模块和库
from __future__ import absolute_import, unicode_literals
# 定义当前代码版本和许可证
__version__ = '0.42.1'
__license__ = 'MIT'

# 导入需要的模块和库
import marshal
import re
import tempfile
import threading
import time
from hashlib import md5
from math import log

# 导入自定义模块和库
from . import finalseg
from ._compat import *

# 根据操作系统类型选择文件替换函数
if os.name == 'nt':
    from shutil import move as _replace_file
else:
    _replace_file = os.rename

# 获取绝对路径的函数
_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))

# 设置默认字典和字典文件名
DEFAULT_DICT = None
DEFAULT_DICT_NAME = "dict.txt"

# 设置日志输出到控制台
log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(log_console)

# 定义全局变量
DICT_WRITING = {}
pool = None

# 定义正则表达式模式
re_userdict = re.compile('^(.+?)( [0-9]+)?( [a-z]+)?$', re.U)
re_eng = re.compile('[a-zA-Z0-9]', re.U)
re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)", re.U)
re_skip_default = re.compile("(\r\n|\s)", re.U)

# 设置日志级别的函数
def setLogLevel(log_level):
    default_logger.setLevel(log_level)

# 定义 Tokenizer 类
class Tokenizer(object):

    def __init__(self, dictionary=DEFAULT_DICT):
        # 初始化 Tokenizer 对象
        self.lock = threading.RLock()
        if dictionary == DEFAULT_DICT:
            self.dictionary = dictionary
        else:
            self.dictionary = _get_abs_path(dictionary)
        self.FREQ = {}
        self.total = 0
        self.user_word_tag_tab = {}
        self.initialized = False
        self.tmp_dir = None
        self.cache_file = None

    def __repr__(self):
        return '<Tokenizer dictionary=%r>' % self.dictionary

    @staticmethod
    # 生成词频字典，统计词频和总词频
    def gen_pfdict(f):
        # 初始化词频字典和总词频
        lfreq = {}
        ltotal = 0
        # 解析文件名
        f_name = resolve_filename(f)
        # 遍历文件的每一行
        for lineno, line in enumerate(f, 1):
            try:
                # 去除空格并解码为 utf-8 格式
                line = line.strip().decode('utf-8')
                # 拆分单词和频率
                word, freq = line.split(' ')[:2]
                freq = int(freq)
                # 更新词频字典和总词频
                lfreq[word] = freq
                ltotal += freq
                # 遍历单词的每个字符
                for ch in xrange(len(word)):
                    wfrag = word[:ch + 1]
                    # 如果子串不在词频字典中，则添加进去
                    if wfrag not in lfreq:
                        lfreq[wfrag] = 0
            except ValueError:
                # 抛出值错误异常
                raise ValueError(
                    'invalid dictionary entry in %s at Line %s: %s' % (f_name, lineno, line))
        # 关闭文件
        f.close()
        # 返回词频字典和总词频
        return lfreq, ltotal

    # 检查是否已初始化
    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    # 计算最大概率路径
    def calc(self, sentence, DAG, route):
        N = len(sentence)
        route[N] = (0, 0)
        logtotal = log(self.total)
        # 逆序遍历句子
        for idx in xrange(N - 1, -1, -1):
            # 计算最大概率路径
            route[idx] = max((log(self.FREQ.get(sentence[idx:x + 1]) or 1) -
                              logtotal + route[x + 1][0], x) for x in DAG[idx])

    # 获取有向无环图
    def get_DAG(self, sentence):
        # 检查是否已初始化
        self.check_initialized()
        DAG = {}
        N = len(sentence)
        # 遍历句子
        for k in xrange(N):
            tmplist = []
            i = k
            frag = sentence[k]
            # 构建有向无环图
            while i < N and frag in self.FREQ:
                if self.FREQ[frag]:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        # 返回有向无环图
        return DAG
    # 对输入的句子进行全模式分词，返回分词结果
    def __cut_all(self, sentence):
        # 获取句子的有向无环图（DAG）
        dag = self.get_DAG(sentence)
        # 初始化变量
        old_j = -1
        eng_scan = 0
        eng_buf = u''
        # 遍历 DAG
        for k, L in iteritems(dag):
            # 如果正在扫描英文单词并且当前字符不是英文字符，则结束扫描
            if eng_scan == 1 and not re_eng.match(sentence[k]):
                eng_scan = 0
                yield eng_buf
            # 如果当前字符对应的词只有一个字符且位置在上一个词的后面
            if len(L) == 1 and k > old_j:
                word = sentence[k:L[0] + 1]
                # 如果是英文单词
                if re_eng.match(word):
                    if eng_scan == 0:
                        eng_scan = 1
                        eng_buf = word
                    else:
                        eng_buf += word
                # 如果不是英文单词
                if eng_scan == 0:
                    yield word
                old_j = L[0]
            else:
                # 遍历当前字符对应的词
                for j in L:
                    if j > k:
                        yield sentence[k:j + 1]
                        old_j = j
        # 如果最后一个词是英文单词，则返回该单词
        if eng_scan == 1:
            yield eng_buf

    # 对输入的句子进行非 HMM 分词，返回分词结果
    def __cut_DAG_NO_HMM(self, sentence):
        # 获取句子的有向无环图（DAG）
        DAG = self.get_DAG(sentence)
        route = {}
        # 计算最佳路径
        self.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        # 遍历最佳路径
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            # 如果是英文单词且长度为1，则加入缓存
            if re_eng.match(l_word) and len(l_word) == 1:
                buf += l_word
                x = y
            else:
                # 如果缓存不为空，则返回缓存中的内容
                if buf:
                    yield buf
                    buf = ''
                yield l_word
                x = y
        # 处理最后一个词
        if buf:
            yield buf
            buf = ''
    # 对输入的句子进行分词处理，返回一个有向无环图（DAG）
    def __cut_DAG(self, sentence):
        # 获取句子的有向无环图（DAG）
        DAG = self.get_DAG(sentence)
        # 用于记录最优路径
        route = {}
        # 计算最优路径
        self.calc(sentence, DAG, route)
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
                # 如果当前词不是单字词
                if buf:
                    # 如果缓存不为空
                    if len(buf) == 1:
                        # 如果缓存只有一个字，则直接输出
                        yield buf
                        buf = ''
                    else:
                        # 如果缓存不止一个字
                        if not self.FREQ.get(buf):
                            # 如果缓存不在词典中，则进行分词处理
                            recognized = finalseg.cut(buf)
                            for t in recognized:
                                yield t
                        else:
                            # 如果缓存在词典中，则逐个输出
                            for elem in buf:
                                yield elem
                        buf = ''
                # 输出当前词
                yield l_word
            x = y

        # 处理最后的缓存
        if buf:
            if len(buf) == 1:
                yield buf
            elif not self.FREQ.get(buf):
                recognized = finalseg.cut(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield elem
    # 定义一个方法，用于对包含中文字符的句子进行分词
    def cut(self, sentence, cut_all=False, HMM=True, use_paddle=False):
        """
        The main function that segments an entire sentence that contains
        Chinese characters into separated words.

        Parameter:
            - sentence: The str(unicode) to be segmented.
            - cut_all: Model type. True for full pattern, False for accurate pattern.
            - HMM: Whether to use the Hidden Markov Model.
        """
        # 检查是否安装了 PaddlePaddle
        is_paddle_installed = check_paddle_install['is_paddle_installed']
        # 将输入的句子转换为字符串
        sentence = strdecode(sentence)
        # 如果使用 PaddlePaddle 并且已安装
        if use_paddle and is_paddle_installed:
            # 如果句子为空，则在 PaddlePaddle 中会引发核心异常
            if sentence is None or len(sentence) == 0:
                return
            # 导入预测模块
            import jieba.lac_small.predict as predict
            # 获取句子的分词结果
            results = predict.get_sent(sentence)
            # 遍历分词结果
            for sent in results:
                if sent is None:
                    continue
                yield sent
            return
        # 定义匹配中文字符的正则表达式
        re_han = re_han_default
        # 定义匹配非中文字符的正则表达式
        re_skip = re_skip_default
        # 根据参数选择不同的分词方法
        if cut_all:
            cut_block = self.__cut_all
        elif HMM:
            cut_block = self.__cut_DAG
        else:
            cut_block = self.__cut_DAG_NO_HMM
        # 使用正则表达式将句子分块
        blocks = re_han.split(sentence)
        # 遍历每个块
        for blk in blocks:
            if not blk:
                continue
            # 如果块中包含中文字符
            if re_han.match(blk):
                # 对块进行分词
                for word in cut_block(blk):
                    yield word
            else:
                # 如果块中不包含中文字符
                tmp = re_skip.split(blk)
                for x in tmp:
                    if re_skip.match(x):
                        yield x
                    elif not cut_all:
                        for xx in x:
                            yield xx
                    else:
                        yield x
    # 对句子进行细分，用于搜索引擎
    def cut_for_search(self, sentence, HMM=True):
        # 使用默认的 HMM 模型对句子进行分词
        words = self.cut(sentence, HMM=HMM)
        # 遍历分词后的结果
        for w in words:
            # 如果词的长度大于2
            if len(w) > 2:
                # 遍历词的每个字符，组成二元语法
                for i in xrange(len(w) - 1):
                    gram2 = w[i:i + 2]
                    # 如果二元语法在词典中存在
                    if self.FREQ.get(gram2):
                        # 返回二元语法
                        yield gram2
            # 如果词的长度大于3
            if len(w) > 3:
                # 遍历词的每个字符，组成三元语法
                for i in xrange(len(w) - 2):
                    gram3 = w[i:i + 3]
                    # 如果三元语法在词典中存在
                    if self.FREQ.get(gram3):
                        # 返回三元语法
                        yield gram3
            # 返回词
            yield w

    # 对句子进行分词，并返回列表形式
    def lcut(self, *args, **kwargs):
        return list(self.cut(*args, **kwargs))

    # 对句子进行细分，用于搜索引擎，并返回列表形式
    def lcut_for_search(self, *args, **kwargs):
        return list(self.cut_for_search(*args, **kwargs))

    # 备份 lcut 方法
    _lcut = lcut
    # 备份 lcut_for_search 方法
    _lcut_for_search = lcut_for_search

    # 对句子进行分词，不使用 HMM 模型
    def _lcut_no_hmm(self, sentence):
        return self.lcut(sentence, False, False)

    # 对句子进行分词，使用所有可能的分词结果
    def _lcut_all(self, sentence):
        return self.lcut(sentence, True)

    # 对句子进行细分，用于搜索引擎，不使用 HMM 模型
    def _lcut_for_search_no_hmm(self, sentence):
        return self.lcut_for_search(sentence, False)

    # 获取词典文件
    def get_dict_file(self):
        # 如果词典为默认词典，则返回默认词典文件
        if self.dictionary == DEFAULT_DICT:
            return get_module_res(DEFAULT_DICT_NAME)
        # 否则返回指定的词典文件
        else:
            return open(self.dictionary, 'rb')
    # 加载用户自定义词典以提高识别率
    def load_userdict(self, f):
        '''
        Load personalized dict to improve detect rate.

        Parameter:
            - f : A plain text file contains words and their ocurrences.
                  Can be a file-like object, or the path of the dictionary file,
                  whose encoding must be utf-8.

        Structure of dict file:
        word1 freq1 word_type1
        word2 freq2 word_type2
        ...
        Word type may be ignored
        '''
        # 检查是否已初始化
        self.check_initialized()
        # 如果 f 是字符串类型，则将其作为文件名打开
        if isinstance(f, string_types):
            f_name = f
            f = open(f, 'rb')
        else:
            f_name = resolve_filename(f)
        # 遍历文件的每一行
        for lineno, ln in enumerate(f, 1):
            line = ln.strip()
            # 如果行不是文本类型，则尝试解码为 utf-8 编码
            if not isinstance(line, text_type):
                try:
                    line = line.decode('utf-8').lstrip('\ufeff')
                except UnicodeDecodeError:
                    raise ValueError('dictionary file %s must be utf-8' % f_name)
            # 如果行为空，则跳过
            if not line:
                continue
            # 使用正则表达式匹配行中的单词、频率和标签
            # 匹配不会为 None，因为至少有一个字符
            word, freq, tag = re_userdict.match(line).groups()
            # 如果频率不为 None，则去除空格
            if freq is not None:
                freq = freq.strip()
            # 如果标签不为 None，则去除空格
            if tag is not None:
                tag = tag.strip()
            # 将单词、频率和标签添加到词典中
            self.add_word(word, freq, tag)
    # 将一个单词添加到字典中
    def add_word(self, word, freq=None, tag=None):
        """
        Add a word to dictionary.

        freq and tag can be omitted, freq defaults to be a calculated value
        that ensures the word can be cut out.
        """
        # 检查字典是否已经初始化
        self.check_initialized()
        # 将单词转换为字符串
        word = strdecode(word)
        # 如果频率未提供，则使用 suggest_freq 方法计算一个默认值
        freq = int(freq) if freq is not None else self.suggest_freq(word, False)
        # 将单词及其频率添加到 FREQ 字典中
        self.FREQ[word] = freq
        # 更新总频率
        self.total += freq
        # 如果提供了标签，则将单词与标签关联
        if tag:
            self.user_word_tag_tab[word] = tag
        # 遍历单词的每个字符
        for ch in xrange(len(word)):
            # 获取单词的部分片段
            wfrag = word[:ch + 1]
            # 如果部分片段不在 FREQ 字典中，则将其添加，并初始化频率为 0
            if wfrag not in self.FREQ:
                self.FREQ[wfrag] = 0
        # 如果频率为 0，则强制将单词进行分割
        if freq == 0:
            finalseg.add_force_split(word)

    # 删除字典中的一个单词
    def del_word(self, word):
        """
        Convenient function for deleting a word.
        """
        # 调用 add_word 方法，将单词频率设为 0，实现删除操作
        self.add_word(word, 0)
    # 建议词频，用于强制将单词中的字符连接或拆分
    def suggest_freq(self, segment, tune=False):
        """
        Suggest word frequency to force the characters in a word to be
        joined or splitted.

        Parameter:
            - segment : The segments that the word is expected to be cut into,
                        If the word should be treated as a whole, use a str.
            - tune : If True, tune the word frequency.

        Note that HMM may affect the final result. If the result doesn't change,
        set HMM=False.
        """
        # 检查是否已初始化
        self.check_initialized()
        # 将总词频转换为浮点数
        ftotal = float(self.total)
        # 初始化频率为1
        freq = 1
        # 如果segment是字符串类型
        if isinstance(segment, string_types):
            # 将segment赋值给word
            word = segment
            # 对分词结果进行遍历
            for seg in self.cut(word, HMM=False):
                # 计算频率
                freq *= self.FREQ.get(seg, 1) / ftotal
            # 更新频率
            freq = max(int(freq * self.total) + 1, self.FREQ.get(word, 1))
        else:
            # 将segment中的元素转换为字符串
            segment = tuple(map(strdecode, segment))
            # 将segment中的元素连接成一个字符串
            word = ''.join(segment)
            # 对segment中的元素进行遍历
            for seg in segment:
                # 计算频率
                freq *= self.FREQ.get(seg, 1) / ftotal
            # 更新频率
            freq = min(int(freq * self.total), self.FREQ.get(word, 0))
        # 如果需要调整频率
        if tune:
            # 添加单词及其频率
            self.add_word(word, freq)
        # 返回频率
        return freq
    def tokenize(self, unicode_sentence, mode="default", HMM=True):
        """
        Tokenize a sentence and yields tuples of (word, start, end)

        Parameter:
            - sentence: the str(unicode) to be segmented.
            - mode: "default" or "search", "search" is for finer segmentation.
            - HMM: whether to use the Hidden Markov Model.
        """
        # 检查输入参数是否为 unicode 类型
        if not isinstance(unicode_sentence, text_type):
            raise ValueError("jieba: the input parameter should be unicode.")
        # 初始化起始位置
        start = 0
        # 根据不同模式进行分词
        if mode == 'default':
            # 对句子进行分词，返回每个词及其起始和结束位置
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                yield (w, start, start + width)
                start += width
        else:
            # 对句子进行细粒度分词
            for w in self.cut(unicode_sentence, HMM=HMM):
                width = len(w)
                # 对长度大于2的词进行二元分词
                if len(w) > 2:
                    for i in xrange(len(w) - 1):
                        gram2 = w[i:i + 2]
                        if self.FREQ.get(gram2):
                            yield (gram2, start + i, start + i + 2)
                # 对长度大于3的词进行三元分词
                if len(w) > 3:
                    for i in xrange(len(w) - 2):
                        gram3 = w[i:i + 3]
                        if self.FREQ.get(gram3):
                            yield (gram3, start + i, start + i + 3)
                yield (w, start, start + width)
                start += width

    def set_dictionary(self, dictionary_path):
        # 使用锁确保线程安全
        with self.lock:
            # 获取字典文件的绝对路径
            abs_path = _get_abs_path(dictionary_path)
            # 检查字典文件是否存在
            if not os.path.isfile(abs_path):
                raise Exception("jieba: file does not exist: " + abs_path)
            # 设置字典路径
            self.dictionary = abs_path
            # 标记为未初始化
            self.initialized = False
# 创建默认的 Tokenizer 实例
dt = Tokenizer()

# 全局函数

# 获取指定键的频率值，如果不存在则返回默认值
get_FREQ = lambda k, d=None: dt.FREQ.get(k, d)
# 向分词器添加新词
add_word = dt.add_word
# 计算
calc = dt.calc
# 分词
cut = dt.cut
# 精确分词
lcut = dt.lcut
# 搜索引擎分词
cut_for_search = dt.cut_for_search
# 搜索引擎精确分词
lcut_for_search = dt.lcut_for_search
# 删除指定词
del_word = dt.del_word
# 获取 DAG
get_DAG = dt.get_DAG
# 获取词典文件
get_dict_file = dt.get_dict_file
# 初始化
initialize = dt.initialize
# 加载用户自定义词典
load_userdict = dt.load_userdict
# 设置词典
set_dictionary = dt.set_dictionary
# 建议词频
suggest_freq = dt.suggest_freq
# 分词
tokenize = dt.tokenize
# 用户自定义词性标注表
user_word_tag_tab = dt.user_word_tag_tab

# 定义私有函数

# 全模式分词
def _lcut_all(s):
    return dt._lcut_all(s)

# 精确模式分词
def _lcut(s):
    return dt._lcut(s)

# 精确模式分词，不使用 HMM
def _lcut_no_hmm(s):
    return dt._lcut_no_hmm(s)

# 全模式分词
def _lcut_all(s):
    return dt._lcut_all(s)

# 搜索引擎模式分词
def _lcut_for_search(s):
    return dt._lcut_for_search(s)

# 搜索引擎模式分词，不使用 HMM
def _lcut_for_search_no_hmm(s):
    return dt._lcut_for_search_no_hmm(s)

# 并行分词
def _pcut(sentence, cut_all=False, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    if cut_all:
        result = pool.map(_lcut_all, parts)
    elif HMM:
        result = pool.map(_lcut, parts)
    else:
        result = pool.map(_lcut_no_hmm, parts)
    for r in result:
        for w in r:
            yield w

# 并行搜索引擎分词
def _pcut_for_search(sentence, HMM=True):
    parts = strdecode(sentence).splitlines(True)
    if HMM:
        result = pool.map(_lcut_for_search, parts)
    else:
        result = pool.map(_lcut_for_search_no_hmm, parts)
    for r in result:
        for w in r:
            yield w

# 启用并行模式
def enable_parallel(processnum=None):
    """
    Change the module's `cut` and `cut_for_search` functions to the
    parallel version.

    Note that this only works using dt, custom Tokenizer
    instances are not supported.
    """
    global pool, dt, cut, cut_for_search
    from multiprocessing import cpu_count
    if os.name == 'nt':
        raise NotImplementedError(
            "jieba: parallel mode only supports posix system")
    else:
        from multiprocessing import Pool
    dt.check_initialized()
    # 如果未指定进程数，则使用 CPU 核心数作为默认值
    if processnum is None:
        processnum = cpu_count()
    # 创建进程池对象，用于并行处理任务
    pool = Pool(processnum)
    # 将 _pcut 函数赋值给 cut 变量，用于分词
    cut = _pcut
    # 将 _pcut_for_search 函数赋值给 cut_for_search 变量，用于搜索时的分词
    cut_for_search = _pcut_for_search
# 禁用并行处理，重置全局变量 pool, dt, cut, cut_for_search
def disable_parallel():
    # 声明要修改的全局变量
    global pool, dt, cut, cut_for_search
    # 如果存在并行处理池
    if pool:
        # 关闭并行处理池
        pool.close()
        # 将并行处理池设为 None
        pool = None
    # 重置 cut 变量为 dt.cut
    cut = dt.cut
    # 重置 cut_for_search 变量为 dt.cut_for_search
    cut_for_search = dt.cut_for_search
```