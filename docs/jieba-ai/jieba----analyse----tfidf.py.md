# `jieba\jieba\analyse\tfidf.py`

```
# 设置文件编码为 UTF-8
# 导入绝对引入的未来特性
import os
# 导入结巴分词库
import jieba
import jieba.posseg
# 从操作符模块中导入 itemgetter 函数
from operator import itemgetter

# 定义一个 lambda 函数，用于获取模块路径
_get_module_path = lambda path: os.path.normpath(os.path.join(os.getcwd(),
                                                 os.path.dirname(__file__), path))
# 获取绝对路径
_get_abs_path = jieba._get_abs_path

# 默认 IDF 文件路径
DEFAULT_IDF = _get_module_path("idf.txt")

# 关键词提取器类
class KeywordExtractor(object):

    # 停用词集合
    STOP_WORDS = set((
        "the", "of", "is", "and", "to", "in", "that", "we", "for", "an", "are",
        "by", "be", "as", "on", "with", "can", "if", "from", "which", "you", "it",
        "this", "then", "at", "have", "all", "not", "one", "has", "or", "that"
    ))

    # 设置停用词
    def set_stop_words(self, stop_words_path):
        # 获取停用词文件的绝对路径
        abs_path = _get_abs_path(stop_words_path)
        # 如果文件不存在，则抛出异常
        if not os.path.isfile(abs_path):
            raise Exception("jieba: file does not exist: " + abs_path)
        # 读取文件内容并解码为 UTF-8 格式
        content = open(abs_path, 'rb').read().decode('utf-8')
        # 将停用词添加到停用词集合中
        for line in content.splitlines():
            self.stop_words.add(line)

    # 提取关键词
    def extract_tags(self, *args, **kwargs):
        # 抛出未实现异常
        raise NotImplementedError

# IDF 加载器类
class IDFLoader(object):

    # 初始化方法
    def __init__(self, idf_path=None):
        self.path = ""
        self.idf_freq = {}
        self.median_idf = 0.0
        # 如果提供了 IDF 文件路径，则设置新路径
        if idf_path:
            self.set_new_path(idf_path)

    # 设置新的 IDF 文件路径
    def set_new_path(self, new_idf_path):
        # 如果路径不同，则更新路径
        if self.path != new_idf_path:
            self.path = new_idf_path
            # 读取文件内容并解码为 UTF-8 格式
            content = open(new_idf_path, 'rb').read().decode('utf-8')
            self.idf_freq = {}
            # 解析文件内容，构建 IDF 频率字典
            for line in content.splitlines():
                word, freq = line.strip().split(' ')
                self.idf_freq[word] = float(freq)
            # 计算 IDF 中位数
            self.median_idf = sorted(
                self.idf_freq.values())[len(self.idf_freq) // 2]

    # 获取 IDF 频率字典和中位数 IDF
    def get_idf(self):
        return self.idf_freq, self.median_idf

# TF-IDF 关键词提取器类
class TFIDF(KeywordExtractor):
    # 初始化函数，用于创建一个新的TF-IDF对象
    def __init__(self, idf_path=None):
        # 设置分词器为结巴的默认分词器
        self.tokenizer = jieba.dt
        # 设置词性标注器为结巴的默认词性标注器
        self.postokenizer = jieba.posseg.dt
        # 复制停用词列表到对象的停用词属性中
        self.stop_words = self.STOP_WORDS.copy()
        # 创建一个IDFLoader对象，用于加载IDF文件，默认为DEFAULT_IDF
        self.idf_loader = IDFLoader(idf_path or DEFAULT_IDF)
        # 获取IDF频率和中位数IDF值
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()
    
    # 设置新的IDF文件路径
    def set_idf_path(self, idf_path):
        # 获取新的IDF文件的绝对路径
        new_abs_path = _get_abs_path(idf_path)
        # 如果新的IDF文件路径不存在，则抛出异常
        if not os.path.isfile(new_abs_path):
            raise Exception("jieba: file does not exist: " + new_abs_path)
        # 设置新的IDF文件路径到IDFLoader对象中
        self.idf_loader.set_new_path(new_abs_path)
        # 获取新的IDF频率和中位数IDF值
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()
    # 从句子中使用 TF-IDF 算法提取关键词
    def extract_tags(self, sentence, topK=20, withWeight=False, allowPOS=(), withFlag=False):
        """
        从句子中使用 TF-IDF 算法提取关键词。
        参数：
            - topK: 返回前多少个关键词。`None` 表示返回所有可能的词。
            - withWeight: 如果为 True，返回一个 (word, weight) 列表；
                          如果为 False，返回一个词列表。
            - allowPOS: 允许的词性列表，例如 ['ns', 'n', 'vn', 'v', 'nr']。
                        如果 w 的词性不在列表中，将被过滤。
            - withFlag: 仅在 allowPOS 不为空时有效。
                        如果为 True，返回一个类似 posseg.cut 的 (word, weight) 列表；
                        如果为 False，返回一个词列表。
        """
        # 如果 allowPOS 不为空，将其转换为不可变集合
        if allowPOS:
            allowPOS = frozenset(allowPOS)
            # 使用 postokenizer 对句子进行分词
            words = self.postokenizer.cut(sentence)
        else:
            # 使用 tokenizer 对句子进行分词
            words = self.tokenizer.cut(sentence)
        # 初始化词频字典
        freq = {}
        # 遍历分词结果
        for w in words:
            # 如果 allowPOS 不为空
            if allowPOS:
                # 如果词性不在允许的词性列表中，跳过
                if w.flag not in allowPOS:
                    continue
                # 如果不需要词性标记
                elif not withFlag:
                    w = w.word
            # 获取词或词性
            wc = w.word if allowPOS and withFlag else w
            # 如果词长度小于 2 或者在停用词列表中，跳过
            if len(wc.strip()) < 2 or wc.lower() in self.stop_words:
                continue
            # 更新词频
            freq[w] = freq.get(w, 0.0) + 1.0
        # 计算总词频
        total = sum(freq.values())
        # 根据 TF-IDF 计算权重
        for k in freq:
            kw = k.word if allowPOS and withFlag else k
            freq[k] *= self.idf_freq.get(kw, self.median_idf) / total

        # 根据权重排序关键词
        if withWeight:
            tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(freq, key=freq.__getitem__, reverse=True)
        # 返回前 topK 个关键词
        if topK:
            return tags[:topK]
        else:
            return tags
```