# `ChatRWKV\tokenizer\rwkv_tokenizer.py`

```
# 导入必要的模块
import os, sys, time, random

# 打印说明文档
print('''
#######################################################################################################################

This tokenizer is not used in any RWKV models yet. I plan to use it for the future multilang RWKV models.

Benefits:

* Good support of most languages, from European to CJK to Arabic and Hindi and more.

* Clean vocab. Good for code too. Vocab size = 65536 (use 0 for 
    # 初始化方法，接受文件名参数
    def __init__(self, file_name):
        # 初始化索引到标记的字典
        self.idx2token = {}
        # 创建一个空列表 sorted，必须已经排序
        sorted = [] # must be already sorted
        # 读取文件的所有行
        lines = open(file_name, "r", encoding="utf-8").readlines()
        # 遍历文件的每一行
        for l in lines:
            # 获取索引值
            idx = int(l[:l.index(' ')])
            # 获取标记值
            x = eval(l[l.index(' '):l.rindex(' ')])
            # 如果标记是字符串，则编码成 UTF-8 格式的字节流
            x = x.encode("utf-8") if isinstance(x, str) else x
            # 断言标记是字节流
            assert isinstance(x, bytes)
            # 断言字节流的长度与文件中记录的长度相同
            assert len(x) == int(l[l.rindex(' '):])
            # 将标记添加到 sorted 列表中
            sorted += [x]
            # 将索引和标记的映射关系添加到 idx2token 字典中
            self.idx2token[idx] = x

        # 初始化标记到索引的字典
        self.token2idx = {}
        # 遍历索引到标记的字典
        for k, v in self.idx2token.items():
            # 将标记到索引的映射关系添加到 token2idx 字典中
            self.token2idx[v] = int(k)

        # 预先计算一些用于快速匹配的表
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        # 遍历已排序的列表（倒序 - 先匹配更长的标记）
        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            # 如果标记长度大于等于 2
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                # 将标记添加到表中对应位置
                self.table[s0][s1] += [s]
                # 更新标记长度
                self.wlen[s0] = max(self.wlen[s0], len(s))
                # 将 s1 添加到 s0 对应的 good 集合中
                self.good[s0].add(s1)

    # 将字节流编码成标记列表
    def encodeBytes(self, src: bytes) -> list[int]:
        # 获取源字节流的长度
        src_len: int = len(src)
        # 初始化标记列表
        tokens: list[int] = []
        # 初始化索引
        i: int = 0
        # 遍历源字节流
        while i < src_len:
            # 获取当前字节
            s: bytes = src[i : i + 1]

            # 如果索引小于源字节流长度减 1
            if i < src_len - 1:
                # 获取下一个字节的整数值
                s1: int = int(src[i + 1])
                # 获取当前字节的整数值
                s0: int = int(src[i])
                # 如果 s1 在 s0 对应的 good 集合中
                if s1 in self.good[s0]:
                    # 获取匹配的标记
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        # 尝试从表中找到匹配的标记
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            # 将标记索引添加到 tokens 列表中
            tokens.append(self.token2idx[s])
            # 更新索引
            i += len(s)

        # 返回标记列表
        return tokens

    # 将标记列表解码成字节流
    def decodeBytes(self, tokens):
        # 将标记列表中的标记解码成字节流并拼接起来
        return b''.join(map(lambda i: self.idx2token[i], tokens))
    # 定义一个方法，用于将输入的字符串编码成字节流，然后调用encodeBytes方法进行编码
    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))
    
    # 定义一个方法，用于将输入的tokens解码成字符串，然后调用decodeBytes方法进行解码
    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')
    
    # 定义一个方法，用于打印tokens中的每个元素
    def printTokens(self, tokens):
        # 遍历tokens中的每个元素
        for i in tokens:
            # 将索引i转换成对应的token
            s = self.idx2token[i]
            # 尝试将token解码成字符串，如果失败则跳过
            try:
                s = s.decode('utf-8')
            except:
                pass
            # 打印token的repr形式和索引i，以空格结尾
            print(f'{repr(s)}{i}', end=' ')
            # 打印tokens中每个元素的repr形式和索引i
            # print(repr(s), i)
        # 打印完所有元素后换行
        print()
# 定义一个 TRIE 类，用于实现 Trie 树数据结构
class TRIE:
    # 定义类的属性，包括字符、指向子节点的列表、值的集合、前缀节点
    __slots__ = tuple("ch,to,values,front".split(","))
    to:list
    values:set
    # 初始化方法，设置节点的字符、子节点列表、值的集合、前缀节点
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    # 重写 repr 方法，返回节点的字符串表示
    def __repr__(self):
        fr = self
        ret = []
        # 遍历节点的前缀节点，将字符添加到列表中
        while(fr!=None):
            if(fr.ch!=None):
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>"%(ret[::-1], self.values)
    
    # 添加方法，向 Trie 树中添加键值对
    def add(self, key:bytes, idx:int=0, val=None):
        # 如果索引等于键的长度，表示已经遍历完整个键
        if(idx == len(key)):
            # 如果值为空，则将值设置为键本身
            if(val is None):
                val = key
            # 将值添加到节点的值集合中
            self.values.add(val)
            return self
        # 获取当前索引对应的字符
        ch = key[idx]
        # 如果当前字符对应的子节点为空，则创建一个新的子节点
        if(self.to[ch] is None):
            self.to[ch] = TRIE(front=self, ch=ch)
        # 递归调用添加方法，继续向下一个子节点添加键值对
        return self.to[ch].add(key, idx=idx+1, val=val)
    
    # 查找最长匹配方法，查找 Trie 树中与给定键最长匹配的节点
    def find_longest(self, key:bytes, idx:int=0):
        u:TRIE = self
        ch:int = key[idx]
        # 遍历 Trie 树，直到找到最长匹配的节点
        while(u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            # 如果当前节点有值，则记录下当前索引、节点和节点的值
            if(u.values):
                ret = idx, u, u.values
            # 如果已经遍历完整个键，则跳出循环
            if(idx==len(key)):
                break
            ch = key[idx]
        return ret

# 定义 TRIE_TOKENIZER 类
class TRIE_TOKENIZER():
    # 初始化方法，接受文件名参数
    def __init__(self, file_name):
        # 初始化索引到标记的字典
        self.idx2token = {}
        # 创建空列表用于存储已排序的标记
        sorted = [] # must be already sorted
        # 打开文件，按行读取内容
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # 遍历文件的每一行
        for l in lines:
            # 从每行中提取索引和标记
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            # 如果标记是字符串，则编码为 utf-8
            x = x.encode("utf-8") if isinstance(x, str) else x
            # 断言标记是字节类型
            assert isinstance(x, bytes)
            # 断言标记长度与行中指定的长度相同
            assert len(x) == int(l[l.rindex(' '):])
            # 将标记添加到已排序列表中
            sorted += [x]
            # 将索引和标记添加到索引到标记的字典中
            self.idx2token[idx] = x

        # 初始化标记到索引的字典
        self.token2idx = {}
        # 遍历索引到标记的字典
        for k,v in self.idx2token.items():
            # 将标记到索引的映射添加到标记到索引的字典中
            self.token2idx[v] = int(k)

        # 初始化 TRIE 树
        self.root = TRIE()
        # 遍历标记到索引的字典
        for t, i in self.token2idx.items():
            # 将标记和索引添加到 TRIE 树中
            _ = self.root.add(t, val=(t, i))

    # 将字节编码为标记序列
    def encodeBytes(self, src:bytes):
        # 初始化索引
        idx:int = 0
        # 初始化标记列表
        tokens = []
        # 循环直到索引达到字节长度
        while (idx < len(src)):
            # 保存当前索引
            _idx:int = idx
            # 在 TRIE 树中查找最长匹配的标记
            idx, _, values = self.root.find_longest(src, idx)
            # 断言找到了不同的索引
            assert(idx != _idx)
            # 获取匹配的标记
            _, token = next(iter(values))            
            # 将标记添加到标记列表中
            tokens.append(token)
        # 返回标记列表
        return tokens

    # 将标记序列解码为字节
    def decodeBytes(self, tokens):
        # 将标记列表连接为字节序列
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    # 将字符串编码为标记序列
    def encode(self, src):
        # 将字符串编码为 utf-8，然后调用 encodeBytes 方法
        return self.encodeBytes(src.encode("utf-8"))

    # 将标记序列解码为字符串
    def decode(self, tokens):
        # 将标记序列解码为字节，然后解码为 utf-8 编码的字符串
        return self.decodeBytes(tokens).decode('utf-8')

    # 打印标记序列
    def printTokens(self, tokens):
        # 遍历标记序列
        for i in tokens:
            # 获取标记
            s = self.idx2token[i]
            # 尝试将标记解码为 utf-8 编码的字符串
            try:
                s = s.decode('utf-8')
            except:
                pass
            # 打印标记和索引
            print(f'{repr(s)}{i}', end=' ')
        # 打印换行符
        print()
# 导入所需的模块
import time

# 定义一个函数，用于读取 ZIP 文件内容并返回文件名到数据的字典
def read_zip(fname):
    bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    zip.close()  # 关闭 ZIP 对象
    return fdict  # 返回结果字典

# 定义一个 RWKV_TOKENIZER 类的实例
TOKENIZER = RWKV_TOKENIZER('rwkv_vocab_v20230424.txt')

# 定义一个 TRIE_TOKENIZER 类的实例
TRIE_TEST = TRIE_TOKENIZER('rwkv_vocab_v20230424.txt')

# 定义一个字符串变量
src = '''起業家イーロン・マスク氏...'''  # 一段日语文本

# 打印字符串内容
print(src)

# 打印字符串长度
print(f'\n{len(src)} chars')

# 对字符串进行编码
tokens = TOKENIZER.encode(src)

# 断言编码后的字符串能够被正确解码
assert TOKENIZER.decode(tokens) == src

# 打印空行
print()

# 打印编码后的 tokens
TOKENIZER.printTokens(tokens)

# 打印 tokens 的长度
print(f'\n{len(tokens)} tokens\n')

# 将字符串内容重复 20 次
src = src * 20
src_len = len(src)
print(f'Benchmark {src_len} tokens...')

# 定义一个函数，用于对编码和解码进行性能测试
def benchmark(XXX):
    min_t = 1e100
    for i in range(10):
        t_begin = time.time_ns()
        tokens = XXX.encode(src)
        min_t = min(time.time_ns() - t_begin, min_t)
    print('Encode', round(src_len / min_t * 1e3, 3), 'MB/s')

    min_t = 1e100
    for i in range(10):
        t_begin = time.time_ns()
        sss = XXX.decode(tokens)
        min_t = min(time.time_ns() - t_begin, min_t)
    print('Decode', round(src_len / min_t * 1e3, 3), 'MB/s')

# 对 TOKENIZER 进行性能测试
benchmark(TOKENIZER)

# 对 TRIE_TEST 进行性能测试
benchmark(TRIE_TEST)
# 单元测试
print('Unit test...')

# 定义测试用例
QQQ = ['', ' ', 'Õ\U000683b8', b'\xe6\xaa\x81'.decode('utf-8')]

# 生成随机字符串并添加到测试用例中
for TRIAL in range(500):
    x = ''
    for xx in [
        ['0',' '],
        ['0','1'],
        ['0','1',' '],
        ['0','1',' ','00','11','  ','000','111','   '],
        list('01 \n\r\t,.;!?:\'\"-=你好')
    ]:
        for i in range(256):
            x += random.choice(xx)
    QQQ += [x]

# 生成空格字符串并添加到测试用例中
for i in range(5000):
    QQQ += [' ' * i]

# 生成随机字符并添加到测试用例中
for TRIAL in range(5000):
    x = chr(random.randrange(0, 256))
    x = x * random.randrange(1, 32)
    QQQ += [x]

# 生成不合法的 UTF-8 字符并添加到测试用例中
for TRIAL in range(99999):
    x = chr(random.randrange(256, 1114112))
    x = x * random.randrange(1, 4)
    try:
        tmp = x.encode("utf-8")
        QQQ += [x]
    except:
        pass

# 添加 UTF-8 解码能力和压力测试的说明文本到测试用例中
QQQ += ['''
UTF-8 decoder capability and stress test
----------------------------------------

Markus Kuhn <http://www.cl.cam.ac.uk/~mgk25/> - 2015-08-28 - CC BY 4.0

This test file can help you examine, how your UTF-8 decoder handles
various types of correct, malformed, or otherwise interesting UTF-8
sequences. This file is not meant to be a conformance test. It does
not prescribe any particular outcome. Therefore, there is no way to
"pass" or "fail" this test file, even though the text does suggest a
preferable decoder behaviour at some places. Its aim is, instead, to
help you think about, and test, the behaviour of your UTF-8 decoder on a
systematic collection of unusual inputs. Experience so far suggests
that most first-time authors of UTF-8 decoders find at least one
serious problem in their decoder using this file.

The test lines below cover boundary conditions, malformed UTF-8
sequences, as well as correctly encoded UTF-8 sequences of Unicode code
points that should never occur in a correct UTF-8 file.
# 根据 ISO 10646-1:2000 的规定，UTF-8 接收设备应该以相同的方式解释“格式错误的序列”和“不在采用子集内的字符”，并且接收设备应该向用户指示“不在采用子集内的字符”。UTF-8 解码器中常用的方法是用替换字符（U+FFFD）来替换任何格式错误的 UTF-8 序列，这个字符看起来有点像倒置的问号，或者类似的符号。可能是一个好主意，从视觉上区分格式错误的 UTF-8 序列和正确编码的 Unicode 字符，即使 ISO 10646-1 没有强制要求。无论如何，忽略格式错误的序列或不可用的字符都不符合 ISO 10646 的规定，会使调试更加困难，并且可能导致用户混淆。

# 请检查格式错误的 UTF-8 序列是否（1）被表示，（2）是否被一个单一的替换字符（或等效信号）所代表，以及（3）在非法的 UTF-8 序列之后的引号是否被正确显示，即在任何格式错误的序列之后立即进行正确的重新同步。这个文件的最后一行写着“THE END”，所以如果你没有看到这个，你的解码器在之前出现了崩溃，这应该总是令人担忧的原因。

# 这个文件中的所有行都恰好是 79 个字符长（加上换行符）。此外，所有行都以“|”结尾，除了两个测试行 2.1.1 和 2.2.1，它们包含不可打印的 ASCII 控制字符 U+0000 和 U+007F。如果你用等宽字体显示这个文件，这些“|”字符应该都对齐在第 79 列（右边缘）。这样可以快速测试你的 UTF-8 解码器是否在每行中找到了正确数量的字符，也就是每个格式错误的序列是否被一个单一的替换字符所替换。
# 注意：在这里使用的“malformed sequence”是指不完整的字节序列，也可以使用替换字符来表示每个不完整的字节序列
# 如果你在解码器中采用了这种策略，请忽略“|”列

# 测试开始
# 1. 一些正确的 UTF-8 文本
# 你应该看到希腊单词'kosme': "κόσμε"

# 2. 边界条件测试用例
# 2.1 一定长度的第一个可能序列
# 2.1.1 1 字节 (U-00000000): "�"
# 2.1.2 2 字节 (U-00000080): ""
# 2.1.3 3 字节 (U-00000800): "ࠀ"
# 2.1.4 4 字节 (U-00010000): "𐀀"
# 2.1.5 5 字节 (U-00200000): "�����"
# 2.1.6 6 字节 (U-04000000): "������"

# 2.2 一定长度的最后一个可能序列
# 2.2.1 1 字节 (U-0000007F): ""
# 定义不同字节数的 Unicode 编码的边界条件
2.2.2  2 bytes (U-000007FF):        "߿"                                       |
2.2.3  3 bytes (U-0000FFFF):        "￿"                                       |
2.2.4  4 bytes (U-001FFFFF):        "����"                                       |
2.2.5  5 bytes (U-03FFFFFF):        "�����"                                       |
2.2.6  6 bytes (U-7FFFFFFF):        "������"                                       |
                                                                              |
2.3  其他边界条件                                                |
                                                                              |
2.3.1  U-0000D7FF = ed 9f bf = "퟿"                                            |
2.3.2  U-0000E000 = ee 80 80 = ""                                            |
2.3.3  U-0000FFFD = ef bf bd = "�"                                            |
2.3.4  U-0010FFFF = f4 8f bf bf = "􏿿"                                         |
2.3.5  U-00110000 = f4 90 80 80 = "����"                                         |
                                                                              |
3  格式错误的序列                                                        |
                                                                              |
3.1  意外的续补字节                                            |
                                                                              |
每个意外的续补字节应该被单独标记为一个格式错误的序列。         |
                                                                              |
3.1.1  第一个续补字节 0x80: "�"                                      |
3.1.2  最后一个续补字节 0xbf: "�"                                      |
                                                                              |
# 定义了各种情况下的 UTF-8 编码的字节序列，用于解析和处理 UTF-8 编码的文本数据
# 3.1.3 - 3.1.8 定义了不同长度的 UTF-8 编码的 continuation bytes
# 3.1.9 定义了所有 64 个可能的 continuation bytes（0x80-0xbf）
# 3.2 定义了 UTF-8 编码中的孤立起始字符的情况
# 3.2.1 定义了所有 2 字节序列的起始字节（0xc0-0xdf）后面跟着空格字符的情况
# 3.2.2  所有3字节序列（0xe0-0xef）的前16个字节，每个后面跟着一个空格字符：
# "� � � � � � � � � � � � � � � � "
# 3.2.3  所有4字节序列（0xf0-0xf7）的前8个字节，每个后面跟着一个空格字符：
# "� � � � � � � � "
# 3.2.4  所有5字节序列（0xf8-0xfb）的前4个字节，每个后面跟着一个空格字符：
# "� � � � "
# 3.2.5  所有6字节序列（0xfc-0xfd）的前2个字节，每个后面跟着一个空格字符：
# "� � "
# 3.3  最后一个连续字节缺失的序列
# 所有不完整序列的字节应该被标记为单个格式错误的序列，即你应该只看到一个替换字符
# 以下是对接下来的 10 个测试中每个字符的描述
# 3.3.1  2 字节序列，最后一个字节丢失（U+0000）："�"
# 3.3.2  3 字节序列，最后一个字节丢失（U+0000）："��"
# 3.3.3  4 字节序列，最后一个字节丢失（U+0000）："���"
# 3.3.4  5 字节序列，最后一个字节丢失（U+0000）："����"
# 3.3.5  6 字节序列，最后一个字节丢失（U+0000）："�����"
# 3.3.6  2 字节序列，最后一个字节丢失（U-000007FF）："�"
# 3.3.7  3 字节序列，最后一个字节丢失（U-0000FFFF）："�"
# 3.3.8  4 字节序列，最后一个字节丢失（U-001FFFFF）："���"
# 3.3.9  5 字节序列，最后一个字节丢失（U-03FFFFFF）："����"
# 3.3.10 6 字节序列，最后一个字节丢失（U-7FFFFFFF）："�����"

# 3.4 连接不完整序列
# 将 3.3 中的 10 个序列连接起来，你应该看到 10 个错误的序列被标记为："�����������������������������"

# 3.5 不可能出现的字节
# 以下两个字节不可能出现在正确的 UTF-8 字符串中
# 定义特殊字符 fe 和 ff
fe = "�"
ff = "�"
# 定义特殊字符 fe fe ff ff
fe fe ff ff = "����"
# 过长序列
# 以下序列在 Unicode 2.0 标准中不是格式错误的。然而，它们比必要的长度长，正确的 UTF-8 编码器不允许产生它们。
# "安全的 UTF-8 解码器"应该拒绝它们，就像格式错误的序列一样，有两个原因：
# (1) 如果过长序列不被视为有效的字符表示，有助于更快地发现问题，有助于调试应用程序。
# (2) 过长序列提供了字符的替代表示，可能被恶意使用来绕过仅检查 ASCII 字符的过滤器。
# 例如，一个 2 字节编码的换行符 (LF) 不会被只计算 0x0a 字节的行计数器捕捉到，但它仍然会在后续的不安全的 UTF-8 解码器中被处理为换行符。
# 从安全性的角度来看，UTF-8 序列的 ASCII 兼容性也意味着，ASCII 字符 *只* 允许由范围在 0x00-0x7f 的 ASCII 字节表示。
# 为了确保 ASCII 兼容性的这一方面，只使用拒绝过长 UTF-8 序列的 "安全的 UTF-8 解码器"，其中存在更短的编码。
# 以下是对 ASCII 字符 "/" 的五种过长表示的示例
# 如果使用安全的 UTF-8 解码器，所有这些过长表示都应该被拒绝，例如通过替换为替换字符
# 如果你在下面看到斜杠，那么你没有安全的 UTF-8 解码器！

# U+002F = c0 af = "��"
# U+002F = e0 80 af = "���"
# U+002F = f0 80 80 af = "����"
# U+002F = f8 80 80 80 af = "�����"
# U+002F = fc 80 80 80 80 af = "������"

# 下面是使用给定字节数表示的仍然会导致过长序列的最高 Unicode 值
# 这是对安全的 UTF-8 解码器的边界测试。所有五个字符都应该被拒绝，就像是格式错误的 UTF-8 序列一样。

# 4.2.1 U-0000007F = c1 bf = "��"
# 4.2.2 U-000007FF = e0 9f bf = "���"
# 4.2.3 U-0000FFFF = f0 8f bf bf = "����"
# 定义了一系列 UTF-8 编码的字符，包括了不合法的、过长表示的 NUL 字符和单个 UTF-16 代理项
# 这些字符应该被拒绝，因为它们不代表有效的 ISO 10646 字符，而且一个接受它们的 UTF-8 解码器可能会引入类似过长 UTF-8 序列的安全问题
# 过长表示的 NUL 字符应该被拒绝，不应该被当作 ASCII NUL 字符对待
# 单个 UTF-16 代理项也应该被拒绝
# UTF-16 编码中的特殊情况
# 单个 UTF-16 代理项
5.1.1  U+D800 = ed a0 80 = "���"  # 单个 UTF-16 代理项的编码
5.1.2  U+DB7F = ed ad bf = "���"  # 单个 UTF-16 代理项的编码
5.1.3  U+DB80 = ed ae 80 = "���"  # 单个 UTF-16 代理项的编码
5.1.4  U+DBFF = ed af bf = "���"  # 单个 UTF-16 代理项的编码
5.1.5  U+DC00 = ed b0 80 = "���"  # 单个 UTF-16 代理项的编码
5.1.6  U+DF80 = ed be 80 = "���"  # 单个 UTF-16 代理项的编码
5.1.7  U+DFFF = ed bf bf = "���"  # 单个 UTF-16 代理项的编码

# 成对的 UTF-16 代理项
5.2.1  U+D800 U+DC00 = ed a0 80 ed b0 80 = "������"  # 成对 UTF-16 代理项的编码
5.2.2  U+D800 U+DFFF = ed a0 80 ed bf bf = "������"  # 成对 UTF-16 代理项的编码
5.2.3  U+DB7F U+DC00 = ed ad bf ed b0 80 = "������"  # 成对 UTF-16 代理项的编码
5.2.4  U+DB7F U+DFFF = ed ad bf ed bf bf = "������"  # 成对 UTF-16 代理项的编码
5.2.5  U+DB80 U+DC00 = ed ae 80 ed b0 80 = "������"  # 成对 UTF-16 代理项的编码
5.2.6  U+DB80 U+DFFF = ed ae 80 ed bf bf = "������"  # 成对 UTF-16 代理项的编码
5.2.7  U+DBFF U+DC00 = ed af bf ed b0 80 = "������"  # 成对 UTF-16 代理项的编码
5.2.8  U+DBFF U+DFFF = ed af bf ed bf bf = "������"  # 成对 UTF-16 代理项的编码

# 非字符代码位置
# 以下“非字符”由应用程序“保留内部使用”，根据 Unicode 校正声明 #9，这些“非字符”“不应该互换”。Unicode 校正声明 #9 删除了这些规定
5.3 Noncharacter code positions
# 这部分代码是一段文档，描述了UTF-8数据中的非字符代码点的潜在安全风险以及在内部使用中可能触发的操作
# 例如在一些文件API中，16位字符可能使用整数值-1（U+FFFF）来表示文件结束（EOF）或错误条件
# 在一些UTF-16接收器中，代码点U+FFFE可能触发字节交换操作（在UTF-16LE和UTF-16BE之间转换）
# 对于这些非字符的内部使用，可能希望在UTF-8解码器中阻止这些代码点，因为它们在传入的UTF-8数据中不应该合法地出现，并且可能在后续处理中触发不安全的行为
# 特别问题的非字符在16位应用程序中是U+FFFE和U+FFFF
# 其他非字符包括U+FDD0到U+FDEF
# 以下是一个UTF-8编码的样本纯文本文件，包含了各种语言和符号的示例
# 该文件包含了一些数学、科学、语言学和字典学的示例
# 以及APL编程语言的示例
5.3.4  U+nFFFE U+nFFFF (for n = 1..10)                                        |
                                                                              |
       "🿾🿿𯿾𯿿𿿾𿿿񏿾񏿿񟿾񟿿񯿾񯿿񿿾񿿿򏿾򏿿                                    |
        򟿾򟿿򯿾򯿿򿿾򿿿󏿾󏿿󟿾󟿿󯿾󯿿󿿾󿿿􏿾􏿿"                                   |
                                                                              |
THE END                                                                       |
# 以上是一些十六进制编码的字符，可能是UTF-8编码的一部分

UTF-8 encoded sample plain-text file
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

Markus Kuhn [ˈmaʳkʊs kuːn] <http://www.cl.cam.ac.uk/~mgk25/> — 2002-07-25 CC BY
# 作者信息和版权信息

The ASCII compatible UTF-8 encoding used in this plain-text file
is defined in Unicode, ISO 10646-1, and RFC 2279.
# 该纯文本文件使用的ASCII兼容的UTF-8编码在Unicode、ISO 10646-1和RFC 2279中定义

Using Unicode/UTF-8, you can write in emails and source code things such as

Mathematics and sciences:
# 数学和科学示例

  ∮ E⋅da = Q,  n → ∞, ∑ f(i) = ∏ g(i),      ⎧⎡⎛┌─────┐⎞⎤⎫
                                            ⎪⎢⎜│a²+b³ ⎟⎥⎪
  ∀x∈ℝ: ⌈x⌉ = −⌊−x⌋, α ∧ ¬β = ¬(¬α ∨ β),    ⎪⎢⎜│───── ⎟⎥⎪
                                            ⎪⎢⎜⎷ c₈   ⎟⎥⎪
  ℕ ⊆ ℕ₀ ⊂ ℤ ⊂ ℚ ⊂ ℝ ⊂ ℂ,                   ⎨⎢⎜       ⎟⎥⎬
                                            ⎪⎢⎜ ∞     ⎟⎥⎪
  ⊥ < a ≠ b ≡ c ≤ d ≪ ⊤ ⇒ (⟦A⟧ ⇔ ⟪B⟫),      ⎪⎢⎜ ⎲     ⎟⎥⎪
                                            ⎪⎢⎜ ⎳aⁱ-bⁱ⎟⎥⎪
  2H₂ + O₂ ⇌ 2H₂O, R = 4.7 kΩ, ⌀ 200 mm     ⎩⎣⎝i=1    ⎠⎦⎭
# 数学和科学示例

Linguistics and dictionaries:
# 语言学和字典学示例

  ði ıntəˈnæʃənəl fəˈnɛtık əsoʊsiˈeıʃn
  Y [ˈʏpsilɔn], Yen [jɛn], Yoga [ˈjoːgɑ]
# 语言学和字典学示例

APL:
# APL编程语言示例

  ((V⍳V)=⍳⍴V)/V←,V    ⌷←⍳→⍴∆∇⊃‾⍎⍕⌈
# APL编程语言示例
# 在纯文本文件中实现更美观的排版

  ╔══════════════════════════════════════════╗
  ║                                          ║
  ║   • ‘single’ and “double” quotes         ║
  ║                                          ║
  ║   • Curly apostrophes: “We’ve been here” ║
  ║                                          ║
  ║   • Latin-1 apostrophe and accents: '´`  ║
  ║                                          ║
  ║   • ‚deutsche‘ „Anführungszeichen“       ║
  ║                                          ║
  ║   • †, ‡, ‰, •, 3–4, —, −5/+5, ™, …      ║
  ║                                          ║
  ║   • ASCII safety test: 1lI|, 0OD, 8B     ║
  ║                      ╭─────────╮         ║
  ║   • the euro symbol: │ 14.95 € │         ║
  ║                      ╰─────────╯         ║
  ╚══════════════════════════════════════════╝

# 合并字符：

  STARGΛ̊TE SG-1, a = v̇ = r̈, a⃑ ⊥ b⃑
# 希腊语（多音节）：

  希腊国歌：

  我认识你从剑的刃口
  那可怕的剑，
  我认识你从脸庞
  用暴力测量大地。

  从希腊人的骨骼中
  圣洁的
  和像第一次那样勇敢
  你好，你好，自由！

  公元前4世纪德摩斯特尼斯的演讲：

  雅典人啊，我看到的并不是同样的事情，
  当我回顾事情并听到言辞时；
  因为有人在谈论惩罚菲利普，
  有人在处理这些事情，以便我们不会先误解。
  因此，我认为那些说这样话的人除了议题，
  他们没有给你们呈现真相。但我，我曾经
  为城市和自己的安全和惩罚菲利普，
  我非常清楚；因为对我来说，这两者早已
  不是新鲜事；但现在我确信我们能够
  充分预见第一步，以便我们能够拯救盟友。
  因为如果这一点确实存在，那么我们就可以确定
  谁将受到惩罚以及如何审视；但在正确假设之前，
  我认为对于结果进行推测是徒劳的。

  德摩斯特尼斯，第三奥林匹亚

格鲁吉亚语：

  从Unicode会议邀请函中：

  请立即注册参加第十届全球Unicode会议，
  该会议将于3月10-12日举行，
  在德国慕尼黑。会议将聚集全球专家，
  他们在互联网和Unicode等领域拥有专业知识，
  国际化和本地化，Unicode在操作系统中的使用，
  以及在程序，字体，文本处理和多语言计算机系统中的使用。
# 俄语文本，Unicode 会议邀请函
Russian:

  From a Unicode conference invitation:

  Зарегистрируйтесь сейчас на Десятую Международную Конференцию по
  Unicode, которая состоится 10-12 марта 1997 года в Майнце в Германии.
  Конференция соберет широкий круг экспертов по  вопросам глобального
  Интернета и Unicode, локализации и интернационализации, воплощению и
  применению Unicode в различных операционных системах и программных
  приложениях, шрифтах, верстке и многоязычных компьютерных системах.

# 泰语文本，三国演义中的诗歌
Thai (UCS Level 2):

  Excerpt from a poetry on The Romance of The Three Kingdoms (a Chinese
  classic 'San Gua'):

  [----------------------------|------------------------]
    ๏ แผ่นดินฮั่นเสื่อมโทรมแสนสังเวช  พระปกเกศกองบู๊กู้ขึ้นใหม่
  สิบสองกษัตริย์ก่อนหน้าแลถัดไป       สององค์ไซร้โง่เขลาเบาปัญญา
    ทรงนับถือขันทีเป็นที่พึ่ง           บ้านเมืองจึงวิปริตเป็นนักหนา
  โฮจิ๋นเรียกทัพทั่วหัวเมืองมา         หมายจะฆ่ามดชั่วตัวสำคัญ
    เหมือนขับไสไล่เสือจากเคหา      รับหมาป่าเข้ามาเลยอาสัญ
  ฝ่ายอ้องอุ้นยุแยกให้แตกกัน          ใช้สาวนั้นเป็นชนวนชื่นชวนใจ
    พลันลิฉุยกุยกีกลับก่อเหตุ          ช่างอาเพศจริงหนาฟ้าร้องไห้
  ต้องรบราฆ่าฟันจนบรรลัย           ฤๅหาใครค้ำชูกู้บรรลังก์ ฯ

  (The above is a two-column text. If combining characters are handled
  correctly, the lines of the second column should be aligned with the
  | character above.)

# 埃塞俄比亚语文本，阿姆哈拉语的谚语
Ethiopian:

  Proverbs in the Amharic language:

  ሰማይ አይታረስ ንጉሥ አይከሰስ።
  ብላ ካለኝ እንደአባቴ በቆመጠኝ።
  ጌጥ ያለቤቱ ቁምጥና ነው።
  ደሀ በሕልሙ ቅቤ ባይጠጣ ንጣት በገደለው።
  የአፍ ወለምታ በቅቤ አይታሽም።
  አይጥ በበላ ዳዋ ተመታ።
  ሲተረጉሙ ይደረግሙ።
  ቀስ በቀስ፥ ዕንቁላል በእግሩ ይሄዳል።
  ድር ቢያብር አንበሳ ያስር።
  ሰው እንደቤቱ እንጅ እንደ ጉረቤቱ አይተዳደርም።
  እግዜር የከፈተውን ጉሮሮ ሳይዘጋው አይድርም።
  የጎረቤት ሌባ፥ ቢያዩት ይስቅ ባያዩት ያጠልቅ።
  ሥራ ከመፍታት ልጄን ላፋታት።
  ዓባይ ማደሪያ የለው፥ ግንድ ይዞ ይዞራል።
  የእስላም አገሩ መካ የአሞራ አገሩ ዋርካ።
  ተንጋሎ ቢተፉ ተመልሶ ባፉ።
  ወዳጅህ ማር ቢሆን ጨርስህ አትላሰው።
  እግርህን በፍራሽህ ልክ ዘርጋ።
# 这部分代码是一些乱码和特殊字符的展示，没有实际的程序代码含义
# 这部分代码是一系列的文本，包含了不同语言的句子，用于测试文本对齐和显示的效果
# 这部分代码并不需要注释，因为它只是用于测试文本显示效果的文本数据
# 巴西葡萄牙语：我可以吃玻璃，不会伤害我。
# 佛得角克里奥尔语（佛得角）：我可以吃玻璃，不会伤害我。
# 帕皮阿门图语：我可以吃玻璃，它不会伤害我。
# 意大利语：我可以吃玻璃，不会伤害我。
# 米兰语：我可以吃玻璃，它不会伤害我。
# 罗马语：我可以吃玻璃，不会伤害我。
# 那不勒斯语：我可以吃玻璃，它不会伤害我。
# 威尼斯语：我可以吃玻璃，它不会伤害我。
# 热那亚语（热那亚）：我可以吃玻璃，它不会伤害我。
# 西西里语：我可以吃玻璃，不会伤害我。
# 卡皮纳德斯语（撒丁岛）：（需要）
# 卢古多雷斯语（撒丁岛）：（需要）
# 罗曼什语（格里松）：我可以吃玻璃，它不会伤害我。
# 罗姆人/吉普赛语：（需要）
# 罗马尼亚语：我可以吃玻璃，它不会伤害我。
# 世界语：我可以吃玻璃，它不会伤害我。
# 皮克特语：（需要）
# 布列塔尼语：（需要）
# 康沃尔语：我可以吃玻璃，它不会伤害我。
# 威尔士语：我可以吃玻璃，它不会伤害我。
# 曼岛盖尔语：我可以吃玻璃，它不会伤害我。
# 古爱尔兰语（欧甘文）：我可以吃玻璃，它不会伤害我。
# 古爱尔兰语（拉丁文）：我可以吃玻璃，它不会伤害我。
# 爱尔兰语：我可以吃玻璃，它不会伤害我。
# 阿尔斯特盖尔语：我可以吃玻璃，它不会伤害我。
# 苏格兰盖尔语：我可以吃玻璃，它不会伤害我。
# 古英语（符文）：我可以吃玻璃，它不会伤害我。
# 古英语（拉丁文）：我可以吃玻璃，它不会伤害我。
# 中古英语：我可以吃玻璃，它不会伤害我。
# 英语：我可以吃玻璃，它不会伤害我。
# 英语（国际音标）：[aɪ kæn iːt glɑːs ænd ɪt dɐz nɒt hɜːt miː]（标准英音）
# 英语（盲文）：⠊⠀⠉⠁⠝⠀⠑⠁⠞⠀⠛⠇⠁⠎⠎⠀⠁⠝⠙⠀⠊⠞⠀⠙⠕⠑⠎⠝⠞⠀⠓⠥⠗⠞⠀⠍⠑
# 牙买加语：我可以吃玻璃，它永远不会伤害我。
# 苏格兰-英语/多里克语：我可以吃玻璃，它不会伤害我们。
# 格拉斯哥语：（需要）
# 哥特语（4）：我可以吃玻璃，它不会伤害我。
# 古挪威语（符文）：我可以吃玻璃，它不会伤害我。
# Old Norse (Latin)的翻译
Old Norse (Latin): Ek get etið gler án þess að verða sár.
# Norsk / Norwegian (Nynorsk)的翻译
Norsk / Norwegian (Nynorsk): Eg kan eta glas utan å skada meg.
# Norsk / Norwegian (Bokmål)的翻译
Norsk / Norwegian (Bokmål): Jeg kan spise glass uten å skade meg.
# Føroyskt / Faroese的翻译
Føroyskt / Faroese: Eg kann eta glas, skaðaleysur.
# Íslenska / Icelandic的翻译
Íslenska / Icelandic: Ég get etið gler án þess að meiða mig.
# Svenska / Swedish的翻译
Svenska / Swedish: Jag kan äta glas utan att skada mig.
# Dansk / Danish的翻译
Dansk / Danish: Jeg kan spise glas, det gør ikke ondt på mig.
# Sønderjysk的翻译
Sønderjysk: Æ ka æe glass uhen at det go mæ naue.
# Frysk / Frisian的翻译
Frysk / Frisian: Ik kin glês ite, it docht me net sear.
# Nederlands / Dutch的翻译
Nederlands / Dutch: Ik kan glas eten, het doet mĳ geen kwaad.
# Kirchröadsj/Bôchesserplat的翻译
Kirchröadsj/Bôchesserplat: Iech ken glaas èèse, mer 't deet miech jing pieng.
# Afrikaans的翻译
Afrikaans: Ek kan glas eet, maar dit doen my nie skade nie.
# Lëtzebuergescht / Luxemburgish的翻译
Lëtzebuergescht / Luxemburgish: Ech kan Glas iessen, daat deet mir nët wei.
# Deutsch / German的翻译
Deutsch / German: Ich kann Glas essen, ohne mir zu schaden.
# Ruhrdeutsch的翻译
Ruhrdeutsch: Ich kann Glas verkasematuckeln, ohne dattet mich wat jucken tut.
# Langenfelder Platt的翻译
Langenfelder Platt: Isch kann Jlaas kimmeln, uuhne datt mich datt weh dääd.
# Lausitzer Mundart ("Lusatian")的翻译
Lausitzer Mundart ("Lusatian"): Ich koann Gloos assn und doas dudd merr ni wii.
# Odenwälderisch的翻译
Odenwälderisch: Iech konn glaasch voschbachteln ohne dass es mir ebbs daun doun dud.
# Sächsisch / Saxon的翻译
Sächsisch / Saxon: 'sch kann Glos essn, ohne dass'sch mer wehtue.
# Pfälzisch的翻译
Pfälzisch: Isch konn Glass fresse ohne dasses mer ebbes ausmache dud.
# Schwäbisch / Swabian的翻译
Schwäbisch / Swabian: I kå Glas frässa, ond des macht mr nix!
# Deutsch (Voralberg)的翻译
Deutsch (Voralberg): I ka glas eassa, ohne dass mar weh tuat.
# Bayrisch / Bavarian的翻译
Bayrisch / Bavarian: I koh Glos esa, und es duard ma ned wei.
# Allemannisch的翻译
Allemannisch: I kaun Gloos essen, es tuat ma ned weh.
# Schwyzerdütsch (Zürich)的翻译
Schwyzerdütsch (Zürich): Ich chan Glaas ässe, das schadt mir nöd.
# Schwyzerdütsch (Luzern)的翻译
Schwyzerdütsch (Luzern): Ech cha Glâs ässe, das schadt mer ned.
# Plautdietsch的翻译
Plautdietsch: (NEEDED)
# Hungarian的翻译
Hungarian: Meg tudom enni az üveget, nem lesz tőle bajom.
# Suomi / Finnish的翻译
Suomi / Finnish: Voin syödä lasia, se ei vahingoita minua.
# Sami (Northern)的翻译
Sami (Northern): Sáhtán borrat lása, dat ii leat bávččas.
# Erzian的翻译
Erzian: Мон ярсан суликадо, ды зыян эйстэнзэ а ули.
# Northern Karelian的翻译
Northern Karelian: Mie voin syvvä lasie ta minla ei ole kipie.
# 南卡累利亚语：我可以吃玻璃而不伤害我。
# 维普斯语：（需要）
# 沃蒂语：（需要）
# 利沃尼亚语：（需要）
# 爱沙尼亚语：我可以吃玻璃，对我没有任何伤害。
# 拉脱维亚语：我可以吃玻璃，它不会伤害我。
# 立陶宛语：我可以吃玻璃，它不会伤害我。
# 古普鲁士语：（需要）
# 索布语（温迪什语）：（需要）
# 捷克语：我可以吃玻璃，不会伤害我。
# 斯洛伐克语：我可以吃玻璃。不会伤害我。
# 波兰语：我可以吃玻璃，它不会伤害我。
# 斯洛文尼亚语：我可以吃玻璃，而不会伤害我。
# 波斯尼亚语、克罗地亚语、黑山语和塞尔维亚语（拉丁文）：我可以吃玻璃，对我没有伤害。
# 波斯尼亚语、黑山语和塞尔维亚语（西里尔文）：我可以吃玻璃，对我没有伤害。
# 马其顿语：我可以吃玻璃，不会伤害我。
# 俄语：我可以吃玻璃，它对我没有害处。
# 白俄罗斯语（西里尔文）：我可以吃玻璃，它对我没有伤害。
# 白俄罗斯语（拉丁文）：我可以吃玻璃，它对我没有伤害。
# 乌克兰语：我可以吃玻璃，它对我没有伤害。
# 保加利亚语：我可以吃玻璃，它对我没有伤害。
# 格鲁吉亚语：我吃了玻璃，没有伤害我。
# 亚美尼亚语：我可以吃玻璃，它对我没有伤害。
# 阿尔巴尼亚语：我可以吃玻璃，对我没有伤害。
# 土耳其语：我可以吃玻璃，对我没有伤害。
# 土耳其语（奥斯曼语）：我可以吃玻璃，对我没有伤害。
# 鞑靼语：我可以吃玻璃，但它不会伤害我。
# 乌兹别克语（罗马文）：我可以吃玻璃，但它不会伤害我。
# 乌兹别克语（西里尔文）：我可以吃玻璃，但它不会伤害我。
# 孟加拉语：我可以吃玻璃，对我没有伤害。
# 马拉地语（男性）：我可以吃玻璃，对我没有伤害。
# 马拉地语（女性）：我可以吃玻璃，对我没有伤害。
# 卡纳达语：我可以吃玻璃，没有伤害我。
# 印地语（男性）：我可以吃玻璃，对我没有伤害。
# 印地语（女性）：我可以吃玻璃，对我没有伤害。
# 马拉雅拉姆语：我可以吃玻璃，它不会伤害我。
# 泰米尔语：我可以吃玻璃，它不会伤害我。
# Telugu: 我不能吃玻璃，也不会受伤。
Sinhalese: මම වීදුරු කෑමට හැකියි. එයින් මට කිසි හානියක් සිදු නොවේ.
Urdu(3): میں کانچ کھا سکتا ہوں اور مجھے تکلیف نہیں ہوتی ۔
Pashto(3): زه شيشه خوړلې شم، هغه ما نه خوږوي
Farsi / Persian(3): .من می توانم بدونِ احساس درد شيشه بخورم
Arabic(3): أنا قادر على أكل الزجاج و هذا لا يؤلمني.
Aramaic: (NEEDED)
Maltese: Nista' niekol il-ħġieġ u ma jagħmilli xejn.
Hebrew(3): אני יכול לאכול זכוכית וזה לא מזיק לי.
Yiddish(3): איך קען עסן גלאָז און עס טוט מיר נישט װײ.
Judeo-Arabic: (NEEDED)
Ladino: (NEEDED)
Gǝʼǝz: (NEEDED)
Amharic: (NEEDED)
Twi: Metumi awe tumpan, ɜnyɜ me hwee.
Hausa (Latin): Inā iya taunar gilāshi kuma in gamā lāfiyā.
Hausa (Ajami) (2): إِنا إِىَ تَونَر غِلَاشِ كُمَ إِن غَمَا لَافِىَا
Yoruba(4): Mo lè je̩ dígí, kò ní pa mí lára.
Lingala: Nakokí kolíya biténi bya milungi, ekosála ngáí mabé tɛ́.
(Ki)Swahili: Naweza kula bilauri na sikunyui.
Malay: Saya boleh makan kaca dan ia tidak mencederakan saya.
Tagalog: Kaya kong kumain nang bubog at hindi ako masaktan.
Chamorro: Siña yo' chumocho krestat, ti ha na'lalamen yo'.
Fijian: Au rawa ni kana iloilo, ia au sega ni vakacacani kina.
Javanese: Aku isa mangan beling tanpa lara.
Burmese (Unicode 4.0): က္ယ္ဝန္‌တော္‌၊က္ယ္ဝန္‌မ မ္ယက္‌စားနုိင္‌သည္‌။ ၎က္ရောင္‌့ ထိခုိက္‌မ္ဟု မရ္ဟိပာ။ (9)
Burmese (Unicode 5.0): ကျွန်တော် ကျွန်မ မှန်စားနိုင်တယ်။ ၎င်းကြောင့် ထိခိုက်မှုမရှိပါ။ (9)
Vietnamese (quốc ngữ): Tôi có thể ăn thủy tinh mà không hại gì.
Vietnamese (nôm) (4): 些 𣎏 世 咹 水 晶 𦓡 空 𣎏 害 咦
Khmer: ខ្ញុំអាចញុំកញ្ចក់បាន ដោយគ្មានបញ្ហារ
Lao: ຂອ້ຍກິນແກ້ວໄດ້ໂດຍທີ່ມັນບໍ່ໄດ້ເຮັດໃຫ້ຂອ້ຍເຈັບ.
Thai: ฉันกินกระจกได้ แต่มันไม่ทำให้ฉันเจ็บ
Mongolian (Cyrillic): Би шил идэй чадна, надад хортой биш
Mongolian (Classic) (5): ᠪᠢ ᠰᠢᠯᠢ ᠢᠳᠡᠶᠦ ᠴᠢᠳᠠᠨᠠ ᠂ ᠨᠠᠳᠤᠷ ᠬᠣᠤᠷᠠᠳᠠᠢ ᠪᠢᠰᠢ
Dzongkha: (NEEDED)
Nepali: ﻿म काँच खान सक्छू र मलाई केहि नी हुन्‍न् ।
Tibetan: ཤེལ་སྒོ་ཟ་ནས་ང་ན་གི་མ་རེད།
Chinese: 我能吞下玻璃而不伤身体。
Chinese (Traditional): 我能吞下玻璃而不傷身體。
# 定义一系列不同语言的字符串
Taiwanese(6): Góa ē-tàng chia̍h po-lê, mā bē tio̍h-siong.
Japanese: 私はガラスを食べられます。それは私を傷つけません。
Korean: 나는 유리를 먹을 수 있어요. 그래도 아프지 않아요
Bislama: Mi save kakae glas, hemi no save katem mi.
Hawaiian: Hiki iaʻu ke ʻai i ke aniani; ʻaʻole nō lā au e ʻeha.
Marquesan: E koʻana e kai i te karahi, mea ʻā, ʻaʻe hauhau.
Inuktitut (10): ᐊᓕᒍᖅ ᓂᕆᔭᕌᖓᒃᑯ ᓱᕋᙱᑦᑐᓐᓇᖅᑐᖓ
Chinook Jargon: Naika məkmək kakshət labutay, pi weyk ukuk munk-sik nay.
Navajo: Tsésǫʼ yishą́ągo bííníshghah dóó doo shił neezgai da.
Cherokee (and Cree, Chickasaw, Cree, Micmac, Ojibwa, Lakota, Náhuatl, Quechua, Aymara, and other American languages): (NEEDED)
Garifuna: (NEEDED)
Gullah: (NEEDED)
Lojban: mi kakne le nu citka le blaci .iku'i le se go'i na xrani mi
Nórdicg: Ljœr ye caudran créneþ ý jor cẃran.
''']
        
# 遍历字符串列表
for q in QQQ:
    # 对每个字符串进行编码
    tokens = TOKENIZER.encode(q)
    # 如果编码后的字符串与原始字符串不一致，则输出错误信息
    if q != TOKENIZER.decode(tokens):
        print('ERROR', q)
    # 如果编码后的字符串与使用TRIE_TEST编码的结果不一致，则输出错误信息
    if str(tokens) != str(TRIE_TEST.encode(q)):
        print('ERROR', q)

# 输出所有字符串编码均正常的信息
print('All OK\n')
```