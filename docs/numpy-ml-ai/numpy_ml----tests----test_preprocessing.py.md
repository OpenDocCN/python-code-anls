# `numpy-ml\numpy_ml\tests\test_preprocessing.py`

```
# 禁用 flake8 检查
from collections import Counter

# 导入 huffman 模块
import huffman
# 导入 numpy 模块
import numpy as np

# 从 scipy.fftpack 模块导入 dct 函数
from scipy.fftpack import dct

# 从 sklearn.preprocessing 模块导入 StandardScaler 类
from sklearn.preprocessing import StandardScaler
# 从 sklearn.feature_extraction.text 模块导入 TfidfVectorizer 类
from sklearn.feature_extraction.text import TfidfVectorizer

# 尝试导入 librosa.core.time_frequency 模块中的 fft_frequencies 函数，如果导入失败则导入 librosa 模块中的 fft_frequencies 函数
try:
    from librosa.core.time_frequency import fft_frequencies
except ImportError:
    # 对于 librosa 版本 >= 0.8.0
    from librosa import fft_frequencies
# 从 librosa.feature 模块导入 mfcc 函数
from librosa.feature import mfcc as lr_mfcc
# 从 librosa.util 模块导入 frame 函数
from librosa.util import frame
# 从 librosa.filters 模块导入 mel 函数
from librosa.filters import mel

# 导入 numpy_ml 模块中的相关实现
from numpy_ml.preprocessing.general import Standardizer
from numpy_ml.preprocessing.nlp import HuffmanEncoder, TFIDFEncoder
from numpy_ml.preprocessing.dsp import (
    DCT,
    DFT,
    mfcc,
    to_frames,
    mel_filterbank,
    dft_bins,
)
from numpy_ml.utils.testing import random_paragraph


# 测试 Huffman 编码
def test_huffman(N=15):
    np.random.seed(12345)

    i = 0
    while i < N:
        # 生成随机单词数量
        n_words = np.random.randint(1, 100)
        # 生成随机段落
        para = random_paragraph(n_words)
        # 创建 Huffman 编码器对象
        HT = HuffmanEncoder()
        # 对段落进行编码
        HT.fit(para)
        # 获取编码后的字典
        my_dict = HT._item2code
        # 使用 gold-standard 的 huffman 模块生成编码字典
        their_dict = huffman.codebook(Counter(para).items())

        # 检查两个字典是否一致
        for k, v in their_dict.items():
            fstr = "their_dict['{}'] = {}, but my_dict['{}'] = {}"
            assert k in my_dict, "key `{}` not in my_dict".format(k)
            assert my_dict[k] == v, fstr.format(k, v, k, my_dict[k])
        print("PASSED")
        i += 1


# 测试标准化器
def test_standardizer(N=15):
    np.random.seed(12345)

    i = 0
    while i < N:
        # 随机生成是否计算均值和标准差的标志
        mean = bool(np.random.randint(2))
        std = bool(np.random.randint(2))
        # 随机生成矩阵的行数和列数
        N = np.random.randint(2, 100)
        M = np.random.randint(2, 100)
        # 生成随机矩阵
        X = np.random.rand(N, M)

        # 创建标准化器对象
        S = Standardizer(with_mean=mean, with_std=std)
        # 对数据进行拟合
        S.fit(X)
        # 进行标准化处理
        mine = S.transform(X)

        # 使用 sklearn 中的 StandardScaler 进行标准化
        theirs = StandardScaler(with_mean=mean, with_std=std)
        gold = theirs.fit_transform(X)

        # 检查两种标准化结果是否接近
        np.testing.assert_almost_equal(mine, gold)
        print("PASSED")
        i += 1
# 测试 TF-IDF 编码器的功能，生成 N 个测试用例
def test_tfidf(N=15):
    # 设置随机种子
    np.random.seed(12345)

    # 初始化计数器 i
    i = 0
    # 循环生成 N 个测试用例
    while i < N:
        # 初始化文档列表
        docs = []
        # 随机生成文档数量
        n_docs = np.random.randint(1, 10)
        # 生成每个文档的内容
        for d in range(n_docs):
            # 随机生成每个文档的行数
            n_lines = np.random.randint(1, 1000)
            # 生成每行的随机段落
            lines = [random_paragraph(np.random.randint(1, 10)) for _ in range(n_lines)]
            # 将每行段落连接成文档
            docs.append("\n".join([" ".join(l) for l in lines]))

        # 随机选择是否平滑 IDF
        smooth = bool(np.random.randint(2))

        # 初始化 TF-IDF 编码器对象
        tfidf = TFIDFEncoder(
            lowercase=True,
            min_count=0,
            smooth_idf=smooth,
            max_tokens=None,
            input_type="strings",
            filter_stopwords=False,
        )
        # 初始化 sklearn 中的 TfidfVectorizer 对象作为对照
        gold = TfidfVectorizer(
            input="content",
            norm=None,
            use_idf=True,
            lowercase=True,
            smooth_idf=smooth,
            sublinear_tf=False,
        )

        # 对 TF-IDF 编码器进行拟合
        tfidf.fit(docs)
        # 获取 TF-IDF 编码结果
        mine = tfidf.transform(ignore_special_chars=True)
        # 获取 sklearn 中 TfidfVectorizer 的结果并转换为数组
        theirs = gold.fit_transform(docs).toarray()

        # 断言 TF-IDF 编码结果与对照结果的近似相等
        np.testing.assert_almost_equal(mine, theirs)
        # 打印测试通过信息
        print("PASSED")
        # 计数器加一
        i += 1


# 测试 DCT 变换的功能，生成 N 个测试用例
def test_dct(N=15):
    # 设置随机种子
    np.random.seed(12345)

    # 初始化计数器 i
    i = 0
    # 循环生成 N 个测试用例
    while i < N:
        # 随机生成信号长度 N
        N = np.random.randint(2, 100)
        # 随机生成信号
        signal = np.random.rand(N)
        # 随机选择是否正交
        ortho = bool(np.random.randint(2))
        # 计算自定义的 DCT 变换
        mine = DCT(signal, orthonormal=ortho)
        # 计算 scipy 中的 DCT 变换作为对照
        theirs = dct(signal, norm="ortho" if ortho else None)

        # 断言自定义 DCT 变换结果与对照结果的近似相等
        np.testing.assert_almost_equal(mine, theirs)
        # 打印测试通过信息
        print("PASSED")
        # 计数器加一
        i += 1


# 测试 DFT 变换的功能，生成 N 个测试用例
def test_dft(N=15):
    # 设置随机种子
    np.random.seed(12345)

    # 初始化计数器 i
    i = 0
    # 循环生成 N 个测试用例
    while i < N:
        # 随机生成信号长度 N
        N = np.random.randint(2, 100)
        # 随机生成信号
        signal = np.random.rand(N)
        # 计算自定义的 DFT 变换
        mine = DFT(signal)
        # 计算 numpy 中的快速傅里叶变换作为对照
        theirs = np.fft.rfft(signal)

        # 断言自定义 DFT 变换结果的实部与对照结果的实部的近似相等
        np.testing.assert_almost_equal(mine.real, theirs.real)
        # 打印测试通过信息
        print("PASSED")
        # 计数器加一
        i += 1


# 测试 MFCC 的功能，生成 N 个测试用例
def test_mfcc(N=1):
    """Broken"""
    # 设置随机种子
    np.random.seed(12345)

    # 初始化计数器 i
    i = 0
    # 当 i 小于 N 时执行循环
    while i < N:
        # 生成一个介于500到1000之间的随机整数，赋值给 N
        N = np.random.randint(500, 1000)
        # 生成一个介于50到100之间的随机整数，赋值给 fs
        fs = np.random.randint(50, 100)
        # 设置 MFCC 参数
        n_mfcc = 12
        window_len = 100
        stride_len = 50
        n_filters = 20
        window_dur = window_len / fs
        stride_dur = stride_len / fs
        # 生成一个长度为 N 的随机信号
        signal = np.random.rand(N)

        # 计算自己实现的 MFCC 特征
        mine = mfcc(
            signal,
            fs=fs,
            window="hann",
            window_duration=window_dur,
            stride_duration=stride_dur,
            lifter_coef=0,
            alpha=0,
            n_mfccs=n_mfcc,
            normalize=False,
            center=True,
            n_filters=n_filters,
            replace_intercept=False,
        )

        # 使用库函数计算 MFCC 特征
        theirs = lr_mfcc(
            signal,
            sr=fs,
            n_mels=n_filters,
            n_mfcc=n_mfcc,
            n_fft=window_len,
            hop_length=stride_len,
            htk=True,
        ).T

        # 检查两种方法计算的 MFCC 特征是否接近
        np.testing.assert_almost_equal(mine, theirs, decimal=4)
        # 打印“PASSED”表示通过测试
        print("PASSED")
        # i 自增
        i += 1
# 测试帧处理函数
def test_framing(N=15):
    # 设置随机种子
    np.random.seed(12345)

    # 初始化循环计数器
    i = 0
    # 循环执行 N 次
    while i < N:
        # 生成随机信号长度
        N = np.random.randint(500, 100000)
        # 生成随机窗口长度
        window_len = np.random.randint(10, 100)
        # 生成随机步长
        stride_len = np.random.randint(1, 50)
        # 生成随机信号
        signal = np.random.rand(N)

        # 调用自定义的帧处理函数 to_frames
        mine = to_frames(signal, window_len, stride_len, writeable=False)
        # 调用库函数 frame 进行帧处理
        theirs = frame(signal, frame_length=window_len, hop_length=stride_len).T

        # 断言两者长度相等
        assert len(mine) == len(theirs), "len(mine) = {}, len(theirs) = {}".format(
            len(mine), len(theirs)
        )
        # 使用 np.testing.assert_almost_equal 检查两者是否几乎相等
        np.testing.assert_almost_equal(mine, theirs)
        # 打印测试通过信息
        print("PASSED")
        # 更新循环计数器
        i += 1


# 测试离散傅立叶变换频率分辨率函数
def test_dft_bins(N=15):
    # 设置随机种子
    np.random.seed(12345)

    # 初始化循环计数器
    i = 0
    # 循环执行 N 次
    while i < N:
        # 生成随机信号长度
        N = np.random.randint(500, 100000)
        # 生成随机采样频率
        fs = np.random.randint(50, 1000)

        # 调用自定义的频率分辨率函数 dft_bins
        mine = dft_bins(N, fs=fs, positive_only=True)
        # 调用库函数 fft_frequencies 计算频率分辨率
        theirs = fft_frequencies(fs, N)
        # 使用 np.testing.assert_almost_equal 检查两者是否几乎相等
        np.testing.assert_almost_equal(mine, theirs)
        # 打印测试通过信息
        print("PASSED")
        # 更新循环计数器
        i += 1


# 测试梅尔滤波器组函数
def test_mel_filterbank(N=15):
    # 设置随机种子
    np.random.seed(12345)

    # 初始化循环计数器
    i = 0
    # 循环执行 N 次
    while i < N:
        # 生成随机采样频率
        fs = np.random.randint(50, 10000)
        # 生成随机滤波器数量
        n_filters = np.random.randint(2, 20)
        # 生成随机窗口长度
        window_len = np.random.randint(10, 100)
        # 生成随机是否归一化参数
        norm = np.random.randint(2)

        # 调用自定义的梅尔滤波器组函数 mel_filterbank
        mine = mel_filterbank(
            window_len, n_filters, fs, min_freq=0, max_freq=None, normalize=bool(norm)
        )

        # 调用库函数 mel 计算梅尔滤波器组
        theirs = mel(
            fs,
            n_fft=window_len,
            n_mels=n_filters,
            htk=True,
            norm="slaney" if norm == 1 else None,
        )

        # 使用 np.testing.assert_almost_equal 检查两者是否几乎相等
        np.testing.assert_almost_equal(mine, theirs)
        # 打印测试通过信息
        print("PASSED")
        # 更新循环计数器
        i += 1
```