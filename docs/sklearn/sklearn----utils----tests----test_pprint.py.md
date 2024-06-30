# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_pprint.py`

```
# 导入正则表达式模块
import re
# 导入美化打印模块中的 PrettyPrinter 类
from pprint import PrettyPrinter

# 导入 numpy 库并使用 np 别名
import numpy as np

# 导入 sklearn 内部的打印工具 _EstimatorPrettyPrinter 类
from sklearn.utils._pprint import _EstimatorPrettyPrinter
# 导入 sklearn 的 LogisticRegressionCV 类
from sklearn.linear_model import LogisticRegressionCV
# 导入 sklearn 的管道构建工具 make_pipeline 函数
from sklearn.pipeline import make_pipeline
# 导入 sklearn 的 BaseEstimator 和 TransformerMixin 类
from sklearn.base import BaseEstimator, TransformerMixin
# 导入 sklearn 的特征选择模块 SelectKBest 和 chi2 函数
from sklearn.feature_selection import SelectKBest, chi2
# 导入 sklearn 的 config_context 模块

# 忽略 flake8 工具的检查，以解决代码行过长的问题
# ruff: noqa
    # 初始化函数，用于创建一个文本特征提取器对象
    def __init__(
        self,
        input="content",  # 输入参数，默认为"content"，表示输入是文本内容
        encoding="utf-8",  # 文本编码方式，默认为UTF-8
        decode_error="strict",  # 解码错误处理方式，默认为严格模式
        strip_accents=None,  # 是否去除重音符号，默认不去除
        lowercase=True,  # 是否将文本转换为小写，默认转换为小写
        preprocessor=None,  # 预处理器函数，默认为空，不进行额外预处理
        tokenizer=None,  # 分词器函数，默认为空，使用默认的分词方式
        stop_words=None,  # 停用词列表或者None，默认为None，即不使用停用词
        token_pattern=r"(?u)\b\w\w+\b",  # 分词的正则表达式模式，默认为匹配至少两个字符的单词
        ngram_range=(1, 1),  # n-gram的范围，默认为单个词语
        analyzer="word",  # 分析器类型，默认为词级分析器
        max_df=1.0,  # 词频高于此阈值的特征将被忽略，默认为不忽略任何特征
        min_df=1,  # 词频低于此阈值的特征将被忽略，默认为不忽略任何特征
        max_features=None,  # 最大特征数，默认为不限制特征数
        vocabulary=None,  # 词汇表，可以是列表或者字典，默认为None
        binary=False,  # 如果为True，则特征矩阵将是二进制的，默认为False
        dtype=np.int64,  # 特征矩阵的数据类型，默认为64位整数
    ):
        self.input = input  # 将参数input赋值给对象的input属性
        self.encoding = encoding  # 将参数encoding赋值给对象的encoding属性
        self.decode_error = decode_error  # 将参数decode_error赋值给对象的decode_error属性
        self.strip_accents = strip_accents  # 将参数strip_accents赋值给对象的strip_accents属性
        self.lowercase = lowercase  # 将参数lowercase赋值给对象的lowercase属性
        self.preprocessor = preprocessor  # 将参数preprocessor赋值给对象的preprocessor属性
        self.tokenizer = tokenizer  # 将参数tokenizer赋值给对象的tokenizer属性
        self.analyzer = analyzer  # 将参数analyzer赋值给对象的analyzer属性
        self.token_pattern = token_pattern  # 将参数token_pattern赋值给对象的token_pattern属性
        self.stop_words = stop_words  # 将参数stop_words赋值给对象的stop_words属性
        self.max_df = max_df  # 将参数max_df赋值给对象的max_df属性
        self.min_df = min_df  # 将参数min_df赋值给对象的min_df属性
        self.max_features = max_features  # 将参数max_features赋值给对象的max_features属性
        self.ngram_range = ngram_range  # 将参数ngram_range赋值给对象的ngram_range属性
        self.vocabulary = vocabulary  # 将参数vocabulary赋值给对象的vocabulary属性
        self.binary = binary  # 将参数binary赋值给对象的binary属性
        self.dtype = dtype  # 将参数dtype赋值给对象的dtype属性
class Pipeline(BaseEstimator):
    # 定义 Pipeline 类，继承自 BaseEstimator

    def __init__(self, steps, memory=None):
        # 初始化方法，接受 steps 参数和可选的 memory 参数
        self.steps = steps  # 设置 steps 实例变量
        self.memory = memory  # 设置 memory 实例变量


class SVC(BaseEstimator):
    # 定义 SVC 类，继承自 BaseEstimator

    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="auto_deprecated",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        random_state=None,
    ):
        # 初始化方法，接受多个参数来配置 SVC 模型的属性
        self.kernel = kernel  # 设置 kernel 实例变量
        self.degree = degree  # 设置 degree 实例变量
        self.gamma = gamma  # 设置 gamma 实例变量
        self.coef0 = coef0  # 设置 coef0 实例变量
        self.tol = tol  # 设置 tol 实例变量
        self.C = C  # 设置 C 实例变量
        self.shrinking = shrinking  # 设置 shrinking 实例变量
        self.probability = probability  # 设置 probability 实例变量
        self.cache_size = cache_size  # 设置 cache_size 实例变量
        self.class_weight = class_weight  # 设置 class_weight 实例变量
        self.verbose = verbose  # 设置 verbose 实例变量
        self.max_iter = max_iter  # 设置 max_iter 实例变量
        self.decision_function_shape = decision_function_shape  # 设置 decision_function_shape 实例变量
        self.random_state = random_state  # 设置 random_state 实例变量


class PCA(BaseEstimator):
    # 定义 PCA 类，继承自 BaseEstimator

    def __init__(
        self,
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
    ):
        # 初始化方法，接受多个参数来配置 PCA 模型的属性
        self.n_components = n_components  # 设置 n_components 实例变量
        self.copy = copy  # 设置 copy 实例变量
        self.whiten = whiten  # 设置 whiten 实例变量
        self.svd_solver = svd_solver  # 设置 svd_solver 实例变量
        self.tol = tol  # 设置 tol 实例变量
        self.iterated_power = iterated_power  # 设置 iterated_power 实例变量
        self.random_state = random_state  # 设置 random_state 实例变量


class NMF(BaseEstimator):
    # 定义 NMF 类，继承自 BaseEstimator

    def __init__(
        self,
        n_components=None,
        init=None,
        solver="cd",
        beta_loss="frobenius",
        tol=1e-4,
        max_iter=200,
        random_state=None,
        alpha=0.0,
        l1_ratio=0.0,
        verbose=0,
        shuffle=False,
    ):
        # 初始化方法，接受多个参数来配置 NMF 模型的属性
        self.n_components = n_components  # 设置 n_components 实例变量
        self.init = init  # 设置 init 实例变量
        self.solver = solver  # 设置 solver 实例变量
        self.beta_loss = beta_loss  # 设置 beta_loss 实例变量
        self.tol = tol  # 设置 tol 实例变量
        self.max_iter = max_iter  # 设置 max_iter 实例变量
        self.random_state = random_state  # 设置 random_state 实例变量
        self.alpha = alpha  # 设置 alpha 实例变量
        self.l1_ratio = l1_ratio  # 设置 l1_ratio 实例变量
        self.verbose = verbose  # 设置 verbose 实例变量
        self.shuffle = shuffle  # 设置 shuffle 实例变量


class SimpleImputer(BaseEstimator):
    # 定义 SimpleImputer 类，继承自 BaseEstimator

    def __init__(
        self,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
        verbose=0,
        copy=True,
    ):
        # 初始化方法，接受多个参数来配置 SimpleImputer 模型的属性
        self.missing_values = missing_values  # 设置 missing_values 实例变量
        self.strategy = strategy  # 设置 strategy 实例变量
        self.fill_value = fill_value  # 设置 fill_value 实例变量
        self.verbose = verbose  # 设置 verbose 实例变量
        self.copy = copy  # 设置 copy 实例变量


def test_basic(print_changed_only_false):
    # 测试函数，接受一个参数 print_changed_only_false

    # Basic pprint test
    lr = LogisticRegression()  # 创建一个 LogisticRegression 实例 lr
    expected = """
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)"""
    expected = expected[1:]  # 移除字符串开头的第一个换行符 \n
    # 使用断言检查 lr 对象的字符串表示是否与预期的字符串相等
    assert lr.__repr__() == expected
def test_gridsearch(print_changed_only_false):
    # 定义一个参数网格，包含两个参数字典，用于 GridSearchCV
    param_grid = [
        {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
    ]
    # 创建一个 GridSearchCV 对象，使用 SVC 作为基础分类器，参数包括参数网格和交叉验证折数为 5
    gs = GridSearchCV(SVC(), param_grid, cv=5)

    # 预期的 GridSearchCV 对象的字符串表示，包含了具体的参数设置，需要保留格式整齐
    expected = """
GridSearchCV(estimator=SVC(C=1.0, break_ties=False, cache_size=200,
                          class_weight=None, coef0=0.0,
                          decision_function_shape='ovr', degree=3,
                          gamma='scale', kernel='rbf', max_iter=-1,
                          probability=False, random_state=None, shrinking=True,
                          tol=0.001, verbose=False),
             param_grid=[{'kernel': ['rbf'], 'gamma': [0.001, 0.0001], 'C': [1, 10, 100, 1000]},
                         {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}],
             cv=5)"""

    expected = expected[1:]  # 移除开头的换行符
    # 断言 GridSearchCV 对象的字符串表示符合预期的格式
    assert gs.__repr__() == expected
ultimate = GridSearchCV(cv=5, error_score='raise-deprecating',  # 创建一个网格搜索交叉验证对象，指定参数如交叉验证折数为5
             estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,  # 使用支持向量机作为基础评估器，设置初始参数
                           decision_function_shape='ovr', degree=3,
                           gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False),
             iid='warn', n_jobs=None,  # 其他参数设置如不并行处理任务
             param_grid=[{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],  # 定义参数网格，包括C和gamma的多组取值
                          'kernel': ['rbf']},  # 使用RBF核的情况
                         {'C': [1, 10, 100, 1000], 'kernel': ['linear']}],  # 使用线性核的情况
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,  # 其他设置如预分配任务数，重新拟合最佳模型等
             scoring=None, verbose=0)"""

    expected = expected[1:]  # 移除字符串开头的第一个换行符
    assert gs.__repr__() == expected  # 断言检查生成的网格搜索对象的字符串表示是否符合预期


def test_gridsearch_pipeline(print_changed_only_false):
    # render a pipeline inside a gridsearch
    pp = _EstimatorPrettyPrinter(compact=True, indent=1, indent_at_name=True)  # 创建一个漂亮打印器对象，用于打印估计器

    pipeline = Pipeline([("reduce_dim", PCA()), ("classify", SVC())])  # 创建一个包含PCA和SVC的管道对象
    N_FEATURES_OPTIONS = [2, 4, 8]  # 定义不同的特征数选项
    C_OPTIONS = [1, 10, 100, 1000]  # 定义不同的C参数选项
    param_grid = [  # 定义管道中可能的参数网格
        {
            "reduce_dim": [PCA(iterated_power=7), NMF()],  # 第一个步骤使用PCA或者NMF降维
            "reduce_dim__n_components": N_FEATURES_OPTIONS,  # 指定降维后的特征数选项
            "classify__C": C_OPTIONS,  # 指定分类器的C参数选项
        },
        {
            "reduce_dim": [SelectKBest(chi2)],  # 第二个步骤使用SelectKBest进行特征选择
            "reduce_dim__k": N_FEATURES_OPTIONS,  # 指定SelectKBest的k值选项
            "classify__C": C_OPTIONS,  # 指定分类器的C参数选项
        },
    ]
    gspipline = GridSearchCV(pipeline, cv=3, n_jobs=1, param_grid=param_grid)  # 创建管道的网格搜索交叉验证对象，指定参数如交叉验证折数为3
    expected = """
# 创建一个 GridSearchCV 对象，用于交叉验证和参数优化
GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=Pipeline(memory=None,
                                steps=[('reduce_dim',
                                        PCA(copy=True, iterated_power='auto',
                                            n_components=None,
                                            random_state=None,
                                            svd_solver='auto', tol=0.0,
                                            whiten=False)),
                                       ('classify',
                                        SVC(C=1.0, cache_size=200,
                                            class_weight=None, coef0=0.0,
                                            decision_function_shape='ovr',
                                            degree=3, gamma='auto_deprecated',
                                            kernel='rbf', max_iter=-1,
                                            probability=False,
                                            random_state=None, shrinking=True,
                                            tol=0.001, verbose=False))]),
             iid='warn', n_jobs=1,
             # 定义参数网格，包括两个字典，每个字典包含不同的参数组合
             param_grid=[{'classify__C': [1, 10, 100, 1000],
                          'reduce_dim': [PCA(copy=True, iterated_power=7,
                                             n_components=None,
                                             random_state=None,
                                             svd_solver='auto', tol=0.0,
                                             whiten=False),
                                         NMF(alpha=0.0, beta_loss='frobenius',
                                             init=None, l1_ratio=0.0,
                                             max_iter=200, n_components=None,
                                             random_state=None, shuffle=False,
                                             solver='cd', tol=0.0001,
                                             verbose=0)],
                          'reduce_dim__n_components': [2, 4, 8]},
                         {'classify__C': [1, 10, 100, 1000],
                          'reduce_dim': [SelectKBest(k=10,
                                                     score_func=<function chi2 at some_address>)],
                          'reduce_dim__k': [2, 4, 8]}],
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)

# 移除字符串 expected 中的第一个换行符 '\n'
expected = expected[1:]
# 使用 pprint 模块格式化输出 gspipline 对象，以便进行比较
repr_ = pp.pformat(gspipline)
# 为了便于复现，移除 '<function chi2 at 0x.....>' 地址信息，将其统一替换为 '<function chi2 at some_address>'
repr_ = re.sub("function chi2 at 0x.*>", "function chi2 at some_address>", repr_)
# 使用断言确保格式化后的 repr_ 与 expected 相等
assert repr_ == expected
    # 创建一个包含数字范围内所有元素的字典，键和值相同
    vocabulary = {i: i for i in range(n_max_elements_to_show)}
    # 使用指定的词汇表（即上一行创建的字典）初始化一个计数向量化器对象
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    
    # 初始化一个预期输出字符串（通常是正则表达式或者多行文本的预期输出）
    expected = r"""
    # 创建一个 CountVectorizer 对象，用于将文本转换为词频矩阵
    vectorizer = CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                                 dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
                                 lowercase=True, max_df=1.0, max_features=None, min_df=1,
                                 ngram_range=(1, 1), preprocessor=None, stop_words=None,
                                 strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                                 tokenizer=None,
                                 vocabulary={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
                                             8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14,
                                             15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20,
                                             21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26,
                                             27: 27, 28: 28, 29: 29})
    
    # 移除预期输出字符串中的第一个换行符
    expected = expected[1:]  # remove first \n
    # 断言 CountVectorizer 对象的格式化字符串是否与预期输出字符串相匹配
    assert pp.pformat(vectorizer) == expected

    # 创建一个新的 vocabulary 字典，包含从 0 到 n_max_elements_to_show 的整数映射
    vocabulary = {i: i for i in range(n_max_elements_to_show + 1)}
    # 使用新的 vocabulary 创建 CountVectorizer 对象
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    
    # 移除预期输出字符串中的第一个换行符
    expected = r"""
CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
                lowercase=True, max_df=1.0, max_features=None, min_df=1,
                ngram_range=(1, 1), preprocessor=None, stop_words=None,
                strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                tokenizer=None,
                vocabulary={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
                            8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14,
                            15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20,
                            21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26,
                            27: 27, 28: 28, 29: 29, ...})"""
    
    # 移除预期输出字符串中的第一个换行符
    expected = expected[1:]  # remove first \n
    # 断言新创建的 CountVectorizer 对象的格式化字符串是否与预期输出字符串相匹配
    assert pp.pformat(vectorizer) == expected

    # 使用列表创建一个 param_grid 字典，其中包含键为 "C"，值为从 0 到 n_max_elements_to_show 的整数列表
    param_grid = {"C": list(range(n_max_elements_to_show))}
    # 使用 param_grid 字典创建一个 GridSearchCV 对象，用于参数优化
    gs = GridSearchCV(SVC(), param_grid)
    
    expected = """
GridSearchCV(cv='warn', error_score='raise-deprecating',
             estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                           decision_function_shape='ovr', degree=3,
                           gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False),
             iid='warn', n_jobs=None,
             param_grid={'C': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                               27, 28, 29]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)"""
    
    # 移除预期输出字符串中的第一个换行符
    expected = expected[1:]  # remove first \n
    # 断言 GridSearchCV 对象的格式化字符串是否与预期输出字符串相匹配
    assert pp.pformat(gs) == expected

    # 使用列表创建一个 param_grid 字典，其中包含键为 "C"，值为从 0 到 n_max_elements_to_show + 1 的整数列表
    param_grid = {"C": list(range(n_max_elements_to_show + 1))}
    # 创建一个网格搜索交叉验证对象，使用支持向量机 (SVC) 作为基本模型，并使用给定的参数网格
    gs = GridSearchCV(SVC(), param_grid)
    # 设置期望的值为空字符串，这里可能是为了后续使用的初始设置或占位符
    expected = """
# 创建一个 GridSearchCV 对象，用于网格搜索和交叉验证
GridSearchCV(cv='warn',  # 使用警告级别的交叉验证
             error_score='raise-deprecating',  # 遇到警告时抛出错误
             estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                           decision_function_shape='ovr', degree=3,
                           gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False),  # 使用默认参数初始化 SVC 估计器
             iid='warn',  # 根据不同的版本，iid 参数可能会引发警告
             n_jobs=None,  # 不使用并行计算
             param_grid={'C': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                               27, 28, 29, ...]},  # 设置参数网格，待搜索的参数值列表
             pre_dispatch='2*n_jobs',  # 预分配任务时使用的工作数的倍数
             refit=True,  # 使用最佳参数重新拟合估计器
             return_train_score=False,  # 不返回训练集的评分
             scoring=None,  # 不使用评分函数
             verbose=0)  # 关闭详细输出模式

# 从期望结果字符串中移除第一个换行符
expected = expected[1:]  # 移除第一个换行符 \n
assert pp.pformat(gs) == expected
    # 在这里定义一个多行字符串，用于存储预期的文本内容
    expected = """
# 创建一个 LogisticRegression 对象，设置了多个参数的默认值
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter...,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)"""
# 去除期望字符串的第一个换行符 '\n'
expected = expected[1:]  # remove first \n
# 断言期望字符串与 LogisticRegression 对象的 __repr__ 方法返回的字符串相等，且长度为非空白字符数减去 4
assert expected == lr.__repr__(N_CHAR_MAX=n_nonblank - 4)

# 使用 N_CHAR_MAX 等于非空白字符数减去 2 进行测试：
# - 省略号的左右两侧在同一行，但添加省略号会使得 repr 字符串变长，因此这里不添加省略号。
expected = """
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)"""
# 去除期望字符串的第一个换行符 '\n'
expected = expected[1:]  # remove first \n
# 断言期望字符串与 LogisticRegression 对象的 __repr__ 方法返回的字符串相等，且长度为非空白字符数减去 2
assert expected == lr.__repr__(N_CHAR_MAX=n_nonblank - 2)
    # 定义一个名为 DummyEstimator 的类，它继承自 TransformerMixin 和 BaseEstimator
    class DummyEstimator(TransformerMixin, BaseEstimator):
        
        # 类变量，记录 __repr__ 方法被调用的次数
        nb_times_repr_called = 0

        # 初始化方法，接受一个 estimator 参数
        def __init__(self, estimator=None):
            self.estimator = estimator

        # 重写 __repr__ 方法，每次被调用时增加调用次数，并返回父类的字符串表示
        def __repr__(self):
            DummyEstimator.nb_times_repr_called += 1
            return super().__repr__()

        # transform 方法，返回传入的 X，参数 copy 未使用（忽略测试覆盖）
        def transform(self, X, copy=None):  # pragma: no cover
            return X

    # 创建一个名为 estimator 的 DummyEstimator 实例
    estimator = DummyEstimator(
        # 使用 make_pipeline 函数创建一个 Pipeline 对象作为参数传入 DummyEstimator
        make_pipeline(DummyEstimator(DummyEstimator()), DummyEstimator(), "passthrough")
    )

    # 使用 config_context 上下文管理器，设置 print_changed_only 参数为 False
    with config_context(print_changed_only=False):
        # 调用 repr 函数，触发 estimator 的 __repr__ 方法
        repr(estimator)
        # 获取调用 DummyEstimator 类的 __repr__ 方法的次数
        nb_repr_print_changed_only_false = DummyEstimator.nb_times_repr_called

    # 重置 DummyEstimator 类的 __repr__ 方法被调用次数为 0
    DummyEstimator.nb_times_repr_called = 0

    # 使用 config_context 上下文管理器，设置 print_changed_only 参数为 True
    with config_context(print_changed_only=True):
        # 调用 repr 函数，再次触发 estimator 的 __repr__ 方法
        repr(estimator)
        # 获取调用 DummyEstimator 类的 __repr__ 方法的次数
        nb_repr_print_changed_only_true = DummyEstimator.nb_times_repr_called

    # 断言，验证两种 print_changed_only 参数设置下 __repr__ 方法的调用次数相同
    assert nb_repr_print_changed_only_false == nb_repr_print_changed_only_true
```