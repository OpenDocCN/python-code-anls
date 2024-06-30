# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\tests\test_base.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 pytest 库，用于单元测试
from numpy.testing import assert_array_equal  # 导入 NumPy 测试模块中的数组相等断言

from sklearn.base import BaseEstimator  # 导入 sklearn 基类模块中的 BaseEstimator 类
from sklearn.feature_selection._base import SelectorMixin  # 导入 sklearn 特征选择模块中的 SelectorMixin 类
from sklearn.utils.fixes import CSC_CONTAINERS  # 导入 sklearn 工具修复模块中的 CSC_CONTAINERS 常量


class StepSelector(SelectorMixin, BaseEstimator):
    """保留每个步长为 `step` 的特征（从索引 0 开始）。

    如果 `step < 1`，则不选择任何特征。
    """

    def __init__(self, step=2):
        self.step = step  # 初始化步长参数

    def fit(self, X, y=None):
        X = self._validate_data(X, accept_sparse="csc")  # 验证输入数据 X 的格式为稀疏矩阵
        return self  # 返回自身对象

    def _get_support_mask(self):
        mask = np.zeros(self.n_features_in_, dtype=bool)  # 创建一个布尔类型的全零掩码数组
        if self.step >= 1:
            mask[:: self.step] = True  # 按步长设置掩码数组的值为 True
        return mask  # 返回生成的掩码数组


support = [True, False] * 5  # 创建一个布尔类型的支持列表
support_inds = [0, 2, 4, 6, 8]  # 创建一个索引列表
X = np.arange(20).reshape(2, 10)  # 创建一个 2x10 的 NumPy 数组
Xt = np.arange(0, 20, 2).reshape(2, 5)  # 创建一个转换后的 2x5 的 NumPy 数组
Xinv = X.copy()  # 复制 X 数组到 Xinv 数组
Xinv[:, 1::2] = 0  # 将 Xinv 数组的奇数列置为零
y = [0, 1]  # 创建一个标签列表
feature_names = list("ABCDEFGHIJ")  # 创建特征名称列表
feature_names_t = feature_names[::2]  # 创建转换后的特征名称列表
feature_names_inv = np.array(feature_names)  # 创建特征名称的 NumPy 数组版本
feature_names_inv[1::2] = ""  # 将特征名称数组的奇数索引位置置空


def test_transform_dense():
    sel = StepSelector()  # 创建 StepSelector 类的实例 sel
    Xt_actual = sel.fit(X, y).transform(X)  # 使用 sel 对象拟合并转换 X 数组
    Xt_actual2 = StepSelector().fit_transform(X, y)  # 直接拟合并转换 X 数组
    assert_array_equal(Xt, Xt_actual)  # 断言转换后的数组 Xt 与预期结果相等
    assert_array_equal(Xt, Xt_actual2)  # 断言转换后的数组 Xt 与预期结果相等

    # 检查 dtype 是否匹配
    assert np.int32 == sel.transform(X.astype(np.int32)).dtype  # 断言转换后的数组类型为 np.int32
    assert np.float32 == sel.transform(X.astype(np.float32)).dtype  # 断言转换后的数组类型为 np.float32

    # 检查 1 维列表和其他 dtype：
    names_t_actual = sel.transform([feature_names])  # 使用 sel 对象转换特征名称列表
    assert_array_equal(feature_names_t, names_t_actual.ravel())  # 断言转换后的特征名称与预期结果相等

    # 检查错误的形状是否引发错误
    with pytest.raises(ValueError):
        sel.transform(np.array([[1], [2]]))


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_transform_sparse(csc_container):
    X_sp = csc_container(X)  # 使用给定的 csc_container 将 X 转换为稀疏矩阵
    sel = StepSelector()  # 创建 StepSelector 类的实例 sel
    Xt_actual = sel.fit(X_sp).transform(X_sp)  # 使用 sel 对象拟合并转换稀疏矩阵 X_sp
    Xt_actual2 = sel.fit_transform(X_sp)  # 直接拟合并转换稀疏矩阵 X_sp
    assert_array_equal(Xt, Xt_actual.toarray())  # 断言转换后的稀疏矩阵数组 Xt 与预期结果相等
    assert_array_equal(Xt, Xt_actual2.toarray())  # 断言转换后的稀疏矩阵数组 Xt 与预期结果相等

    # 检查 dtype 是否匹配
    assert np.int32 == sel.transform(X_sp.astype(np.int32)).dtype  # 断言转换后的数组类型为 np.int32
    assert np.float32 == sel.transform(X_sp.astype(np.float32)).dtype  # 断言转换后的数组类型为 np.float32

    # 检查错误的形状是否引发错误
    with pytest.raises(ValueError):
        sel.transform(np.array([[1], [2]]))


def test_inverse_transform_dense():
    sel = StepSelector()  # 创建 StepSelector 类的实例 sel
    Xinv_actual = sel.fit(X, y).inverse_transform(Xt)  # 使用 sel 对象拟合并反向转换 Xt 数组
    assert_array_equal(Xinv, Xinv_actual)  # 断言反向转换后的数组 Xinv 与预期结果相等

    # 检查 dtype 是否匹配
    assert np.int32 == sel.inverse_transform(Xt.astype(np.int32)).dtype  # 断言反向转换后的数组类型为 np.int32
    assert np.float32 == sel.inverse_transform(Xt.astype(np.float32)).dtype  # 断言反向转换后的数组类型为 np.float32

    # 检查 1 维列表和其他 dtype：
    names_inv_actual = sel.inverse_transform([feature_names_t])  # 使用 sel 对象反向转换特征名称列表
    assert_array_equal(feature_names_inv, names_inv_actual.ravel())  # 断言反向转换后的特征名称与预期结果相等

    # 检查错误的形状是否引发错误
    with pytest.raises(ValueError):
        sel.inverse_transform(np.array([[1], [2]]))
# 使用 pytest 的 parametrize 装饰器，对测试函数 test_inverse_transform_sparse 参数化，参数为 CSC_CONTAINERS 中的每个元素
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_inverse_transform_sparse(csc_container):
    # 使用 csc_container 创建稀疏矩阵 X_sp 和 Xt_sp
    X_sp = csc_container(X)
    Xt_sp = csc_container(Xt)
    
    # 创建 StepSelector 实例
    sel = StepSelector()
    
    # 对 X_sp 进行拟合，并进行逆变换得到 Xinv_actual
    Xinv_actual = sel.fit(X_sp).inverse_transform(Xt_sp)
    
    # 断言 Xinv 与 Xinv_actual 的内容是否相等
    assert_array_equal(Xinv, Xinv_actual.toarray())

    # 检查逆变换后的数据类型是否匹配
    assert np.int32 == sel.inverse_transform(Xt_sp.astype(np.int32)).dtype
    assert np.float32 == sel.inverse_transform(Xt_sp.astype(np.float32)).dtype

    # 检查形状不匹配时是否引发 ValueError 异常
    with pytest.raises(ValueError):
        sel.inverse_transform(np.array([[1], [2]]))


# 测试获取支持信息的函数
def test_get_support():
    # 创建 StepSelector 实例
    sel = StepSelector()
    
    # 对 X, y 进行拟合
    sel.fit(X, y)
    
    # 断言支持信息 support 是否与 sel.get_support() 返回的结果相等
    assert_array_equal(support, sel.get_support())
    
    # 断言支持信息 support_inds 是否与 sel.get_support(indices=True) 返回的结果相等
    assert_array_equal(support_inds, sel.get_support(indices=True))


# 测试输出 DataFrame 的函数
def test_output_dataframe():
    """检查输出的 DataFrame 的数据类型是否与输入的 DataFrame 一致。"""
    # 导入 pytest 并检查是否存在 pandas 库
    pd = pytest.importorskip("pandas")

    # 创建输入数据 X，包含不同数据类型的列
    X = pd.DataFrame(
        {
            "a": pd.Series([1.0, 2.4, 4.5], dtype=np.float32),
            "b": pd.Series(["a", "b", "a"], dtype="category"),
            "c": pd.Series(["j", "b", "b"], dtype="category"),
            "d": pd.Series([3.0, 2.4, 1.2], dtype=np.float64),
        }
    )

    # 遍历不同的步骤数 [2, 3]
    for step in [2, 3]:
        # 创建 StepSelector 实例，并设置输出为 pandas DataFrame
        sel = StepSelector(step=step).set_output(transform="pandas")
        
        # 对输入数据 X 进行拟合
        sel.fit(X)

        # 对拟合后的数据进行变换
        output = sel.transform(X)
        
        # 检查输出的每列数据类型是否与输入 X 的每列数据类型相等
        for name, dtype in output.dtypes.items():
            assert dtype == X.dtypes[name]

    # 当 step=0 时，将不选择任何特征
    sel0 = StepSelector(step=0).set_output(transform="pandas")
    sel0.fit(X, y)

    # 预期会发出 UserWarning 警告信息
    msg = "No features were selected"
    with pytest.warns(UserWarning, match=msg):
        output0 = sel0.transform(X)

    # 断言输出的索引与输入 X 的索引相等
    assert_array_equal(output0.index, X.index)
    
    # 断言输出的形状为 (样本数, 0)，即没有选中任何特征
    assert output0.shape == (X.shape[0], 0)
```