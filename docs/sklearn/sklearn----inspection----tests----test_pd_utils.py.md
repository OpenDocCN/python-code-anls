# `D:\src\scipysrc\scikit-learn\sklearn\inspection\tests\test_pd_utils.py`

```
# 导入所需的库和模块
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

# 导入需要测试的函数和类
from sklearn.inspection._pd_utils import _check_feature_names, _get_feature_index
from sklearn.utils._testing import _convert_container

# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例
@pytest.mark.parametrize(
    "feature_names, array_type, expected_feature_names",
    [
        (None, "array", ["x0", "x1", "x2"]),  # 参数化测试用例：array 类型，预期特征名为 ["x0", "x1", "x2"]
        (None, "dataframe", ["a", "b", "c"]),  # 参数化测试用例：dataframe 类型，预期特征名为 ["a", "b", "c"]
        (np.array(["a", "b", "c"]), "array", ["a", "b", "c"]),  # 参数化测试用例：自定义特征名，预期特征名为 ["a", "b", "c"]
    ],
)
def test_check_feature_names(feature_names, array_type, expected_feature_names):
    # 生成一个随机的 10x3 的 NumPy 数组
    X = np.random.randn(10, 3)
    # 如果没有指定特征名，使用默认列名 ["a", "b", "c"] 对 X 进行容器转换
    column_names = ["a", "b", "c"]
    X = _convert_container(X, constructor_name=array_type, columns_name=column_names)
    # 调用 _check_feature_names 函数进行特征名验证
    feature_names_validated = _check_feature_names(X, feature_names)
    # 断言验证结果与预期特征名一致
    assert feature_names_validated == expected_feature_names


# 测试检查特征名中存在重复的情况
def test_check_feature_names_error():
    # 生成一个随机的 10x3 的 NumPy 数组
    X = np.random.randn(10, 3)
    # 指定存在重复特征名的情况
    feature_names = ["a", "b", "c", "a"]
    # 预期的错误消息
    msg = "feature_names should not contain duplicates."
    # 使用 pytest.raises 断言捕获 ValueError 异常，并验证错误消息
    with pytest.raises(ValueError, match=msg):
        _check_feature_names(X, feature_names)


# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例
@pytest.mark.parametrize("fx, idx", [(0, 0), (1, 1), ("a", 0), ("b", 1), ("c", 2)])
def test_get_feature_index(fx, idx):
    # 已知的特征名列表
    feature_names = ["a", "b", "c"]
    # 调用 _get_feature_index 函数，验证特征名对应的索引是否与预期索引一致
    assert _get_feature_index(fx, feature_names) == idx


# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例
@pytest.mark.parametrize(
    "fx, feature_names, err_msg",
    [
        ("a", None, "Cannot plot partial dependence for feature 'a'"),  # 参数化测试用例：特征名为 'a'，没有特征名列表
        ("d", ["a", "b", "c"], "Feature 'd' not in feature_names"),  # 参数化测试用例：特征名为 'd'，特征名列表为 ["a", "b", "c"]
    ],
)
def test_get_feature_names_error(fx, feature_names, err_msg):
    # 使用 pytest.raises 断言捕获 ValueError 异常，并验证错误消息
    with pytest.raises(ValueError, match=err_msg):
        _get_feature_index(fx, feature_names)
```