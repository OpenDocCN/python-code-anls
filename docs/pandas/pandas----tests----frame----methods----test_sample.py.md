# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_sample.py`

```
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm
import pandas.core.common as com


class TestSample:
    @pytest.fixture
    def obj(self, frame_or_series):
        # 根据传入的 frame_or_series 类型，生成对应的随机数组或随机矩阵
        if frame_or_series is Series:
            arr = np.random.default_rng(2).standard_normal(10)
        else:
            arr = np.random.default_rng(2).standard_normal((10, 10))
        return frame_or_series(arr, dtype=None)

    @pytest.mark.parametrize("test", list(range(10)))
    def test_sample(self, test, obj):
        # 修复问题: 2419
        # 检查 random_state 参数的行为
        # 确保当接收到种子或随机状态时的稳定性 -- 运行 10 次

        # 使用固定的种子生成随机数
        seed = np.random.default_rng(2).integers(0, 100)
        # 检验随机种子在抽样时的一致性
        tm.assert_equal(
            obj.sample(n=4, random_state=seed), obj.sample(n=4, random_state=seed)
        )

        tm.assert_equal(
            obj.sample(frac=0.7, random_state=seed),
            obj.sample(frac=0.7, random_state=seed),
        )

        # 使用不同的随机种子生成随机数
        tm.assert_equal(
            obj.sample(n=4, random_state=np.random.default_rng(test)),
            obj.sample(n=4, random_state=np.random.default_rng(test)),
        )

        tm.assert_equal(
            obj.sample(frac=0.7, random_state=np.random.default_rng(test)),
            obj.sample(frac=0.7, random_state=np.random.default_rng(test)),
        )

        # 使用相同的随机种子但设置 frac=2 来测试替换抽样
        tm.assert_equal(
            obj.sample(
                frac=2,
                replace=True,
                random_state=np.random.default_rng(test),
            ),
            obj.sample(
                frac=2,
                replace=True,
                random_state=np.random.default_rng(test),
            ),
        )

        # 多次抽样测试，检验结果是否一致
        os1, os2 = [], []
        for _ in range(2):
            os1.append(obj.sample(n=4, random_state=test))
            os2.append(obj.sample(frac=0.7, random_state=test))
        tm.assert_equal(*os1)
        tm.assert_equal(*os2)

    def test_sample_lengths(self, obj):
        # 检查抽样结果的长度是否正确
        assert len(obj.sample(n=4)) == 4
        assert len(obj.sample(frac=0.34)) == 3
        assert len(obj.sample(frac=0.36)) == 4

    def test_sample_invalid_random_state(self, obj):
        # 检查当 random_state 参数无效时是否会引发错误
        msg = (
            "random_state must be an integer, array-like, a BitGenerator, Generator, "
            "a numpy RandomState, or None"
        )
        with pytest.raises(ValueError, match=msg):
            obj.sample(random_state="a_string")

    def test_sample_wont_accept_n_and_frac(self, obj):
        # 当同时给出 n 和 frac 时应该抛出错误
        msg = "Please enter a value for `frac` OR `n`, not both"
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, frac=0.3)
    def test_sample_requires_positive_n_frac(self, obj):
        # 测试函数：确保 `n` 和 `frac` 都需要是正数
        with pytest.raises(
            ValueError,
            match="A negative number of rows requested. Please provide `n` >= 0",
        ):
            obj.sample(n=-3)
        with pytest.raises(
            ValueError,
            match="A negative number of rows requested. Please provide `frac` >= 0",
        ):
            obj.sample(frac=-0.3)

    def test_sample_requires_integer_n(self, obj):
        # 测试函数：确保 `n` 是整数，不接受浮点数
        with pytest.raises(ValueError, match="Only integers accepted as `n` values"):
            obj.sample(n=3.2)

    def test_sample_invalid_weight_lengths(self, obj):
        # 测试函数：确保权重长度正确
        msg = "Weights and axis to be sampled must be of same length"
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, weights=[0, 1])

        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, weights=[0.5] * 11)

        with pytest.raises(ValueError, match="Fewer non-zero entries in p than size"):
            obj.sample(n=4, weights=Series([0, 0, 0.2]))

    def test_sample_negative_weights(self, obj):
        # 测试函数：检查不接受负权重值
        bad_weights = [-0.1] * 10
        msg = "weight vector many not include negative values"
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, weights=bad_weights)

    def test_sample_inf_weights(self, obj):
        # 测试函数：检查不接受正无穷和负无穷的权重值
        weights_with_inf = [0.1] * 10
        weights_with_inf[0] = np.inf
        msg = "weight vector may not include `inf` values"
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, weights=weights_with_inf)

        weights_with_ninf = [0.1] * 10
        weights_with_ninf[0] = -np.inf
        with pytest.raises(ValueError, match=msg):
            obj.sample(n=3, weights=weights_with_ninf)

    def test_sample_zero_weights(self, obj):
        # 测试函数：检查所有权重为零时抛出错误
        zero_weights = [0] * 10
        with pytest.raises(ValueError, match="Invalid weights: weights sum to zero"):
            obj.sample(n=3, weights=zero_weights)

    def test_sample_missing_weights(self, obj):
        # 测试函数：检查所有权重缺失时抛出错误
        nan_weights = [np.nan] * 10
        with pytest.raises(ValueError, match="Invalid weights: weights sum to zero"):
            obj.sample(n=3, weights=nan_weights)

    def test_sample_none_weights(self, obj):
        # 测试函数：检查 `None` 被替换为零的情况
        weights_with_None = [None] * 10
        weights_with_None[5] = 0.5
        tm.assert_equal(
            obj.sample(n=1, axis=0, weights=weights_with_None), obj.iloc[5:6]
        )

    @pytest.mark.parametrize(
        "func_str,arg",
        [
            ("np.array", [2, 3, 1, 0]),
            ("np.random.MT19937", 3),
            ("np.random.PCG64", 11),
        ],
    )
    # 测试函数，用于验证样本抽样功能中的随机状态设置
    def test_sample_random_state(self, func_str, arg, frame_or_series):
        # 创建一个 DataFrame 对象，包含两列，各自从10到19和20到29递增
        obj = DataFrame({"col1": range(10, 20), "col2": range(20, 30)})
        # 获取适当的对象，可以是 DataFrame 或 Series
        obj = tm.get_obj(obj, frame_or_series)
        # 使用指定的随机状态生成器对对象进行抽样，返回抽样结果
        result = obj.sample(n=3, random_state=eval(func_str)(arg))
        # 使用 pandas 的随机状态函数包装器生成相同随机状态的期望结果的抽样
        expected = obj.sample(n=3, random_state=com.random_state(eval(func_str)(arg)))
        # 断言结果与期望相等
        tm.assert_equal(result, expected)

    # 测试函数，用于验证样本抽样功能中的生成器用法
    def test_sample_generator(self, frame_or_series):
        # 创建一个对象，其内容为从0到99的数组
        obj = frame_or_series(np.arange(100))
        # 使用指定的随机数生成器生成器对象
        rng = np.random.default_rng(2)

        # 连续调用应该会推进种子状态，结果不应完全相同
        result1 = obj.sample(n=50, random_state=rng)
        result2 = obj.sample(n=50, random_state=rng)
        assert not (result1.index.values == result2.index.values).all()

        # 使用相同的生成器初始化必须产生相同的结果
        # 连续调用应该会推进种子状态
        result1 = obj.sample(n=50, random_state=np.random.default_rng(11))
        result2 = obj.sample(n=50, random_state=np.random.default_rng(11))
        # 断言结果相等
        tm.assert_equal(result1, result2)

    # 测试函数，用于验证非替换抽样中的上采样设置
    def test_sample_upsampling_without_replacement(self, frame_or_series):
        # 创建一个 DataFrame 对象，包含单列，其值为 'a', 'b', 'c'
        obj = DataFrame({"A": list("abc")})
        # 获取适当的对象，可以是 DataFrame 或 Series
        obj = tm.get_obj(obj, frame_or_series)

        # 设置错误消息，用于在不正确的参数设置时引发 ValueError
        msg = (
            "Replace has to be set to `True` when "
            "upsampling the population `frac` > 1."
        )
        # 使用 pytest 断言引发的异常包含特定消息
        with pytest.raises(ValueError, match=msg):
            obj.sample(frac=2, replace=False)
class TestSampleDataFrame:
    # Tests which are relevant only for DataFrame, so these are
    # as fully parametrized as they can get.

    def test_sample_axis1(self):
        # Check weights with axis = 1
        easy_weight_list = [0] * 3  # 创建一个长度为3的列表，初始值为0
        easy_weight_list[2] = 1  # 将列表第三个元素改为1，其它元素为0

        df = DataFrame(
            {"col1": range(10, 20), "col2": range(20, 30), "colString": ["a"] * 10}
        )  # 创建一个DataFrame对象，包含三列数据

        sample1 = df.sample(n=1, axis=1, weights=easy_weight_list)  # 对DataFrame进行抽样，axis=1，使用权重列表
        tm.assert_frame_equal(sample1, df[["colString"]])  # 断言抽样结果与预期的DataFrame子集相等

        # Test default axes
        tm.assert_frame_equal(
            df.sample(n=3, random_state=42), df.sample(n=3, axis=0, random_state=42)
        )  # 断言使用不同方式调用抽样函数的结果相等

    def test_sample_aligns_weights_with_frame(self):
        # Test that function aligns weights with frame
        df = DataFrame({"col1": [5, 6, 7], "col2": ["a", "b", "c"]}, index=[9, 5, 3])
        ser = Series([1, 0, 0], index=[3, 5, 9])

        tm.assert_frame_equal(df.loc[[3]], df.sample(1, weights=ser))  # 断言使用权重进行抽样后的DataFrame与预期相等

        # Weights have index values to be dropped because not in
        # sampled DataFrame
        ser2 = Series([0.001, 0, 10000], index=[3, 5, 10])
        tm.assert_frame_equal(df.loc[[3]], df.sample(1, weights=ser2))  # 断言使用包含不在抽样DataFrame中的索引的权重抽样结果与预期相等

        # Weights have empty values to be filed with zeros
        ser3 = Series([0.01, 0], index=[3, 5])
        tm.assert_frame_equal(df.loc[[3]], df.sample(1, weights=ser3))  # 断言使用包含空值的权重抽样结果与预期相等

        # No overlap in weight and sampled DataFrame indices
        ser4 = Series([1, 0], index=[1, 2])

        with pytest.raises(ValueError, match="Invalid weights: weights sum to zero"):
            df.sample(1, weights=ser4)  # 使用不合法权重进行抽样应引发异常

    def test_sample_is_copy(self):
        # GH#27357, GH#30784: ensure the result of sample is an actual copy and
        # doesn't track the parent dataframe
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=["a", "b", "c"]
        )
        df2 = df.sample(3)

        with tm.assert_produces_warning(None):
            df2["d"] = 1  # 断言修改抽样结果的列会触发警告，表明抽样结果是原始DataFrame的副本

    def test_sample_does_not_modify_weights(self):
        # GH-42843
        result = np.array([np.nan, 1, np.nan])
        expected = result.copy()
        ser = Series([1, 2, 3])

        # Test numpy array weights won't be modified in place
        ser.sample(weights=result)
        tm.assert_numpy_array_equal(result, expected)  # 断言在使用numpy数组作为权重时，权重数组不会被就地修改

        # Test DataFrame column won't be modified in place
        df = DataFrame({"values": [1, 1, 1], "weights": [1, np.nan, np.nan]})
        expected = df["weights"].copy()

        df.sample(frac=1.0, replace=True, weights="weights")
        result = df["weights"]
        tm.assert_series_equal(result, expected)  # 断言在使用DataFrame列作为权重时，权重列不会被就地修改

    def test_sample_ignore_index(self):
        # GH 38581
        df = DataFrame(
            {"col1": range(10, 20), "col2": range(20, 30), "colString": ["a"] * 10}
        )
        result = df.sample(3, ignore_index=True)  # 测试使用ignore_index参数进行抽样后的结果

        expected_index = Index(range(3))
        tm.assert_index_equal(result.index, expected_index, exact=True)  # 断言抽样结果的索引与预期的索引相等
```