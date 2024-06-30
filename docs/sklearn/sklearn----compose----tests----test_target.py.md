# `D:\src\scipysrc\scikit-learn\sklearn\compose\tests\test_target.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试

from sklearn import datasets  # 导入sklearn中的数据集模块
from sklearn.base import BaseEstimator, TransformerMixin, clone  # 导入基础估计器、转换器混合类和克隆函数
from sklearn.compose import TransformedTargetRegressor  # 导入转换目标回归器
from sklearn.dummy import DummyRegressor  # 导入虚拟回归器
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit  # 导入线性回归和正交匹配追踪模型
from sklearn.pipeline import Pipeline  # 导入管道工具
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # 导入函数转换器和标准缩放器
from sklearn.utils._testing import assert_allclose, assert_no_warnings  # 导入测试函数

friedman = datasets.make_friedman1(random_state=0)  # 生成Friedman1数据集


def test_transform_target_regressor_error():
    X, y = friedman  # 从数据集中获取特征X和目标y

    # 在转换目标回归器中使用线性回归器和标准缩放器同时提供转换器和函数
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(),
        transformer=StandardScaler(),
        func=np.exp,
        inverse_func=np.log,
    )

    with pytest.raises(
        ValueError,
        match="'transformer' and functions 'func'/'inverse_func' cannot both be set.",
    ):
        regr.fit(X, y)

    # 使用不支持样本权重的正交匹配追踪模型进行拟合
    sample_weight = np.ones((y.shape[0],))
    regr = TransformedTargetRegressor(
        regressor=OrthogonalMatchingPursuit(), transformer=StandardScaler()
    )

    with pytest.raises(
        TypeError,
        match=r"fit\(\) got an unexpected " "keyword argument 'sample_weight'",
    ):
        regr.fit(X, y, sample_weight=sample_weight)

    # 仅提供(func, inverse_func)中的一个，但另一个未提供
    regr = TransformedTargetRegressor(func=np.exp)
    with pytest.raises(
        ValueError,
        match="When 'func' is provided, 'inverse_func' must also be provided",
    ):
        regr.fit(X, y)

    regr = TransformedTargetRegressor(inverse_func=np.log)
    with pytest.raises(
        ValueError,
        match="When 'inverse_func' is provided, 'func' must also be provided",
    ):
        regr.fit(X, y)


def test_transform_target_regressor_invertible():
    X, y = friedman  # 从数据集中获取特征X和目标y

    # 使用线性回归器，给定func=np.sqrt和inverse_func=np.log，并检查是否可逆
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.sqrt,
        inverse_func=np.log,
        check_inverse=True,
    )

    with pytest.warns(
        UserWarning,
        match=(
            "The provided functions or"
            " transformer are not strictly inverse of each other."
        ),
    ):
        regr.fit(X, y)

    # 使用线性回归器，给定func=np.sqrt和inverse_func=np.log，不检查是否可逆
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.sqrt, inverse_func=np.log
    )
    regr.set_params(check_inverse=False)
    assert_no_warnings(regr.fit, X, y)


def _check_standard_scaled(y, y_pred):
    y_mean = np.mean(y, axis=0)  # 计算y的均值
    y_std = np.std(y, axis=0)  # 计算y的标准差
    assert_allclose((y - y_mean) / y_std, y_pred)  # 断言y的标准化结果与预测值y_pred的接近程度


def _check_shifted_by_one(y, y_pred):
    assert_allclose(y + 1, y_pred)  # 断言y加一后与预测值y_pred的接近程度


def test_transform_target_regressor_functions():
    X, y = friedman  # 从数据集中获取特征X和目标y

    # 使用线性回归器，给定func=np.log和inverse_func=np.exp进行转换
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.log, inverse_func=np.exp
    )

    y_pred = regr.fit(X, y).predict(X)  # 拟合模型并预测
    # 检查转换器的输出
    # 对 y 进行转换，并压缩为一维数组
    y_tran = regr.transformer_.transform(y.reshape(-1, 1)).squeeze()
    # 断言预测值和转换后的对数值相近
    assert_allclose(np.log(y), y_tran)
    # 断言原始值经过逆转换后与原始 y 值相近
    assert_allclose(
        y, regr.transformer_.inverse_transform(y_tran.reshape(-1, 1)).squeeze()
    )
    # 断言 y 的形状与预测值的形状相同
    assert y.shape == y_pred.shape
    # 断言预测值与通过逆函数反向转换的回归器预测值相近
    assert_allclose(y_pred, regr.inverse_func(regr.regressor_.predict(X)))
    # 检查回归器的输出
    # 使用线性回归器拟合转换后的 X 和 regr.func(y) 的组合
    lr = LinearRegression().fit(X, regr.func(y))
    # 断言回归器的系数与线性回归器训练得到的系数相近
    assert_allclose(regr.regressor_.coef_.ravel(), lr.coef_.ravel())
# 定义测试函数，用于测试多输出的转换目标回归器功能
def test_transform_target_regressor_functions_multioutput():
    # 从friedman数据集中获取输入特征X
    X = friedman[0]
    # 从friedman数据集中获取输出目标y，包括两个列：第一列是原始y，第二列是y的平方加1
    y = np.vstack((friedman[1], friedman[1] ** 2 + 1)).T
    # 创建转换目标回归器对象，使用线性回归器作为基础回归器，log作为转换函数，exp作为逆转换函数
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.log, inverse_func=np.exp
    )
    # 对X和y进行拟合并预测
    y_pred = regr.fit(X, y).predict(X)
    # 检查转换器的输出
    y_tran = regr.transformer_.transform(y)
    assert_allclose(np.log(y), y_tran)  # 断言转换后的y与log(y)的接近程度
    assert_allclose(y, regr.transformer_.inverse_transform(y_tran))  # 断言逆转换后的y与原始y的接近程度
    assert y.shape == y_pred.shape  # 断言y和y_pred的形状一致
    assert_allclose(y_pred, regr.inverse_func(regr.regressor_.predict(X)))  # 断言预测值与逆转换后的预测值的接近程度
    # 检查回归器的输出
    lr = LinearRegression().fit(X, regr.func(y))  # 使用线性回归器拟合转换后的y
    assert_allclose(regr.regressor_.coef_.ravel(), lr.coef_.ravel())  # 断言回归系数的接近程度


@pytest.mark.parametrize(
    "X,y", [friedman, (friedman[0], np.vstack((friedman[1], friedman[1] ** 2 + 1)).T)]
)
# 参数化测试函数，测试一维转换器
def test_transform_target_regressor_1d_transformer(X, y):
    # 所有的scikit-learn转换器都期望2D数据。FunctionTransformer使用validate=False可以解除对输入为2D向量的检查。
    # 我们使用一个1D和2D的y数组来检查数据形状的一致性。
    transformer = FunctionTransformer(
        func=lambda x: x + 1, inverse_func=lambda x: x - 1
    )
    # 创建转换目标回归器对象，使用线性回归器作为基础回归器，指定转换器为上面定义的FunctionTransformer
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), transformer=transformer
    )
    # 对X和y进行拟合并预测
    y_pred = regr.fit(X, y).predict(X)
    assert y.shape == y_pred.shape  # 断言y和y_pred的形状一致
    # 检查前向变换的一致性
    y_tran = regr.transformer_.transform(y)
    _check_shifted_by_one(y, y_tran)  # 自定义函数检查y是否加了1
    assert y.shape == y_pred.shape  # 再次断言y和y_pred的形状一致
    # 检查逆变换的一致性
    assert_allclose(y, regr.transformer_.inverse_transform(y_tran).squeeze())  # 断言逆变换后的y与原始y的接近程度
    # 检查回归器的一致性
    lr = LinearRegression()
    transformer2 = clone(transformer)
    lr.fit(X, transformer2.fit_transform(y))  # 使用线性回归器拟合转换后的y
    y_lr_pred = lr.predict(X)
    assert_allclose(y_pred, transformer2.inverse_transform(y_lr_pred))  # 断言预测值与逆转换后的预测值的接近程度
    assert_allclose(regr.regressor_.coef_, lr.coef_)  # 断言回归系数的接近程度


@pytest.mark.parametrize(
    "X,y", [friedman, (friedman[0], np.vstack((friedman[1], friedman[1] ** 2 + 1)).T)]
)
# 参数化测试函数，测试二维转换器
def test_transform_target_regressor_2d_transformer(X, y):
    # 检查只接受2D数组的转换器与1D/2D y数组的一致性。
    transformer = StandardScaler()
    # 创建转换目标回归器对象，使用线性回归器作为基础回归器，指定转换器为上面定义的StandardScaler
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), transformer=transformer
    )
    # 对X和y进行拟合并预测
    y_pred = regr.fit(X, y).predict(X)
    assert y.shape == y_pred.shape  # 断言y和y_pred的形状一致
    # 检查前向变换的一致性
    if y.ndim == 1:  # 如果y是1维的，则创建一个2维数组并压缩结果
        y_tran = regr.transformer_.transform(y.reshape(-1, 1))
    else:
        y_tran = regr.transformer_.transform(y)
    _check_standard_scaled(y, y_tran.squeeze())  # 自定义函数检查y是否标准化
    assert y.shape == y_pred.shape  # 再次断言y和y_pred的形状一致
    # 检查逆变换的一致性
    assert_allclose(y, regr.transformer_.inverse_transform(y_tran).squeeze())  # 断言逆变换后的y与原始y的接近程度
    # 检查回归器的一致性
    lr = LinearRegression()
    # 克隆一个变压器对象，确保原始对象不被修改
    transformer2 = clone(transformer)
    
    # 检查 y 的维度是否为 1，如果是则需要创建一个二维数组并压缩结果
    if y.ndim == 1:
        # 使用线性回归模型拟合数据，将 y 重塑为二维数组并进行变压器转换，然后压缩结果
        lr.fit(X, transformer2.fit_transform(y.reshape(-1, 1)).squeeze())
        # 使用线性回归模型预测 X 数据，并将结果重塑为二维数组
        y_lr_pred = lr.predict(X).reshape(-1, 1)
        # 使用逆变压器将预测结果转换回原始数据空间，并压缩结果
        y_pred2 = transformer2.inverse_transform(y_lr_pred).squeeze()
    else:
        # 使用线性回归模型拟合数据，直接对 y 进行变压器转换
        lr.fit(X, transformer2.fit_transform(y))
        # 使用线性回归模型预测 X 数据
        y_lr_pred = lr.predict(X)
        # 使用逆变压器将预测结果转换回原始数据空间
        y_pred2 = transformer2.inverse_transform(y_lr_pred)
    
    # 断言两个预测结果的近似程度
    assert_allclose(y_pred, y_pred2)
    
    # 断言线性回归模型的系数与拟合的系数的近似程度
    assert_allclose(regr.regressor_.coef_, lr.coef_)
def test_transform_target_regressor_2d_transformer_multioutput():
    # 检查是否与仅接受2D数组和2D y数组的转换器一致性
    X = friedman[0]  # 从friedman数据集中获取特征数据
    y = np.vstack((friedman[1], friedman[1] ** 2 + 1)).T  # 从friedman数据集中获取目标数据，并创建多输出的目标数组
    transformer = StandardScaler()  # 创建标准化转换器
    regr = TransformedTargetRegressor(  # 创建转换目标回归器
        regressor=LinearRegression(), transformer=transformer
    )
    y_pred = regr.fit(X, y).predict(X)  # 对模型进行拟合和预测
    assert y.shape == y_pred.shape  # 断言预测结果形状与目标数据形状一致

    # 一致性的正向转换
    y_tran = regr.transformer_.transform(y)  # 使用转换器对目标数据进行转换
    _check_standard_scaled(y, y_tran)  # 调用检查标准化函数，确保转换正确
    assert y.shape == y_pred.shape  # 再次断言预测结果形状与目标数据形状一致

    # 一致性的逆向转换
    assert_allclose(y, regr.transformer_.inverse_transform(y_tran).squeeze())  # 断言逆向转换后的数据与原始目标数据接近
    lr = LinearRegression()  # 创建线性回归模型
    transformer2 = clone(transformer)  # 克隆标准化转换器
    lr.fit(X, transformer2.fit_transform(y))  # 对线性回归模型进行拟合
    y_lr_pred = lr.predict(X)  # 使用线性回归模型进行预测
    assert_allclose(y_pred, transformer2.inverse_transform(y_lr_pred))  # 断言预测结果与逆转换后的结果接近
    assert_allclose(regr.regressor_.coef_, lr.coef_)  # 断言转换目标回归器的回归器系数与线性回归模型的系数一致


def test_transform_target_regressor_3d_target():
    # 非回归测试，用于检查处理3D目标的转换器
    X = friedman[0]  # 从friedman数据集中获取特征数据
    y = np.tile(friedman[1].reshape(-1, 1, 1), [1, 3, 2])  # 创建3D目标数据并使用numpy的tile函数重复数组

    def flatten_data(data):
        return data.reshape(data.shape[0], -1)  # 定义用于扁平化数据的函数

    def unflatten_data(data):
        return data.reshape(data.shape[0], -1, 2)  # 定义用于反扁平化数据的函数

    transformer = FunctionTransformer(func=flatten_data, inverse_func=unflatten_data)  # 创建功能转换器
    regr = TransformedTargetRegressor(  # 创建转换目标回归器
        regressor=LinearRegression(), transformer=transformer
    )
    y_pred = regr.fit(X, y).predict(X)  # 对模型进行拟合和预测
    assert y.shape == y_pred.shape  # 断言预测结果形状与目标数据形状一致


def test_transform_target_regressor_multi_to_single():
    X = friedman[0]  # 从friedman数据集中获取特征数据
    y = np.transpose([friedman[1], (friedman[1] ** 2 + 1)])  # 转置并创建多到单一目标的数组

    def func(y):
        out = np.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)  # 定义用于处理目标数据的函数
        return out[:, np.newaxis]

    def inverse_func(y):
        return y  # 定义逆函数

    tt = TransformedTargetRegressor(  # 创建转换目标回归器
        func=func, inverse_func=inverse_func, check_inverse=False
    )
    tt.fit(X, y)  # 对模型进行拟合
    y_pred_2d_func = tt.predict(X)  # 使用模型进行预测
    assert y_pred_2d_func.shape == (100, 1)  # 断言预测结果形状为(100, 1)

    # 强制函数只返回一维数组
    def func(y):
        return np.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)

    tt = TransformedTargetRegressor(  # 创建转换目标回归器
        func=func, inverse_func=inverse_func, check_inverse=False
    )
    tt.fit(X, y)  # 对模型进行拟合
    y_pred_1d_func = tt.predict(X)  # 使用模型进行预测
    assert y_pred_1d_func.shape == (100, 1)  # 断言预测结果形状为(100, 1)

    assert_allclose(y_pred_1d_func, y_pred_2d_func)  # 断言两种函数预测结果接近


class DummyCheckerArrayTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        assert isinstance(X, np.ndarray)  # 断言输入特征为numpy数组
        return self

    def transform(self, X):
        assert isinstance(X, np.ndarray)  # 断言输入特征为numpy数组
        return X  # 返回输入特征

    def inverse_transform(self, X):
        assert isinstance(X, np.ndarray)  # 断言输入特征为numpy数组
        return X  # 返回输入特征
class DummyCheckerListRegressor(DummyRegressor):
    # 继承自 DummyRegressor 的自定义回归器类，用于检查输入是否为列表
    def fit(self, X, y, sample_weight=None):
        # 断言 X 是列表类型
        assert isinstance(X, list)
        # 调用父类的 fit 方法进行模型拟合
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        # 断言 X 是列表类型
        assert isinstance(X, list)
        # 调用父类的 predict 方法进行预测
        return super().predict(X)


def test_transform_target_regressor_ensure_y_array():
    # 确保传递给转换器的目标 y 总是一个 numpy 数组。类似地，如果传递的 X 是列表，
    # 我们检查预测器接收它的方式。
    X, y = friedman
    # 创建 TransformedTargetRegressor 对象
    tt = TransformedTargetRegressor(
        transformer=DummyCheckerArrayTransformer(),
        regressor=DummyCheckerListRegressor(),
        check_inverse=False,
    )
    # 使用 X 和 y 的列表形式拟合 TransformedTargetRegressor
    tt.fit(X.tolist(), y.tolist())
    # 使用 X 的列表形式进行预测
    tt.predict(X.tolist())
    # 断言拟合时传递的 X 是列表，但 y 不是列表
    with pytest.raises(AssertionError):
        tt.fit(X, y.tolist())
    # 断言预测时传递的 X 是列表
    with pytest.raises(AssertionError):
        tt.predict(X)


class DummyTransformer(TransformerMixin, BaseEstimator):
    """Dummy transformer which count how many time fit was called."""

    def __init__(self, fit_counter=0):
        self.fit_counter = fit_counter

    def fit(self, X, y=None):
        # 统计 fit 方法被调用的次数
        self.fit_counter += 1
        return self

    def transform(self, X):
        # 返回输入的 X，不做任何转换
        return X

    def inverse_transform(self, X):
        # 返回输入的 X，不做任何逆转换
        return X


@pytest.mark.parametrize("check_inverse", [False, True])
def test_transform_target_regressor_count_fit(check_inverse):
    # 用于测试问题 #11618 的回归测试
    # 检查我们对转换器仅调用一次 fit 方法
    X, y = friedman
    # 创建 TransformedTargetRegressor 对象
    ttr = TransformedTargetRegressor(
        transformer=DummyTransformer(), check_inverse=check_inverse
    )
    # 拟合 TransformedTargetRegressor
    ttr.fit(X, y)
    # 断言转换器的 fit_counter 等于 1
    assert ttr.transformer_.fit_counter == 1


class DummyRegressorWithExtraFitParams(DummyRegressor):
    # 继承自 DummyRegressor 的自定义回归器类，带有额外的 fit 参数

    def fit(self, X, y, sample_weight=None, check_input=True):
        # 在以下测试中，我们强制将 check_input 设为 false，确保它被实际传递给回归器
        assert not check_input
        # 调用父类的 fit 方法进行模型拟合
        return super().fit(X, y, sample_weight)


def test_transform_target_regressor_pass_fit_parameters():
    X, y = friedman
    # 创建 TransformedTargetRegressor 对象
    regr = TransformedTargetRegressor(
        regressor=DummyRegressorWithExtraFitParams(), transformer=DummyTransformer()
    )
    # 使用 check_input=False 参数拟合 TransformedTargetRegressor
    regr.fit(X, y, check_input=False)
    # 断言转换器的 fit_counter 等于 1
    assert regr.transformer_.fit_counter == 1


def test_transform_target_regressor_route_pipeline():
    X, y = friedman

    # 创建 TransformedTargetRegressor 对象
    regr = TransformedTargetRegressor(
        regressor=DummyRegressorWithExtraFitParams(), transformer=DummyTransformer()
    )
    # 定义 Pipeline 的估算器列表
    estimators = [("normalize", StandardScaler()), ("est", regr)]

    # 创建 Pipeline 对象
    pip = Pipeline(estimators)
    # 使用额外参数 **{"est__check_input": False} 拟合 Pipeline
    pip.fit(X, y, **{"est__check_input": False})

    # 断言转换器的 fit_counter 等于 1
    assert regr.transformer_.fit_counter == 1
    # 定义一个预测方法，接受输入参数 X 和一个检查输入的布尔值参数，默认为 True
    def predict(self, X, check_input=True):
        # 在下面的测试中，确保将检查输入参数设置为 False
        # 将 predict_called 属性设置为 True，表示预测方法被调用过
        self.predict_called = True
        # 使用断言确保 check_input 参数为 False，如果为 True，会触发 AssertionError
        assert not check_input
        # 调用父类的 predict 方法进行预测，并返回其结果
        return super().predict(X)
# 定义一个测试函数，用于测试转换目标回归器在传递额外预测参数时的行为。
def test_transform_target_regressor_pass_extra_predict_parameters():
    # 从数据集 friedman 中获取特征 X 和目标值 y
    X, y = friedman
    # 创建一个转换目标回归器对象，使用自定义的 DummyRegressorWithExtraPredictParams 作为回归器，
    # DummyTransformer 作为变换器
    regr = TransformedTargetRegressor(
        regressor=DummyRegressorWithExtraPredictParams(), transformer=DummyTransformer()
    )

    # 对转换目标回归器进行拟合，传入特征 X 和目标值 y
    regr.fit(X, y)
    # 使用转换目标回归器进行预测，传入特征 X，关闭输入检查选项（check_input=False）
    regr.predict(X, check_input=False)
    # 断言回归器的 predict 方法被调用过
    assert regr.regressor_.predict_called
```