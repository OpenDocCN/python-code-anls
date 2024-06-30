# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_plot\tests\test_common_curve_display.py`

```
# 导入所需的库
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试

# 导入机器学习相关模块和类
from sklearn.base import ClassifierMixin, clone  # 导入ClassifierMixin类和clone函数
from sklearn.calibration import CalibrationDisplay  # 导入CalibrationDisplay类
from sklearn.compose import make_column_transformer  # 导入make_column_transformer函数
from sklearn.datasets import load_iris  # 导入load_iris函数，用于加载鸢尾花数据集
from sklearn.exceptions import NotFittedError  # 导入NotFittedError异常类
from sklearn.linear_model import LogisticRegression  # 导入LogisticRegression类
from sklearn.metrics import (
    ConfusionMatrixDisplay,  # 导入ConfusionMatrixDisplay类
    DetCurveDisplay,  # 导入DetCurveDisplay类
    PrecisionRecallDisplay,  # 导入PrecisionRecallDisplay类
    PredictionErrorDisplay,  # 导入PredictionErrorDisplay类
    RocCurveDisplay,  # 导入RocCurveDisplay类
)
from sklearn.pipeline import make_pipeline  # 导入make_pipeline函数
from sklearn.preprocessing import StandardScaler  # 导入StandardScaler类
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 导入DecisionTreeClassifier和DecisionTreeRegressor类


@pytest.fixture(scope="module")
def data():
    return load_iris(return_X_y=True)  # 返回加载的鸢尾花数据集


@pytest.fixture(scope="module")
def data_binary(data):
    X, y = data
    return X[y < 2], y[y < 2]  # 返回二元分类问题的数据子集


@pytest.mark.parametrize(
    "Display",
    [CalibrationDisplay, DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay],
)
def test_display_curve_error_classifier(pyplot, data, data_binary, Display):
    """检查当仅支持二元分类时是否会引发适当的错误。"""
    X, y = data
    X_binary, y_binary = data_binary
    clf = DecisionTreeClassifier().fit(X, y)

    # Case 1: 多类分类器与多类目标
    msg = "Expected 'estimator' to be a binary classifier. Got 3 classes instead."
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(clf, X, y)

    # Case 2: 多类分类器与二元目标
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(clf, X_binary, y_binary)

    # Case 3: 二元分类器与多类目标
    clf = DecisionTreeClassifier().fit(X_binary, y_binary)
    msg = "The target y is not binary. Got multiclass type of target."
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(clf, X, y)


@pytest.mark.parametrize(
    "Display",
    [CalibrationDisplay, DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay],
)
def test_display_curve_error_regression(pyplot, data_binary, Display):
    """检查在回归问题中是否会引发错误。"""

    # Case 1: 回归器
    X, y = data_binary
    regressor = DecisionTreeRegressor().fit(X, y)

    msg = "Expected 'estimator' to be a binary classifier. Got DecisionTreeRegressor"
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(regressor, X, y)

    # Case 2: 回归目标
    classifier = DecisionTreeClassifier().fit(X, y)
    # 强制`y_true`被视为回归问题
    y = y + 0.5
    msg = "The target y is not binary. Got continuous type of target."
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(classifier, X, y)
    with pytest.raises(ValueError, match=msg):
        Display.from_predictions(y, regressor.fit(X, y).predict(X))


@pytest.mark.parametrize(
    "response_method, msg",
    [
        (
            "predict_proba",
            "MyClassifier has none of the following attributes: predict_proba.",
        ),  # 检查 MyClassifier 是否具有 predict_proba 属性，如果没有则返回相应的错误信息
        (
            "decision_function",
            "MyClassifier has none of the following attributes: decision_function.",
        ),  # 检查 MyClassifier 是否具有 decision_function 属性，如果没有则返回相应的错误信息
        (
            "auto",
            (
                "MyClassifier has none of the following attributes: predict_proba,"
                " decision_function."
            ),  # 检查 MyClassifier 是否具有 predict_proba 和 decision_function 属性中的任何一个或两者都没有，如果没有则返回相应的错误信息
        ),
        (
            "bad_method",
            "MyClassifier has none of the following attributes: bad_method.",
        ),  # 检查 MyClassifier 是否具有 bad_method 属性，如果没有则返回相应的错误信息
    ],
@pytest.mark.parametrize(
    "Display", [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay]
)


# 使用 pytest 的 parametrize 装饰器，为测试函数指定多个参数化的 Display 类型



def test_display_curve_error_no_response(
    pyplot,
    data_binary,
    response_method,
    msg,
    Display,
):


# 定义测试函数 test_display_curve_error_no_response，检查当请求的响应方法在给定训练好的分类器中未定义时，是否正确引发错误
# 接受参数 pyplot, data_binary, response_method, msg 和 Display



X, y = data_binary


# 从 data_binary 中解包获取特征集 X 和标签集 y



class MyClassifier(ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = [0, 1]
        return self


# 定义名为 MyClassifier 的自定义分类器类，继承自 ClassifierMixin
# 实现 fit 方法，用于模拟一个简单的分类器，设置 self.classes_ 属性并返回自身



clf = MyClassifier().fit(X, y)


# 创建 MyClassifier 的实例 clf，并调用 fit 方法拟合数据 X, y



with pytest.raises(AttributeError, match=msg):
    Display.from_estimator(clf, X, y, response_method=response_method)


# 使用 pytest 的 raises 断言检查在调用 Display.from_estimator 方法时是否引发 AttributeError 异常，并匹配错误消息 msg



@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])


# 使用 pytest 的 parametrize 装饰器，为测试函数 test_display_curve_estimator_name_multiple_calls 指定多个参数化的 constructor_name



clf = LogisticRegression().fit(X, y)


# 创建 LogisticRegression 的实例 clf，并调用 fit 方法拟合数据 X, y



y_pred = clf.predict_proba(X)[:, 1]


# 使用拟合好的 clf 对象，预测数据 X 的概率，并取出概率的第二列



assert constructor_name in ("from_estimator", "from_predictions")


# 使用断言确保 constructor_name 在 ("from_estimator", "from_predictions") 中



if constructor_name == "from_estimator":
    disp = Display.from_estimator(clf, X, y, name=clf_name)
else:
    disp = Display.from_predictions(y, y_pred, name=clf_name)


# 根据 constructor_name 的值，调用 Display 的不同方法创建 disp 对象
# 若 constructor_name 为 "from_estimator"，则使用 clf, X, y 和 clf_name 调用 Display.from_estimator 方法
# 否则，使用 y, y_pred 和 clf_name 调用 Display.from_predictions 方法



assert disp.estimator_name == clf_name


# 使用断言检查 disp 对象的 estimator_name 属性是否与 clf_name 相等



pyplot.close("all")


# 关闭 pyplot 的所有图形窗口，清理状态



disp.plot()


# 调用 disp 对象的 plot 方法，绘制曲线图



assert clf_name in disp.line_.get_label()


# 使用断言检查 clf_name 是否在 disp 对象的 line_ 属性的标签中



clf_name = "another_name"
disp.plot(name=clf_name)


# 更新 clf_name 为 "another_name"，并调用 disp 对象的 plot 方法，传入参数 name=clf_name



@pytest.mark.parametrize(
    "clf",
    [
        LogisticRegression(),
        make_pipeline(StandardScaler(), LogisticRegression()),
        make_pipeline(
            make_column_transformer((StandardScaler(), [0, 1])), LogisticRegression()
        ),
    ],
)
@pytest.mark.parametrize(
    "Display", [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay]
)


# 使用 pytest 的 parametrize 装饰器，为测试函数 test_display_curve_not_fitted_errors 指定多个参数化的 clf 和 Display



model = clone(clf)


# 克隆 clf，创建 model 对象



with pytest.raises(NotFittedError):
    Display.from_estimator(model, X, y)


# 使用 pytest 的 raises 断言检查调用 Display.from_estimator 方法时是否引发 NotFittedError 异常



model.fit(X, y)


# 调用 model 对象的 fit 方法，拟合数据 X, y



assert model.__class__.__name__ in disp.line_.get_label()


# 使用断言检查 model 对象的类名是否在 disp 对象的 line_ 属性的标签中



assert disp.estimator_name == model.__class__.__name__


# 使用断言检查 disp 对象的 estimator_name 属性是否与 model 对象的类名相等



@pytest.mark.parametrize(
    "Display", [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay]
)


# 使用 pytest 的 parametrize 装饰器，为测试函数 test_display_curve_n_samples_consistency 指定多个参数化的 Display 类型
    """
    检查当 `y_pred` 或 `sample_weight` 长度不一致时引发的错误。
    """
    # 从二进制数据中获取特征向量 X 和目标向量 y
    X, y = data_binary
    # 使用决策树分类器拟合数据
    classifier = DecisionTreeClassifier().fit(X, y)

    # 错误信息字符串
    msg = "Found input variables with inconsistent numbers of samples"

    # 测试：确保 Display.from_estimator 在不一致输入时会引发 ValueError 错误
    with pytest.raises(ValueError, match=msg):
        # 测试样例：使用 X 的部分数据和完整的 y 数据
        Display.from_estimator(classifier, X[:-2], y)
    with pytest.raises(ValueError, match=msg):
        # 测试样例：使用完整的 X 数据和部分的 y 数据
        Display.from_estimator(classifier, X, y[:-2])
    with pytest.raises(ValueError, match=msg):
        # 测试样例：使用完整的 X 数据和部分的 y 数据以及样本权重
        Display.from_estimator(classifier, X, y, sample_weight=np.ones(X.shape[0] - 2))
# 使用 pytest.mark.parametrize 装饰器为 test_display_curve_error_pos_label 函数参数化，每次使用不同的 Display 类
@pytest.mark.parametrize(
    "Display", [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay]
)
# 定义测试函数 test_display_curve_error_pos_label，用于检查在未指定 pos_label 时的错误消息一致性
def test_display_curve_error_pos_label(pyplot, data_binary, Display):
    """Check consistence of error message when `pos_label` should be specified."""
    # 从测试数据中获取特征 X 和标签 y
    X, y = data_binary
    # 修改 y 的值，加 10
    y = y + 10

    # 使用决策树分类器拟合数据
    classifier = DecisionTreeClassifier().fit(X, y)
    # 对 X 预测得到概率值并取最后一列
    y_pred = classifier.predict_proba(X)[:, -1]
    # 期望的错误消息格式
    msg = r"y_true takes value in {10, 11} and pos_label is not specified"
    # 断言应该抛出 ValueError 异常，并且异常消息与期望的格式匹配
    with pytest.raises(ValueError, match=msg):
        Display.from_predictions(y, y_pred)


# 使用 pytest.mark.parametrize 装饰器为 test_classifier_display_curve_named_constructor_return_type 函数参数化，每次使用不同的 Display 类和构造函数
@pytest.mark.parametrize(
    "Display",
    [
        CalibrationDisplay,
        DetCurveDisplay,
        PrecisionRecallDisplay,
        RocCurveDisplay,
        PredictionErrorDisplay,
        ConfusionMatrixDisplay,
    ],
)
@pytest.mark.parametrize(
    "constructor",
    ["from_predictions", "from_estimator"],
)
# 定义测试函数 test_classifier_display_curve_named_constructor_return_type，检查命名构造函数在子类化时返回的正确类型
def test_classifier_display_curve_named_constructor_return_type(
    pyplot, data_binary, Display, constructor
):
    """Check that named constructors return the correct type when subclassed.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/27675
    """
    # 从测试数据中获取特征 X 和标签 y
    X, y = data_binary

    # 这里可以是任何值 - 我们只需检查命名构造函数的返回类型，因此在此处唯一的要求是不出错地实例化该类
    y_pred = y

    # 使用逻辑回归分类器拟合数据
    classifier = LogisticRegression().fit(X, y)

    # 定义 Display 的子类，继承自 Display
    class SubclassOfDisplay(Display):
        pass

    # 根据构造函数参数选择不同的方法创建 curve 对象
    if constructor == "from_predictions":
        curve = SubclassOfDisplay.from_predictions(y, y_pred)
    else:  # constructor == "from_estimator"
        curve = SubclassOfDisplay.from_estimator(classifier, X, y)

    # 断言 curve 对象的类型是 SubclassOfDisplay 的实例
    assert isinstance(curve, SubclassOfDisplay)
```