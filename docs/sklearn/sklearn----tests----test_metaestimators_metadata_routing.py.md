# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_metaestimators_metadata_routing.py`

```
# 导入必要的库和模块
import copy  # 导入 copy 模块，用于对象的深复制
import re  # 导入 re 模块，用于正则表达式的操作

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于单元测试

from sklearn import config_context  # 导入 sklearn 中的 config_context 模块
from sklearn.base import is_classifier  # 导入 sklearn 中的 is_classifier 函数
from sklearn.calibration import CalibratedClassifierCV  # 导入 sklearn 中的 CalibratedClassifierCV 类
from sklearn.compose import TransformedTargetRegressor  # 导入 sklearn 中的 TransformedTargetRegressor 类
from sklearn.covariance import GraphicalLassoCV  # 导入 sklearn 中的 GraphicalLassoCV 类
from sklearn.ensemble import (  # 导入 sklearn 中的集成学习方法相关类
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
)
from sklearn.exceptions import UnsetMetadataPassedError  # 导入 sklearn 中的 UnsetMetadataPassedError 异常类
from sklearn.experimental import (  # 导入 sklearn 中的实验性模块
    enable_halving_search_cv,  # noqa
    enable_iterative_imputer,  # noqa
)
from sklearn.feature_selection import (  # 导入 sklearn 中的特征选择相关类
    RFE,
    RFECV,
    SelectFromModel,
    SequentialFeatureSelector,
)
from sklearn.impute import IterativeImputer  # 导入 sklearn 中的 IterativeImputer 类
from sklearn.linear_model import (  # 导入 sklearn 中的线性模型相关类
    ElasticNetCV,
    LarsCV,
    LassoCV,
    LassoLarsCV,
    LogisticRegressionCV,
    MultiTaskElasticNetCV,
    MultiTaskLassoCV,
    OrthogonalMatchingPursuitCV,
    RANSACRegressor,
    RidgeClassifierCV,
    RidgeCV,
)
from sklearn.model_selection import (  # 导入 sklearn 中的模型选择相关类
    FixedThresholdClassifier,
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    TunedThresholdClassifierCV,
)
from sklearn.multiclass import (  # 导入 sklearn 中的多类别分类器相关类
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)
from sklearn.multioutput import (  # 导入 sklearn 中的多输出相关类
    ClassifierChain,
    MultiOutputClassifier,
    MultiOutputRegressor,
    RegressorChain,
)
from sklearn.semi_supervised import SelfTrainingClassifier  # 导入 sklearn 中的 SelfTrainingClassifier 类
from sklearn.tests.metadata_routing_common import (  # 导入 sklearn 中的测试元数据路由相关模块和函数
    ConsumingClassifier,
    ConsumingRegressor,
    ConsumingScorer,
    ConsumingSplitter,
    NonConsumingClassifier,
    NonConsumingRegressor,
    _Registry,
    assert_request_is_empty,
    check_recorded_metadata,
)
from sklearn.utils.metadata_routing import MetadataRouter  # 导入 sklearn 中的 MetadataRouter 类

rng = np.random.RandomState(42)  # 创建一个指定种子的随机数生成器
N, M = 100, 4  # 设置数据集大小 N 和特征数 M
X = rng.rand(N, M)  # 生成一个 N 行 M 列的随机数矩阵作为特征数据集 X
y = rng.randint(0, 3, size=N)  # 生成一个长度为 N 的随机整数数组作为目标变量 y
y_binary = (y >= 1).astype(int)  # 根据 y 生成二元分类目标变量 y_binary
classes = np.unique(y)  # 获取目标变量 y 中的唯一类别值
y_multi = rng.randint(0, 3, size=(N, 3))  # 生成一个大小为 N×3 的随机整数矩阵作为多类别目标变量 y_multi
classes_multi = [np.unique(y_multi[:, i]) for i in range(y_multi.shape[1])]  # 获取 y_multi 每列的唯一类别值
metadata = rng.randint(0, 10, size=N)  # 生成一个长度为 N 的随机整数数组作为元数据
sample_weight = rng.rand(N)  # 生成一个长度为 N 的随机数数组作为样本权重
groups = np.array([0, 1] * (len(y) // 2))  # 生成一个长度为 N 的数组作为分组标识符

@pytest.fixture(autouse=True)
def enable_slep006():
    """为所有测试启用 SLEP006。"""
    with config_context(enable_metadata_routing=True):  # 使用 config_context 启用元数据路由
        yield  # 返回上下文管理器

METAESTIMATORS: list = [  # 定义一个元算法估计器列表
    {
        "metaestimator": MultiOutputRegressor,  # 元估计器类型为 MultiOutputRegressor
        "estimator_name": "estimator",  # 估计器名称为 "estimator"
        "estimator": "regressor",  # 具体的估计器为 "regressor"
        "X": X,  # 特征数据集 X
        "y": y_multi,  # 多类别目标变量 y_multi
        "estimator_routing_methods": ["fit", "partial_fit"],  # 支持的估计器方法包括 "fit" 和 "partial_fit"
    },
    {
        "metaestimator": MultiOutputClassifier,  # 元估计器类型为 MultiOutputClassifier
        "estimator_name": "estimator",  # 估计器名称为 "estimator"
        "estimator": "classifier",  # 具体的估计器为 "classifier"
        "X": X,  # 特征数据集 X
        "y": y_multi,  # 多类别目标变量 y_multi
        "estimator_routing_methods": ["fit", "partial_fit"],  # 支持的估计器方法包括 "fit" 和 "partial_fit"
        "method_args": {"partial_fit": {"classes": classes_multi}},  # 针对 "partial_fit" 方法的参数配置
    },
    {
        "metaestimator": CalibratedClassifierCV,
        "estimator_name": "estimator",
        "estimator": "classifier",
        "X": X,
        "y": y,
        "estimator_routing_methods": ["fit"],
        "preserves_metadata": "subset",
    },
    
    {
        "metaestimator": ClassifierChain,
        "estimator_name": "base_estimator",
        "estimator": "classifier",
        "X": X,
        "y": y_multi,
        "estimator_routing_methods": ["fit"],
    },
    
    {
        "metaestimator": RegressorChain,
        "estimator_name": "base_estimator",
        "estimator": "regressor",
        "X": X,
        "y": y_multi,
        "estimator_routing_methods": ["fit"],
    },
    
    {
        "metaestimator": LogisticRegressionCV,
        "X": X,
        "y": y,
        "scorer_name": "scoring",
        "scorer_routing_methods": ["fit", "score"],
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    
    {
        "metaestimator": GridSearchCV,
        "estimator_name": "estimator",
        "estimator": "classifier",
        "init_args": {"param_grid": {"alpha": [0.1, 0.2]}},
        "X": X,
        "y": y,
        "estimator_routing_methods": ["fit"],
        "preserves_metadata": "subset",
        "scorer_name": "scoring",
        "scorer_routing_methods": ["fit", "score"],
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    
    {
        "metaestimator": RandomizedSearchCV,
        "estimator_name": "estimator",
        "estimator": "classifier",
        "init_args": {"param_distributions": {"alpha": [0.1, 0.2]}},
        "X": X,
        "y": y,
        "estimator_routing_methods": ["fit"],
        "preserves_metadata": "subset",
        "scorer_name": "scoring",
        "scorer_routing_methods": ["fit", "score"],
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    
    {
        "metaestimator": HalvingGridSearchCV,
        "estimator_name": "estimator",
        "estimator": "classifier",
        "init_args": {"param_grid": {"alpha": [0.1, 0.2]}},
        "X": X,
        "y": y,
        "estimator_routing_methods": ["fit"],
        "preserves_metadata": "subset",
        "scorer_name": "scoring",
        "scorer_routing_methods": ["fit", "score"],
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    
    {
        "metaestimator": HalvingRandomSearchCV,
        "estimator_name": "estimator",
        "estimator": "classifier",
        "init_args": {"param_distributions": {"alpha": [0.1, 0.2]}},
        "X": X,
        "y": y,
        "estimator_routing_methods": ["fit"],
        "preserves_metadata": "subset",
        "scorer_name": "scoring",
        "scorer_routing_methods": ["fit", "score"],
        "cv_name": "cv",
        "cv_routing_methods": ["fit"],
    },
    
    
    
    # 定义了多个字典，每个字典描述了不同的元估计器及其配置
    {
        "metaestimator": <MetaEstimatorClass>,
        "estimator_name": "<Name>",
        "estimator": "<Type>",
        "X": X,
        "y": <TargetVariable>,
        "estimator_routing_methods": ["fit"],
        "preserves_metadata": "subset",
        "init_args": {"param_grid" 或 "param_distributions": {"alpha": [0.1, 0.2]}},
        "scorer_name": "scoring",
        "scorer_routing_methods": ["fit", "score"],
        "cv_name": "cv",
        "cv_routing_methods": ["fit"]
    }
    
    
    这段代码定义了多个字典，每个字典描述了不同的机器学习元估计器（MetaEstimator）及其相关的配置信息，用于模型训练和评估。
    {
        "metaestimator": FixedThresholdClassifier,  # 使用固定阈值的分类器作为元估计器
        "estimator_name": "estimator",  # 估计器名称为 "estimator"
        "estimator": "classifier",  # 估计器类型为分类器
        "X": X,  # 输入特征矩阵 X
        "y": y_binary,  # 二元目标变量 y
        "estimator_routing_methods": ["fit"],  # 支持的估计器方法为 "fit"
        "preserves_metadata": "subset",  # 保留元数据的子集
    },
    {
        "metaestimator": TunedThresholdClassifierCV,  # 使用交叉验证调整阈值的分类器作为元估计器
        "estimator_name": "estimator",  # 估计器名称为 "estimator"
        "estimator": "classifier",  # 估计器类型为分类器
        "X": X,  # 输入特征矩阵 X
        "y": y_binary,  # 二元目标变量 y
        "estimator_routing_methods": ["fit"],  # 支持的估计器方法为 "fit"
        "preserves_metadata": "subset",  # 保留元数据的子集
    },
    {
        "metaestimator": OneVsRestClassifier,  # 使用一对多分类策略的分类器作为元估计器
        "estimator_name": "estimator",  # 估计器名称为 "estimator"
        "estimator": "classifier",  # 估计器类型为分类器
        "X": X,  # 输入特征矩阵 X
        "y": y,  # 多类别目标变量 y
        "estimator_routing_methods": ["fit", "partial_fit"],  # 支持的估计器方法为 "fit" 和 "partial_fit"
        "method_args": {"partial_fit": {"classes": classes}},  # 部分拟合方法的参数设置
    },
    {
        "metaestimator": OneVsOneClassifier,  # 使用一对一分类策略的分类器作为元估计器
        "estimator_name": "estimator",  # 估计器名称为 "estimator"
        "estimator": "classifier",  # 估计器类型为分类器
        "X": X,  # 输入特征矩阵 X
        "y": y,  # 多类别目标变量 y
        "estimator_routing_methods": ["fit", "partial_fit"],  # 支持的估计器方法为 "fit" 和 "partial_fit"
        "preserves_metadata": "subset",  # 保留元数据的子集
        "method_args": {"partial_fit": {"classes": classes}},  # 部分拟合方法的参数设置
    },
    {
        "metaestimator": OutputCodeClassifier,  # 使用输出编码的分类器作为元估计器
        "estimator_name": "estimator",  # 估计器名称为 "estimator"
        "estimator": "classifier",  # 估计器类型为分类器
        "init_args": {"random_state": 42},  # 初始化参数，设置随机种子为 42
        "X": X,  # 输入特征矩阵 X
        "y": y,  # 目标变量 y
        "estimator_routing_methods": ["fit"],  # 支持的估计器方法为 "fit"
    },
    {
        "metaestimator": SelectFromModel,  # 使用模型选择的方法作为元估计器
        "estimator_name": "estimator",  # 估计器名称为 "estimator"
        "estimator": "classifier",  # 估计器类型为分类器
        "X": X,  # 输入特征矩阵 X
        "y": y,  # 目标变量 y
        "estimator_routing_methods": ["fit", "partial_fit"],  # 支持的估计器方法为 "fit" 和 "partial_fit"
        "method_args": {"partial_fit": {"classes": classes}},  # 部分拟合方法的参数设置
    },
    {
        "metaestimator": OrthogonalMatchingPursuitCV,  # 使用正交匹配追踪的交叉验证作为元估计器
        "X": X,  # 输入特征矩阵 X
        "y": y,  # 目标变量 y
        "cv_name": "cv",  # 交叉验证名称为 "cv"
        "cv_routing_methods": ["fit"],  # 支持的交叉验证方法为 "fit"
    },
    {
        "metaestimator": ElasticNetCV,  # 使用弹性网络的交叉验证作为元估计器
        "X": X,  # 输入特征矩阵 X
        "y": y,  # 目标变量 y
        "cv_name": "cv",  # 交叉验证名称为 "cv"
        "cv_routing_methods": ["fit"],  # 支持的交叉验证方法为 "fit"
    },
    {
        "metaestimator": LassoCV,  # 使用Lasso的交叉验证作为元估计器
        "X": X,  # 输入特征矩阵 X
        "y": y,  # 目标变量 y
        "cv_name": "cv",  # 交叉验证名称为 "cv"
        "cv_routing_methods": ["fit"],  # 支持的交叉验证方法为 "fit"
    },
    {
        "metaestimator": MultiTaskElasticNetCV,  # 使用多任务弹性网络的交叉验证作为元估计器
        "X": X,  # 输入特征矩阵 X
        "y": y_multi,  # 多任务目标变量 y_multi
        "cv_name": "cv",  # 交叉验证名称为 "cv"
        "cv_routing_methods": ["fit"],  # 支持的交叉验证方法为 "fit"
    },
    {
        "metaestimator": MultiTaskLassoCV,  # 使用多任务Lasso的交叉验证作为元估计器
        "X": X,  # 输入特征矩阵 X
        "y": y_multi,  # 多任务目标变量 y_multi
        "cv_name": "cv",  # 交叉验证名称为 "cv"
        "cv_routing_methods": ["fit"],  # 支持的交叉验证方法为 "fit"
    },
    {
        "metaestimator": LarsCV,  # 使用LARS的交叉验证作为元估计器
        "X": X,  # 输入特征矩阵 X
        "y": y,  # 目标变量 y
        "cv_name": "cv",  # 交叉验证名称为 "cv"
        "cv_routing_methods": ["fit"],  # 支持的交叉验证方法为 "fit"
    },
    {
        "metaestimator": LassoLarsCV,  # 使用Lasso LARS的交叉验证作为元估计器
        "X": X,  # 输入特征矩阵 X
        "y": y,  # 目标变量 y
        "cv_name": "cv",  # 交叉验证名称为 "cv"
        "cv_routing_methods": ["fit"],  # 支持的交叉验证方法为 "fit"
    },
    `
    {
        "metaestimator": RANSACRegressor,  // 使用 RANSAC 回归器作为元估计器
        "estimator_name": "estimator",     // 元估计器名称设定为 "estimator"
        "estimator": "regressor",          // 元估计器类型为回归器
        "init_args": {"min_samples": 0.5}, // 初始参数设定为 {"min_samples": 0.5}
        "X": X,                            // 输入特征 X
        "y": y,                            // 输出标签 y
        "preserves_metadata": "subset",    // 保留部分元数据
        "estimator_routing_methods": ["fit", "predict", "score"],  // 支持的估计器方法包括 "fit", "predict", "score"
        "method_mapping": {"fit": ["fit", "score"]}  // 方法映射，fit 方法映射到 ["fit", "score"]
    },
    {
        "metaestimator": IterativeImputer,  // 使用 IterativeImputer 作为元估计器
        "estimator_name": "estimator",     // 元估计器名称设定为 "estimator"
        "estimator": "regressor",          // 元估计器类型为回归器
        "init_args": {"skip_complete": False},  // 初始参数设定为 {"skip_complete": False}
        "X": X,                            // 输入特征 X
        "y": y,                            // 输出标签 y
        "estimator_routing_methods": ["fit"]  // 支持的估计器方法只有 "fit"
    },
    {
        "metaestimator": BaggingClassifier, // 使用 BaggingClassifier 作为元估计器
        "estimator_name": "estimator",     // 元估计器名称设定为 "estimator"
        "estimator": "classifier",         // 元估计器类型为分类器
        "X": X,                            // 输入特征 X
        "y": y,                            // 输出标签 y
        "preserves_metadata": False,       // 不保留元数据
        "estimator_routing_methods": ["fit"]  // 支持的估计器方法只有 "fit"
    },
    {
        "metaestimator": BaggingRegressor, // 使用 BaggingRegressor 作为元估计器
        "estimator_name": "estimator",     // 元估计器名称设定为 "estimator"
        "estimator": "regressor",          // 元估计器类型为回归器
        "X": X,                            // 输入特征 X
        "y": y,                            // 输出标签 y
        "preserves_metadata": False,       // 不保留元数据
        "estimator_routing_methods": ["fit"]  // 支持的估计器方法只有 "fit"
    },
    {
        "metaestimator": RidgeCV,           // 使用 RidgeCV 作为元估计器
        "X": X,                             // 输入特征 X
        "y": y,                             // 输出标签 y
        "scorer_name": "scoring",           // 评分器名称设定为 "scoring"
        "scorer_routing_methods": ["fit"]   // 支持的评分器方法只有 "fit"
    },
    {
        "metaestimator": RidgeClassifierCV,  // 使用 RidgeClassifierCV 作为元估计器
        "X": X,                             // 输入特征 X
        "y": y,                             // 输出标签 y
        "scorer_name": "scoring",           // 评分器名称设定为 "scoring"
        "scorer_routing_methods": ["fit"]   // 支持的评分器方法只有 "fit"
    },
    {
        "metaestimator": RidgeCV,           // 使用 RidgeCV 作为元估计器
        "X": X,                             // 输入特征 X
        "y": y,                             // 输出标签 y
        "scorer_name": "scoring",           // 评分器名称设定为 "scoring"
        "scorer_routing_methods": ["fit"],  // 支持的评分器方法只有 "fit"
        "cv_name": "cv",                    // 交叉验证器名称设定为 "cv"
        "cv_routing_methods": ["fit"]       // 支持的交叉验证器方法只有 "fit"
    },
    {
        "metaestimator": RidgeClassifierCV,  // 使用 RidgeClassifierCV 作为元估计器
        "X": X,                             // 输入特征 X
        "y": y,                             // 输出标签 y
        "scorer_name": "scoring",           // 评分器名称设定为 "scoring"
        "scorer_routing_methods": ["fit"],  // 支持的评分器方法只有 "fit"
        "cv_name": "cv",                    // 交叉验证器名称设定为 "cv"
        "cv_routing_methods": ["fit"]       // 支持的交叉验证器方法只有 "fit"
    },
    {
        "metaestimator": GraphicalLassoCV,   // 使用 GraphicalLassoCV 作为元估计器
        "X": X,                              // 输入特征 X
        "y": y,                              // 输出标签 y
        "cv_name": "cv",                     // 交叉验证器名称设定为 "cv"
        "cv_routing_methods": ["fit"]        // 支持的交叉验证器方法只有 "fit"
    },
    {
        "metaestimator": TransformedTargetRegressor,  // 使用 TransformedTargetRegressor 作为元估计器
        "estimator": "regressor",                   // 元估计器类型为回归器
        "estimator_name": "regressor",              // 元估计器名称设定为 "regressor"
        "X": X,                                     // 输入特征 X
        "y": y,                                     // 输出标签 y
        "estimator_routing_methods": ["fit", "predict"]  // 支持的估计器方法包括 "fit", "predict"
    },
# 列表包含所有要测试的元估计器及其设置

# 键说明：
# - metaestimator: 要测试的元估计器
# - estimator_name: 子估计器参数的名称
# - estimator: 子估计器类型，可以是"regressor"或"classifier"
# - init_args: 传递给元估计器构造函数的参数
# - X: 用于拟合和预测的 X 数据
# - y: 用于拟合的 y 数据
# - estimator_routing_methods: 检查路由到子估计器的所有方法的列表
# - preserves_metadata:
#     - True（默认）: 元估计器将元数据无修改地传递给子估计器。我们检查子估计器记录的值是否与传递给元估计器的值相同。
#     - False: 不检查值，仅检查是否传递了预期名称/键的元数据。
#     - "subset": 我们检查子估计器记录的元数据是否是传递给元估计器的元数据的子集。
# - scorer_name: 评分器参数的名称
# - scorer_routing_methods: 检查路由到评分器的所有方法的列表
# - cv_name: CV 分割器参数的名称
# - cv_routing_methods: 检查路由到分割器的所有方法的列表
# - method_args: 一个字典，定义传递给方法的额外参数，例如将 `classes` 传递给 `partial_fit`。
# - method_mapping: 一个字典，形式为 `{caller: [callee1, ...]}`，指示应调用哪些 `.set_{method}_request` 方法来设置请求值。如果不存在，则假定一对一映射。

# pytest 使用的 ID，以在运行测试时获取有意义的详细消息
METAESTIMATOR_IDS = [str(row["metaestimator"].__name__) for row in METAESTIMATORS]

# 不支持的估计器列表
UNSUPPORTED_ESTIMATORS = [
    AdaBoostClassifier(),
    AdaBoostRegressor(),
    RFE(ConsumingClassifier()),
    RFECV(ConsumingClassifier()),
    SelfTrainingClassifier(ConsumingClassifier()),
    SequentialFeatureSelector(ConsumingClassifier()),
]

def get_init_args(metaestimator_info, sub_estimator_consumes):
    """获取元估计器的初始化参数

    这是一个辅助函数，用于从 METAESTIMATORS 列表中获取元估计器的初始化参数。如果不需要初始化参数，则返回空字典。

    参数
    ----------
    metaestimator_info : dict
        来自 METAESTIMATORS 的元估计器信息

    sub_estimator_consumes : bool
        子估计器是否消耗元数据

    返回
    -------
    kwargs : dict
        元估计器的初始化参数

    (estimator, estimator_registry) : (estimator, registry)
        子估计器及其对应的注册表

    (scorer, scorer_registry) : (scorer, registry)
        评分器及其对应的注册表

    (cv, cv_registry) : (CV 分割器, registry)
        CV 分割器及其对应的注册表
    """
    # 从 metaestimator_info 中获取初始化参数字典，如果不存在则为空字典
    kwargs = metaestimator_info.get("init_args", {})
    # 初始化 estimator 和 estimator_registry 为 None
    estimator, estimator_registry = None, None
    # 初始化 scorer 和 scorer_registry 为 None
    scorer, scorer_registry = None, None
    # 初始化 cv 和 cv_registry 为 None
    cv, cv_registry = None, None
    
    # 检查 metaestimator_info 是否包含 "estimator" 键
    if "estimator" in metaestimator_info:
        # 获取 estimator_name
        estimator_name = metaestimator_info["estimator_name"]
        # 初始化 estimator_registry 为 _Registry 的实例
        estimator_registry = _Registry()
        # 获取 sub_estimator_type
        sub_estimator_type = metaestimator_info["estimator"]
        
        # 如果 sub_estimator_consumes 为真
        if sub_estimator_consumes:
            # 根据 sub_estimator_type 类型选择 ConsumingRegressor 或 ConsumingClassifier
            if sub_estimator_type == "regressor":
                estimator = ConsumingRegressor(estimator_registry)
            elif sub_estimator_type == "classifier":
                estimator = ConsumingClassifier(estimator_registry)
            else:
                # 抛出异常，指示未允许的 sub_estimator_type
                raise ValueError("Unpermitted `sub_estimator_type`.")  # pragma: nocover
        else:
            # 根据 sub_estimator_type 类型选择 NonConsumingRegressor 或 NonConsumingClassifier
            if sub_estimator_type == "regressor":
                estimator = NonConsumingRegressor()
            elif sub_estimator_type == "classifier":
                estimator = NonConsumingClassifier()
            else:
                # 抛出异常，指示未允许的 sub_estimator_type
                raise ValueError("Unpermitted `sub_estimator_type`.")  # pragma: nocover
        
        # 将 estimator 添加到 kwargs 字典中，键为 estimator_name
        kwargs[estimator_name] = estimator
    
    # 检查 metaestimator_info 是否包含 "scorer_name" 键
    if "scorer_name" in metaestimator_info:
        # 获取 scorer_name
        scorer_name = metaestimator_info["scorer_name"]
        # 初始化 scorer_registry 为 _Registry 的实例
        scorer_registry = _Registry()
        # 创建 ConsumingScorer 的实例，传入 scorer_registry
        scorer = ConsumingScorer(registry=scorer_registry)
        # 将 scorer 添加到 kwargs 字典中，键为 scorer_name
        kwargs[scorer_name] = scorer
    
    # 检查 metaestimator_info 是否包含 "cv_name" 键
    if "cv_name" in metaestimator_info:
        # 获取 cv_name
        cv_name = metaestimator_info["cv_name"]
        # 初始化 cv_registry 为 _Registry 的实例
        cv_registry = _Registry()
        # 创建 ConsumingSplitter 的实例，传入 cv_registry
        cv = ConsumingSplitter(registry=cv_registry)
        # 将 cv 添加到 kwargs 字典中，键为 cv_name
        kwargs[cv_name] = cv
    
    # 返回元组，包含 kwargs 字典及三个元组：(estimator, estimator_registry), (scorer, scorer_registry), (cv, cv_registry)
    return (
        kwargs,
        (estimator, estimator_registry),
        (scorer, scorer_registry),
        (cv, cv_registry),
    )
# 定义函数 `set_requests`，设置子估计器的请求方法。
def set_requests(estimator, *, method_mapping, methods, metadata_name, value=True):
    # 遍历传入的方法列表
    for caller in methods:
        # 对于每个调用方法，查找其对应的被调用方法列表，如果没有映射，则默认为其自身
        for callee in method_mapping.get(caller, [caller]):
            # 根据被调用方法名动态获取方法对象 `set_{callee}_request`
            set_request_for_method = getattr(estimator, f"set_{callee}_request")
            # 调用获取到的方法，设置请求参数 `metadata_name` 的值为 `value`
            set_request_for_method(**{metadata_name: value})
            # 如果估计器是分类器，并且被调用方法是 "partial_fit"
            if is_classifier(estimator) and callee == "partial_fit":
                # 针对分类器额外设置 `classes=True`
                set_request_for_method(classes=True)


# 使用 pytest 的参数化装饰器，测试不支持的估计器的元数据路由
@pytest.mark.parametrize("estimator", UNSUPPORTED_ESTIMATORS)
def test_unsupported_estimators_get_metadata_routing(estimator):
    """测试对于尚未实现路由的元估计器，get_metadata_routing 抛出 NotImplementedError 异常。"""
    with pytest.raises(NotImplementedError):
        estimator.get_metadata_routing()


# 使用 pytest 的参数化装饰器，测试不支持的估计器在元数据路由启用时的 fit 方法
@pytest.mark.parametrize("estimator", UNSUPPORTED_ESTIMATORS)
def test_unsupported_estimators_fit_with_metadata(estimator):
    """测试在启用元数据路由且传递了元数据时，对于尚未实现路由的元估计器，fit 方法抛出 NotImplementedError 异常。"""
    with pytest.raises(NotImplementedError):
        try:
            # 尝试调用 fit 方法，传入样本数据和可能的样本权重
            estimator.fit([[1]], [1], sample_weight=[1])
        except TypeError:
            # 如果不是所有元估计器都支持样本权重，则跳过该测试
            raise NotImplementedError


# 测试 _Registry 对象的复制行为
def test_registry_copy():
    """测试确保 _Registry 对象在新实例中不会被复制。"""
    # 创建两个 _Registry 实例 a 和 b
    a = _Registry()
    b = _Registry()
    # 断言 a 和 b 是不同的对象
    assert a is not b
    # 断言 a 是其浅拷贝的同一个实例
    assert a is copy.copy(a)
    # 断言 a 是其深拷贝的同一个实例
    assert a is copy.deepcopy(a)


# 使用 pytest 的参数化装饰器，测试默认请求的设置
@pytest.mark.parametrize("metaestimator", METAESTIMATORS, ids=METAESTIMATOR_IDS)
def test_default_request(metaestimator):
    """检查默认请求是否为空并且类型正确。"""
    # 获取元估计器的类对象和初始化参数
    cls = metaestimator["metaestimator"]
    kwargs, *_ = get_init_args(metaestimator, sub_estimator_consumes=True)
    # 使用初始化参数创建元估计器实例
    instance = cls(**kwargs)
    # 如果 metaestimator 中包含 "cv_name"
    if "cv_name" in metaestimator:
        # 我们的 GroupCV 分割器默认请求组，但在这个测试中应该忽略
        exclude = {"splitter": ["split"]}
    else:
        exclude = None
    # 断言确保 instance.get_metadata_routing() 返回一个空值或空集合
    assert_request_is_empty(instance.get_metadata_routing(), exclude=exclude)
    # 断言确保 instance.get_metadata_routing() 返回的对象是 MetadataRouter 类的实例
    assert isinstance(instance.get_metadata_routing(), MetadataRouter)
# 使用 pytest.mark.parametrize 装饰器参数化测试函数，对每个 METAESTIMATORS 执行测试
@pytest.mark.parametrize("metaestimator", METAESTIMATORS, ids=METAESTIMATOR_IDS)
def test_error_on_missing_requests_for_sub_estimator(metaestimator):
    # 测试当子估计器的请求未设置时是否会引发 UnsetMetadataPassedError 异常
    if "estimator" not in metaestimator:
        # 如果 metaestimator 中不包含 "estimator" 键，表示这个测试对没有子估计器的元估计器无意义
        return

    # 从 metaestimator 中获取需要的参数和数据
    cls = metaestimator["metaestimator"]
    X = metaestimator["X"]
    y = metaestimator["y"]
    routing_methods = metaestimator["estimator_routing_methods"]

    # 遍历元估计器定义的路由方法
    for method_name in routing_methods:
        # 对于每个路由方法，检查两个关键字 "sample_weight" 和 "metadata"
        for key in ["sample_weight", "metadata"]:
            # 调用 get_init_args 函数获取初始化参数，包括估计器和评分器
            kwargs, (estimator, _), (scorer, _), *_ = get_init_args(
                metaestimator, sub_estimator_consumes=True
            )
            # 如果存在评分器，则设置评分请求中的关键字为 True
            if scorer:
                scorer.set_score_request(**{key: True})
            # 从样本权重和元数据中选择当前关键字对应的值
            val = {"sample_weight": sample_weight, "metadata": metadata}[key]
            # 构建当前方法所需的关键字参数
            method_kwargs = {key: val}
            # 实例化元估计器类
            instance = cls(**kwargs)
            # 构建错误消息，说明为什么会引发 UnsetMetadataPassedError 异常
            msg = (
                f"[{key}] are passed but are not explicitly set as requested or not"
                f" requested for {estimator.__class__.__name__}.{method_name}"
            )
            # 使用 pytest.raises 检查是否抛出了 UnsetMetadataPassedError 异常，并验证错误消息
            with pytest.raises(UnsetMetadataPassedError, match=re.escape(msg)):
                # 获取当前方法的引用
                method = getattr(instance, method_name)
                if "fit" not in method_name:
                    # 在调用 fit 方法之前设置请求
                    set_requests(
                        estimator,
                        method_mapping=metaestimator.get("method_mapping", {}),
                        methods=["fit"],
                        metadata_name=key,
                    )
                    # 调用元估计器的 fit 方法
                    instance.fit(X, y, **method_kwargs)
                # 确保请求在测试执行后被取消设置，以防它们在设置 fit 方法时被设置为副作用
                set_requests(
                    estimator,
                    method_mapping=metaestimator.get("method_mapping", {}),
                    methods=["fit"],
                    metadata_name=key,
                    value=None,
                )
                try:
                    # 对于除 fit 和 partial_fit 之外的方法，调用时不应传递 y 参数
                    method(X, y, **method_kwargs)
                except TypeError:
                    method(X, **method_kwargs)


# 使用 pytest.mark.parametrize 装饰器参数化测试函数，对每个 METAESTIMATORS 执行测试
@pytest.mark.parametrize("metaestimator", METAESTIMATORS, ids=METAESTIMATOR_IDS)
def test_setting_request_on_sub_estimator_removes_error(metaestimator):
    # 当在子估计器上显式请求元数据时，不应有错误发生
    # 如果 metaestimator 中没有 "estimator" 键，则退出函数，因为后续的操作依赖于子估计器的存在
    if "estimator" not in metaestimator:
        # 这个测试仅适用于具有子估计器的元估计器，例如 MyMetaEstimator(estimator=MySubEstimator())
        return

    # 从 metaestimator 中获取元估计器类
    cls = metaestimator["metaestimator"]
    # 从 metaestimator 中获取 X 数据
    X = metaestimator["X"]
    # 从 metaestimator 中获取 y 数据
    y = metaestimator["y"]
    # 从 metaestimator 中获取估计器路由方法列表
    routing_methods = metaestimator["estimator_routing_methods"]
    # 从 metaestimator 中获取方法映射字典，如果不存在则为空字典
    method_mapping = metaestimator.get("method_mapping", {})
    # 从 metaestimator 中获取是否保留元数据的标志，默认为 True
    preserves_metadata = metaestimator.get("preserves_metadata", True)

    # 遍历估计器路由方法列表
    for method_name in routing_methods:
        # 遍历 ["sample_weight", "metadata"] 列表
        for key in ["sample_weight", "metadata"]:
            # 根据 key 获取对应的值，构建方法参数字典
            val = {"sample_weight": sample_weight, "metadata": metadata}[key]
            method_kwargs = {key: val}

            # 调用 get_init_args 函数获取初始化参数
            kwargs, (estimator, registry), (scorer, _), (cv, _) = get_init_args(
                metaestimator, sub_estimator_consumes=True
            )

            # 如果 scorer 存在，则设置 scorer 的请求
            if scorer:
                set_requests(
                    scorer, method_mapping={}, methods=["score"], metadata_name=key
                )

            # 如果 cv 存在，则设置 cv 的分割请求
            if cv:
                cv.set_split_request(groups=True, metadata=True)

            # 在子估计器上调用 `set_{method}_request({metadata}==True)` 方法
            set_requests(
                estimator,
                method_mapping=method_mapping,
                methods=[method_name],
                metadata_name=key,
            )

            # 使用获取的初始化参数实例化元估计器类
            instance = cls(**kwargs)
            # 获取元估计器实例的特定方法
            method = getattr(instance, method_name)
            # 获取 metaestimator 中的方法参数字典中指定方法名的附加参数
            extra_method_args = metaestimator.get("method_args", {}).get(
                method_name, {}
            )

            # 如果方法名不是 "fit"，则在调用方法之前先调用 instance 的 fit 方法
            if "fit" not in method_name:
                instance.fit(X, y)

            try:
                # 调用方法，参数包括 X, y 和其他方法参数
                method(X, y, **method_kwargs, **extra_method_args)
            except TypeError:
                # 如果抛出 TypeError，则调用方法只传递 X 和其他方法参数
                method(X, **method_kwargs, **extra_method_args)

            # 断言 registry 不为空，以确保记录了至少一个估计器
            assert registry
            # 如果 preserves_metadata 是 "subset"，则 split_params 是 method_kwargs 的键集合，否则为空元组
            split_params = (
                method_kwargs.keys() if preserves_metadata == "subset" else ()
            )

            # 遍历 registry 中的每个估计器，检查记录的元数据
            for estimator in registry:
                check_recorded_metadata(
                    estimator,
                    method=method_name,
                    parent=method_name,
                    split_params=split_params,
                    **method_kwargs,
                )
# 使用 pytest 的参数化装饰器，对每个 metaestimator 进行参数化测试
@pytest.mark.parametrize("metaestimator", METAESTIMATORS, ids=METAESTIMATOR_IDS)
def test_non_consuming_estimator_works(metaestimator):
    # 测试非消费型估计器是否正常工作，即在未设置任何请求的情况下，元估计器能够正常工作。
    # 这是对 https://github.com/scikit-learn/scikit-learn/issues/28239 的回归测试。

    # 如果 metaestimator 不包含键 "estimator"，则跳过测试
    if "estimator" not in metaestimator:
        return

    def set_request(estimator, method_name):
        # 设置请求的辅助函数，例如在估计器上调用 set_fit_request
        if is_classifier(estimator) and method_name == "partial_fit":
            estimator.set_partial_fit_request(classes=True)

    # 获取 metaestimator 中的元估计器类
    cls = metaestimator["metaestimator"]
    # 获取 metaestimator 中的输入数据 X 和标签 y
    X = metaestimator["X"]
    y = metaestimator["y"]
    # 获取 metaestimator 中的估计器路由方法列表
    routing_methods = metaestimator["estimator_routing_methods"]

    # 遍历估计器路由方法列表
    for method_name in routing_methods:
        # 调用 get_init_args 函数，获取初始化参数
        kwargs, (estimator, _), (_, _), (_, _) = get_init_args(
            metaestimator, sub_estimator_consumes=False
        )
        # 使用获取的参数实例化元估计器
        instance = cls(**kwargs)
        # 调用 set_request 函数，设置估计器的请求
        set_request(estimator, method_name)
        # 获取方法名称对应的方法对象
        method = getattr(instance, method_name)
        # 获取额外的方法参数
        extra_method_args = metaestimator.get("method_args", {}).get(method_name, {})
        
        # 如果方法名称中不包含 "fit"，则调用实例的 fit 方法
        if "fit" not in method_name:
            instance.fit(X, y, **extra_method_args)
        
        # 尝试调用方法，并确保不会引发路由错误
        try:
            # `fit` 和 `partial_fit` 方法接受 y 参数，其他方法不接受
            method(X, y, **extra_method_args)
        except TypeError:
            method(X, **extra_method_args)


# 使用 pytest 的参数化装饰器，对每个 metaestimator 进行参数化测试
@pytest.mark.parametrize("metaestimator", METAESTIMATORS, ids=METAESTIMATOR_IDS)
def test_metadata_is_routed_correctly_to_scorer(metaestimator):
    """测试确保任何请求的元数据正确路由到交叉验证估计器中的底层评分器。"""
    # 如果 metaestimator 中不包含键 "scorer_name"，则跳过测试
    if "scorer_name" not in metaestimator:
        return

    # 获取 metaestimator 中的元估计器类
    cls = metaestimator["metaestimator"]
    # 获取 metaestimator 中的评分器路由方法列表
    routing_methods = metaestimator["scorer_routing_methods"]
    # 获取 metaestimator 中的方法映射
    method_mapping = metaestimator.get("method_mapping", {})
    # 遍历给定的路由方法列表
    for method_name in routing_methods:
        # 调用函数获取初始化参数，包括元估计器、子估计器消耗标志、评分器和注册表、交叉验证对象及其消耗标志
        kwargs, (estimator, _), (scorer, registry), (cv, _) = get_init_args(
            metaestimator, sub_estimator_consumes=True
        )
        # 设置评分器的请求，指定使用样本权重
        scorer.set_score_request(sample_weight=True)
        # 如果存在交叉验证对象，设置其分割请求，包括分组信息和元数据
        if cv:
            cv.set_split_request(groups=True, metadata=True)
        # 如果存在估计器，设置其请求，指定使用样本权重作为元数据
        if estimator is not None:
            set_requests(
                estimator,
                method_mapping=method_mapping,
                methods=[method_name],
                metadata_name="sample_weight",
            )
        # 根据参数实例化一个类对象
        instance = cls(**kwargs)
        # 获取类实例中对应方法名的方法对象
        method = getattr(instance, method_name)
        # 准备方法调用的关键字参数，包括样本权重
        method_kwargs = {"sample_weight": sample_weight}
        # 如果方法名不包含'fit'，则调用实例的fit方法进行拟合
        if "fit" not in method_name:
            instance.fit(X, y)
        # 调用方法执行预期操作，传入X、y数据和方法关键字参数
        method(X, y, **method_kwargs)

        # 确保注册表不为空
        assert registry
        # 遍历注册表中的每个评分器对象，检查记录的元数据是否符合预期
        for _scorer in registry:
            check_recorded_metadata(
                obj=_scorer,
                method="score",
                parent=method_name,
                split_params=("sample_weight",),
                **method_kwargs,
            )
@pytest.mark.parametrize("metaestimator", METAESTIMATORS, ids=METAESTIMATOR_IDS)
def test_metadata_is_routed_correctly_to_splitter(metaestimator):
    """Test that any requested metadata is correctly routed to the underlying
    splitters in CV estimators.
    """
    # 检查 metaestimator 中是否包含 cv_routing_methods 键
    if "cv_routing_methods" not in metaestimator:
        # 如果不包含，说明该测试仅适用于接受 CV splitter 的元估计器
        return

    # 获取 metaestimator 中的相关信息
    cls = metaestimator["metaestimator"]
    routing_methods = metaestimator["cv_routing_methods"]
    X_ = metaestimator["X"]
    y_ = metaestimator["y"]

    # 遍历 routing_methods 列表
    for method_name in routing_methods:
        # 调用 get_init_args 函数获取初始化参数
        kwargs, (estimator, _), (scorer, _), (cv, registry) = get_init_args(
            metaestimator, sub_estimator_consumes=True
        )
        # 如果 estimator 存在，则设置 fit 请求的 sample_weight 和 metadata 为 False
        if estimator:
            estimator.set_fit_request(sample_weight=False, metadata=False)
        # 如果 scorer 存在，则设置 score 请求的 sample_weight 和 metadata 为 False
        if scorer:
            scorer.set_score_request(sample_weight=False, metadata=False)
        # 设置 cv 的 split 请求的 groups 为 True，metadata 为 True
        cv.set_split_request(groups=True, metadata=True)
        # 使用 cls 类创建一个实例 instance，传入 kwargs 中的参数
        instance = cls(**kwargs)
        # 构建 method_kwargs 字典，设置 groups 和 metadata 参数
        method_kwargs = {"groups": groups, "metadata": metadata}
        # 获取 instance 对象的 method_name 方法
        method = getattr(instance, method_name)
        # 调用 method 方法，传入 X_, y_ 和 method_kwargs 参数
        method(X_, y_, **method_kwargs)
        # 断言 registry 不为空
        assert registry
        # 遍历 registry 列表中的 _splitter 对象
        for _splitter in registry:
            # 调用 check_recorded_metadata 函数，检查记录的 metadata
            check_recorded_metadata(
                obj=_splitter, method="split", parent=method_name, **method_kwargs
            )
```