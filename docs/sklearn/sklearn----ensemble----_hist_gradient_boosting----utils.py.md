# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\utils.py`

```
# 导入所需模块和类
"""This module contains utility routines."""
from ...base import is_classifier
from .binning import _BinMapper

# 定义函数，返回另一个库中未拟合的估算器，具有匹配的超参数
def get_equivalent_estimator(estimator, lib="lightgbm", n_classes=None):
    """Return an unfitted estimator from another lib with matching hyperparams.
    
    This utility function takes care of renaming the sklearn parameters into
    their LightGBM, XGBoost or CatBoost equivalent parameters.
    
    # unmapped XGB parameters:
    # - min_samples_leaf
    # - min_data_in_bin
    # - min_split_gain (there is min_split_loss though?)
    
    # unmapped Catboost parameters:
    # max_leaves
    # min_*
    """

    # 检查库的合法性，仅支持 'lightgbm', 'xgboost', 'catboost'
    if lib not in ("lightgbm", "xgboost", "catboost"):
        raise ValueError(
            "accepted libs are lightgbm, xgboost, and catboost.  got {}".format(lib)
        )

    # 获取 sklearn 参数
    sklearn_params = estimator.get_params()

    # 如果损失函数为 'auto'，则抛出异常
    if sklearn_params["loss"] == "auto":
        raise ValueError(
            "auto loss is not accepted. We need to know if "
            "the problem is binary or multiclass classification."
        )
    
    # 如果启用了 early_stopping，则抛出异常
    if sklearn_params["early_stopping"]:
        raise NotImplementedError("Early stopping should be deactivated.")

    # 映射 LightGBM 损失函数到 sklearn 的损失函数
    lightgbm_loss_mapping = {
        "squared_error": "regression_l2",
        "absolute_error": "regression_l1",
        "log_loss": "binary" if n_classes == 2 else "multiclass",
        "gamma": "gamma",
        "poisson": "poisson",
    }

    # 设置 LightGBM 参数
    lightgbm_params = {
        "objective": lightgbm_loss_mapping[sklearn_params["loss"]],
        "learning_rate": sklearn_params["learning_rate"],
        "n_estimators": sklearn_params["max_iter"],
        "num_leaves": sklearn_params["max_leaf_nodes"],
        "max_depth": sklearn_params["max_depth"],
        "min_data_in_leaf": sklearn_params["min_samples_leaf"],
        "reg_lambda": sklearn_params["l2_regularization"],
        "max_bin": sklearn_params["max_bins"],
        "min_data_in_bin": 1,
        "min_sum_hessian_in_leaf": 1e-3,
        "min_split_gain": 0,
        "verbosity": 10 if sklearn_params["verbose"] else -10,
        "boost_from_average": True,
        "enable_bundle": False,  # also makes feature order consistent
        "subsample_for_bin": _BinMapper().subsample,
        "poisson_max_delta_step": 1e-12,
        "feature_fraction_bynode": sklearn_params["max_features"],
    }

    # 如果损失函数为 'log_loss' 并且类别数大于 2，则调整参数
    if sklearn_params["loss"] == "log_loss" and n_classes > 2:
        # 在多类别损失中，LightGBM 将 hessian 乘以 2
        lightgbm_params["min_sum_hessian_in_leaf"] *= 2
        # LightGBM 3.0 引入了不同的 hessian 缩放方式
        if n_classes is not None:
            lightgbm_params["learning_rate"] *= n_classes / (n_classes - 1)

    # XGB
    # XGBoost损失函数映射表，将sklearn参数中的损失函数映射到XGBoost的相应损失函数
    xgboost_loss_mapping = {
        "squared_error": "reg:linear",
        "absolute_error": "LEAST_ABSOLUTE_DEV_NOT_SUPPORTED",
        "log_loss": "reg:logistic" if n_classes == 2 else "multi:softmax",
        "gamma": "reg:gamma",
        "poisson": "count:poisson",
    }

    # XGBoost参数字典，设置训练模型的各种参数
    xgboost_params = {
        "tree_method": "hist",  # 使用直方图算法构建树
        "grow_policy": "lossguide",  # 使用损失导向增长策略，以便设置最大叶子节点数
        "objective": xgboost_loss_mapping[sklearn_params["loss"]],  # 设置优化目标为对应的XGBoost损失函数
        "learning_rate": sklearn_params["learning_rate"],  # 学习率
        "n_estimators": sklearn_params["max_iter"],  # 树的数量或迭代次数
        "max_leaves": sklearn_params["max_leaf_nodes"],  # 最大叶子节点数
        "max_depth": sklearn_params["max_depth"] or 0,  # 树的最大深度
        "lambda": sklearn_params["l2_regularization"],  # L2正则化参数
        "max_bin": sklearn_params["max_bins"],  # 直方图的最大箱数
        "min_child_weight": 1e-3,  # 子节点的最小权重
        "verbosity": 2 if sklearn_params["verbose"] else 0,  # 输出详细程度
        "silent": sklearn_params["verbose"] == 0,  # 是否静默模式
        "n_jobs": -1,  # 并行处理的作业数，-1表示使用所有处理器
        "colsample_bynode": sklearn_params["max_features"],  # 每个节点分裂时用于特征抽样的列数比例
    }

    # CatBoost损失函数映射表，将sklearn参数中的损失函数映射到CatBoost的相应损失函数
    catboost_loss_mapping = {
        "squared_error": "RMSE",
        "absolute_error": "LEAST_ASBOLUTE_DEV_NOT_SUPPORTED",
        "log_loss": "Logloss" if n_classes == 2 else "MultiClass",
        "gamma": None,
        "poisson": "Poisson",
    }

    # CatBoost参数字典，设置训练模型的各种参数
    catboost_params = {
        "loss_function": catboost_loss_mapping[sklearn_params["loss"]],  # 设置优化目标为对应的CatBoost损失函数
        "learning_rate": sklearn_params["learning_rate"],  # 学习率
        "iterations": sklearn_params["max_iter"],  # 迭代次数
        "depth": sklearn_params["max_depth"],  # 树的深度
        "reg_lambda": sklearn_params["l2_regularization"],  # L2正则化参数
        "max_bin": sklearn_params["max_bins"],  # 直方图的最大箱数
        "feature_border_type": "Median",  # 特征边界类型
        "leaf_estimation_method": "Newton",  # 叶子估计方法
        "verbose": bool(sklearn_params["verbose"]),  # 是否输出详细信息
    }

    # 根据所选的机器学习库选择相应的模型并返回
    if lib == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        if is_classifier(estimator):
            return LGBMClassifier(**lightgbm_params)  # 返回LightGBM分类器模型
        else:
            return LGBMRegressor(**lightgbm_params)  # 返回LightGBM回归器模型

    elif lib == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        if is_classifier(estimator):
            return XGBClassifier(**xgboost_params)  # 返回XGBoost分类器模型
        else:
            return XGBRegressor(**xgboost_params)  # 返回XGBoost回归器模型

    else:
        from catboost import CatBoostClassifier, CatBoostRegressor

        if is_classifier(estimator):
            return CatBoostClassifier(**catboost_params)  # 返回CatBoost分类器模型
        else:
            return CatBoostRegressor(**catboost_params)  # 返回CatBoost回归器模型
```