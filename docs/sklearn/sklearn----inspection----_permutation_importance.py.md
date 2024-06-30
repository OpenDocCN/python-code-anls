# `D:\src\scipysrc\scikit-learn\sklearn\inspection\_permutation_importance.py`

```
# 导入必要的库和模块
"""Permutation importance for estimators."""

import numbers  # 导入用于数字判断的模块

import numpy as np  # 导入NumPy库

from ..ensemble._bagging import _generate_indices  # 导入生成索引的函数
from ..metrics import check_scoring, get_scorer_names  # 导入评分检查和获取评分器名称的函数
from ..model_selection._validation import _aggregate_score_dicts  # 导入聚合评分字典的函数
from ..utils import Bunch, _safe_indexing, check_array, check_random_state  # 导入工具函数和类
from ..utils._param_validation import (  # 导入参数验证相关的模块
    HasMethods,
    Integral,
    Interval,
    RealNotInt,
    StrOptions,
    validate_params,
)
from ..utils.parallel import Parallel, delayed  # 导入并行处理相关的模块


def _weights_scorer(scorer, estimator, X, y, sample_weight):
    # 如果有样本权重，使用样本权重计算评分
    if sample_weight is not None:
        return scorer(estimator, X, y, sample_weight=sample_weight)
    # 否则，直接计算评分
    return scorer(estimator, X, y)


def _calculate_permutation_scores(
    estimator,
    X,
    y,
    sample_weight,
    col_idx,
    random_state,
    n_repeats,
    scorer,
    max_samples,
):
    """Calculate score when `col_idx` is permuted."""
    random_state = check_random_state(random_state)

    # 在 X 的副本上操作，以确保在线程并行时的线程安全性。
    # 当 joblib 后端为 'loky' 或旧的 'multiprocessing' 时，如果 X 很大，它会自动由只读内存映射（memmap）支持。
    # 而 X.copy() 则始终保证返回一个可写的数据结构，其列可以就地洗牌。
    if max_samples < X.shape[0]:
        # 生成样本索引
        row_indices = _generate_indices(
            random_state=random_state,
            bootstrap=False,
            n_population=X.shape[0],
            n_samples=max_samples,
        )
        # 按索引获取对应的行
        X_permuted = _safe_indexing(X, row_indices, axis=0)
        y = _safe_indexing(y, row_indices, axis=0)
        if sample_weight is not None:
            sample_weight = _safe_indexing(sample_weight, row_indices, axis=0)
    else:
        # 如果不需要采样，直接对整个 X 进行操作
        X_permuted = X.copy()

    scores = []
    shuffling_idx = np.arange(X_permuted.shape[0])
    # 重复 n_repeats 次
    for _ in range(n_repeats):
        random_state.shuffle(shuffling_idx)
        if hasattr(X_permuted, "iloc"):
            # 处理 DataFrame 的情况，直接在列上进行洗牌
            col = X_permuted.iloc[shuffling_idx, col_idx]
            col.index = X_permuted.index
            X_permuted[X_permuted.columns[col_idx]] = col
        else:
            # 处理普通数组或矩阵的情况，直接在指定列上进行洗牌
            X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx]
        # 计算每次洗牌后的评分
        scores.append(_weights_scorer(scorer, estimator, X_permuted, y, sample_weight))

    # 聚合多次评分结果
    if isinstance(scores[0], dict):
        scores = _aggregate_score_dicts(scores)
    else:
        scores = np.array(scores)

    return scores


def _create_importances_bunch(baseline_score, permuted_score):
    """Compute the importances as the decrease in score.

    Parameters
    ----------
    baseline_score : ndarrayURL of shape (n_features,)
        The baseline score without permutation.
    permuted_score : ndarrayURL of shape (n_features, n_repeats)
        The permuted scores for the `n` repetitions.

    Returns
    -------
    # 计算特征重要性得分，通过原始基线分数减去置换后的分数得出
    importances = baseline_score - permuted_score
    # 返回一个 Bunch 对象，包含计算得出的特征重要性的统计信息
    return Bunch(
        # 计算每个特征的重要性均值，沿着第二维度（即每列）的平均值
        importances_mean=np.mean(importances, axis=1),
        # 计算每个特征的重要性标准差，沿着第二维度（即每列）的标准差
        importances_std=np.std(importances, axis=1),
        # 原始的特征重要性得分矩阵，维度为 (特征数, 重复次数)
        importances=importances,
    )
@validate_params(
    {
        "estimator": [HasMethods(["fit"])],  # 参数验证：确保estimator有fit方法
        "X": ["array-like"],  # 参数验证：X应该是类数组形式的输入数据
        "y": ["array-like", None],  # 参数验证：y应该是类数组形式的目标数据，或者可以为None
        "scoring": [
            StrOptions(set(get_scorer_names())),  # 参数验证：scoring可以是预定义评分器的字符串选项
            callable,  # 参数验证：或者可以是一个可调用对象
            list,  # 参数验证：或者可以是列表
            tuple,  # 参数验证：或者可以是元组
            dict,  # 参数验证：或者可以是字典
            None,  # 参数验证：或者可以为None
        ],
        "n_repeats": [Interval(Integral, 1, None, closed="left")],  # 参数验证：n_repeats应该是大于等于1的整数
        "n_jobs": [Integral, None],  # 参数验证：n_jobs应该是整数或者为None
        "random_state": ["random_state"],  # 参数验证：random_state应该是随机状态对象
        "sample_weight": ["array-like", None],  # 参数验证：sample_weight应该是类数组形式的权重数据，或者可以为None
        "max_samples": [
            Interval(Integral, 1, None, closed="left"),  # 参数验证：max_samples应该是大于等于1的整数
            Interval(RealNotInt, 0, 1, closed="right"),  # 或者是0到1之间的实数，包含0但不包含1
        ],
    },
    prefer_skip_nested_validation=True,  # 参数验证：优先跳过嵌套验证
)
def permutation_importance(
    estimator,
    X,
    y,
    *,
    scoring=None,  # 特定参数：scoring是用于评估的指标，可以是字符串、可调用对象、列表、元组、字典或None
    n_repeats=5,  # 特定参数：n_repeats是进行特征置换的次数，默认为5
    n_jobs=None,  # 特定参数：n_jobs是并行作业的数量，可以是整数或None
    random_state=None,  # 特定参数：random_state是随机数种子
    sample_weight=None,  # 特定参数：sample_weight是样本权重，可以是类数组形式的数据或None
    max_samples=1.0,  # 特定参数：max_samples是最大样本数或比例，默认为1.0
):
    """Permutation importance for feature evaluation [BRE]_.

    The :term:`estimator` is required to be a fitted estimator. `X` can be the
    data set used to train the estimator or a hold-out set. The permutation
    importance of a feature is calculated as follows. First, a baseline metric,
    defined by :term:`scoring`, is evaluated on a (potentially different)
    dataset defined by the `X`. Next, a feature column from the validation set
    is permuted and the metric is evaluated again. The permutation importance
    is defined to be the difference between the baseline metric and metric from
    permutating the feature column.

    Read more in the :ref:`User Guide <permutation_importance>`.

    Parameters
    ----------
    estimator : object
        An estimator that has already been :term:`fitted` and is compatible
        with :term:`scorer`.

    X : ndarray or DataFrame, shape (n_samples, n_features)
        Data on which permutation importance will be computed.

    y : array-like or None, shape (n_samples, ) or (n_samples, n_classes)
        Targets for supervised or `None` for unsupervised.

    scoring : str, callable, list, tuple, or dict, default=None
        Scorer to use.
        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        Passing multiple scores to `scoring` is more efficient than calling
        `permutation_importance` for each of the scores as it reuses
        predictions to avoid redundant computation.

        If None, the estimator's default scorer is used.

    n_repeats : int, default=5
        Number of times to permute a feature.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel. The computation is done by computing
        permutation score for each columns and parallelized over the columns.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        并行运行的作业数。对每列计算排列分数并在列之间并行化计算。
        `None` 表示除非在 :obj:`joblib.parallel_backend` 上下文中，否则为 1。
        `-1` 表示使用所有处理器。更多细节见 :term:`术语表 <n_jobs>`。

    random_state : int, RandomState instance, default=None
        Pseudo-random number generator to control the permutations of each
        feature.
        Pass an int to get reproducible results across function calls.
        See :term:`Glossary <random_state>`.
        伪随机数生成器，用于控制每个特征的排列。
        传递一个整数以在函数调用之间获得可重现的结果。
        参见 :term:`术语表 <random_state>`。

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights used in scoring.
        在评分中使用的样本权重。
        
        .. versionadded:: 0.24

    max_samples : int or float, default=1.0
        The number of samples to draw from X to compute feature importance
        in each repeat (without replacement).
        从 X 中抽取的样本数量，用于在每次重复中计算特征重要性（不替换）。
        
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.
        - If `max_samples` is equal to `1.0` or `X.shape[0]`, all samples
          will be used.
        
        当使用此选项时可能提供较少准确的重要性估计，但在评估大型数据集上的特征重要性时保持方法的可处理性。
        结合 `n_repeats` 使用，允许控制此方法的计算速度与统计精度的折衷。
        
        .. versionadded:: 1.0

    Returns
    -------
    result : :class:`~sklearn.utils.Bunch` or dict of such instances
        Dictionary-like object, with the following attributes.
        类似字典的对象，具有以下属性。

        importances_mean : ndarray of shape (n_features, )
            Mean of feature importance over `n_repeats`.
            在 `n_repeats` 上特征重要性的平均值。
        
        importances_std : ndarray of shape (n_features, )
            Standard deviation over `n_repeats`.
            在 `n_repeats` 上的标准偏差。
        
        importances : ndarray of shape (n_features, n_repeats)
            Raw permutation importance scores.
            原始排列重要性分数。
        
        If there are multiple scoring metrics in the scoring parameter
        `result` is a dict with scorer names as keys (e.g. 'roc_auc') and
        `Bunch` objects like above as values.
        如果在评分参数中有多个评分指标，则 `result` 是一个以评分器名称为键（如 'roc_auc'），
        以类似上述的 `Bunch` 对象为值的字典。

    References
    ----------
    .. [BRE] :doi:`L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
             2001. <10.1023/A:1010933404324>`
    # 检查 X 是否具有 iloc 属性，如果没有则转换为 NumPy 数组
    if not hasattr(X, "iloc"):
        X = check_array(X, force_all_finite="allow-nan", dtype=None)

    # 根据给定的随机状态检查或创建随机数生成器的实例
    random_state = check_random_state(random_state)
    # 生成一个随机种子，确保每次并行调用 _calculate_permutation_scores 时使用独立的 RandomState 实例
    random_seed = random_state.randint(np.iinfo(np.int32).max + 1)

    # 如果 max_samples 不是整数，则根据比例设置为样本数量的整数值
    if not isinstance(max_samples, numbers.Integral):
        max_samples = int(max_samples * X.shape[0])
    # 如果 max_samples 大于样本数量，则抛出错误
    elif max_samples > X.shape[0]:
        raise ValueError("max_samples must be <= n_samples")

    # 检查评分器，并根据需要初始化评分函数
    scorer = check_scoring(estimator, scoring=scoring)
    # 计算基准得分，考虑样本权重
    baseline_score = _weights_scorer(scorer, estimator, X, y, sample_weight)

    # 使用并行计算方式计算每列特征的排列重要性得分
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_calculate_permutation_scores)(
            estimator,
            X,
            y,
            sample_weight,
            col_idx,
            random_seed,
            n_repeats,
            scorer,
            max_samples,
        )
        for col_idx in range(X.shape[1])
    )

    # 如果基准得分是字典类型，则返回每个评分指标的重要性字典
    if isinstance(baseline_score, dict):
        return {
            name: _create_importances_bunch(
                baseline_score[name],
                # 解包排列后的得分
                np.array([scores[col_idx][name] for col_idx in range(X.shape[1])]),
            )
            for name in baseline_score
        }
    # 否则，返回基准得分和排列得分的重要性 bunch
    else:
        return _create_importances_bunch(baseline_score, np.array(scores))
```