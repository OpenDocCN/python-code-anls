# `D:\src\scipysrc\scikit-learn\sklearn\utils\class_weight.py`

```
# 导入NumPy库，用于处理数组
import numpy as np
# 导入SciPy库中的稀疏矩阵模块，用于稀疏矩阵的支持
from scipy import sparse
# 导入参数验证模块中的指定函数和类
from ._param_validation import StrOptions, validate_params

# 使用参数验证装饰器对compute_class_weight函数进行参数验证
@validate_params(
    {
        "class_weight": [dict, StrOptions({"balanced"}), None],  # class_weight参数接受字典、"balanced"字符串或None
        "classes": [np.ndarray],  # classes参数为NumPy数组，包含数据中出现的所有类别
        "y": ["array-like"],  # y参数为类数组形式，包含每个样本的原始类别标签
    },
    prefer_skip_nested_validation=True,  # 首选跳过嵌套验证
)
def compute_class_weight(class_weight, *, classes, y):
    """Estimate class weights for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, "balanced" or None
        If "balanced", class weights will be given by
        `n_samples / (n_classes * np.bincount(y))`.
        If a dictionary is given, keys are classes and values are corresponding class
        weights.
        If `None` is given, the class weights will be uniform.

    classes : ndarray
        Array of the classes occurring in the data, as given by
        `np.unique(y_org)` with `y_org` the original class labels.

    y : array-like of shape (n_samples,)
        Array of original class labels per sample.

    Returns
    -------
    class_weight_vect : ndarray of shape (n_classes,)
        Array with `class_weight_vect[i]` the weight for i-th class.

    References
    ----------
    The "balanced" heuristic is inspired by
    Logistic Regression in Rare Events Data, King, Zen, 2001.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.class_weight import compute_class_weight
    >>> y = [1, 1, 1, 1, 0, 0]
    >>> compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    array([1.5 , 0.75])
    """
    # 由于循环引用导致的导入错误，延迟导入LabelEncoder
    from ..preprocessing import LabelEncoder

    # 如果y中有超出classes范围的标签，抛出ValueError异常
    if set(y) - set(classes):
        raise ValueError("classes should include all valid labels that can be in y")
    # 如果class_weight为None或空字典，使用均匀的类别权重
    if class_weight is None or len(class_weight) == 0:
        weight = np.ones(classes.shape[0], dtype=np.float64, order="C")
    # 如果class_weight为"balanced"，根据y中各类别的出现频率估算类别权重
    elif class_weight == "balanced":
        # 使用LabelEncoder将y中的类别标签编码为整数
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        # 如果classes中包含不在y中的标签，抛出ValueError异常
        if not all(np.isin(classes, le.classes_)):
            raise ValueError("classes should have valid labels that are in y")

        # 计算每个类别的权重，基于y中每个类别的出现频率
        recip_freq = len(y) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
        weight = recip_freq[le.transform(classes)]
    else:
        # 如果使用者定义了权重字典
        # 创建一个与 classes 数组长度相同的浮点数数组，初始值为 1
        weight = np.ones(classes.shape[0], dtype=np.float64, order="C")
        # 存储未加权的类别列表
        unweighted_classes = []
        # 遍历 classes 数组的索引和值
        for i, c in enumerate(classes):
            # 如果当前类别 c 在 class_weight 字典中有定义
            if c in class_weight:
                # 使用 class_weight 中定义的权重替换默认的权重值
                weight[i] = class_weight[c]
            else:
                # 否则将该类别 c 添加到未加权类别列表
                unweighted_classes.append(c)

        # 计算有权重的类别数量
        n_weighted_classes = len(classes) - len(unweighted_classes)
        # 如果存在未加权的类别，并且加权类别数量不等于 class_weight 的长度
        if unweighted_classes and n_weighted_classes != len(class_weight):
            # 将未加权类别列表转换为用户友好的字符串格式
            unweighted_classes_user_friendly_str = np.array(unweighted_classes).tolist()
            # 抛出值错误，说明未加权的类别不在 class_weight 中
            raise ValueError(
                f"The classes, {unweighted_classes_user_friendly_str}, are not in"
                " class_weight"
            )

    # 返回计算好的权重数组
    return weight
# 基于给定参数进行验证，确保参数的类型和格式正确
@validate_params(
    {
        "class_weight": [dict, list, StrOptions({"balanced"}), None],
        "y": ["array-like", "sparse matrix"],
        "indices": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
# 估算用于不平衡数据集的样本权重
def compute_sample_weight(class_weight, y, *, indices=None):
    """Estimate sample weights by class for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, list of dicts, "balanced", or None
        Weights associated with classes in the form `{class_label: weight}`.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        `[{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}]` instead of
        `[{1:1}, {2:5}, {3:1}, {4:1}]`.

        The `"balanced"` mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data:
        `n_samples / (n_classes * np.bincount(y))`.

        For multi-output, the weights of each column of y will be multiplied.

    y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_outputs)
        Array of original class labels per sample.

    indices : array-like of shape (n_subsample,), default=None
        Array of indices to be used in a subsample. Can be of length less than
        `n_samples` in the case of a subsample, or equal to `n_samples` in the
        case of a bootstrap subsample with repeated indices. If `None`, the
        sample weight will be calculated over the full sample. Only `"balanced"`
        is supported for `class_weight` if this is provided.

    Returns
    -------
    sample_weight_vect : ndarray of shape (n_samples,)
        Array with sample weights as applied to the original `y`.

    Examples
    --------
    >>> from sklearn.utils.class_weight import compute_sample_weight
    >>> y = [1, 1, 1, 1, 0, 0]
    >>> compute_sample_weight(class_weight="balanced", y=y)
    array([0.75, 0.75, 0.75, 0.75, 1.5 , 1.5 ])
    """

    # 确保 y 是二维的。稀疏矩阵已经是二维的。
    if not sparse.issparse(y):
        # 如果 y 不是稀疏矩阵，则至少将其视为一维数组
        y = np.atleast_1d(y)
        # 如果 y 是一维的，则将其重塑为二维的
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
    # 确定 y 的输出列数
    n_outputs = y.shape[1]

    # 如果提供了 indices 且 class_weight 不是 "balanced"，则引发 ValueError 异常
    if indices is not None and class_weight != "balanced":
        raise ValueError(
            "The only valid class_weight for subsampling is 'balanced'. "
            f"Given {class_weight}."
        )
    # 如果输出的数量大于1
    elif n_outputs > 1:
        # 如果类别权重为None或者是字典类型，则抛出数值错误异常
        if class_weight is None or isinstance(class_weight, dict):
            raise ValueError(
                "For multi-output, class_weight should be a list of dicts, or the "
                "string 'balanced'."
            )
        # 如果类别权重是列表类型，并且其长度不等于输出数量，则抛出数值错误异常
        elif isinstance(class_weight, list) and len(class_weight) != n_outputs:
            raise ValueError(
                "For multi-output, number of elements in class_weight should match "
                f"number of outputs. Got {len(class_weight)} element(s) while having "
                f"{n_outputs} outputs."
            )

    # 初始化一个空列表，用于存储扩展后的类别权重
    expanded_class_weight = []
    # 遍历输出的数量
    for k in range(n_outputs):
        # 如果目标值y是稀疏矩阵
        if sparse.issparse(y):
            # 可以逐列地将稀疏矩阵变为密集数组
            y_full = y[:, [k]].toarray().flatten()
        else:
            y_full = y[:, k]
        # 获取目标值y中的唯一类别
        classes_full = np.unique(y_full)
        # 初始化缺失的类别为None
        classes_missing = None

        # 如果类别权重为"balanced"或者输出数量为1，则类别权重为整体的类别权重
        if class_weight == "balanced" or n_outputs == 1:
            class_weight_k = class_weight
        else:
            # 否则类别权重为当前输出k对应的类别权重
            class_weight_k = class_weight[k]

        # 如果有索引值
        if indices is not None:
            # 获取子样本的目标值y_subsample和其唯一类别
            y_subsample = y_full[indices]
            classes_subsample = np.unique(y_subsample)

            # 计算子样本的类别权重，涵盖了所有原始数据中存在的标签，可能存在的缺失类别
            weight_k = np.take(
                compute_class_weight(
                    class_weight_k, classes=classes_subsample, y=y_subsample
                ),
                np.searchsorted(classes_subsample, classes_full),
                mode="clip",
            )

            # 计算缺失的类别集合，即原始数据中存在但子样本中不存在的类别
            classes_missing = set(classes_full) - set(classes_subsample)
        else:
            # 否则直接计算整体数据的类别权重
            weight_k = compute_class_weight(
                class_weight_k, classes=classes_full, y=y_full
            )

        # 将类别权重按照目标值y_full的类别顺序进行排序
        weight_k = weight_k[np.searchsorted(classes_full, y_full)]

        # 如果存在缺失的类别
        if classes_missing:
            # 将缺失的类别权重设置为0
            weight_k[np.isin(y_full, list(classes_missing))] = 0.0

        # 将当前输出k的类别权重添加到扩展类别权重列表中
        expanded_class_weight.append(weight_k)

    # 对所有输出的类别权重进行乘积，得到最终的扩展类别权重，数据类型为浮点数
    expanded_class_weight = np.prod(expanded_class_weight, axis=0, dtype=np.float64)

    # 返回最终的扩展类别权重
    return expanded_class_weight
```