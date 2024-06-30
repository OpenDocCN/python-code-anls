# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\_label.py`

```
# 导入所需的模块和库
import array
import itertools
import warnings
from collections import defaultdict
from numbers import Integral

# 导入第三方库
import numpy as np
import scipy.sparse as sp

# 导入Scikit-Learn相关模块和函数
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import column_or_1d
from ..utils._array_api import _setdiff1d, device, get_namespace
from ..utils._encode import _encode, _unique
from ..utils._param_validation import Interval, validate_params
from ..utils.multiclass import type_of_target, unique_labels
from ..utils.sparsefuncs import min_max_axis
from ..utils.validation import _num_samples, check_array, check_is_fitted

# 定义公开的模块接口
__all__ = [
    "label_binarize",
    "LabelBinarizer",
    "LabelEncoder",
    "MultiLabelBinarizer",
]

# 定义LabelEncoder类，继承TransformerMixin和BaseEstimator
class LabelEncoder(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):
    """Encode target labels with value between 0 and n_classes-1.

    This transformer should be used to encode target values, *i.e.* `y`, and
    not the input `X`.

    Read more in the :ref:`User Guide <preprocessing_targets>`.

    .. versionadded:: 0.12

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Holds the label for each class.

    See Also
    --------
    OrdinalEncoder : Encode categorical features using an ordinal encoding
        scheme.
    OneHotEncoder : Encode categorical features as a one-hot numeric array.

    Examples
    --------
    `LabelEncoder` can be used to normalize labels.

    >>> from sklearn.preprocessing import LabelEncoder
    >>> le = LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6])
    array([0, 0, 1, 2]...)
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])

    It can also be used to transform non-numerical labels (as long as they are
    hashable and comparable) to numerical labels.

    >>> le = LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"])
    array([2, 2, 1]...)
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']

    """

    def fit(self, y):
        """Fit label encoder.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
            Fitted label encoder.
        """
        # 将y转换为单列数组（如果需要，发出警告）
        y = column_or_1d(y, warn=True)
        # 从y中提取唯一值，作为类别标签
        self.classes_ = _unique(y)
        # 返回已拟合的LabelEncoder实例
        return self
    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Encoded labels.
        """
        # 根据输入的目标值 y，将其转换为一维数组或列向量，并发出警告
        y = column_or_1d(y, warn=True)
        # 使用 _unique 函数获取唯一的类别，并将 y 转换为这些类别的索引
        self.classes_, y = _unique(y, return_inverse=True)
        # 返回编码后的标签数组 y
        return y

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Labels as normalized encodings.
        """
        # 检查当前对象是否已经拟合，即是否已经训练过
        check_is_fitted(self)
        # 获取命名空间和可能的设备类型
        xp, _ = get_namespace(y)
        # 将输入的目标值 y 转换为一维数组，并指定数据类型为已知类别的数据类型，发出警告
        y = column_or_1d(y, dtype=self.classes_.dtype, warn=True)
        # 如果输入的 y 为空数组，则返回一个空数组
        if _num_samples(y) == 0:
            return xp.asarray([])

        # 调用 _encode 函数，将 y 编码为与已知类别对应的编码
        return _encode(y, uniques=self.classes_)

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Original encoding.
        """
        # 检查当前对象是否已经拟合，即是否已经训练过
        check_is_fitted(self)
        # 获取命名空间和可能的设备类型
        xp, _ = get_namespace(y)
        # 将输入的目标值 y 转换为一维数组，并发出警告
        y = column_or_1d(y, warn=True)
        # 如果输入的 y 为空数组，则返回一个空数组
        if _num_samples(y) == 0:
            return xp.asarray([])

        # 使用 _setdiff1d 函数找出 y 中未曾见过的标签
        diff = _setdiff1d(
            ar1=y,
            ar2=xp.arange(self.classes_.shape[0], device=device(y)),
            xp=xp,
        )
        # 如果有未曾见过的标签，抛出 ValueError 异常
        if diff.shape[0]:
            raise ValueError("y contains previously unseen labels: %s" % str(diff))
        
        # 将 y 转换为数组，并按照 y 中的索引从 self.classes_ 中取出对应的原始编码
        y = xp.asarray(y)
        return xp.take(self.classes_, y, axis=0)

    def _more_tags(self):
        return {"X_types": ["1dlabels"], "array_api_support": True}
class LabelBinarizer(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):
    """Binarize labels in a one-vs-all fashion.

    Several regression and binary classification algorithms are
    available in scikit-learn. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    At learning time, this simply consists in learning one regressor
    or binary classifier per class. In doing so, one needs to convert
    multi-class labels to binary labels (belong or does not belong
    to the class). `LabelBinarizer` makes this process easy with the
    transform method.

    At prediction time, one assigns the class for which the corresponding
    model gave the greatest confidence. `LabelBinarizer` makes this easy
    with the :meth:`inverse_transform` method.

    Read more in the :ref:`User Guide <preprocessing_targets>`.

    Parameters
    ----------
    neg_label : int, default=0
        Value with which negative labels must be encoded.

    pos_label : int, default=1
        Value with which positive labels must be encoded.

    sparse_output : bool, default=False
        True if the returned array from transform is desired to be in sparse
        CSR format.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Holds the label for each class.

    y_type_ : str
        Represents the type of the target data as evaluated by
        :func:`~sklearn.utils.multiclass.type_of_target`. Possible types are
        'continuous', 'continuous-multioutput', 'binary', 'multiclass',
        'multiclass-multioutput', 'multilabel-indicator', and 'unknown'.

    sparse_input_ : bool
        `True` if the input data to transform is given as a sparse matrix,
         `False` otherwise.

    See Also
    --------
    label_binarize : Function to perform the transform operation of
        LabelBinarizer with fixed classes.
    OneHotEncoder : Encode categorical features using a one-hot aka one-of-K
        scheme.

    Examples
    --------
    >>> from sklearn.preprocessing import LabelBinarizer
    >>> lb = LabelBinarizer()
    >>> lb.fit([1, 2, 6, 4, 2])
    LabelBinarizer()
    >>> lb.classes_
    array([1, 2, 4, 6])
    >>> lb.transform([1, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    Binary targets transform to a column vector

    >>> lb = LabelBinarizer()
    >>> lb.fit_transform(['yes', 'no', 'no', 'yes'])
    array([[1],
           [0],
           [0],
           [1]])

    Passing a 2D matrix for multilabel classification

    >>> import numpy as np
    >>> lb.fit(np.array([[0, 1, 1], [1, 0, 0]]))
    LabelBinarizer()
    >>> lb.classes_
    array([0, 1, 2])
    >>> lb.transform([0, 1, 2, 1])
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [0, 1, 0]])
    """

    _parameter_constraints: dict = {
        "neg_label": [Integral],  # 确定 neg_label 参数类型为整数
        "pos_label": [Integral],  # 确定 pos_label 参数类型为整数
        "sparse_output": ["boolean"],  # 确定 sparse_output 参数类型为布尔值

        "neg_label": [Integral],  # Specifies the type constraint for the neg_label parameter as an integer
        "pos_label": [Integral],  # Specifies the type constraint for the pos_label parameter as an integer
        "sparse_output": ["boolean"],  # Specifies the type constraint for the sparse_output parameter as a boolean
    }

    # 初始化函数，设置负标签、正标签和稀疏输出的选项
    def __init__(self, *, neg_label=0, pos_label=1, sparse_output=False):
        self.neg_label = neg_label  # 设置负标签值
        self.pos_label = pos_label  # 设置正标签值
        self.sparse_output = sparse_output  # 设置稀疏输出选项

    # 使用装饰器定义的_fit_context方法，用于拟合(label binarizer)
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y):
        """Fit label binarizer.

        Parameters
        ----------
        y : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 检查负标签是否大于等于正标签，如果是，则抛出值错误异常
        if self.neg_label >= self.pos_label:
            raise ValueError(
                f"neg_label={self.neg_label} must be strictly less than "
                f"pos_label={self.pos_label}."
            )

        # 如果稀疏输出为真且正标签为0或负标签不为0，则抛出值错误异常
        if self.sparse_output and (self.pos_label == 0 or self.neg_label != 0):
            raise ValueError(
                "Sparse binarization is only supported with non "
                "zero pos_label and zero neg_label, got "
                f"pos_label={self.pos_label} and neg_label={self.neg_label}"
            )

        # 获取y的类型
        self.y_type_ = type_of_target(y, input_name="y")

        # 如果目标数据类型包含"multioutput"，则抛出值错误异常
        if "multioutput" in self.y_type_:
            raise ValueError(
                "Multioutput target data is not supported with label binarization"
            )

        # 如果y的样本数为0，则抛出值错误异常
        if _num_samples(y) == 0:
            raise ValueError("y has 0 samples: %r" % y)

        # 判断y是否为稀疏矩阵
        self.sparse_input_ = sp.issparse(y)
        # 获取类别标签
        self.classes_ = unique_labels(y)
        # 返回实例本身
        return self

    # 拟合并转换函数，用于将多类标签转换为二进制标签
    def fit_transform(self, y):
        """Fit label binarizer/transform multi-class labels to binary labels.

        The output of transform is sometimes referred to as
        the 1-of-K coding scheme.

        Parameters
        ----------
        y : {ndarray, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_classes)
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification. Sparse matrix can be
            CSR, CSC, COO, DOK, or LIL.

        Returns
        -------
        Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            Shape will be (n_samples, 1) for binary problems. Sparse matrix
            will be of CSR format.
        """
        # 调用fit方法拟合标签二值化器，并返回其转换后的结果
        return self.fit(y).transform(y)
    def transform(self, y):
        """
        Transform multi-class labels to binary labels.

        The output of transform is sometimes referred to by some authors as
        the 1-of-K coding scheme.

        Parameters
        ----------
        y : {array, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_classes)
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification. Sparse matrix can be
            CSR, CSC, COO, DOK, or LIL.

        Returns
        -------
        Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            Shape will be (n_samples, 1) for binary problems. Sparse matrix
            will be of CSR format.
        """
        # 确保模型已拟合
        check_is_fitted(self)

        # 检查 y 是否为多标签类型
        y_is_multilabel = type_of_target(y).startswith("multilabel")
        # 如果 y 是多标签类型但模型不支持多标签，则引发错误
        if y_is_multilabel and not self.y_type_.startswith("multilabel"):
            raise ValueError("The object was not fitted with multilabel input.")

        # 调用 label_binarize 函数进行标签二值化转换
        return label_binarize(
            y,
            classes=self.classes_,
            pos_label=self.pos_label,
            neg_label=self.neg_label,
            sparse_output=self.sparse_output,
        )

    def inverse_transform(self, Y, threshold=None):
        """
        Transform binary labels back to multi-class labels.

        Parameters
        ----------
        Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            Target values. All sparse matrices are converted to CSR before
            inverse transformation.

        threshold : float, default=None
            Threshold used in the binary and multi-label cases.

            Use 0 when ``Y`` contains the output of :term:`decision_function`
            (classifier).
            Use 0.5 when ``Y`` contains the output of :term:`predict_proba`.

            If None, the threshold is assumed to be half way between
            neg_label and pos_label.

        Returns
        -------
        y : {ndarray, sparse matrix} of shape (n_samples,)
            Target values. Sparse matrix will be of CSR format.

        Notes
        -----
        In the case when the binary labels are fractional
        (probabilistic), :meth:`inverse_transform` chooses the class with the
        greatest value. Typically, this allows to use the output of a
        linear model's :term:`decision_function` method directly as the input
        of :meth:`inverse_transform`.
        """
        # 确保模型已拟合
        check_is_fitted(self)

        # 如果未提供阈值，则默认为 pos_label 和 neg_label 的中间值
        if threshold is None:
            threshold = (self.pos_label + self.neg_label) / 2.0

        # 根据模型类型调用相应的反转换函数
        if self.y_type_ == "multiclass":
            y_inv = _inverse_binarize_multiclass(Y, self.classes_)
        else:
            y_inv = _inverse_binarize_thresholding(
                Y, self.y_type_, self.classes_, threshold
            )

        # 如果输入是稀疏矩阵，则转换成 CSR 格式
        if self.sparse_input_:
            y_inv = sp.csr_matrix(y_inv)
        elif sp.issparse(y_inv):
            y_inv = y_inv.toarray()

        # 返回反转换后的目标值
        return y_inv
    # 定义一个方法 `_more_tags`，返回一个字典对象
    def _more_tags(self):
        # 返回包含键为 "X_types"，值为包含字符串 "1dlabels" 的列表的字典
        return {"X_types": ["1dlabels"]}
# 使用 @validate_params 装饰器验证参数，确保参数的类型和取值符合指定的规范
@validate_params(
    {
        "y": ["array-like", "sparse matrix"],  # 参数 y 应为 array-like 或稀疏矩阵类型
        "classes": ["array-like"],  # 参数 classes 应为 array-like 类型
        "neg_label": [Interval(Integral, None, None, closed="neither")],  # 参数 neg_label 应为整数类型
        "pos_label": [Interval(Integral, None, None, closed="neither")],  # 参数 pos_label 应为整数类型
        "sparse_output": ["boolean"],  # 参数 sparse_output 应为布尔类型
    },
    prefer_skip_nested_validation=True,  # 设置优先跳过嵌套验证
)
# 定义函数 label_binarize，用于按一对多的方式对标签进行二元化
def label_binarize(y, *, classes, neg_label=0, pos_label=1, sparse_output=False):
    """Binarize labels in a one-vs-all fashion.

    Several regression and binary classification algorithms are
    available in scikit-learn. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    This function makes it possible to compute this transformation for a
    fixed set of class labels known ahead of time.

    Parameters
    ----------
    y : array-like or sparse matrix
        Sequence of integer labels or multilabel data to encode.

    classes : array-like of shape (n_classes,)
        Uniquely holds the label for each class.

    neg_label : int, default=0
        Value with which negative labels must be encoded.

    pos_label : int, default=1
        Value with which positive labels must be encoded.

    sparse_output : bool, default=False,
        Set to true if output binary array is desired in CSR sparse format.

    Returns
    -------
    Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
        Shape will be (n_samples, 1) for binary problems. Sparse matrix will
        be of CSR format.

    See Also
    --------
    LabelBinarizer : Class used to wrap the functionality of label_binarize and
        allow for fitting to classes independently of the transform operation.

    Examples
    --------
    >>> from sklearn.preprocessing import label_binarize
    >>> label_binarize([1, 6], classes=[1, 2, 4, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    The class ordering is preserved:

    >>> label_binarize([1, 6], classes=[1, 6, 4, 2])
    array([[1, 0, 0, 0],
           [0, 1, 0, 0]])

    Binary targets transform to a column vector

    >>> label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes'])
    array([[1],
           [0],
           [0],
           [1]])
    """
    # 如果 y 不是 list 类型，则使用 check_array 函数进行处理
    if not isinstance(y, list):
        # XXX Workaround that will be removed when list of list format is
        # dropped
        y = check_array(
            y, input_name="y", accept_sparse="csr", ensure_2d=False, dtype=None
        )
    else:
        # 如果 y 是空列表，则抛出 ValueError 异常
        if _num_samples(y) == 0:
            raise ValueError("y has 0 samples: %r" % y)
    # 如果 neg_label 大于等于 pos_label，则抛出 ValueError 异常
    if neg_label >= pos_label:
        raise ValueError(
            "neg_label={0} must be strictly less than pos_label={1}.".format(
                neg_label, pos_label
            )
        )
    # 如果 sparse_output 为真且 (pos_label == 0 或者 neg_label != 0)，则引发值错误异常
    if sparse_output and (pos_label == 0 or neg_label != 0):
        raise ValueError(
            "Sparse binarization is only supported with non "
            "zero pos_label and zero neg_label, got "
            "pos_label={0} and neg_label={1}"
            "".format(pos_label, neg_label)
        )

    # 为了处理在密集情况下 pos_label == 0 的情况
    pos_switch = pos_label == 0
    # 如果 pos_switch 为真，则将 pos_label 赋值为 -neg_label
    if pos_switch:
        pos_label = -neg_label

    # 确定 y 的类型
    y_type = type_of_target(y)
    # 如果 y 的类型包含 "multioutput"，则抛出值错误异常，不支持标签二值化
    if "multioutput" in y_type:
        raise ValueError(
            "Multioutput target data is not supported with label binarization"
        )
    # 如果 y 的类型为 "unknown"，则抛出值错误异常，目标数据类型未知
    if y_type == "unknown":
        raise ValueError("The type of target data is not known")

    # 计算样本数 n_samples，根据 y 是否为稀疏矩阵来决定
    n_samples = y.shape[0] if sp.issparse(y) else len(y)
    # 类别数 n_classes 为 classes 的长度
    n_classes = len(classes)
    # 将 classes 转换为 numpy 数组
    classes = np.asarray(classes)

    # 如果 y 的类型为 "binary"
    if y_type == "binary":
        # 如果类别数 n_classes 为 1
        if n_classes == 1:
            # 如果 sparse_output 为真，返回一个 (n_samples, 1) 的稀疏矩阵
            if sparse_output:
                return sp.csr_matrix((n_samples, 1), dtype=int)
            # 否则创建一个全为 0 的 numpy 数组 Y，并赋值为 neg_label，然后返回 Y
            else:
                Y = np.zeros((len(y), 1), dtype=int)
                Y += neg_label
                return Y
        # 如果类别数大于等于 3，则将 y_type 设为 "multiclass"
        elif len(classes) >= 3:
            y_type = "multiclass"

    # 对类别进行排序得到 sorted_class
    sorted_class = np.sort(classes)
    
    # 如果 y 的类型为 "multilabel-indicator"
    if y_type == "multilabel-indicator":
        # 计算 y 的类别数 y_n_classes，如果 y 是稀疏矩阵，使用 y.shape[1]，否则使用 len(y[0])
        y_n_classes = y.shape[1] if hasattr(y, "shape") else len(y[0])
        # 如果 classes 的大小与 y_n_classes 不匹配，引发值错误异常
        if classes.size != y_n_classes:
            raise ValueError(
                "classes {0} mismatch with the labels {1} found in the data".format(
                    classes, unique_labels(y)
                )
            )

    # 如果 y 的类型为 "binary" 或者 "multiclass"
    if y_type in ("binary", "multiclass"):
        # 将 y 转换为列向量或者一维数组
        y = column_or_1d(y)

        # 从 y 中挑选出已知的标签
        y_in_classes = np.isin(y, classes)
        y_seen = y[y_in_classes]
        # 在 sorted_class 中搜索 y_seen 的索引
        indices = np.searchsorted(sorted_class, y_seen)
        # 计算 indptr，将索引转换为偏移数组
        indptr = np.hstack((0, np.cumsum(y_in_classes)))

        # 创建一个和 indices 大小相同的数据数组 data，填充为 pos_label
        data = np.empty_like(indices)
        data.fill(pos_label)
        # 创建一个稀疏矩阵 Y，形状为 (n_samples, n_classes)
        Y = sp.csr_matrix((data, indices, indptr), shape=(n_samples, n_classes))
    # 如果 y 的类型为 "multilabel-indicator"
    elif y_type == "multilabel-indicator":
        # 将 y 转换为稀疏矩阵 Y
        Y = sp.csr_matrix(y)
        # 如果 pos_label 不等于 1，则创建一个和 Y.data 大小相同的数据数组 data，填充为 pos_label
        if pos_label != 1:
            data = np.empty_like(Y.data)
            data.fill(pos_label)
            Y.data = data
    else:
        # 否则，引发值错误异常，不支持该类型的目标数据与标签二值化
        raise ValueError(
            "%s target data is not supported with label binarization" % y_type
        )

    # 如果不是稀疏输出
    if not sparse_output:
        # 将稀疏矩阵 Y 转换为密集数组
        Y = Y.toarray()
        # 将 Y 转换为整型数组
        Y = Y.astype(int, copy=False)

        # 如果 neg_label 不等于 0，则将 Y 中为 0 的元素赋值为 neg_label
        if neg_label != 0:
            Y[Y == 0] = neg_label

        # 如果 pos_switch 为真，则将 Y 中为 pos_label 的元素赋值为 0
        if pos_switch:
            Y[Y == pos_label] = 0
    else:
        # 否则，将 Y.data 转换为整型数组
        Y.data = Y.data.astype(int, copy=False)

    # 保持标签的顺序
    if np.any(classes != sorted_class):
        # 在 sorted_class 中搜索 classes 的索引
        indices = np.searchsorted(sorted_class, classes)
        # 按照索引重新排列 Y 的列
        Y = Y[:, indices]

    # 如果 y 的类型为 "binary"
    if y_type == "binary":
        # 如果 sparse_output 为真，提取 Y 的最后一列
        if sparse_output:
            Y = Y.getcol(-1)
        # 否则，提取 Y 的倒数第二维为 1 的列，并将其形状改为 (-1, 1)
        else:
            Y = Y[:, -1].reshape((-1, 1))

    # 返回处理后的 Y
    return Y
# 逆转标签二进制化的多类别转换函数
def _inverse_binarize_multiclass(y, classes):
    """Inverse label binarization transformation for multiclass.

    Multiclass uses the maximal score instead of a threshold.
    """
    # 将类别转换为NumPy数组
    classes = np.asarray(classes)

    # 如果y是稀疏矩阵
    if sp.issparse(y):
        # 将y转换为CSR格式的稀疏矩阵
        y = y.tocsr()
        # 获取矩阵的行数和列数
        n_samples, n_outputs = y.shape
        # 创建输出索引数组
        outputs = np.arange(n_outputs)
        # 计算每行的最大值和最小值
        row_max = min_max_axis(y, 1)[1]
        # 计算每行的非零元素个数
        row_nnz = np.diff(y.indptr)

        # 将每行的最大值重复以便与数据比较
        y_data_repeated_max = np.repeat(row_max, row_nnz)
        # 找出每行中取得最大值的所有索引
        y_i_all_argmax = np.flatnonzero(y_data_repeated_max == y.data)

        # 处理最后一行最大值为0的特殊情况
        if row_max[-1] == 0:
            y_i_all_argmax = np.append(y_i_all_argmax, [len(y.data)])

        # 获取每行第一个最大值的索引
        index_first_argmax = np.searchsorted(y_i_all_argmax, y.indptr[:-1])
        # 获取每行的第一个最大值的索引
        y_ind_ext = np.append(y.indices, [0])
        y_i_argmax = y_ind_ext[y_i_all_argmax[index_first_argmax]]
        # 处理所有元素为0的行
        y_i_argmax[np.where(row_nnz == 0)[0]] = 0

        # 处理包含负数的最大值为0的行
        samples = np.arange(n_samples)[(row_nnz > 0) & (row_max.ravel() == 0)]
        for i in samples:
            ind = y.indices[y.indptr[i] : y.indptr[i + 1]]
            y_i_argmax[i] = classes[np.setdiff1d(outputs, ind)][0]

        return classes[y_i_argmax]
    else:
        # 返回取每行最大值的索引对应的类别
        return classes.take(y.argmax(axis=1), mode="clip")


def _inverse_binarize_thresholding(y, output_type, classes, threshold):
    """Inverse label binarization transformation using thresholding."""

    # 如果输出类型为二进制且y为二维且列数大于2，则引发值错误异常
    if output_type == "binary" and y.ndim == 2 and y.shape[1] > 2:
        raise ValueError("output_type='binary', but y.shape = {0}".format(y.shape))

    # 如果输出类型不为二进制且y的列数不等于类别数，则引发值错误异常
    if output_type != "binary" and y.shape[1] != len(classes):
        raise ValueError(
            "The number of class is not equal to the number of dimension of y."
        )

    # 将类别转换为NumPy数组
    classes = np.asarray(classes)

    # 执行阈值处理
    if sp.issparse(y):
        if threshold > 0:
            # 如果阈值大于0且y的格式不是"csr"或"csc"，则将y转换为CSR格式
            if y.format not in ("csr", "csc"):
                y = y.tocsr()
            # 将y的数据转换为布尔数组，表示是否大于阈值
            y.data = np.array(y.data > threshold, dtype=int)
            # 消除零元素
            y.eliminate_zeros()
        else:
            # 否则将y转换为布尔数组，表示是否大于阈值
            y = np.array(y.toarray() > threshold, dtype=int)
    else:
        # 将y转换为布尔数组，表示是否大于阈值
        y = np.array(y > threshold, dtype=int)

    # 逆转换数据
    if output_type == "binary":
        # 如果输出类型为二进制
        if sp.issparse(y):
            # 如果y为稀疏矩阵，将其转换为密集数组
            y = y.toarray()
        # 如果y为二维且列数为2，则返回对应类别的值
        if y.ndim == 2 and y.shape[1] == 2:
            return classes[y[:, 1]]
        else:
            # 否则返回y对应的类别
            if len(classes) == 1:
                return np.repeat(classes[0], len(y))
            else:
                return classes[y.ravel()]

    elif output_type == "multilabel-indicator":
        # 如果输出类型为多标签指示器，则直接返回y
        return y
    else:
        # 如果输出类型不被支持，抛出值错误异常，提示不支持的格式
        raise ValueError("{0} format is not supported".format(output_type))
class MultiLabelBinarizer(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):
    """Transform between iterable of iterables and a multilabel format.

    Although a list of sets or tuples is a very intuitive format for multilabel
    data, it is unwieldy to process. This transformer converts between this
    intuitive format and the supported multilabel format: a (samples x classes)
    binary matrix indicating the presence of a class label.

    Parameters
    ----------
    classes : array-like of shape (n_classes,), default=None
        Indicates an ordering for the class labels.
        All entries should be unique (cannot contain duplicate classes).

    sparse_output : bool, default=False
        Set to True if output binary array is desired in CSR sparse format.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        A copy of the `classes` parameter when provided.
        Otherwise it corresponds to the sorted set of classes found
        when fitting.

    See Also
    --------
    OneHotEncoder : Encode categorical features using a one-hot aka one-of-K
        scheme.

    Examples
    --------
    >>> from sklearn.preprocessing import MultiLabelBinarizer
    >>> mlb = MultiLabelBinarizer()
    >>> mlb.fit_transform([(1, 2), (3,)])
    array([[1, 1, 0],
           [0, 0, 1]])
    >>> mlb.classes_
    array([1, 2, 3])

    >>> mlb.fit_transform([{'sci-fi', 'thriller'}, {'comedy'}])
    array([[0, 1, 1],
           [1, 0, 0]])
    >>> list(mlb.classes_)
    ['comedy', 'sci-fi', 'thriller']

    A common mistake is to pass in a list, which leads to the following issue:

    >>> mlb = MultiLabelBinarizer()
    >>> mlb.fit(['sci-fi', 'thriller', 'comedy'])
    MultiLabelBinarizer()
    >>> mlb.classes_
    array(['-', 'c', 'd', 'e', 'f', 'h', 'i', 'l', 'm', 'o', 'r', 's', 't',
        'y'], dtype=object)

    To correct this, the list of labels should be passed in as:

    >>> mlb = MultiLabelBinarizer()
    >>> mlb.fit([['sci-fi', 'thriller', 'comedy']])
    MultiLabelBinarizer()
    >>> mlb.classes_
    array(['comedy', 'sci-fi', 'thriller'], dtype=object)
    """

    _parameter_constraints: dict = {
        "classes": ["array-like", None],  # 参数约束：类别标签的数组形式，可以为 None
        "sparse_output": ["boolean"],     # 参数约束：稀疏输出的布尔值
    }

    def __init__(self, *, classes=None, sparse_output=False):
        self.classes = classes            # 初始化类别标签
        self.sparse_output = sparse_output  # 初始化稀疏输出标志

    @_fit_context(prefer_skip_nested_validation=True)
    # 定义一个方法 fit，用于训练多标签二值化器，并存储 classes_
    def fit(self, y):
        """Fit the label sets binarizer, storing :term:`classes_`.

        Parameters
        ----------
        y : iterable of iterables
            A set of labels (any orderable and hashable object) for each
            sample. If the `classes` parameter is set, `y` will not be
            iterated.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # 清空缓存字典
        self._cached_dict = None

        # 如果没有提供预定义的类别（classes），则从y中推断类别并排序
        if self.classes is None:
            classes = sorted(set(itertools.chain.from_iterable(y)))
        # 如果提供了classes参数，确保其不包含重复类别，否则引发 ValueError
        elif len(set(self.classes)) < len(self.classes):
            raise ValueError(
                "The classes argument contains duplicate "
                "classes. Remove these duplicates before passing "
                "them to MultiLabelBinarizer."
            )
        else:
            # 否则使用已经提供的类别
            classes = self.classes
        
        # 确定类别的数据类型，如果所有类别都是整数，则选择整数类型，否则选择对象类型
        dtype = int if all(isinstance(c, int) for c in classes) else object
        # 初始化 self.classes_ 为一个长度为类别数的空数组，并将类别数据填充进去
        self.classes_ = np.empty(len(classes), dtype=dtype)
        self.classes_[:] = classes
        # 返回已经拟合好的估计器
        return self

    # 应用装饰器 _fit_context，用于适应上下文并且忽略嵌套验证
    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, y):
        """Fit the label sets binarizer and transform the given label sets.

        Parameters
        ----------
        y : iterable of iterables
            A set of labels (any orderable and hashable object) for each
            sample. If the `classes` parameter is set, `y` will not be
            iterated.

        Returns
        -------
        y_indicator : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            A matrix such that `y_indicator[i, j] = 1` iff `classes_[j]`
            is in `y[i]`, and 0 otherwise. Sparse matrix will be of CSR
            format.
        """
        # 如果已经提供了类别信息，则直接调用 fit 方法拟合并转换标签集合 y
        if self.classes is not None:
            return self.fit(y).transform(y)

        # 清空缓存字典
        self._cached_dict = None

        # 使用 defaultdict(int) 自动增加新类别的映射
        class_mapping = defaultdict(int)
        class_mapping.default_factory = class_mapping.__len__
        # 对标签进行转换，使用 class_mapping 对象
        yt = self._transform(y, class_mapping)

        # 按照映射的值排序类别，并重新排列列
        tmp = sorted(class_mapping, key=class_mapping.get)

        # 确定类别的数据类型，如果所有类别都是整数，则选择整数类型，否则选择对象类型
        dtype = int if all(isinstance(c, int) for c in tmp) else object
        # 初始化类别映射为一个长度为 tmp 的数组，并将 tmp 数据填充进去
        class_mapping = np.empty(len(tmp), dtype=dtype)
        class_mapping[:] = tmp
        # 使用 np.unique 函数获取类别和其反向索引，保存在 self.classes_ 和 inverse 中
        self.classes_, inverse = np.unique(class_mapping, return_inverse=True)
        # 确保 yt.indices 保持其当前的数据类型
        yt.indices = np.asarray(inverse[yt.indices], dtype=yt.indices.dtype)

        # 如果不需要稀疏矩阵输出，则将 yt 转换为数组
        if not self.sparse_output:
            yt = yt.toarray()

        # 返回转换后的标签矩阵
        return yt
    # 定义一个方法，用于将给定的标签集合转换成稀疏矩阵表示
    def transform(self, y):
        """Transform the given label sets.

        Parameters
        ----------
        y : iterable of iterables
            A set of labels (any orderable and hashable object) for each
            sample. If the `classes` parameter is set, `y` will not be
            iterated.

        Returns
        -------
        y_indicator : array or CSR matrix, shape (n_samples, n_classes)
            A matrix such that `y_indicator[i, j] = 1` iff `classes_[j]` is in
            `y[i]`, and 0 otherwise.
        """
        # 检查模型是否已经拟合
        check_is_fitted(self)

        # 构建类别到索引的映射
        class_to_index = self._build_cache()
        
        # 使用映射将标签集合进行转换
        yt = self._transform(y, class_to_index)

        # 如果不需要稀疏输出，则转换为密集数组
        if not self.sparse_output:
            yt = yt.toarray()

        # 返回转换后的结果
        return yt

    def _build_cache(self):
        # 如果缓存为空，则构建类别到索引的字典并缓存
        if self._cached_dict is None:
            self._cached_dict = dict(zip(self.classes_, range(len(self.classes_))))

        # 返回缓存的字典
        return self._cached_dict

    def _transform(self, y, class_mapping):
        """Transforms the label sets with a given mapping.

        Parameters
        ----------
        y : iterable of iterables
            A set of labels (any orderable and hashable object) for each
            sample. If the `classes` parameter is set, `y` will not be
            iterated.

        class_mapping : Mapping
            Maps from label to column index in label indicator matrix.

        Returns
        -------
        y_indicator : sparse matrix of shape (n_samples, n_classes)
            Label indicator matrix. Will be of CSR format.
        """
        # 初始化用于稀疏矩阵的索引和指针
        indices = array.array("i")
        indptr = array.array("i", [0])
        # 存储未知标签的集合
        unknown = set()
        
        # 遍历每个样本的标签集合
        for labels in y:
            index = set()
            # 对每个标签进行处理，将其映射到对应的列索引
            for label in labels:
                try:
                    index.add(class_mapping[label])
                except KeyError:
                    # 如果标签未知，则记录在未知集合中
                    unknown.add(label)
            # 扩展索引列表
            indices.extend(index)
            # 更新指针列表
            indptr.append(len(indices))
        
        # 如果存在未知标签，则发出警告
        if unknown:
            warnings.warn(
                "unknown class(es) {0} will be ignored".format(sorted(unknown, key=str))
            )
        
        # 创建稀疏矩阵的数据部分，所有数据点的值为1
        data = np.ones(len(indices), dtype=int)

        # 返回稀疏矩阵表示的标签指示器
        return sp.csr_matrix(
            (data, indices, indptr), shape=(len(indptr) - 1, len(class_mapping))
        )
    # 将指示矩阵转换为标签集合的方法

    # 检查模型是否已经拟合
    check_is_fitted(self)

    # 检查指示矩阵的列数是否与已知类别数相匹配
    if yt.shape[1] != len(self.classes_):
        raise ValueError(
            "Expected indicator for {0} classes, but got {1}".format(
                len(self.classes_), yt.shape[1]
            )
        )

    # 如果指示矩阵是稀疏矩阵，则进行处理
    if sp.issparse(yt):
        yt = yt.tocsr()  # 转换为压缩稀疏行格式
        # 如果数据中包含除了0和1之外的值，抛出异常
        if len(yt.data) != 0 and len(np.setdiff1d(yt.data, [0, 1])) > 0:
            raise ValueError("Expected only 0s and 1s in label indicator.")
        # 返回每个样本的标签集合，其中每个标签是对应类别的名称
        return [
            tuple(self.classes_.take(yt.indices[start:end]))
            for start, end in zip(yt.indptr[:-1], yt.indptr[1:])
        ]
    else:
        # 如果指示矩阵是密集矩阵，则进行处理
        unexpected = np.setdiff1d(yt, [0, 1])
        # 如果数据中包含除了0和1之外的值，抛出异常
        if len(unexpected) > 0:
            raise ValueError(
                "Expected only 0s and 1s in label indicator. Also got {0}".format(
                    unexpected
                )
            )
        # 返回每个样本的标签集合，其中每个标签是对应类别的名称
        return [tuple(self.classes_.compress(indicators)) for indicators in yt]

    # 返回额外的标签信息
    def _more_tags(self):
        return {"X_types": ["2dlabels"]}
```