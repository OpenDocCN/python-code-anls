# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_multiclass.py`

```
from itertools import product  # 导入 itertools 模块中的 product 函数，用于计算可迭代对象的笛卡尔积

import numpy as np  # 导入 NumPy 库，并使用 np 别名
import pytest  # 导入 pytest 测试框架
from scipy.sparse import issparse  # 从 scipy.sparse 导入 issparse 函数，用于检查对象是否为稀疏矩阵

from sklearn import config_context, datasets  # 导入 sklearn 库中的 config_context 和 datasets 模块
from sklearn.model_selection import ShuffleSplit  # 从 sklearn.model_selection 导入 ShuffleSplit 类，用于交叉验证
from sklearn.svm import SVC  # 从 sklearn.svm 导入 SVC 类，支持向量机模型
from sklearn.utils._array_api import yield_namespace_device_dtype_combinations  # 从 sklearn.utils._array_api 导入 yield_namespace_device_dtype_combinations 函数
from sklearn.utils._testing import (  # 从 sklearn.utils._testing 导入多个函数和类
    _array_api_for_tests,  # 用于测试的 API
    _convert_container,  # 转换容器
    assert_allclose,  # 断言两个对象是否接近
    assert_array_almost_equal,  # 断言两个数组是否几乎相等
    assert_array_equal,  # 断言两个数组是否完全相等
)
from sklearn.utils.estimator_checks import _NotAnArray  # 从 sklearn.utils.estimator_checks 导入 _NotAnArray 类
from sklearn.utils.fixes import (  # 从 sklearn.utils.fixes 导入多个容器类型
    COO_CONTAINERS,  # COO 格式的容器
    CSC_CONTAINERS,  # CSC 格式的容器
    CSR_CONTAINERS,  # CSR 格式的容器
    DOK_CONTAINERS,  # DOK 格式的容器
    LIL_CONTAINERS,  # LIL 格式的容器
)
from sklearn.utils.metaestimators import _safe_split  # 从 sklearn.utils.metaestimators 导入 _safe_split 函数，安全地切分数据集
from sklearn.utils.multiclass import (  # 从 sklearn.utils.multiclass 导入多分类相关函数和类
    _ovr_decision_function,  # 一对多决策函数
    check_classification_targets,  # 检查分类目标
    class_distribution,  # 计算类分布
    is_multilabel,  # 检查是否为多标签
    type_of_target,  # 获取目标类型
    unique_labels,  # 获取唯一标签
)

multilabel_explicit_zero = np.array([[0, 1], [1, 0]])  # 创建一个 NumPy 数组，表示显式零的多标签矩阵
multilabel_explicit_zero[:, 0] = 0  # 将多标签矩阵的第一列设置为零

def _generate_sparse(  # 定义生成稀疏矩阵的函数 _generate_sparse
    data,  # 输入数据
    sparse_containers=tuple(  # 稀疏容器类型的元组，默认为多种格式的容器
        COO_CONTAINERS  # COO 格式的容器
        + CSC_CONTAINERS  # CSC 格式的容器
        + CSR_CONTAINERS  # CSR 格式的容器
        + DOK_CONTAINERS  # DOK 格式的容器
        + LIL_CONTAINERS  # LIL 格式的容器
    ),
    dtypes=(bool, int, np.int8, np.uint8, float, np.float32),  # 数据类型的元组，默认包含布尔值和数值类型
):
    return [  # 返回一个列表，包含不同类型和格式的稀疏矩阵
        sparse_container(data, dtype=dtype)  # 使用不同的稀疏容器和数据类型创建稀疏矩阵
        for sparse_container in sparse_containers  # 遍历稀疏容器类型
        for dtype in dtypes  # 遍历数据类型
    ]

EXAMPLES = {  # 创建一个示例字典
    "multilabel-indicator": [  # 多标签指示器示例的键值对列表
        # 当数据格式为稀疏或密集时有效，通过 CSR 格式进行测试标识
        *_generate_sparse(  # 展开 _generate_sparse 函数的返回值作为列表的一部分
            np.random.RandomState(42).randint(2, size=(10, 10)),  # 生成随机整数矩阵
            sparse_containers=CSR_CONTAINERS,  # 使用 CSR 格式的稀疏容器
            dtypes=(int,),  # 仅使用整数数据类型
        ),
        [[0, 1], [1, 0]],  # 一个多标签指示器的列表示例
        [[0, 1]],  # 另一个多标签指示器的列表示例
        *_generate_sparse(  # 展开 _generate_sparse 函数的返回值作为列表的一部分
            multilabel_explicit_zero,  # 使用显式零多标签矩阵作为输入数据
            sparse_containers=CSC_CONTAINERS,  # 使用 CSC 格式的稀疏容器
            dtypes=(int,),  # 仅使用整数数据类型
        ),
        *_generate_sparse([[0, 1], [1, 0]]),  # 展开 _generate_sparse 函数的返回值作为列表的一部分，使用默认稀疏容器和数据类型
        *_generate_sparse([[0, 0], [0, 0]]),  # 展开 _generate_sparse 函数的返回值作为列表的一部分，使用默认稀疏容器和数据类型
        *_generate_sparse([[0, 1]]),  # 展开 _generate_sparse 函数的返回值作为列表的一部分，使用默认稀疏容器和数据类型
        # 仅在数据为密集形式时有效
        [[-1, 1], [1, -1]],  # 密集形式的矩阵示例
        np.array([[-1, 1], [1, -1]]),  # 密集形式的矩阵示例
        np.array([[-3, 3], [3, -3]]),  # 密集形式的矩阵示例
        _NotAnArray(np.array([[-3, 3], [3, -3]])),  # 表示不是数组的示例
    ],
    "multiclass": [  # 多类分类示例的键值对列表
        [1, 0, 2, 2, 1, 4, 2, 4, 4, 4],  # 多类分类的标签列表示例
        np.array([1, 0, 2]),  # 多类分类的标签数组示例
        np.array([1, 0, 2], dtype=np.int8),  # 多类分类的标签数组示例，指定数据类型为 int8
        np.array([1, 0, 2], dtype=np.uint8),  # 多类分类的标签数组示例，指定数据类型为 uint8
        np.array([1, 0, 2], dtype=float),  # 多类分类的标签数组示例，指定数据类型为 float
        np.array([1, 0, 2], dtype=np.float32),  # 多类分类的标签数组示例，指定数据类型为 float32
        np.array([[1], [0], [2]]),  # 多类分类的标签二维数组示例
        _NotAnArray(np.array([1, 0, 2])),  # 表示不是数组的示例
        [0, 1, 2],  # 多类分类的标签列表示例
        ["a", "b", "c"],  # 多类分类的标签列表示例
        np.array(["a", "b", "c"]),  # 多类分类的标签数组示例
        np.array(["a", "b", "c"], dtype=object),  # 多类分类的标签数组示例，指定数据类型为 object
        np.array(["a", "b", "c"], dtype=
    # 多类别多输出数据集示例
    "multiclass-multioutput": [
        # 整数列表示例
        [[1, 0, 2, 2], [1, 4, 2, 4]],
        # 字符串列表示例
        [["a", "b"], ["c", "d"]],
        # 二维整数 NumPy 数组示例
        np.array([[1, 0, 2, 2], [1, 4, 2, 4]]),
        # 二维 int8 类型 NumPy 数组示例
        np.array([[1, 0, 2, 2], [1, 4, 2, 4]], dtype=np.int8),
        # 二维 uint8 类型 NumPy 数组示例
        np.array([[1, 0, 2, 2], [1, 4, 2, 4]], dtype=np.uint8),
        # 二维浮点数 NumPy 数组示例
        np.array([[1, 0, 2, 2], [1, 4, 2, 4]], dtype=float),
        # 二维 float32 类型 NumPy 数组示例
        np.array([[1, 0, 2, 2], [1, 4, 2, 4]], dtype=np.float32),
        # 稀疏数据生成函数返回值示例
        *_generate_sparse(
            [[1, 0, 2, 2], [1, 4, 2, 4]],
            sparse_containers=CSC_CONTAINERS + CSR_CONTAINERS,
            dtypes=(int, np.int8, np.uint8, float, np.float32),
        ),
        # 字符串二维数组示例
        np.array([["a", "b"], ["c", "d"]]),
        # 字符串二维数组示例
        np.array([["a", "b"], ["c", "d"]]),
        # 对象类型的字符串二维数组示例
        np.array([["a", "b"], ["c", "d"]], dtype=object),
        # 三维整数 NumPy 数组示例
        np.array([[1, 0, 2]]),
        # 不是数组类型示例
        _NotAnArray(np.array([[1, 0, 2]])),
    ],

    # 二元分类数据集示例
    "binary": [
        # 一维整数数组示例
        [0, 1],
        # 一维整数数组示例
        [1, 1],
        # 空数组示例
        [],
        # 一维整数数组示例
        [0],
        # 一维布尔类型 NumPy 数组示例
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool),
        # 一维 int8 类型 NumPy 数组示例
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.int8),
        # 一维 uint8 类型 NumPy 数组示例
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.uint8),
        # 一维浮点数 NumPy 数组示例
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=float),
        # 一维 float32 类型 NumPy 数组示例
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.float32),
        # 二维整数数组示例
        np.array([[0], [1]]),
        # 不是数组类型示例
        _NotAnArray(np.array([[0], [1]])),
        # 一维整数数组示例
        [1, -1],
        # 一维整数数组示例
        [3, 5],
        # 字符串数组示例
        ["a"],
        # 字符串数组示例
        ["a", "b"],
        # 字符串数组示例
        ["abc", "def"],
        # 字符串数组示例
        np.array(["abc", "def"]),
        # 字符串数组示例
        ["a", "b"],
        # 对象类型的字符串数组示例
        np.array(["abc", "def"], dtype=object),
    ],

    # 连续值数据集示例
    "continuous": [
        # 一维浮点数数组示例
        [1e-5],
        # 一维浮点数数组示例
        [0, 0.5],
        # 二维浮点数 NumPy 数组示例
        np.array([[0], [0.5]]),
        # 二维 float32 类型 NumPy 数组示例
        np.array([[0], [0.5]], dtype=np.float32),
    ],

    # 连续值多输出数据集示例
    "continuous-multioutput": [
        # 二维浮点数 NumPy 数组示例
        np.array([[0, 0.5], [0.5, 0]]),
        # 二维 float32 类型 NumPy 数组示例
        np.array([[0, 0.5], [0.5, 0]], dtype=np.float32),
        # 一维浮点数数组生成稀疏数据示例
        *_generate_sparse(
            [[0, 0.5], [0.5, 0]],
            sparse_containers=CSC_CONTAINERS + CSR_CONTAINERS,
            dtypes=(float, np.float32),
        ),
        # 一维浮点数数组生成稀疏数据示例
        *_generate_sparse(
            [[0, 0.5]],
            sparse_containers=CSC_CONTAINERS + CSR_CONTAINERS,
            dtypes=(float, np.float32),
        ),
    ],

    # 未知类型数据集示例
    "unknown": [
        # 空列表示例
        [[]],
        # 对象类型的空数组示例
        np.array([[]], dtype=object),
        # 空元组示例
        [()],
        # 不支持的序列嵌套示例
        np.array([np.array([]), np.array([1, 2, 3])], dtype=object),
        # 序列嵌套示例
        [np.array([]), np.array([1, 2, 3])],
        # 集合示例
        [{1, 2, 3}, {1, 2}],
        # 冻结集合示例
        [frozenset([1, 2, 3]), frozenset([1, 2])],
        # 字典示例
        [{0: "a", 1: "b"}, {0: "a"}],
        # 0 维数组示例
        np.array(0),
        # 空第二维数组示例
        np.array([[], []]),
        # 三维数组示例
        np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),
    ],
}

ARRAY_API_EXAMPLES = {
    "multilabel-indicator": [
        np.random.RandomState(42).randint(2, size=(10, 10)),  # 创建一个10x10的随机整数数组
        [[0, 1], [1, 0]],  # 二维列表，表示二分类标签
        [[0, 1]],  # 二维列表，表示多标签问题中的一个标签
        multilabel_explicit_zero,  # 函数或变量 multilabel_explicit_zero 的引用
        [[0, 0], [0, 0]],  # 全为零的二维列表，表示没有标签
        [[-1, 1], [1, -1]],  # 二维列表，表示多分类标签
        np.array([[-1, 1], [1, -1]]),  # 二维 NumPy 数组，表示多分类标签
        np.array([[-3, 3], [3, -3]]),  # 二维 NumPy 数组，表示多分类标签
        _NotAnArray(np.array([[-3, 3], [3, -3]])),  # _NotAnArray 类的实例，包含一个 NumPy 数组
    ],
    "multiclass": [
        [1, 0, 2, 2, 1, 4, 2, 4, 4, 4],  # 多类分类的标签列表
        np.array([1, 0, 2]),  # 一维 NumPy 数组，表示多类分类的标签
        np.array([1, 0, 2], dtype=np.int8),  # 一维 NumPy 数组，使用 int8 类型
        np.array([1, 0, 2], dtype=np.uint8),  # 一维 NumPy 数组，使用 uint8 类型
        np.array([1, 0, 2], dtype=float),  # 一维 NumPy 数组，使用 float 类型
        np.array([1, 0, 2], dtype=np.float32),  # 一维 NumPy 数组，使用 float32 类型
        np.array([[1], [0], [2]]),  # 二维 NumPy 数组，表示多类分类的标签
        _NotAnArray(np.array([1, 0, 2])),  # _NotAnArray 类的实例，包含一个 NumPy 数组
        [0, 1, 2],  # 多类分类的标签列表
    ],
    "multiclass-multioutput": [
        [[1, 0, 2, 2], [1, 4, 2, 4]],  # 二维列表，表示多输出的多类分类标签
        np.array([[1, 0, 2, 2], [1, 4, 2, 4]]),  # 二维 NumPy 数组，表示多输出的多类分类标签
        np.array([[1, 0, 2, 2], [1, 4, 2, 4]], dtype=np.int8),  # 二维 NumPy 数组，使用 int8 类型
        np.array([[1, 0, 2, 2], [1, 4, 2, 4]], dtype=np.uint8),  # 二维 NumPy 数组，使用 uint8 类型
        np.array([[1, 0, 2, 2], [1, 4, 2, 4]], dtype=float),  # 二维 NumPy 数组，使用 float 类型
        np.array([[1, 0, 2, 2], [1, 4, 2, 4]], dtype=np.float32),  # 二维 NumPy 数组，使用 float32 类型
        np.array([[1, 0, 2]]),  # 二维 NumPy 数组，表示多输出的多类分类标签
        _NotAnArray(np.array([[1, 0, 2]])),  # _NotAnArray 类的实例，包含一个 NumPy 数组
    ],
    "binary": [
        [0, 1],  # 二元分类标签列表
        [1, 1],  # 二元分类标签列表
        [],  # 空列表，表示没有标签
        [0],  # 一元分类标签列表
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1]),  # 一维 NumPy 数组，表示二元分类标签
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool),  # 一维 NumPy 数组，使用 bool 类型
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.int8),  # 一维 NumPy 数组，使用 int8 类型
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.uint8),  # 一维 NumPy 数组，使用 uint8 类型
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=float),  # 一维 NumPy 数组，使用 float 类型
        np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.float32),  # 一维 NumPy 数组，使用 float32 类型
        np.array([[0], [1]]),  # 二维 NumPy 数组，表示二元分类标签
        _NotAnArray(np.array([[0], [1]])),  # _NotAnArray 类的实例，包含一个 NumPy 数组
        [1, -1],  # 二元分类标签列表
        [3, 5],  # 二元分类标签列表
    ],
    "continuous": [
        [1e-5],  # 连续数值的列表
        [0, 0.5],  # 连续数值的列表
        np.array([[0], [0.5]]),  # 二维 NumPy 数组，表示连续数值
        np.array([[0], [0.5]], dtype=np.float32),  # 二维 NumPy 数组，使用 float32 类型，表示连续数值
    ],
    "continuous-multioutput": [
        np.array([[0, 0.5], [0.5, 0]]),  # 二维 NumPy 数组，表示多输出的连续数值
        np.array([[0, 0.5], [0.5, 0]], dtype=np.float32),  # 二维 NumPy 数组，使用 float32 类型，表示多输出的连续数值
        np.array([[0, 0.5]]),  # 二维 NumPy 数组，表示多输出的连续数值
    ],
    "unknown": [
        [[]],  # 包含空列表的列表
        [()],  # 包含空元组的列表
        np.array(0),  # 标量 NumPy 数组
        np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]),  # 三维 NumPy 数组
    ],
}


NON_ARRAY_LIKE_EXAMPLES = [
    {1, 2, 3},  # 集合，不是类数组对象的示例
    {0: "a", 1: "b"},  # 字典，不是类数组对象的示例
    {0: [5], 1: [5]},  # 字典，包含列表，不是类数组对象的示例
    "abc",  # 字符串，不是类数组对象的示例
    frozenset([1, 2, 3]),  # frozenset，不是类数组对象的示例
    None,  # None 类型，不是类数组对象的示例
]

MULTILABEL_SEQUENCES = [
    [[1], [2], [0, 1]],  # 二维列表，表示多标签序列
    [(), (2), (0, 1)],  # 元组和列表组成的列表，表示多标签序列
    np.array([[], [1, 2]], dtype="object"),  # 二维 NumPy 数组，使用
    # 对输入的数组进行测试，确保 unique_labels 函数返回的结果与预期的一维数组相等
    assert_array_equal(unique_labels(np.array([[0, 0, 1], [0, 0, 0]])), np.arange(3))

    # 多个数组作为参数传递给 unique_labels 函数进行测试
    assert_array_equal(unique_labels([4, 0, 2], range(5)), np.arange(5))
    assert_array_equal(unique_labels((0, 1, 2), (0,), (2, 1)), np.arange(3))

    # 用二进制指示矩阵进行边界情况测试，预期会触发 ValueError 异常
    with pytest.raises(ValueError):
        unique_labels([4, 0, 2], np.ones((5, 5)))
    with pytest.raises(ValueError):
        unique_labels(np.ones((5, 4)), np.ones((5, 5)))

    # 对于全为1的矩阵作为输入，确保 unique_labels 函数返回的结果与预期的一维数组相等
    assert_array_equal(unique_labels(np.ones((4, 5)), np.ones((5, 5))), np.arange(5))
def test_unique_labels_non_specific():
    # Test unique_labels with a variety of collected examples

    # Smoke test for all supported format
    for format in ["binary", "multiclass", "multilabel-indicator"]:
        # Iterate over examples for each format
        for y in EXAMPLES[format]:
            unique_labels(y)

    # We don't support those format at the moment
    # Test for examples that are not array-like
    for example in NON_ARRAY_LIKE_EXAMPLES:
        with pytest.raises(ValueError):
            unique_labels(example)

    # Test for unsupported y_types
    for y_type in [
        "unknown",
        "continuous",
        "continuous-multioutput",
        "multiclass-multioutput",
    ]:
        for example in EXAMPLES[y_type]:
            with pytest.raises(ValueError):
                unique_labels(example)


def test_unique_labels_mixed_types():
    # Mix with binary or multiclass and multilabel
    mix_clf_format = product(
        EXAMPLES["multilabel-indicator"], EXAMPLES["multiclass"] + EXAMPLES["binary"]
    )

    # Test combinations of multilabel and multiclass/binary examples
    for y_multilabel, y_multiclass in mix_clf_format:
        with pytest.raises(ValueError):
            unique_labels(y_multiclass, y_multilabel)
        with pytest.raises(ValueError):
            unique_labels(y_multilabel, y_multiclass)

    # Additional specific cases to raise ValueError
    with pytest.raises(ValueError):
        unique_labels([[1, 2]], [["a", "d"]])

    with pytest.raises(ValueError):
        unique_labels(["1", 2])

    with pytest.raises(ValueError):
        unique_labels([["1", 2], [1, 3]])

    with pytest.raises(ValueError):
        unique_labels([["1", "2"], [2, 3]])


def test_is_multilabel():
    # Iterate over different groups of examples
    for group, group_examples in EXAMPLES.items():
        dense_exp = group == "multilabel-indicator"

        # Check each example in the group
        for example in group_examples:
            # Determine if the example is a sparse multilabel-indicator
            sparse_exp = dense_exp and issparse(example)

            # Check if example or its array representation meet criteria for sparse multilabel
            if issparse(example) or (
                hasattr(example, "__array__")
                and np.asarray(example).ndim == 2
                and np.asarray(example).dtype.kind in "biuf"
                and np.asarray(example).shape[1] > 0
            ):
                examples_sparse = [
                    sparse_container(example)
                    for sparse_container in (
                        COO_CONTAINERS
                        + CSC_CONTAINERS
                        + CSR_CONTAINERS
                        + DOK_CONTAINERS
                        + LIL_CONTAINERS
                    )
                ]
                # Assert if example is correctly identified as sparse multilabel
                for exmpl_sparse in examples_sparse:
                    assert sparse_exp == is_multilabel(
                        exmpl_sparse
                    ), f"is_multilabel({exmpl_sparse!r}) should be {sparse_exp}"

            # Convert sparse examples to dense before testing
            if issparse(example):
                example = example.toarray()

            # Assert if example is correctly identified as dense multilabel
            assert dense_exp == is_multilabel(
                example
            ), f"is_multilabel({example!r}) should be {dense_exp}"
    # 定义字符串元组，包含三个元素：'array_namespace', 'device', 'dtype_name'
    "array_namespace, device, dtype_name",
    # 调用函数 yield_namespace_device_dtype_combinations() 生成命名空间、设备和数据类型的组合
    yield_namespace_device_dtype_combinations(),
def test_is_multilabel_array_api_compliance(array_namespace, device, dtype_name):
    # 获取特定数组 API 的函数对象
    xp = _array_api_for_tests(array_namespace, device)

    # 遍历数组 API 示例字典中的每个组和对应的示例
    for group, group_examples in ARRAY_API_EXAMPLES.items():
        # 确定当前组是否为多标签指示器组
        dense_exp = group == "multilabel-indicator"
        # 遍历当前组的示例
        for example in group_examples:
            # 如果示例的数据类型为浮点数，则按指定的数据类型进行转换
            if np.asarray(example).dtype.kind == "f":
                example = np.asarray(example, dtype=dtype_name)
            else:
                example = np.asarray(example)
            # 使用特定数组 API 将示例转换为特定设备上的数组对象
            example = xp.asarray(example, device=device)

            # 在数组 API 分发为真的上下文中进行断言
            with config_context(array_api_dispatch=True):
                # 断言当前示例是否符合预期的多标签属性
                assert dense_exp == is_multilabel(
                    example
                ), f"is_multilabel({example!r}) should be {dense_exp}"


def test_check_classification_targets():
    # 遍历示例字典中的每种类型
    for y_type in EXAMPLES.keys():
        # 对于特定类型的 y，如果是未知类型或连续类型的多输出，则引发值错误
        if y_type in ["unknown", "continuous", "continuous-multioutput"]:
            for example in EXAMPLES[y_type]:
                msg = "Unknown label type: "
                with pytest.raises(ValueError, match=msg):
                    check_classification_targets(example)
        else:
            # 对于其他类型的 y，检查分类目标
            for example in EXAMPLES[y_type]:
                check_classification_targets(example)


# @ignore_warnings
def test_type_of_target():
    # 遍历示例字典中的每个组及其示例
    for group, group_examples in EXAMPLES.items():
        # 对于每个示例，检查其类型是否符合预期
        for example in group_examples:
            assert (
                type_of_target(example) == group
            ), "type_of_target(%r) should be %r, got %r" % (
                example,
                group,
                type_of_target(example),
            )

    # 对于非数组样例，验证是否引发值错误并匹配预期的正则表达式消息
    for example in NON_ARRAY_LIKE_EXAMPLES:
        msg_regex = r"Expected array-like \(array or non-string sequence\).*"
        with pytest.raises(ValueError, match=msg_regex):
            type_of_target(example)

    # 对于多标签序列，验证是否引发值错误并匹配特定消息
    for example in MULTILABEL_SEQUENCES:
        msg = (
            "You appear to be using a legacy multi-label data "
            "representation. Sequence of sequences are no longer supported;"
            " use a binary array or sparse matrix instead."
        )
        with pytest.raises(ValueError, match=msg):
            type_of_target(example)


def test_type_of_target_pandas_sparse():
    pd = pytest.importorskip("pandas")

    # 创建稀疏数组，验证是否引发值错误并匹配特定消息
    y = pd.arrays.SparseArray([1, np.nan, np.nan, 1, np.nan])
    msg = "y cannot be class 'SparseSeries' or 'SparseArray'"
    with pytest.raises(ValueError, match=msg):
        type_of_target(y)


def test_type_of_target_pandas_nullable():
    """Check that type_of_target works with pandas nullable dtypes."""
    pd = pytest.importorskip("pandas")

    # 对于每种可空数据类型，验证 type_of_target 的行为
    for dtype in ["Int32", "Float32"]:
        y_true = pd.Series([1, 0, 2, 3, 4], dtype=dtype)
        assert type_of_target(y_true) == "multiclass"

        y_true = pd.Series([1, 0, 1, 0], dtype=dtype)
        assert type_of_target(y_true) == "binary"

    # 创建包含浮点数的 DataFrame，验证 type_of_target 的行为
    y_true = pd.DataFrame([[1.4, 3.1], [3.1, 1.4]], dtype="Float32")
    assert type_of_target(y_true) == "continuous-multioutput"
    # 创建一个 Pandas DataFrame，其中包含两行两列的整数数据，类型为 Int32
    y_true = pd.DataFrame([[0, 1], [1, 1]], dtype="Int32")
    # 使用 type_of_target 函数检查 y_true 的数据类型，期望结果是 "multilabel-indicator"
    assert type_of_target(y_true) == "multilabel-indicator"
    
    # 创建另一个 Pandas DataFrame，包含两行两列的整数数据，类型为 Int32
    y_true = pd.DataFrame([[1, 2], [3, 1]], dtype="Int32")
    # 使用 type_of_target 函数再次检查 y_true 的数据类型，期望结果是 "multiclass-multioutput"
    assert type_of_target(y_true) == "multiclass-multioutput"
@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_unique_labels_pandas_nullable(dtype):
    """Checks that unique_labels work with pandas nullable dtypes.

    Non-regression test for gh-25634.
    """
    pd = pytest.importorskip("pandas")  # 导入并检查 pandas 库是否存在

    # 创建包含预期数据的 pandas Series 对象，使用指定的 dtype
    y_true = pd.Series([1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=dtype)
    y_predicted = pd.Series([0, 0, 1, 1, 0, 1, 1, 1, 1], dtype="int64")

    # 调用 unique_labels 函数，计算实际的标签列表
    labels = unique_labels(y_true, y_predicted)
    # 断言实际标签与预期标签相等
    assert_array_equal(labels, [0, 1])


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_class_distribution(csc_container):
    y = np.array(
        [
            [1, 0, 0, 1],
            [2, 2, 0, 1],
            [1, 3, 0, 1],
            [4, 2, 0, 1],
            [2, 0, 0, 1],
            [1, 3, 0, 1],
        ]
    )
    # 使用给定的 csc_container 构建稀疏矩阵 y_sp
    data = np.array([1, 2, 1, 4, 2, 1, 0, 2, 3, 2, 3, 1, 1, 1, 1, 1, 1])
    indices = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 5])
    indptr = np.array([0, 6, 11, 11, 17])
    y_sp = csc_container((data, indices, indptr), shape=(6, 4))

    # 调用 class_distribution 函数，获取各个类别的分布信息
    classes, n_classes, class_prior = class_distribution(y)
    classes_sp, n_classes_sp, class_prior_sp = class_distribution(y_sp)
    classes_expected = [[1, 2, 4], [0, 2, 3], [0], [1]]
    n_classes_expected = [3, 3, 1, 1]
    class_prior_expected = [[3 / 6, 2 / 6, 1 / 6], [1 / 3, 1 / 3, 1 / 3], [1.0], [1.0]]

    # 遍历每个类别，断言计算的结果与预期结果相等
    for k in range(y.shape[1]):
        assert_array_almost_equal(classes[k], classes_expected[k])
        assert_array_almost_equal(n_classes[k], n_classes_expected[k])
        assert_array_almost_equal(class_prior[k], class_prior_expected[k])

        assert_array_almost_equal(classes_sp[k], classes_expected[k])
        assert_array_almost_equal(n_classes_sp[k], n_classes_expected[k])
        assert_array_almost_equal(class_prior_sp[k], class_prior_expected[k])

    # 使用显式的样本权重再次测试
    (classes, n_classes, class_prior) = class_distribution(
        y, [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    )
    (classes_sp, n_classes_sp, class_prior_sp) = class_distribution(
        y, [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    )
    class_prior_expected = [[4 / 9, 3 / 9, 2 / 9], [2 / 9, 4 / 9, 3 / 9], [1.0], [1.0]]

    # 再次断言计算的结果与预期结果相等
    for k in range(y.shape[1]):
        assert_array_almost_equal(classes[k], classes_expected[k])
        assert_array_almost_equal(n_classes[k], n_classes_expected[k])
        assert_array_almost_equal(class_prior[k], class_prior_expected[k])

        assert_array_almost_equal(classes_sp[k], classes_expected[k])
        assert_array_almost_equal(n_classes_sp[k], n_classes_expected[k])
        assert_array_almost_equal(class_prior_sp[k], class_prior_expected[k])


def test_safe_split_with_precomputed_kernel():
    # 创建两个支持向量分类器对象，一个使用默认核函数，另一个使用预先计算的核矩阵作为核函数
    clf = SVC()
    clfp = SVC(kernel="precomputed")

    # 加载鸢尾花数据集，并计算数据集 X 的核矩阵 K
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    K = np.dot(X, X.T)

    # 创建一个 ShuffleSplit 交叉验证器对象，用于后续的安全分割测试
    cv = ShuffleSplit(test_size=0.25, random_state=0)
    # 将数据集按照交叉验证的第一折划分为训练集和测试集
    train, test = list(cv.split(X))[0]

    # 从分类器和数据中安全地获取训练集的子集
    X_train, y_train = _safe_split(clf, X, y, train)
    # 从内核函数和数据中安全地获取训练集的子集
    K_train, y_train2 = _safe_split(clfp, K, y, train)
    # 断言内核矩阵 K_train 等于 X_train 乘以其转置的近似值
    assert_array_almost_equal(K_train, np.dot(X_train, X_train.T))
    # 断言训练集标签 y_train 等于 y_train2 的近似值
    assert_array_almost_equal(y_train, y_train2)

    # 从分类器和数据中安全地获取测试集的子集
    X_test, y_test = _safe_split(clf, X, y, test, train)
    # 从内核函数和数据中安全地获取测试集的子集
    K_test, y_test2 = _safe_split(clfp, K, y, test, train)
    # 断言内核矩阵 K_test 等于 X_test 乘以 X_train 转置的近似值
    assert_array_almost_equal(K_test, np.dot(X_test, X_train.T))
    # 断言测试集标签 y_test 等于 y_test2 的近似值
    assert_array_almost_equal(y_test, y_test2)
# 定义测试函数 test_ovr_decision_function，用于测试一对多分类器的决策函数
def test_ovr_decision_function():
    # 设置模拟的预测结果，每个数组表示一个样本的类别预测
    predictions = np.array([[0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1]])

    # 设置每个预测结果的置信度
    confidences = np.array(
        [[-1e16, 0, -1e16], [1.0, 2.0, -3.0], [-5.0, 2.0, 5.0], [-0.5, 0.2, 0.5]]
    )

    # 类别数目
    n_classes = 3

    # 调用 _ovr_decision_function 计算决策值
    dec_values = _ovr_decision_function(predictions, confidences, n_classes)

    # 检查决策值是否在票数的 0.5 范围内
    votes = np.array([[1, 0, 2], [1, 1, 1], [1, 0, 2], [1, 0, 2]])
    assert_allclose(votes, dec_values, atol=0.5)

    # 检查预测结果是否符合预期
    expected_prediction = np.array([2, 1, 2, 2])
    assert_array_equal(np.argmax(dec_values, axis=1), expected_prediction)

    # 第三和第四个样本具有相同的票数，但第三个样本的置信度更高，这应反映在决策值上
    assert dec_values[2, 2] > dec_values[3, 2]

    # 检查子集不变性
    dec_values_one = [
        _ovr_decision_function(
            np.array([predictions[i]]), np.array([confidences[i]]), n_classes
        )[0]
        for i in range(4)
    ]
    assert_allclose(dec_values, dec_values_one, atol=1e-6)


# TODO(1.7): Change to ValueError when byte labels is deprecated.
# 使用 pytest 的参数化装饰器定义测试函数 test_labels_in_bytes_format
@pytest.mark.parametrize("input_type", ["list", "array"])
def test_labels_in_bytes_format(input_type):
    # 检查当标签以字节编码形式提供时是否会引发错误
    # 非回归测试，参考：https://github.com/scikit-learn/scikit-learn/issues/16980
    target = _convert_container([b"a", b"b"], input_type)
    # 设置警告消息内容
    err_msg = (
        "Support for labels represented as bytes is deprecated in v1.5 and will"
        " error in v1.7. Convert the labels to a string or integer format."
    )
    # 使用 pytest 的 warn 函数检查是否会发出 FutureWarning 并匹配错误消息
    with pytest.warns(FutureWarning, match=err_msg):
        type_of_target(target)
```