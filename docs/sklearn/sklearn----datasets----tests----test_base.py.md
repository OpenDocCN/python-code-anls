# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\test_base.py`

```
# 导入所需的库和模块
import io  # 输入输出操作
import os  # 操作系统功能
import shutil  # 文件操作
import tempfile  # 临时文件和目录操作
import warnings  # 警告控制
from functools import partial  # 函数工具：部分应用函数
from importlib import resources  # 资源管理器
from pathlib import Path  # 对文件和目录进行面向对象编程
from pickle import dumps, loads  # 对象序列化和反序列化
from unittest.mock import Mock  # 单元测试模拟对象
from urllib.error import HTTPError  # 处理 HTTP 错误

import numpy as np  # 科学计算库
import pytest  # Python 测试框架

from sklearn.datasets import (  # 导入 Scikit-learn 数据集相关模块和函数
    clear_data_home,
    get_data_home,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_files,
    load_iris,
    load_linnerud,
    load_sample_image,
    load_sample_images,
    load_wine,
)
from sklearn.datasets._base import (  # 导入 Scikit-learn 数据集基础模块
    RemoteFileMetadata,
    _fetch_remote,
    load_csv_data,
    load_gzip_compressed_csv_data,
)
from sklearn.datasets.tests.test_common import check_as_frame  # 导入 Scikit-learn 测试工具
from sklearn.preprocessing import scale  # 数据预处理：特征标准化
from sklearn.utils import Bunch  # 一个简单的类，包含一组命名的属性

class _DummyPath:
    """Minimal class that implements the os.PathLike interface."""

    def __init__(self, path):
        self.path = path

    def __fspath__(self):
        return self.path

# 移除指定路径下的目录
def _remove_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)

# 定义 data_home 的 pytest fixture，用于测试数据集的临时目录
@pytest.fixture(scope="module")
def data_home(tmpdir_factory):
    tmp_file = str(tmpdir_factory.mktemp("scikit_learn_data_home_test"))  # 创建临时目录
    yield tmp_file  # 返回临时目录路径
    _remove_dir(tmp_file)  # 清理临时目录

# 定义 load_files_root 的 pytest fixture，用于加载文件时的临时目录
@pytest.fixture(scope="module")
def load_files_root(tmpdir_factory):
    tmp_file = str(tmpdir_factory.mktemp("scikit_learn_load_files_test"))  # 创建临时目录
    yield tmp_file  # 返回临时目录路径
    _remove_dir(tmp_file)  # 清理临时目录

# 定义用于测试的分类目录 1 的 pytest fixture
@pytest.fixture
def test_category_dir_1(load_files_root):
    test_category_dir1 = tempfile.mkdtemp(dir=load_files_root)  # 在指定目录下创建临时目录
    sample_file = tempfile.NamedTemporaryFile(dir=test_category_dir1, delete=False)  # 在临时目录中创建临时文件
    sample_file.write(b"Hello World!\n")  # 向临时文件写入内容
    sample_file.close()  # 关闭临时文件
    yield str(test_category_dir1)  # 返回临时目录路径
    _remove_dir(test_category_dir1)  # 清理临时目录

# 定义用于测试的分类目录 2 的 pytest fixture
@pytest.fixture
def test_category_dir_2(load_files_root):
    test_category_dir2 = tempfile.mkdtemp(dir=load_files_root)  # 在指定目录下创建临时目录
    yield str(test_category_dir2)  # 返回临时目录路径
    _remove_dir(test_category_dir2)  # 清理临时目录

# 参数化测试 data_home 函数，测试数据集的家目录行为
@pytest.mark.parametrize("path_container", [None, Path, _DummyPath])
def test_data_home(path_container, data_home):
    # 根据不同的 path_container 设置 data_home 对象
    if path_container is not None:
        data_home = path_container(data_home)
    data_home = get_data_home(data_home=data_home)  # 获取数据集的家目录路径
    assert data_home == data_home  # 断言路径相等
    assert os.path.exists(data_home)  # 断言路径存在

    # 清空数据集的家目录及其内容
    if path_container is not None:
        data_home = path_container(data_home)
    clear_data_home(data_home=data_home)  # 清空数据集的家目录
    assert not os.path.exists(data_home)  # 断言路径不存在

    # 如果路径不存在，则重新创建数据集的家目录
    data_home = get_data_home(data_home=data_home)  # 获取数据集的家目录路径
    assert os.path.exists(data_home)  # 断言路径存在

# 测试默认情况下加载空文件集
def test_default_empty_load_files(load_files_root):
    res = load_files(load_files_root)  # 加载指定目录下的文件集
    assert len(res.filenames) == 0  # 断言文件名列表为空
    assert len(res.target_names) == 0  # 断言目标名称列表为空
    assert res.DESCR is None  # 断言描述信息为空
# 测试默认加载文件的函数
def test_default_load_files(test_category_dir_1, test_category_dir_2, load_files_root):
    # 调用 load_files 函数加载文件
    res = load_files(load_files_root)
    # 断言返回的文件名列表长度为 1
    assert len(res.filenames) == 1
    # 断言返回的目标名称列表长度为 2
    assert len(res.target_names) == 2
    # 断言返回的描述信息为 None
    assert res.DESCR is None
    # 断言返回的数据内容为 ["Hello World!\n"]
    assert res.data == [b"Hello World!\n"]


# 测试加载文件并指定类别、描述和编码的函数
def test_load_files_w_categories_desc_and_encoding(
    test_category_dir_1, test_category_dir_2, load_files_root
):
    # 获取类别信息
    category = os.path.abspath(test_category_dir_1).split(os.sep).pop()
    # 调用 load_files 函数加载文件，指定描述、类别和编码
    res = load_files(
        load_files_root, description="test", categories=[category], encoding="utf-8"
    )

    # 断言返回的文件名列表长度为 1
    assert len(res.filenames) == 1
    # 断言返回的目标名称列表长度为 1
    assert len(res.target_names) == 1
    # 断言返回的描述信息为 "test"
    assert res.DESCR == "test"
    # 断言返回的数据内容为 ["Hello World!\n"]
    assert res.data == ["Hello World!\n"]


# 测试加载文件但不加载内容的函数
def test_load_files_wo_load_content(
    test_category_dir_1, test_category_dir_2, load_files_root
):
    # 调用 load_files 函数加载文件，但不加载内容
    res = load_files(load_files_root, load_content=False)
    # 断言返回的文件名列表长度为 1
    assert len(res.filenames) == 1
    # 断言返回的目标名称列表长度为 2
    assert len(res.target_names) == 2
    # 断言返回的描述信息为 None
    assert res.DESCR is None
    # 断言返回的数据内容为 None
    assert res.get("data") is None


# 使用 pytest 的参数化功能，测试允许的文件扩展名
@pytest.mark.parametrize("allowed_extensions", ([".txt"], [".txt", ".json"]))
def test_load_files_allowed_extensions(tmp_path, allowed_extensions):
    """Check the behaviour of `allowed_extension` in `load_files`."""
    # 创建临时文件夹及文件
    d = tmp_path / "sub"
    d.mkdir()
    files = ("file1.txt", "file2.json", "file3.json", "file4.md")
    paths = [d / f for f in files]
    for p in paths:
        p.write_bytes(b"hello")
    # 调用 load_files 函数加载指定扩展名的文件
    res = load_files(tmp_path, allowed_extensions=allowed_extensions)
    # 断言返回的文件名集合与期望的文件路径集合一致
    assert set([str(p) for p in paths if p.suffix in allowed_extensions]) == set(
        res.filenames
    )


# 使用 pytest 的参数化功能，测试加载 CSV 数据的函数
@pytest.mark.parametrize(
    "filename, expected_n_samples, expected_n_features, expected_target_names",
    [
        ("wine_data.csv", 178, 13, ["class_0", "class_1", "class_2"]),
        ("iris.csv", 150, 4, ["setosa", "versicolor", "virginica"]),
        ("breast_cancer.csv", 569, 30, ["malignant", "benign"]),
    ],
)
def test_load_csv_data(
    filename, expected_n_samples, expected_n_features, expected_target_names
):
    # 调用 load_csv_data 函数加载 CSV 数据
    actual_data, actual_target, actual_target_names = load_csv_data(filename)
    # 断言返回的数据形状符合预期的样本数和特征数
    assert actual_data.shape[0] == expected_n_samples
    assert actual_data.shape[1] == expected_n_features
    # 断言返回的目标数据形状符合预期的样本数
    assert actual_target.shape[0] == expected_n_samples
    # 断言返回的目标名称列表与预期的一致
    np.testing.assert_array_equal(actual_target_names, expected_target_names)


# 测试加载带有描述信息的 CSV 数据的函数
def test_load_csv_data_with_descr():
    data_file_name = "iris.csv"
    descr_file_name = "iris.rst"

    # 分别调用 load_csv_data 函数加载带描述和不带描述的 CSV 数据
    res_without_descr = load_csv_data(data_file_name=data_file_name)
    res_with_descr = load_csv_data(
        data_file_name=data_file_name, descr_file_name=descr_file_name
    )
    # 断言带描述数据的长度为 4（包括描述信息）
    assert len(res_with_descr) == 4
    # 断言不带描述数据的长度为 3
    assert len(res_without_descr) == 3

    # 使用 numpy 的数组比较功能，断言带描述数据的内容与不带描述数据的内容相同
    np.testing.assert_array_equal(res_with_descr[0], res_without_descr[0])
    np.testing.assert_array_equal(res_with_descr[1], res_without_descr[1])
    np.testing.assert_array_equal(res_with_descr[2], res_without_descr[2])
    # 断言最后一个元素以 ".. _iris_dataset:" 开头
    assert res_with_descr[-1].startswith(".. _iris_dataset:")
@pytest.mark.parametrize(
    "filename, kwargs, expected_shape",
    [
        ("diabetes_data_raw.csv.gz", {}, [442, 10]),  # 定义测试参数：文件名、关键字参数、期望的数据形状
        ("diabetes_target.csv.gz", {}, [442]),  # 定义测试参数：文件名、空的关键字参数、期望的数据形状
        ("digits.csv.gz", {"delimiter": ","}, [1797, 65]),  # 定义测试参数：文件名、指定分隔符的关键字参数、期望的数据形状
    ],
)
def test_load_gzip_compressed_csv_data(filename, kwargs, expected_shape):
    actual_data = load_gzip_compressed_csv_data(filename, **kwargs)  # 调用函数加载压缩的 CSV 数据
    assert actual_data.shape == tuple(expected_shape)  # 断言加载的数据形状符合预期


def test_load_gzip_compressed_csv_data_with_descr():
    data_file_name = "diabetes_target.csv.gz"
    descr_file_name = "diabetes.rst"

    expected_data = load_gzip_compressed_csv_data(data_file_name=data_file_name)  # 加载预期的数据
    actual_data, descr = load_gzip_compressed_csv_data(
        data_file_name=data_file_name,
        descr_file_name=descr_file_name,
    )  # 加载数据和描述信息

    np.testing.assert_array_equal(actual_data, expected_data)  # 使用 NumPy 断言数据加载正确
    assert descr.startswith(".. _diabetes_dataset:")  # 断言描述信息以指定内容开头


def test_load_sample_images():
    try:
        res = load_sample_images()  # 尝试加载示例图像数据
        assert len(res.images) == 2  # 断言图像数量为 2
        assert len(res.filenames) == 2  # 断言文件名数量为 2
        images = res.images

        # 断言第一张图像为中国图像
        assert np.all(images[0][0, 0, :] == np.array([174, 201, 231], dtype=np.uint8))
        # 断言第二张图像为花朵图像
        assert np.all(images[1][0, 0, :] == np.array([2, 19, 13], dtype=np.uint8))
        assert res.DESCR  # 断言加载的数据包含描述信息
    except ImportError:
        warnings.warn("Could not load sample images, PIL is not available.")


def test_load_sample_image():
    try:
        china = load_sample_image("china.jpg")  # 尝试加载名为 "china.jpg" 的示例图像
        assert china.dtype == "uint8"  # 断言图像数据类型为 uint8
        assert china.shape == (427, 640, 3)  # 断言图像形状为 (427, 640, 3)
    except ImportError:
        warnings.warn("Could not load sample images, PIL is not available.")


def test_load_diabetes_raw():
    """Test to check that we load a scaled version by default but that we can
    get an unscaled version when setting `scaled=False`."""
    diabetes_raw = load_diabetes(scaled=False)  # 加载未缩放的糖尿病数据
    assert diabetes_raw.data.shape == (442, 10)  # 断言数据部分的形状为 (442, 10)
    assert diabetes_raw.target.size, 442  # 断言目标变量的大小为 442
    assert len(diabetes_raw.feature_names) == 10  # 断言特征名列表长度为 10
    assert diabetes_raw.DESCR  # 断言数据包含描述信息

    diabetes_default = load_diabetes()  # 加载默认缩放的糖尿病数据

    np.testing.assert_allclose(
        scale(diabetes_raw.data) / (442**0.5), diabetes_default.data, atol=1e-04
    )  # 使用 NumPy 断言两个数据集在一定误差范围内相等


@pytest.mark.parametrize(
    "loader_func, data_shape, target_shape, n_target, has_descr, filenames",
    [
        (load_breast_cancer, (569, 30), (569,), 2, True, ["filename"]),  # 定义参数化测试参数：函数、数据形状、目标形状、目标数目、是否有描述信息、文件名列表
        (load_wine, (178, 13), (178,), 3, True, []),  # 同上，不带文件名列表
        (load_iris, (150, 4), (150,), 3, True, ["filename"]),  # 同上，带文件名列表
        (
            load_linnerud,
            (20, 3),
            (20, 3),
            3,
            True,
            ["data_filename", "target_filename"],  # 同上，带两个文件名列表
        ),
        (load_diabetes, (442, 10), (442,), None, True, []),  # 同上，不带文件名列表
        (load_digits, (1797, 64), (1797,), 10, True, []),  # 同上，不带文件名列表
        (partial(load_digits, n_class=9), (1617, 64), (1617,), 10, True, []),  # 同上，不带文件名列表
    ],
)
# 测试数据加载器函数的通用功能
def test_loader(loader_func, data_shape, target_shape, n_target, has_descr, filenames):
    # 调用指定的数据加载器函数，获取数据 Bunch 对象
    bunch = loader_func()

    # 断言返回的对象是 Bunch 类型的实例
    assert isinstance(bunch, Bunch)
    # 断言数据部分的形状符合预期
    assert bunch.data.shape == data_shape
    # 断言目标部分的形状符合预期
    assert bunch.target.shape == target_shape
    # 如果 Bunch 对象有 feature_names 属性，则断言其长度符合预期
    if hasattr(bunch, "feature_names"):
        assert len(bunch.feature_names) == data_shape[1]
    # 如果指定了 n_target，则断言目标名称列表的长度符合预期
    if n_target is not None:
        assert len(bunch.target_names) == n_target
    # 如果需要包含描述信息，则断言 DESCR 属性不为空
    if has_descr:
        assert bunch.DESCR
    # 如果需要检查文件名列表，则断言每个文件名都存在于 Bunch 对象中，并且对应的文件路径有效
    if filenames:
        assert "data_module" in bunch
        assert all(
            [
                f in bunch
                and (resources.files(bunch["data_module"]) / bunch[f]).is_file()
                for f in filenames
            ]
        )


# 使用参数化测试来验证不同 toy 数据集加载函数的返回结果的数据类型
@pytest.mark.parametrize(
    "loader_func, data_dtype, target_dtype",
    [
        (load_breast_cancer, np.float64, int),
        (load_diabetes, np.float64, np.float64),
        (load_digits, np.float64, int),
        (load_iris, np.float64, int),
        (load_linnerud, np.float64, np.float64),
        (load_wine, np.float64, int),
    ],
)
def test_toy_dataset_frame_dtype(loader_func, data_dtype, target_dtype):
    # 调用指定的 toy 数据集加载函数，并获取默认结果
    default_result = loader_func()
    # 使用 check_as_frame 函数检查返回结果的数据类型是否符合预期
    check_as_frame(
        default_result,
        loader_func,
        expected_data_dtype=data_dtype,
        expected_target_dtype=target_dtype,
    )


# 测试 Bunch 对象的序列化和反序列化操作
def test_loads_dumps_bunch():
    # 创建一个简单的 Bunch 对象
    bunch = Bunch(x="x")
    # 将 Bunch 对象转为 pickle 格式，并再次加载
    bunch_from_pkl = loads(dumps(bunch))
    # 修改加载后的 Bunch 对象的属性值
    bunch_from_pkl.x = "y"
    # 断言修改后的属性值通过索引和属性名均能访问到
    assert bunch_from_pkl["x"] == bunch_from_pkl.x


# 测试在不同版本的 scikit-learn 中序列化和反序列化 Bunch 对象的行为差异
def test_bunch_pickle_generated_with_0_16_and_read_with_0_17():
    # 创建一个 Bunch 对象，并手动设置其属性 __dict__["key"]
    bunch = Bunch(key="original")
    # 模拟使用 scikit-learn 0.16 版本序列化后，使用 0.17 版本读取的情况
    bunch.__dict__["key"] = "set from __dict__"
    bunch_from_pkl = loads(dumps(bunch))
    # 断言加载后的 Bunch 对象的 key 属性值符合预期
    assert bunch_from_pkl.key == "original"
    assert bunch_from_pkl["key"] == "original"
    # 修改加载后的 Bunch 对象的属性值，再次断言属性值符合预期
    bunch_from_pkl.key = "changed"
    assert bunch_from_pkl.key == "changed"
    assert bunch_from_pkl["key"] == "changed"


# 测试检查 Bunch 对象的 dir() 方法是否能正确显示属性
def test_bunch_dir():
    # 加载 iris 数据集
    data = load_iris()
    # 断言"data"在 dir(data) 中，验证 dir() 方法能正确显示属性
    assert "data" in dir(data)


# 测试在尝试导入 `load_boston` 时是否会引发 ImportError，验证是否会提示存在伦理问题
def test_load_boston_error():
    """Check that we raise the ethical warning when trying to import `load_boston`."""
    msg = "The Boston housing prices dataset has an ethical problem"
    # 使用 pytest.raises 验证导入 `load_boston` 时是否会抛出指定的 ImportError
    with pytest.raises(ImportError, match=msg):
        from sklearn.datasets import load_boston  # noqa
    # 定义一个错误消息，指出无法导入名为'non_existing_function'的函数，来自'sklearn.datasets'
    msg = "cannot import name 'non_existing_function' from 'sklearn.datasets'"
    # 使用 pytest 库中的 pytest.raises 上下文管理器，期望捕获 ImportError 异常，并且匹配特定的错误消息
    with pytest.raises(ImportError, match=msg):
        # 尝试从 sklearn.datasets 导入名为 non_existing_function 的函数（这里使用 noqa 来忽略 Flake8 的警告）
        from sklearn.datasets import non_existing_function  # noqa
def test_fetch_remote_raise_warnings_with_invalid_url(monkeypatch):
    """测试在 _fetch_remote 中处理无效 URL 的重试机制."""

    # 定义一个无效的 URL
    url = "https://scikit-learn.org/this_file_does_not_exist.tar.gz"
    
    # 创建一个代表无效远程文件的元数据对象
    invalid_remote_file = RemoteFileMetadata("invalid_file", url, None)
    
    # 创建一个模拟的 urlretrieve 函数的 Mock 对象
    urlretrieve_mock = Mock(
        side_effect=HTTPError(
            url=url, code=404, msg="Not Found", hdrs=None, fp=io.BytesIO()
        )
    )
    
    # 将模拟对象设置为 sklearn.datasets._base.urlretrieve 的替代项
    monkeypatch.setattr("sklearn.datasets._base.urlretrieve", urlretrieve_mock)

    # 使用 pytest 的 warn 声明检查，捕获 UserWarning 类型的警告，并匹配消息中的字符串
    with pytest.warns(UserWarning, match="Retry downloading") as record:
        # 使用 pytest 的 raises 声明检查，捕获 HTTPError 类型的异常，并匹配消息中的字符串
        with pytest.raises(HTTPError, match="HTTP Error 404"):
            # 调用 _fetch_remote 函数，预期会触发 HTTPError 异常
            _fetch_remote(invalid_remote_file, n_retries=3, delay=0)

        # 检查 urlretrieve_mock 被调用的次数，预期为 4 次（3 次重试 + 初始调用）
        assert urlretrieve_mock.call_count == 4

        # 遍历记录中的每个警告对象
        for r in record:
            # 检查警告消息是否符合预期格式
            assert str(r.message) == f"Retry downloading from url: {url}"
        
        # 检查总共捕获的警告数量，预期为 3 次（3 次重试）
        assert len(record) == 3
```