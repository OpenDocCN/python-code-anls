# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\test_lfw.py`

```
"""This test for the LFW require medium-size data downloading and processing

If the data has not been already downloaded by running the examples,
the tests won't run (skipped).

If the test are run, the first execution will be long (typically a bit
more than a couple of minutes) but as the dataset loader is leveraging
joblib, successive runs will be fast (less than 200ms).
"""

# 导入所需的库和模块
import random  # 导入随机数模块
from functools import partial  # 导入偏函数模块

import numpy as np  # 导入数值计算模块numpy
import pytest  # 导入单元测试框架pytest

# 导入用于获取LFW数据集的函数
from sklearn.datasets import fetch_lfw_pairs, fetch_lfw_people  
# 导入用于测试的共同函数
from sklearn.datasets.tests.test_common import check_return_X_y  
# 导入用于单元测试的断言函数
from sklearn.utils._testing import assert_array_equal  

# 假名列表，用于模拟人物名称
FAKE_NAMES = [
    "Abdelatif_Smith",
    "Abhati_Kepler",
    "Camara_Alvaro",
    "Chen_Dupont",
    "John_Lee",
    "Lin_Bauman",
    "Onur_Lopez",
]

# 定义模块级别的测试数据目录的fixture
@pytest.fixture(scope="module")
def mock_empty_data_home(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("scikit_learn_empty_test")

    yield data_dir

# 定义模块级别的测试数据目录的fixture，并生成LFW数据集用于测试
@pytest.fixture(scope="module")
def mock_data_home(tmp_path_factory):
    """Test fixture run once and common to all tests of this module"""
    # 导入并检查是否存在PIL.Image库
    Image = pytest.importorskip("PIL.Image")

    # 创建临时目录用于测试数据集
    data_dir = tmp_path_factory.mktemp("scikit_learn_lfw_test")
    lfw_home = data_dir / "lfw_home"
    lfw_home.mkdir(parents=True, exist_ok=True)

    # 设置随机种子
    random_state = random.Random(42)
    np_rng = np.random.RandomState(42)

    # 为每个假名生成随机的JPEG文件
    counts = {}
    for name in FAKE_NAMES:
        folder_name = lfw_home / "lfw_funneled" / name
        folder_name.mkdir(parents=True, exist_ok=True)

        n_faces = np_rng.randint(1, 5)
        counts[name] = n_faces
        for i in range(n_faces):
            file_path = folder_name / (name + "_%04d.jpg" % i)
            uniface = np_rng.randint(0, 255, size=(250, 250, 3))
            img = Image.fromarray(uniface.astype(np.uint8))
            img.save(file_path)

    # 添加一些随机文件用于测试的健壮性
    (lfw_home / "lfw_funneled" / ".test.swp").write_bytes(
        b"Text file to be ignored by the dataset loader."
    )

    # 生成与LFW相同格式的配对元数据文件
    with open(lfw_home / "pairsDevTrain.txt", "wb") as f:
        f.write(b"10\n")
        more_than_two = [name for name, count in counts.items() if count >= 2]
        for i in range(5):
            name = random_state.choice(more_than_two)
            first, second = random_state.sample(range(counts[name]), 2)
            f.write(("%s\t%d\t%d\n" % (name, first, second)).encode())

        for i in range(5):
            first_name, second_name = random_state.sample(FAKE_NAMES, 2)
            first_index = np_rng.choice(np.arange(counts[first_name]))
            second_index = np_rng.choice(np.arange(counts[second_name]))
            f.write(
                (
                    "%s\t%d\t%s\t%d\n"
                    % (first_name, first_index, second_name, second_index)
                ).encode()
            )
    # 将指定路径下的文件 pairsDevTest.txt 写入字节内容，这是一个占位符，不会被测试
    (lfw_home / "pairsDevTest.txt").write_bytes(
        b"Fake place holder that won't be tested"
    )
    # 将指定路径下的文件 pairs.txt 写入字节内容，这同样是一个占位符，不会被测试
    (lfw_home / "pairs.txt").write_bytes(b"Fake place holder that won't be tested")
    
    # 返回数据目录的路径作为生成器的结果
    yield data_dir
# 测试加载空的 LFW 人物数据集时的异常情况
def test_load_empty_lfw_people(mock_empty_data_home):
    # 使用 pytest 断言检查是否引发 OSError 异常
    with pytest.raises(OSError):
        fetch_lfw_people(data_home=mock_empty_data_home, download_if_missing=False)


# 测试加载虚假的 LFW 人物数据集
def test_load_fake_lfw_people(mock_data_home):
    # 调用 fetch_lfw_people 函数加载数据集，设置最小人脸数为3，禁止下载
    lfw_people = fetch_lfw_people(
        data_home=mock_data_home, min_faces_per_person=3, download_if_missing=False
    )

    # 断言人脸图像数据的形状为 (10, 62, 47)
    assert lfw_people.images.shape == (10, 62, 47)
    # 断言数据集的数据形状为 (10, 2914)
    assert lfw_people.data.shape == (10, 2914)

    # 断言目标标签是一个包含人物整数 ID 的数组
    assert_array_equal(lfw_people.target, [2, 0, 1, 0, 2, 0, 2, 1, 1, 2])

    # 断言目标名称可以通过 target_names 数组找到
    expected_classes = ["Abdelatif Smith", "Abhati Kepler", "Onur Lopez"]
    assert_array_equal(lfw_people.target_names, expected_classes)

    # 可以获取未裁剪或颜色转换的原始数据，并且不限制每人图片数量
    lfw_people = fetch_lfw_people(
        data_home=mock_data_home,
        resize=None,
        slice_=None,
        color=True,
        download_if_missing=False,
    )
    # 断言人脸图像数据的形状为 (17, 250, 250, 3)
    assert lfw_people.images.shape == (17, 250, 250, 3)
    # 断言数据集描述以指定字符串开头
    assert lfw_people.DESCR.startswith(".. _labeled_faces_in_the_wild_dataset:")

    # 断言目标标签与之前相同
    assert_array_equal(
        lfw_people.target, [0, 0, 1, 6, 5, 6, 3, 6, 0, 3, 6, 1, 2, 4, 5, 1, 2]
    )
    # 断言目标名称数组与预期数组相同
    assert_array_equal(
        lfw_people.target_names,
        [
            "Abdelatif Smith",
            "Abhati Kepler",
            "Camara Alvaro",
            "Chen Dupont",
            "John Lee",
            "Lin Bauman",
            "Onur Lopez",
        ],
    )

    # 测试 return_X_y 选项
    fetch_func = partial(
        fetch_lfw_people,
        data_home=mock_data_home,
        resize=None,
        slice_=None,
        color=True,
        download_if_missing=False,
    )
    # 检查数据是否符合返回 X 和 y 的要求
    check_return_X_y(lfw_people, fetch_func)


# 测试加载 LFW 人物数据集时设置过于严格的条件
def test_load_fake_lfw_people_too_restrictive(mock_data_home):
    # 使用 pytest 断言检查是否引发 ValueError 异常
    with pytest.raises(ValueError):
        fetch_lfw_people(
            data_home=mock_data_home,
            min_faces_per_person=100,
            download_if_missing=False,
        )


# 测试加载空的 LFW 对比数据集时的异常情况
def test_load_empty_lfw_pairs(mock_empty_data_home):
    # 使用 pytest 断言检查是否引发 OSError 异常
    with pytest.raises(OSError):
        fetch_lfw_pairs(data_home=mock_empty_data_home, download_if_missing=False)


# 测试加载虚假的 LFW 对比数据集
def test_load_fake_lfw_pairs(mock_data_home):
    # 调用 fetch_lfw_pairs 函数加载训练数据集，禁止下载
    lfw_pairs_train = fetch_lfw_pairs(
        data_home=mock_data_home, download_if_missing=False
    )

    # 断言人脸对数据的形状为 (10, 2, 62, 47)
    assert lfw_pairs_train.pairs.shape == (10, 2, 62, 47)

    # 断言目标标签表示人是否相同的数组
    assert_array_equal(lfw_pairs_train.target, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    # 定义预期的分类类别，用于验证数据集中的目标类别是否正确
    expected_classes = ["Different persons", "Same person"]
    # 使用 assert_array_equal 函数检查数据集中的目标类别是否与预期一致
    assert_array_equal(lfw_pairs_train.target_names, expected_classes)
    
    # 从模拟数据源获取 LFW（Labeled Faces in the Wild）数据集的图像对数据
    # 可以请求原始数据，而无需进行任何裁剪或颜色转换
    lfw_pairs_train = fetch_lfw_pairs(
        data_home=mock_data_home,
        resize=None,  # 不进行调整大小
        slice_=None,  # 不进行切片
        color=True,   # 使用彩色图像
        download_if_missing=False,  # 如果数据缺失则不下载
    )
    # 使用 assert 语句验证返回的图像对数组的形状是否为 (10, 2, 250, 250, 3)
    assert lfw_pairs_train.pairs.shape == (10, 2, 250, 250, 3)
    
    # 验证数据集中的目标标签是否与预期相符
    assert_array_equal(lfw_pairs_train.target, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    # 再次验证数据集中的目标类别是否与预期一致
    assert_array_equal(lfw_pairs_train.target_names, expected_classes)
    
    # 使用 assert 语句检查数据集的描述信息是否以特定字符串开头
    assert lfw_pairs_train.DESCR.startswith(".. _labeled_faces_in_the_wild_dataset:")
# 定义一个测试函数，用于检查我们是否正确裁剪图像。
# 这是一个非回归测试，用于验证以下问题的修复：
# https://github.com/scikit-learn/scikit-learn/issues/24942
def test_fetch_lfw_people_internal_cropping(mock_data_home):
    # 如果裁剪没有正确执行，且我们没有调整图像大小，图像将保持原始大小（250x250），
    # 并且图像将无法适应基于 `slice_` 参数预分配的 NumPy 数组。
    slice_ = (slice(70, 195), slice(78, 172))
    # 调用 fetch_lfw_people 函数获取 LFW 数据集，进行裁剪和调整大小操作。
    lfw = fetch_lfw_people(
        data_home=mock_data_home,  # 指定数据集的存储路径
        min_faces_per_person=3,  # 每个人至少包含的图像数量
        download_if_missing=False,  # 如果数据集不存在是否下载
        resize=None,  # 不调整图像大小
        slice_=slice_,  # 指定裁剪的切片范围
    )
    # 断言验证裁剪后的第一张图像的形状是否符合预期
    assert lfw.images[0].shape == (
        slice_[0].stop - slice_[0].start,  # 计算裁剪后图像的高度
        slice_[1].stop - slice_[1].start,  # 计算裁剪后图像的宽度
    )
```