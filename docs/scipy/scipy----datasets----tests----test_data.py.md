# `D:\src\scipysrc\scipy\scipy\datasets\tests\test_data.py`

```
from scipy.datasets._registry import registry
from scipy.datasets._fetchers import data_fetcher
from scipy.datasets._utils import _clear_cache
from scipy.datasets import ascent, face, electrocardiogram, download_all
from numpy.testing import assert_equal, assert_almost_equal
import os
import pytest

try:
    import pooch
except ImportError:
    raise ImportError("Missing optional dependency 'pooch' required "
                      "for scipy.datasets module. Please use pip or "
                      "conda to install 'pooch'.")

# 获取数据存储路径
data_dir = data_fetcher.path  # type: ignore


def _has_hash(path, expected_hash):
    """Check if the provided path has the expected hash."""
    # 如果路径不存在，则返回 False
    if not os.path.exists(path):
        return False
    # 使用 pooch 库检查文件的哈希值是否与预期哈希值相匹配
    return pooch.file_hash(path) == expected_hash


class TestDatasets:

    @pytest.fixture(scope='module', autouse=True)
    def test_download_all(self):
        # This fixture requires INTERNET CONNECTION
        # 下载所有数据集
        download_all()

        yield

    @pytest.mark.fail_slow(10)
    def test_existence_all(self):
        # 检查数据目录下的文件数量是否大于或等于注册表中数据集的数量
        assert len(os.listdir(data_dir)) >= len(registry)

    def test_ascent(self):
        # 检查 ascent 数据集的形状是否为 (512, 512)
        assert_equal(ascent().shape, (512, 512))

        # 检查 ascent 数据集文件的哈希值
        assert _has_hash(os.path.join(data_dir, "ascent.dat"),
                         registry["ascent.dat"])

    def test_face(self):
        # 检查 face 数据集的形状是否为 (768, 1024, 3)
        assert_equal(face().shape, (768, 1024, 3))

        # 检查 face 数据集文件的哈希值
        assert _has_hash(os.path.join(data_dir, "face.dat"),
                         registry["face.dat"])

    def test_electrocardiogram(self):
        # 检查 electrocardiogram 数据集的形状、数据类型和统计信息
        ecg = electrocardiogram()
        assert_equal(ecg.dtype, float)
        assert_equal(ecg.shape, (108000,))
        assert_almost_equal(ecg.mean(), -0.16510875)
        assert_almost_equal(ecg.std(), 0.5992473991177294)

        # 检查 electrocardiogram 数据集文件的哈希值
        assert _has_hash(os.path.join(data_dir, "ecg.dat"),
                         registry["ecg.dat"])


def test_clear_cache(tmp_path):
    # Note: `tmp_path` is a pytest fixture, it handles cleanup
    # 创建一个临时的缓存目录
    dummy_basepath = tmp_path / "dummy_cache_dir"
    dummy_basepath.mkdir()

    # 为虚拟数据集方法创建三个虚拟数据集文件
    dummy_method_map = {}
    for i in range(4):
        dummy_method_map[f"data{i}"] = [f"data{i}.dat"]
        data_filepath = dummy_basepath / f"data{i}.dat"
        data_filepath.write_text("")

    # 清除与数据集方法 data0 相关联的文件
    # 同时测试可调用参数的清除方法，而不是调用列表
    def data0():
        pass
    _clear_cache(datasets=data0, cache_dir=dummy_basepath,
                 method_map=dummy_method_map)
    assert not os.path.exists(dummy_basepath/"data0.dat")

    # 清除与数据集方法 data1 和 data2 相关联的文件
    def data1():
        pass

    def data2():
        pass
    _clear_cache(datasets=[data1, data2], cache_dir=dummy_basepath,
                 method_map=dummy_method_map)
    # 确保 "data1.dat" 文件和 "data2.dat" 文件不存在
    assert not os.path.exists(dummy_basepath/"data1.dat")
    assert not os.path.exists(dummy_basepath/"data2.dat")

    # 清除与数据集方法 "data3" 相关联的多个数据集文件 "data3_0.dat" 和 "data3_1.dat"
    def data4():
        pass

    # 创建文件 "data4_0.dat" 和 "data4_1.dat"
    (dummy_basepath / "data4_0.dat").write_text("")
    (dummy_basepath / "data4_1.dat").write_text("")

    # 更新 dummy_method_map，将数据集方法 "data4" 关联到文件列表 ["data4_0.dat", "data4_1.dat"]
    dummy_method_map["data4"] = ["data4_0.dat", "data4_1.dat"]

    # 清除缓存中与 data4 方法相关的数据集，使用指定的缓存目录和方法映射
    _clear_cache(datasets=[data4], cache_dir=dummy_basepath,
                 method_map=dummy_method_map)

    # 确保 "data4_0.dat" 和 "data4_1.dat" 文件不存在
    assert not os.path.exists(dummy_basepath/"data4_0.dat")
    assert not os.path.exists(dummy_basepath/"data4_1.dat")

    # 错误的数据集方法应该引发 ValueError，因为它在 dummy_method_map 中不存在
    def data5():
        pass

    with pytest.raises(ValueError):
        _clear_cache(datasets=[data5], cache_dir=dummy_basepath,
                     method_map=dummy_method_map)

    # 移除所有数据集的缓存
    _clear_cache(datasets=None, cache_dir=dummy_basepath)

    # 确保 dummy_basepath 目录不存在
    assert not os.path.exists(dummy_basepath)
```