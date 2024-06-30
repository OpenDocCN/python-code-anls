# `D:\src\scipysrc\scipy\scipy\io\tests\test_paths.py`

```
"""
Ensure that we can use pathlib.Path objects in all relevant IO functions.
"""
# 导入必要的模块
from pathlib import Path

# 导入科学计算库及其子模块
import numpy as np

import scipy.io
import scipy.io.wavfile
from scipy._lib._tmpdirs import tempdir
import scipy.sparse


class TestPaths:
    # 创建一个测试类 TestPaths
    data = np.arange(5).astype(np.int64)

    def test_savemat(self):
        # 在临时目录中保存数据到 MATLAB 格式的 .mat 文件
        with tempdir() as temp_dir:
            # 创建保存数据的路径对象 path
            path = Path(temp_dir) / 'data.mat'
            # 使用 scipy.io.savemat 保存数据到指定路径
            scipy.io.savemat(path, {'data': self.data})
            # 断言路径 path 是一个文件
            assert path.is_file()

    def test_loadmat(self):
        # 使用字符串路径保存数据，并使用 pathlib.Path 加载数据
        with tempdir() as temp_dir:
            # 创建保存数据的路径对象 path
            path = Path(temp_dir) / 'data.mat'
            # 使用字符串路径保存数据到 .mat 文件
            scipy.io.savemat(str(path), {'data': self.data})

            # 使用 scipy.io.loadmat 加载数据
            mat_contents = scipy.io.loadmat(path)
            # 断言加载的数据内容与预期一致
            assert (mat_contents['data'] == self.data).all()

    def test_whosmat(self):
        # 使用字符串路径保存数据，并使用 pathlib.Path 加载数据
        with tempdir() as temp_dir:
            # 创建保存数据的路径对象 path
            path = Path(temp_dir) / 'data.mat'
            # 使用字符串路径保存数据到 .mat 文件
            scipy.io.savemat(str(path), {'data': self.data})

            # 使用 scipy.io.whosmat 获取 .mat 文件的变量信息
            contents = scipy.io.whosmat(path)
            # 断言第一个变量的信息与预期一致
            assert contents[0] == ('data', (1, 5), 'int64')

    def test_readsav(self):
        # 使用当前文件所在目录下的数据文件进行读取
        path = Path(__file__).parent / 'data/scalar_string.sav'
        scipy.io.readsav(path)

    def test_hb_read(self):
        # 使用字符串路径保存数据，并使用 pathlib.Path 加载数据
        with tempdir() as temp_dir:
            # 创建稀疏矩阵数据
            data = scipy.sparse.csr_matrix(scipy.sparse.eye(3))
            # 创建保存数据的路径对象 path
            path = Path(temp_dir) / 'data.hb'
            # 使用 scipy.io.hb_write 保存数据到 .hb 文件
            scipy.io.hb_write(str(path), data)

            # 使用 scipy.io.hb_read 加载 .hb 文件中的数据
            data_new = scipy.io.hb_read(path)
            # 断言加载的数据与原始数据一致
            assert (data_new != data).nnz == 0

    def test_hb_write(self):
        # 使用当前测试临时目录下的 .hb 文件写入稀疏矩阵数据
        with tempdir() as temp_dir:
            # 创建稀疏矩阵数据
            data = scipy.sparse.csr_matrix(scipy.sparse.eye(3))
            # 创建保存数据的路径对象 path
            path = Path(temp_dir) / 'data.hb'
            # 使用 scipy.io.hb_write 将数据写入 .hb 文件
            scipy.io.hb_write(path, data)
            # 断言路径 path 是一个文件
            assert path.is_file()

    def test_mmio_read(self):
        # 使用字符串路径保存数据，并使用 pathlib.Path 加载数据
        with tempdir() as temp_dir:
            # 创建稀疏矩阵数据
            data = scipy.sparse.csr_matrix(scipy.sparse.eye(3))
            # 创建保存数据的路径对象 path
            path = Path(temp_dir) / 'data.mtx'
            # 使用 scipy.io.mmwrite 将数据保存到 .mtx 文件
            scipy.io.mmwrite(str(path), data)

            # 使用 scipy.io.mmread 加载 .mtx 文件中的数据
            data_new = scipy.io.mmread(path)
            # 断言加载的数据与原始数据一致
            assert (data_new != data).nnz == 0

    def test_mmio_write(self):
        # 使用当前测试临时目录下的 .mtx 文件写入稀疏矩阵数据
        with tempdir() as temp_dir:
            # 创建稀疏矩阵数据
            data = scipy.sparse.csr_matrix(scipy.sparse.eye(3))
            # 创建保存数据的路径对象 path
            path = Path(temp_dir) / 'data.mtx'
            # 使用 scipy.io.mmwrite 将数据写入 .mtx 文件
            scipy.io.mmwrite(path, data)

    def test_netcdf_file(self):
        # 使用当前文件所在目录下的 NetCDF 数据文件进行读取
        path = Path(__file__).parent / 'data/example_1.nc'
        scipy.io.netcdf_file(path)

    def test_wavfile_read(self):
        # 使用当前文件所在目录下的 WAV 音频文件进行读取
        path = Path(__file__).parent / 'data/test-8000Hz-le-2ch-1byteu.wav'
        scipy.io.wavfile.read(path)
    # 定义一个测试函数，用于测试 WAV 文件写入功能
    def test_wavfile_write(self):
        # 从当前文件路径推断出输入文件路径，并构建 Path 对象
        input_path = Path(__file__).parent / 'data/test-8000Hz-le-2ch-1byteu.wav'
        # 使用 scipy 库读取 WAV 文件，获取采样率和数据
        rate, data = scipy.io.wavfile.read(str(input_path))

        # 使用 tempdir() 创建临时目录，在该目录下构建输出文件路径
        with tempdir() as temp_dir:
            output_path = Path(temp_dir) / input_path.name
            # 使用 scipy 库将数据写入到指定路径的 WAV 文件中
            scipy.io.wavfile.write(output_path, rate, data)
```