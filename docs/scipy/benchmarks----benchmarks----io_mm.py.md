# `D:\src\scipysrc\scipy\benchmarks\benchmarks\io_mm.py`

```
from .common import set_mem_rlimit, run_monitored, get_mem_info
# 从 common 模块导入设置内存限制、监控运行和获取内存信息的函数

from io import BytesIO, StringIO
# 导入用于处理字节流和字符串的模块

import os
import tempfile
# 导入操作系统和临时文件模块

import numpy as np
# 导入 NumPy 库，用于数值计算

from .common import Benchmark, safe_import
# 从 common 模块导入 Benchmark 类和 safe_import 函数

with safe_import():
    import scipy.sparse
    import scipy.io._mmio
    import scipy.io._fast_matrix_market
    from scipy.io._fast_matrix_market import mmwrite
# 使用 safe_import 函数导入 SciPy 的稀疏矩阵和矩阵市场处理相关模块和函数

def generate_coo(size):
    nnz = int(size / (4 + 4 + 8))
    rows = np.arange(nnz, dtype=np.int32)
    cols = np.arange(nnz, dtype=np.int32)
    data = np.random.default_rng().uniform(low=0, high=1.0, size=nnz)
    return scipy.sparse.coo_matrix((data, (rows, cols)), shape=(nnz, nnz))
# 生成 COO 格式的稀疏矩阵，返回一个 SciPy 的 coo_matrix 对象

def generate_csr(size):
    nrows = 1000
    nnz = int((size - (nrows + 1) * 4) / (4 + 8))
    indptr = (np.arange(nrows + 1, dtype=np.float32) / nrows * nnz).astype(np.int32)
    indptr[-1] = nnz
    indices = np.arange(nnz, dtype=np.int32)
    data = np.random.default_rng().uniform(low=0, high=1.0, size=nnz)
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=(nrows, nnz))
# 生成 CSR 格式的稀疏矩阵，返回一个 SciPy 的 csr_matrix 对象

def generate_dense(size):
    nnz = size // 8
    return np.random.default_rng().uniform(low=0, high=1.0, size=(1, nnz))
# 生成稠密矩阵，返回一个 NumPy 数组

class MemUsage(Benchmark):
    param_names = ['size', 'implementation', 'matrix_type']
    timeout = 4*60
    unit = "actual/optimal memory usage ratio"
    # 定义内存使用情况的基准类，设置参数名、超时时间和单位

    @property
    def params(self):
        return [
            list(self._get_size().keys()),
            ['scipy.io', 'scipy.io._mmio', 'scipy.io._fast_matrix_market'],
            ['dense', 'coo']  # + ['csr']
        ]
    # 定义参数组合的属性，包括测试的大小、实现方式和矩阵类型

    def _get_size(self):
        size = {
            '1M': int(1e6),
            '10M': int(10e6),
            '100M': int(100e6),
            '300M': int(300e6),
            # '500M': int(500e6),
            # '1000M': int(1000e6),
        }
        return size
    # 定义测试的不同大小的字典，以字节为单位

    def setup(self, size, implementation, matrix_type):
        set_mem_rlimit()
        self.size = self._get_size()
        size = self.size[size]

        mem_info = get_mem_info()
        try:
            mem_available = mem_info['memavailable']
        except KeyError:
            mem_available = mem_info['memtotal']

        max_size = int(mem_available * 0.7)//4

        if size > max_size:
            raise NotImplementedError()
        # 设置内存限制，并根据测试大小和可用内存确定最大可接受大小

        # 设置临时文件
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.mtx')
        f.close()
        self.filename = f.name
    # 在临时文件中设置文件名，并确保在 setup 完成后文件未被删除

    def teardown(self, size, implementation, matrix_type):
        os.unlink(self.filename)
    # 在测试完成后删除临时文件
    # 跟踪 mmread 操作的内存使用情况，返回每单位大小的内存消耗
    def track_mmread(self, size, implementation, matrix_type):
        # 获取给定大小的矩阵尺寸
        size = self.size[size]

        # 根据矩阵类型生成相应类型的矩阵 a
        if matrix_type == 'coo':
            a = generate_coo(size)
        elif matrix_type == 'dense':
            a = generate_dense(size)
        elif matrix_type == 'csr':
            # 如果是 csr 类型的矩阵，无法直接读取，返回 0
            return 0
        else:
            # 如果矩阵类型不在支持范围内，抛出未实现错误
            raise NotImplementedError

        # 将生成的矩阵 a 写入到文件 self.filename 中，以通用的对称性形式
        mmwrite(self.filename, a, symmetry='general')
        # 删除变量 a，释放内存
        del a

        # 组装字符串 code，其中包含从指定实现中导入 mmread 并读取 self.filename 文件的操作
        code = f"""
        from {implementation} import mmread
        mmread('{self.filename}')
        """
        # 运行监控代码，记录运行时间和内存峰值
        time, peak_mem = run_monitored(code)
        # 返回内存峰值除以矩阵大小后的结果，表示每单位大小的内存消耗
        return peak_mem / size

    # 跟踪 mmwrite 操作的内存使用情况，返回每单位大小的内存消耗
    def track_mmwrite(self, size, implementation, matrix_type):
        # 获取给定大小的矩阵尺寸
        size = self.size[size]

        # 组装字符串 code，包含从指定实现中导入 mmwrite 并生成特定类型矩阵写入 self.filename 的操作
        code = f"""
        import numpy as np
        import scipy.sparse
        from {implementation} import mmwrite
        
        # 定义生成 coo 类型矩阵的函数
        def generate_coo(size):
            nnz = int(size / (4 + 4 + 8))
            rows = np.arange(nnz, dtype=np.int32)
            cols = np.arange(nnz, dtype=np.int32)
            data = np.random.default_rng().uniform(low=0, high=1.0, size=nnz)
            return scipy.sparse.coo_matrix((data, (rows, cols)), shape=(nnz, nnz))

        # 定义生成 csr 类型矩阵的函数
        def generate_csr(size):
            nrows = 1000
            nnz = int((size - (nrows + 1) * 4) / (4 + 8))
            indptr = (np.arange(nrows + 1, dtype=np.float32) / nrows * nnz).astype(np.int32)
            indptr[-1] = nnz
            indices = np.arange(nnz, dtype=np.int32)
            data = np.random.default_rng().uniform(low=0, high=1.0, size=nnz)
            return scipy.sparse.csr_matrix((data, indices, indptr), shape=(nrows, nnz))
        
        # 定义生成 dense 类型矩阵的函数
        def generate_dense(size):
            nnz = size // 8
            return np.random.default_rng().uniform(low=0, high=1.0, size=(1, nnz))

        # 根据矩阵类型生成相应类型的矩阵 a
        a = generate_{matrix_type}({size})
        # 将生成的矩阵 a 写入到文件 self.filename 中，以通用的对称性形式
        mmwrite('{self.filename}', a, symmetry='general')
        """  # noqa: E501
        # 运行监控代码，记录运行时间和内存峰值
        time, peak_mem = run_monitored(code)
        # 返回内存峰值除以矩阵大小后的结果，表示每单位大小的内存消耗
        return peak_mem / size
class IOSpeed(Benchmark):
    """
    Basic speed test. Does not show full potential as
    1) a relatively small matrix is used to keep test duration reasonable
    2) StringIO/BytesIO are noticeably slower than native C++ I/O to an SSD.
    """
    param_names = ['implementation', 'matrix_type']
    params = [
        ['scipy.io', 'scipy.io._mmio', 'scipy.io._fast_matrix_market'],
        ['dense', 'coo']  # + ['csr']
    ]

    def setup(self, implementation, matrix_type):
        # Use a 10MB matrix size to keep the runtimes somewhat short
        self.size = int(10e6)

        # 根据矩阵类型生成相应的矩阵数据
        if matrix_type == 'coo':
            self.a = generate_coo(self.size)
        elif matrix_type == 'dense':
            self.a = generate_dense(self.size)
        elif matrix_type == 'csr':
            self.a = generate_csr(self.size)
        else:
            raise NotImplementedError

        # 将矩阵数据写入 BytesIO 对象，并获取其字符串表示
        bio = BytesIO()
        mmwrite(bio, self.a, symmetry='general')
        self.a_str = bio.getvalue().decode()

    def time_mmread(self, implementation, matrix_type):
        # 如果矩阵类型为 'csr'，则不能直接读取为 csr 格式，只能是 coo 格式
        if matrix_type == 'csr':
            return

        # 根据实现和矩阵类型选择对应的模块
        if implementation == 'scipy.io':
            impl_module = scipy.io
        elif implementation == 'scipy.io._mmio':
            impl_module = scipy.io._mmio
        elif implementation == 'scipy.io._fast_matrix_market':
            impl_module = scipy.io._fast_matrix_market
        else:
            raise NotImplementedError

        # 使用 mmread 方法读取 StringIO 中的字符串
        impl_module.mmread(StringIO(self.a_str))

    def time_mmwrite(self, implementation, matrix_type):
        # 根据实现选择对应的模块
        if implementation == 'scipy.io':
            impl_module = scipy.io
        elif implementation == 'scipy.io._mmio':
            impl_module = scipy.io._mmio
        elif implementation == 'scipy.io._fast_matrix_market':
            impl_module = scipy.io._fast_matrix_market
        else:
            raise NotImplementedError

        # 使用 mmwrite 方法将矩阵数据写入 BytesIO 对象
        impl_module.mmwrite(BytesIO(), self.a, symmetry='general')
```