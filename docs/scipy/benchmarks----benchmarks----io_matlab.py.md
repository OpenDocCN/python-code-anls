# `D:\src\scipysrc\scipy\benchmarks\benchmarks\io_matlab.py`

```
# 从当前目录下的 common 模块导入 set_mem_rlimit, run_monitored, get_mem_info 函数
# 以及从 os、tempfile、io 模块导入其他必要的函数和类
from .common import set_mem_rlimit, run_monitored, get_mem_info
import os
import tempfile
from io import BytesIO

import numpy as np
# 从 common 模块导入 Benchmark 和 safe_import 函数
from .common import Benchmark, safe_import

# 使用安全导入机制导入 scipy.io 中的 savemat 和 loadmat 函数
with safe_import():
    from scipy.io import savemat, loadmat

# Benchmark 类的子类 MemUsage，用于测量内存使用情况
class MemUsage(Benchmark):
    # 参数化测试的参数名称和超时设定
    param_names = ['size', 'compressed']
    timeout = 4*60
    unit = "actual/optimal memory usage ratio"

    # 参数生成器，定义了测试用例的参数组合
    @property
    def params(self):
        return [list(self._get_sizes().keys()), [True, False]]

    # 内部方法，返回不同尺寸的数据大小字典
    def _get_sizes(self):
        sizes = {
            '1M': 1e6,
            '10M': 10e6,
            '100M': 100e6,
            '300M': 300e6,
            # '500M': 500e6,
            # '1000M': 1000e6,
        }
        return sizes

    # 设置测试的前置条件，包括内存限制设置和临时文件的创建
    def setup(self, size, compressed):
        set_mem_rlimit()  # 设置内存限制
        self.sizes = self._get_sizes()  # 获取数据大小字典
        size = int(self.sizes[size])  # 将尺寸参数转换为整数

        mem_info = get_mem_info()  # 获取内存信息
        try:
            mem_available = mem_info['memavailable']
        except KeyError:
            mem_available = mem_info['memtotal']

        max_size = int(mem_available * 0.7)//4  # 计算可用内存的70%并除以4

        if size > max_size:
            raise NotImplementedError()  # 如果数据大小超过可用内存的70%，抛出未实现错误

        # 设置临时文件
        f = tempfile.NamedTemporaryFile(delete=False, suffix='.mat')
        f.close()  # 关闭临时文件
        self.filename = f.name  # 记录临时文件名

    # 清理测试的后置条件，删除临时文件
    def teardown(self, size, compressed):
        os.unlink(self.filename)  # 删除临时文件

    # 追踪 loadmat 函数的性能，加载临时文件中的数据并测量内存峰值
    def track_loadmat(self, size, compressed):
        size = int(self.sizes[size])  # 将尺寸参数转换为整数

        x = np.random.rand(size//8).view(dtype=np.uint8)  # 生成随机数据 x
        savemat(self.filename, dict(x=x), do_compression=compressed, oned_as='row')  # 保存数据到临时文件

        del x  # 删除 x 变量

        # 准备动态生成的代码块，加载临时文件并测量其执行时间和内存峰值
        code = f"""
        from scipy.io import loadmat
        loadmat('{self.filename}')
        """
        time, peak_mem = run_monitored(code)  # 运行监控代码

        return peak_mem / size  # 返回内存峰值与数据大小之比

    # 追踪 savemat 函数的性能，将随机生成的数据保存到临时文件并测量内存峰值
    def track_savemat(self, size, compressed):
        size = int(self.sizes[size])  # 将尺寸参数转换为整数

        # 准备动态生成的代码块，生成随机数据并调用 savemat 函数保存到临时文件
        code = """
        import numpy as np
        from scipy.io import savemat
        x = np.random.rand(%d//8).view(dtype=np.uint8)
        savemat('%s', dict(x=x), do_compression=%r, oned_as='row')
        """ % (size, self.filename, compressed)
        time, peak_mem = run_monitored(code)  # 运行监控代码获取性能数据
        return peak_mem / size  # 返回内存峰值与数据大小之比


# Benchmark 类的子类 StructArr，用于测试结构化数组的性能
class StructArr(Benchmark):
    # 参数化测试的参数列表和参数名称
    params = [
        [(10, 10, 20), (20, 20, 40), (30, 30, 50)],
        [False, True]
    ]
    param_names = ['(vars, fields, structs)', 'compression']

    # 静态方法，用于生成指定大小的结构化数组
    @staticmethod
    def make_structarr(n_vars, n_fields, n_structs):
        var_dict = {}
        for vno in range(n_vars):
            vname = 'var%00d' % vno
            end_dtype = [('f%d' % d, 'i4', 10) for d in range(n_fields)]
            s_arrs = np.zeros((n_structs,), dtype=end_dtype)
            var_dict[vname] = s_arrs
        return var_dict
    # 初始化对象的设置方法，接受变量数目、字段数目和结构数目作为输入参数
    def setup(self, nvfs, compression):
        # 解包 nvfs 元组，分别获取变量数目、字段数目和结构数目
        n_vars, n_fields, n_structs = nvfs

        # 使用 StructArr 类的方法创建一个结构数组，并赋值给对象的 var_dict 属性
        self.var_dict = StructArr.make_structarr(n_vars, n_fields, n_structs)
        
        # 初始化一个 BytesIO 对象并赋值给对象的 str_io 属性
        self.str_io = BytesIO()

        # 调用 savemat 函数将 var_dict 对象保存到 self.str_io 中，根据 compression 参数决定是否压缩
        savemat(self.str_io, self.var_dict, do_compression=compression)

    # 测试保存数据操作的性能方法
    def time_savemat(self, nvfs, compression):
        # 调用 savemat 函数将对象的 var_dict 属性保存到 self.str_io 中，根据 compression 参数决定是否压缩
        savemat(self.str_io, self.var_dict, do_compression=compression)

    # 测试加载数据操作的性能方法
    def time_loadmat(self, nvfs, compression):
        # 调用 loadmat 函数从 self.str_io 中加载数据
        loadmat(self.str_io)
```