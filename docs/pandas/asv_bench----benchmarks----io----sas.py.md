# `D:\src\scipysrc\pandas\asv_bench\benchmarks\io\sas.py`

```
# 从 pathlib 模块中导入 Path 类，用于处理文件路径
# 从 pandas 库中导入 read_sas 函数，用于读取 SAS 文件

# 定义 ROOT 变量，表示测试数据文件夹路径，使用 Path 对象获取当前文件的父级目录向上三级，然后拼接测试数据文件夹路径
ROOT = Path(__file__).parents[3] / "pandas" / "tests" / "io" / "sas" / "data"

# 定义 SAS 类
class SAS:
    # 定义方法 time_read_sas7bdat，用于测试读取 test1.sas7bdat 文件的性能
    def time_read_sas7bdat(self):
        read_sas(ROOT / "test1.sas7bdat")

    # 定义方法 time_read_xpt，用于测试读取 paxraw_d_short.xpt 文件的性能
    def time_read_xpt(self):
        read_sas(ROOT / "paxraw_d_short.xpt")

    # 定义方法 time_read_sas7bdat_2，用于测试以块方式读取 0x00controlbyte.sas7bdat.bz2 文件的性能
    def time_read_sas7bdat_2(self):
        # 使用 read_sas 函数以块方式读取 0x00controlbyte.sas7bdat.bz2 文件，并读取第一个块
        next(read_sas(ROOT / "0x00controlbyte.sas7bdat.bz2", chunksize=11000))

    # 定义方法 time_read_sas7bdat_2_chunked，用于测试以块方式读取 0x00controlbyte.sas7bdat.bz2 文件的性能
    def time_read_sas7bdat_2_chunked(self):
        # 使用 read_sas 函数以块方式读取 0x00controlbyte.sas7bdat.bz2 文件，每次读取 1000 条记录，循环读取直到读取了 10 个块为止
        for i, _ in enumerate(
            read_sas(ROOT / "0x00controlbyte.sas7bdat.bz2", chunksize=1000)
        ):
            if i == 10:
                break
```