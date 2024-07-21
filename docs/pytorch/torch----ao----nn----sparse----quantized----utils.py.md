# `.\pytorch\torch\ao\nn\sparse\quantized\utils.py`

```py
# mypy: allow-untyped-defs
# 导入 threading 模块，用于多线程支持
import threading

# 声明该模块中公开的类名列表，仅包含 "LinearBlockSparsePattern"
__all__ = [
    "LinearBlockSparsePattern"
]

# 检查给定的行块大小和列块大小是否是有效的线性块稀疏模式
def _is_valid_linear_block_sparse_pattern(row_block_size, col_block_size):
    return (row_block_size == 1 and col_block_size == 4) or \
           (row_block_size == 8 and col_block_size == 1)

# 这是一个临时措施，因为当前流程不允许模块特定的块稀疏模式。
# 实际上，没有办法通过量化流程的模块配置来传达稀疏模式。
# 因此，使用全局上下文来传达稀疏模式。
# 一旦流程支持，应该删除这段代码。
class LinearBlockSparsePattern:
    # 使用 threading 模块提供的可重入锁
    rlock = threading.RLock()
    # 默认的行块大小和列块大小
    row_block_size = 1
    col_block_size = 4
    # 先前的行块大小和列块大小
    prev_row_block_size = 1
    prev_col_block_size = 4

    # 初始化方法，设置行块大小和列块大小，并进行有效性断言检查
    def __init__(self, row_block_size=1, col_block_size=4):
        assert _is_valid_linear_block_sparse_pattern(row_block_size, col_block_size)
        # 获取全局锁，确保线程安全地修改全局状态
        LinearBlockSparsePattern.rlock.acquire()
        # 更新先前的行块大小和列块大小
        LinearBlockSparsePattern.prev_row_block_size = LinearBlockSparsePattern.row_block_size
        LinearBlockSparsePattern.prev_col_block_size = LinearBlockSparsePattern.col_block_size
        # 更新全局行块大小和列块大小
        LinearBlockSparsePattern.row_block_size = row_block_size
        LinearBlockSparsePattern.col_block_size = col_block_size

    # 进入上下文管理器时调用的方法，暂时不做任何操作
    def __enter__(self):
        pass

    # 退出上下文管理器时调用的方法，恢复先前的行块大小和列块大小，并释放全局锁
    def __exit__(self, exc_type, exc_value, backtrace):
        LinearBlockSparsePattern.row_block_size = LinearBlockSparsePattern.prev_row_block_size
        LinearBlockSparsePattern.col_block_size = LinearBlockSparsePattern.prev_col_block_size
        LinearBlockSparsePattern.rlock.release()

    # 返回当前的行块大小和列块大小的静态方法
    @staticmethod
    def block_size():
        return LinearBlockSparsePattern.row_block_size, LinearBlockSparsePattern.col_block_size
```