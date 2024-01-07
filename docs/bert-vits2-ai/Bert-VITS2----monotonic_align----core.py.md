# `Bert-VITS2\monotonic_align\core.py`

```

# 导入 numba 模块
import numba

# 使用 numba.jit 装饰器对函数进行即时编译优化
@numba.jit(
    # 指定函数参数类型和顺序
    numba.void(
        numba.int32[:, :, ::1],  # 三维 int32 类型数组
        numba.float32[:, :, ::1],  # 三维 float32 类型数组
        numba.int32[::1],  # 一维 int32 类型数组
        numba.int32[::1],  # 一维 int32 类型数组
    ),
    # 指定函数为无 Python 对象的纯函数
    nopython=True,
    # 指定函数为无全局解释器锁的纯函数
    nogil=True,
)
# 定义函数 maximum_path_jit
def maximum_path_jit(paths, values, t_ys, t_xs):
    # 获取 paths 的第一维大小
    b = paths.shape[0]
    # 初始化最大负值
    max_neg_val = -1e9
    # 遍历 paths 的第一维
    for i in range(int(b)):
        # 获取当前路径
        path = paths[i]
        # 获取当前值
        value = values[i]
        # 获取当前 t_y
        t_y = t_ys[i]
        # 获取当前 t_x
        t_x = t_xs[i]

        # 初始化 v_prev 和 v_cur
        v_prev = v_cur = 0.0
        # 初始化索引
        index = t_x - 1

        # 遍历 y
        for y in range(t_y):
            # 遍历 x
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                # 如果 x 等于 y
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[y - 1, x]
                # 如果 x 等于 0
                if x == 0:
                    if y == 0:
                        v_prev = 0.0
                    else:
                        v_prev = max_neg_val
                else:
                    v_prev = value[y - 1, x - 1]
                value[y, x] += max(v_prev, v_cur)

        # 逆序遍历 y
        for y in range(t_y - 1, -1, -1):
            # 将 path[y, index] 设置为 1
            path[y, index] = 1
            # 如果索引不为 0 并且条件成立
            if index != 0 and (
                index == y or value[y - 1, index] < value[y - 1, index - 1]
            ):
                index = index - 1

```