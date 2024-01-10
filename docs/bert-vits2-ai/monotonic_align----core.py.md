# `Bert-VITS2\monotonic_align\core.py`

```
# 导入 numba 模块
import numba

# 使用 numba.jit 装饰器，指定函数签名和编译选项
@numba.jit(
    numba.void(
        numba.int32[:, :, ::1],
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.int32[::1],
    ),
    nopython=True,
    nogil=True,
)
# 定义函数 maximum_path_jit，接受路径、数值、目标 y 和目标 x 作为参数
def maximum_path_jit(paths, values, t_ys, t_xs):
    # 获取路径的维度
    b = paths.shape[0]
    # 初始化最大负值
    max_neg_val = -1e9
    # 遍历路径
    for i in range(int(b)):
        # 获取当前路径、数值、目标 y 和目标 x
        path = paths[i]
        value = values[i]
        t_y = t_ys[i]
        t_x = t_xs[i]

        # 初始化当前值和前一个值
        v_prev = v_cur = 0.0
        # 初始化索引
        index = t_x - 1

        # 遍历路径中的每个点
        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                # 如果 x 等于 y，则当前值为最大负值，否则为数值中的对应值
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[y - 1, x]
                # 如果 x 等于 0，则前一个值为 0.0，否则为数值中的对应值
                if x == 0:
                    if y == 0:
                        v_prev = 0.0
                    else:
                        v_prev = max_neg_val
                else:
                    v_prev = value[y - 1, x - 1]
                # 更新数值
                value[y, x] += max(v_prev, v_cur)

        # 逆向遍历路径
        for y in range(t_y - 1, -1, -1):
            # 标记路径上的点
            path[y, index] = 1
            # 更新索引
            if index != 0 and (
                index == y or value[y - 1, index] < value[y - 1, index - 1]
            ):
                index = index - 1
```