# `.\pytorch\benchmarks\dynamo\microbenchmarks\fx_microbenchmarks.py`

```py
# 导入时间测量工具模块
import timeit
# 导入 PyTorch FX 模块
import torch.fx

# 定义常量 N 和 K
N = 100000
K = 1000

# 定义一个函数，生成一个包含大量节点的计算图
def huge_graph():
    # 定义内部函数 fn，对输入 x 进行 N 次正弦函数操作
    def fn(x):
        for _ in range(N):
            x = x.sin()
        return x
    
    # 对 fn 函数进行符号化跟踪，返回 FX 图形式的计算图
    return torch.fx.symbolic_trace(fn)

# 主函数入口
def main():
    # 生成一个大型计算图 g
    g = huge_graph()

    # 定义内部函数 fn，迭代计算图 g 中的所有节点
    def fn():
        for n in g.graph.nodes:
            pass
    
    # 通过多次执行 fn 函数，测量迭代 g 中所有节点所需时间
    t = min(timeit.repeat(fn, number=K, repeat=3))
    # 打印迭代操作耗时信息
    print(f"iterating over {N*K} FX nodes took {t:.1f}s ({N*K/t:.0f} nodes/s)")

# 如果当前脚本作为主程序运行，则执行 main 函数
if __name__ == "__main__":
    main()
```