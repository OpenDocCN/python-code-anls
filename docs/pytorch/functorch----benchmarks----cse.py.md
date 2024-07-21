# `.\pytorch\functorch\benchmarks\cse.py`

```
# 导入PyTorch库
import torch
# 导入PyTorch的FX模块
import torch.fx as fx

# 导入functorch库中的make_fx函数
from functorch import make_fx

# 导入torch._functorch.compile_utils模块中的fx_graph_cse函数
from torch._functorch.compile_utils import fx_graph_cse
# 导入torch.profiler模块中的profile和ProfilerActivity类
from torch.profiler import profile, ProfilerActivity


# 定义函数profile_it，用于对给定函数f和输入inp进行CUDA性能分析
def profile_it(f, inp):
    # 执行函数f(inp)五次，预热GPU
    for _ in range(5):
        f(inp)

    # 定义迭代次数itr为5，使用torch.profiler进行CUDA性能分析
    itr = 5
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(itr):
            f(inp)

    # 获取平均时间统计信息
    timing = prof.key_averages()
    cuda_time_total = 0
    for e in timing:
        cuda_time_total = cuda_time_total + e.cuda_time_total
    # 返回平均CUDA执行时间
    return cuda_time_total / itr


# 定义函数profile_function，用于对给定函数f进行性能分析，并打印分析结果
def profile_function(name, f, inp):
    # 使用make_fx函数将函数f转换为FX图
    fx_g = make_fx(f)(inp)

    # 对FX图进行公共子表达式消除（CSE）
    new_g = fx_graph_cse(fx_g.graph)
    new_g = fx.GraphModule(fx_g, new_g)
    
    # 分别对原始FX图和经过CSE处理的FX图进行性能分析
    avg_cuda_time_f = profile_it(fx_g, inp)
    avg_cuda_time_g = profile_it(new_g, inp)
    
    # 计算节点减少数量
    num_node_decrease = len(fx_g.graph.nodes) - len(new_g.graph.nodes)

    # 打印结果：函数名，原始FX图CUDA平均时间，经CSE处理后FX图CUDA平均时间，节点减少数量，原始FX图节点数量
    print(
        f"{name}, {avg_cuda_time_f}, {avg_cuda_time_g}, {num_node_decrease}, {len(fx_g.graph.nodes)}"
    )


# 在GPU上创建一个随机数生成器g_gpu
g_gpu = torch.Generator(device="cuda")
g_gpu.manual_seed(2147483647)
# 生成一个在GPU上的随机输入inp，大小为2^20
inp = torch.randn(2**20, device="cuda", generator=g_gpu)


# 定义函数f1，对输入x进行两次余弦函数的嵌套操作
def f1(x):
    return x.cos().cos()

# 对函数f1进行性能分析
profile_function("f1", f1, inp)


# 定义函数fsum，对输入x进行四次求和操作
def fsum(x):
    a = x.sum()
    b = x.sum()
    c = x.sum()
    d = x.sum()
    return a + b + c + d

# 对函数fsum进行性能分析
profile_function("fsum", fsum, inp)


# 定义函数fconcat，将输入x和自身拼接，并返回其和
def fconcat(x):
    a = torch.cat((x, x))
    b = torch.cat((x, x))
    return a + b

# 对函数fconcat进行性能分析
profile_function("fconcat", fconcat, inp)


# 定义函数fsum2，对输入x进行多次求和操作
def fsum2(x):
    a = x.sum()
    for _ in range(30):
        a = a + x.sum()
    return a

# 对函数fsum2进行性能分析
profile_function("fsum2", fsum2, inp)


# 定义函数fsummulti，对输入x进行多次求和和乘法操作
def fsummulti(x):
    a = 0
    for _ in range(3):
        a = a + x.sum()
        a = a * x.sum()
    return a

# 对函数fsummulti进行性能分析
profile_function("fsummulti", fsummulti, inp)


# 定义函数fsummulti2，对输入x进行多次求和和乘法操作
def fsummulti2(x):
    a = 0
    for _ in range(30):
        a = a + x.sum()
        a = a * x.sum()
    return a

# 对函数fsummulti2进行性能分析
profile_function("fsummulti2", fsummulti2, inp)


# 定义函数fcos，对输入x进行多次余弦函数操作
def fcos(x):
    a = 0
    for _ in range(3):
        a = a + x.cos()
    return a

# 对函数fcos进行性能分析
profile_function("fcos", fcos, inp)


# 定义函数fcos2，对输入x进行多次余弦函数操作
def fcos2(x):
    a = 0
    for _ in range(30):
        a = a + x.cos()
    return a

# 对函数fcos2进行性能分析
profile_function("fcos2", fcos2, inp)
```