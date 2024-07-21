# `.\pytorch\benchmarks\operator_benchmark\pt\unary_test.py`

```
# 导入名为 operator_benchmark 的模块，并使用别名 op_bench
import operator_benchmark as op_bench
# 导入 PyTorch 库
import torch

"""
用于逐点一元运算符的微基准。
"""

# 针对逐点一元操作的配置
unary_ops_configs_short = op_bench.config_list(
    # 定义属性名称和属性值
    attr_names=["M", "N"],
    attrs=[
        [512, 512],
    ],
    # 定义属性的交叉组合配置
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    # 标记为 "short"
    tags=["short"],
)

# 更长的逐点一元操作配置
unary_ops_configs_long = op_bench.cross_product_configs(
    # 定义 M 和 N 的不同值
    M=[256, 1024], N=[256, 1024], device=["cpu", "cuda"], tags=["long"]
)


# 定义 UnaryOpBenchmark 类，继承自 op_bench.TorchBenchmarkBase 类
class UnaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device, op_func):
        # 初始化输入字典，包含一个随机生成的 MxN 大小的张量，放在指定的设备上
        self.inputs = {"input": torch.rand(M, N, device=device)}
        # 设置操作函数
        self.op_func = op_func

    def forward(self, input):
        # 执行操作函数并返回结果
        return self.op_func(input)


# 下面是一系列的操作函数，每个函数接受一个 input 参数，表示输入张量，并执行相应的操作

def bernoulli_(input):
    return input.bernoulli_()

def cauchy_(input):
    return input.cauchy_()

def digamma_(input):
    return input.digamma_()

def exponential_(input):
    return input.exponential_()

def normal_(input):
    return input.normal_()

def random_(input):
    return input.random_()

def sign_(input):
    return input.sign_()

def uniform_(input):
    return input.uniform_()

def half_(input):
    return input.half_()

def long_(input):
    return input.long_()

# 创建包含操作名称和对应操作函数的列表
unary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    # 定义一个包含名称和对应函数或方法的列表
    attrs=[
        # ["abs", torch.abs] 表示 "abs" 方法由 torch 模块的 abs 函数提供
        ["abs", torch.abs],
        # ["abs_", torch.abs_] 表示 "abs_" 方法由 torch 模块的 abs_ 函数提供
        ["abs_", torch.abs_],
        # ["acos", torch.acos] 表示 "acos" 方法由 torch 模块的 acos 函数提供
        ["acos", torch.acos],
        # ["acos_", torch.acos_] 表示 "acos_" 方法由 torch 模块的 acos_ 函数提供
        ["acos_", torch.acos_],
        # ["argsort", torch.argsort] 表示 "argsort" 方法由 torch 模块的 argsort 函数提供
        ["argsort", torch.argsort],
        # ["asin", torch.asin] 表示 "asin" 方法由 torch 模块的 asin 函数提供
        ["asin", torch.asin],
        # ["asin_", torch.asin_] 表示 "asin_" 方法由 torch 模块的 asin_ 函数提供
        ["asin_", torch.asin_],
        # ["atan", torch.atan] 表示 "atan" 方法由 torch 模块的 atan 函数提供
        ["atan", torch.atan],
        # ["atan_", torch.atan_] 表示 "atan_" 方法由 torch 模块的 atan_ 函数提供
        ["atan_", torch.atan_],
        # ["ceil", torch.ceil] 表示 "ceil" 方法由 torch 模块的 ceil 函数提供
        ["ceil", torch.ceil],
        # ["ceil_", torch.ceil_] 表示 "ceil_" 方法由 torch 模块的 ceil_ 函数提供
        ["ceil_", torch.ceil_],
        # ["clone", torch.clone] 表示 "clone" 方法由 torch 模块的 clone 函数提供
        ["clone", torch.clone],
        # ["cos", torch.cos] 表示 "cos" 方法由 torch 模块的 cos 函数提供
        ["cos", torch.cos],
        # ["cos_", torch.cos_] 表示 "cos_" 方法由 torch 模块的 cos_ 函数提供
        ["cos_", torch.cos_],
        # ["cosh", torch.cosh] 表示 "cosh" 方法由 torch 模块的 cosh 函数提供
        ["cosh", torch.cosh],
        # ["digamma", torch.digamma] 表示 "digamma" 方法由 torch 模块的 digamma 函数提供
        ["digamma", torch.digamma],
        # ["erf", torch.erf] 表示 "erf" 方法由 torch 模块的 erf 函数提供
        ["erf", torch.erf],
        # ["erf_", torch.erf_] 表示 "erf_" 方法由 torch 模块的 erf_ 函数提供
        ["erf_", torch.erf_],
        # ["erfc", torch.erfc] 表示 "erfc" 方法由 torch 模块的 erfc 函数提供
        ["erfc", torch.erfc],
        # ["erfc_", torch.erfc_] 表示 "erfc_" 方法由 torch 模块的 erfc_ 函数提供
        ["erfc_", torch.erfc_],
        # ["erfinv", torch.erfinv] 表示 "erfinv" 方法由 torch 模块的 erfinv 函数提供
        ["erfinv", torch.erfinv],
        # ["exp", torch.exp] 表示 "exp" 方法由 torch 模块的 exp 函数提供
        ["exp", torch.exp],
        # ["exp_", torch.exp_] 表示 "exp_" 方法由 torch 模块的 exp_ 函数提供
        ["exp_", torch.exp_],
        # ["expm1", torch.expm1] 表示 "expm1" 方法由 torch 模块的 expm1 函数提供
        ["expm1", torch.expm1],
        # ["expm1_", torch.expm1_] 表示 "expm1_" 方法由 torch 模块的 expm1_ 函数提供
        ["expm1_", torch.expm1_],
        # ["floor", torch.floor] 表示 "floor" 方法由 torch 模块的 floor 函数提供
        ["floor", torch.floor],
        # ["floor_", torch.floor_] 表示 "floor_" 方法由 torch 模块的 floor_ 函数提供
        ["floor_", torch.floor_],
        # ["frac", torch.frac] 表示 "frac" 方法由 torch 模块的 frac 函数提供
        ["frac", torch.frac],
        # ["frac_", torch.frac_] 表示 "frac_" 方法由 torch 模块的 frac_ 函数提供
        ["frac_", torch.frac_],
        # ["hardshrink", torch.hardshrink] 表示 "hardshrink" 方法由 torch 模块的 hardshrink 函数提供
        ["hardshrink", torch.hardshrink],
        # ["lgamma", torch.lgamma] 表示 "lgamma" 方法由 torch 模块的 lgamma 函数提供
        ["lgamma", torch.lgamma],
        # ["log", torch.log] 表示 "log" 方法由 torch 模块的 log 函数提供
        ["log", torch.log],
        # ["log10", torch.log10] 表示 "log10" 方法由 torch 模块的 log10 函数提供
        ["log10", torch.log10],
        # ["log10_", torch.log10_] 表示 "log10_" 方法由 torch 模块的 log10_ 函数提供
        ["log10_", torch.log10_],
        # ["log1p", torch.log1p] 表示 "log1p" 方法由 torch 模块的 log1p 函数提供
        ["log1p", torch.log1p],
        # ["log1p_", torch.log1p_] 表示 "log1p_" 方法由 torch 模块的 log1p_ 函数提供
        ["log1p_", torch.log1p_],
        # ["log2", torch.log2] 表示 "log2" 方法由 torch 模块的 log2 函数提供
        ["log2", torch.log2],
        # ["log2_", torch.log2_] 表示 "log2_" 方法由 torch 模块的 log2_ 函数提供
        ["log2_", torch.log2_],
        # ["log_", torch.log_] 表示 "log_" 方法由 torch 模块的 log_ 函数提供
        ["log_", torch.log_],
        # ["logit", torch.logit] 表示 "logit" 方法由 torch 模块的 logit 函数提供
        ["logit", torch.logit],
        # ["logit_", torch.logit_] 表示 "logit_" 方法由 torch 模块的 logit_ 函数提供
        ["logit_", torch.logit_],
        # ["neg", torch.neg] 表示 "neg" 方法由 torch 模块的 neg 函数提供
        ["neg", torch.neg],
        # ["neg_", torch.neg_] 表示 "neg_" 方法由 torch 模块的 neg_ 函数提供
        ["neg_", torch.neg_],
        # ["reciprocal", torch.reciprocal] 表示 "reciprocal" 方法由 torch 模块的 reciprocal 函数提供
        ["reciprocal", torch.reciprocal],
        # ["reciprocal_", torch.reciprocal_] 表示 "reciprocal_" 方法由 torch 模块的 reciprocal_ 函数提供
        ["reciprocal_", torch.reciprocal_],
        # ["relu", torch.relu] 表示 "relu" 方法由 torch 模块的 relu 函数提供
        ["relu", torch.relu],
        # ["relu_", torch.relu_] 表示 "relu_" 方法由 torch 模块的 relu_ 函数提供
        ["relu_", torch.relu_],
        # ["round", torch.round] 表示 "round" 方法由 torch 模块的 round 函数提供
        ["round", torch.round],
        # ["round_", torch.round_] 表示 "round_" 方法由 torch 模块的 round_ 函数提供
        ["round_", torch.round_],
        # ["rsqrt", torch.rsqrt] 表示 "rsqrt" 方法由 torch 模块的 rsqrt 函数提供
        ["rsqrt", torch.rsqrt],
        # ["rsqrt_", torch.rsqrt_] 表示 "rsqrt_" 方法由 torch 模块的 rsqrt_ 函数提供
        ["rsqrt_",
)

op_bench.generate_pt_tests_from_op_list(
    unary_ops_list, unary_ops_configs_short + unary_ops_configs_long, UnaryOpBenchmark
)



# 调用 op_bench 模块中的 generate_pt_tests_from_op_list 函数，生成基于给定操作列表的性能测试用例
# 参数 unary_ops_list：一元操作列表
# 参数 unary_ops_configs_short + unary_ops_configs_long：包含短和长配置的一元操作配置列表
# 参数 UnaryOpBenchmark：一元操作基准类，用于执行性能测试



if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块
    op_bench.benchmark_runner.main()



# 调用 op_bench 模块中的 benchmark_runner 模块的 main 函数，启动性能基准测试运行器
# 这将运行性能测试并输出结果
```