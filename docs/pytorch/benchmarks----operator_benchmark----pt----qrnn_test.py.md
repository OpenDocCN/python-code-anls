# `.\pytorch\benchmarks\operator_benchmark\pt\qrnn_test.py`

```py
# 导入 operator_benchmark 库作为 op_bench 别名
import operator_benchmark as op_bench
# 导入 torch 库
import torch
# 从 torch 库中导入 nn（神经网络）模块
from torch import nn

"""
RNN 微基准测试。
"""

# 定义 qrnn_configs，包含多个配置项的列表
qrnn_configs = op_bench.config_list(
    attrs=[
        [1, 3, 1],    # 输入大小为 1，隐藏大小为 3，层数为 1
        [5, 7, 4],    # 输入大小为 5，隐藏大小为 7，层数为 4
    ],
    # 设置属性名称对应的名称列表
    attr_names=["I", "H", "NL"],
    # 使用交叉乘积方式生成配置项
    cross_product_configs={
        "B": (True,),             # 偏置始终为 True（针对量化）
        "D": (False, True),       # 双向
        "dtype": (torch.qint8,),  # 目前仅支持 qint8 数据类型
    },
    # 添加标签 "short"
    tags=["short"],
)

# 定义 LSTMBenchmark 类，继承自 op_bench.TorchBenchmarkBase
class LSTMBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, I, H, NL, B, D, dtype):
        sequence_len = 128   # 序列长度
        batch_size = 16      # 批量大小

        # 由于 quantized.dynamic.LSTM 存在问题，因此创建常规 LSTM，并稍后进行量化。参见问题 #31192。
        scale = 1.0 / 256
        zero_point = 0
        # 创建 nn.LSTM 实例 cell_nn
        cell_nn = nn.LSTM(
            input_size=I,         # 输入大小
            hidden_size=H,        # 隐藏大小
            num_layers=NL,        # 层数
            bias=B,               # 是否使用偏置
            batch_first=False,    # 是否批量优先
            dropout=0.0,          # 丢弃率为 0.0
            bidirectional=D,      # 是否双向
        )
        # 使用 nn.Sequential 封装 cell_nn 到 cell_temp
        cell_temp = nn.Sequential(cell_nn)
        # 使用 torch.ao.quantization.quantize_dynamic 对 cell_temp 进行动态量化
        self.cell = torch.ao.quantization.quantize_dynamic(
            cell_temp, {nn.LSTM, nn.Linear}, dtype=dtype
        )[0]

        # 创建随机张量 x，表示输入数据
        x = torch.randn(
            sequence_len, batch_size, I  # 序列长度，批量大小，特征数
        )  # X 中的特征数
        # 创建随机张量 h，表示隐藏状态 h
        h = torch.randn(
            NL * (D + 1), batch_size, H  # 层数 * 双向数，批量大小，隐藏大小
        )  # 隐藏大小
        # 创建随机张量 c，表示细胞状态 c
        c = torch.randn(
            NL * (D + 1), batch_size, H  # 层数 * 双向数，批量大小，隐藏大小
        )  # 隐藏大小

        # 设置 inputs 字典，包含输入 x、隐藏状态 h 和细胞状态 c
        self.inputs = {"x": x, "h": h, "c": c}
        # 设置模块名称为 "QLSTM"
        self.set_module_name("QLSTM")

    # 前向传播函数，接受输入 x、隐藏状态 h 和细胞状态 c，返回 LSTM 的输出
    def forward(self, x, h, c):
        return self.cell(x, (h, c))[0]

# 使用 op_bench.generate_pt_test 生成基准测试
op_bench.generate_pt_test(qrnn_configs, LSTMBenchmark)

# 如果当前脚本作为主程序运行，调用 op_bench.benchmark_runner.main() 运行基准测试
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```