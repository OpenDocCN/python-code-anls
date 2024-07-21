# `.\pytorch\torch\backends\cudnn\rnn.py`

```
# 引入torch.cuda模块，声明允许未类型化的函数定义
import torch.cuda

# 尝试导入torch._C._cudnn模块，用于处理cuDNN相关操作
try:
    from torch._C import _cudnn
except ImportError:
    # 如果导入失败，则_cudnn设为None，这些函数的使用应受torch.backends.cudnn.is_available()保护，
    # 因此在此处不进行任何检查是安全的。
    _cudnn = None  # type: ignore[assignment]

# 根据给定的模式返回相应的cuDNN模式整数值
def get_cudnn_mode(mode):
    if mode == "RNN_RELU":
        return int(_cudnn.RNNMode.rnn_relu)
    elif mode == "RNN_TANH":
        return int(_cudnn.RNNMode.rnn_tanh)
    elif mode == "LSTM":
        return int(_cudnn.RNNMode.lstm)
    elif mode == "GRU":
        return int(_cudnn.RNNMode.gru)
    else:
        # 如果模式未知，则抛出异常
        raise Exception(f"Unknown mode: {mode}")  # noqa: TRY002

# 注意：实际上不再需要这个类（事实上，我们可以序列化dropout状态以获得更好的可重现性），
# 但为了向后兼容旧模型，我们保留了它。
class Unserializable:
    def __init__(self, inner):
        self.inner = inner

    def get(self):
        return self.inner

    def __getstate__(self):
        # 注意：不能返回{}，因为Python2不会调用__setstate__如果值评估为False
        return "<unserializable>"

    def __setstate__(self, state):
        self.inner = None

# 初始化dropout状态
def init_dropout_state(dropout, train, dropout_seed, dropout_state):
    # 获取当前设备的dropout描述符名称
    dropout_desc_name = "desc_" + str(torch.cuda.current_device())
    # 如果是训练模式，则设置dropout概率；否则将dropout概率设为0
    dropout_p = dropout if train else 0
    # 如果dropout状态字典中不存在当前设备的描述符，或者描述符对应的值为None，则进行初始化
    if (dropout_desc_name not in dropout_state) or (
        dropout_state[dropout_desc_name].get() is None
    ):
        # 如果dropout概率为0，则将dropout状态设置为Unserializable(None)
        if dropout_p == 0:
            dropout_state[dropout_desc_name] = Unserializable(None)
        else:
            # 否则，使用torch._cudnn_init_dropout_state初始化dropout状态
            dropout_state[dropout_desc_name] = Unserializable(
                torch._cudnn_init_dropout_state(  # type: ignore[call-arg]
                    dropout_p,
                    train,
                    dropout_seed,
                    self_ty=torch.uint8,
                    device=torch.device("cuda"),
                )
            )
    # 获取当前设备的dropout状态
    dropout_ts = dropout_state[dropout_desc_name].get()
    return dropout_ts
```