# `.\PaddleOCR\ppocr\modeling\necks\intracl.py`

```
# 导入 paddle 库
import paddle
# 从 paddle 库中导入 nn 模块
from paddle import nn

# 定义一个 IntraCLBlock 类，继承自 nn.Layer
class IntraCLBlock(nn.Layer):
    # 定义 forward 方法，用于前向传播
    def forward(self, x):
        # 对输入 x 进行通道缩减操作
        x_new = self.conv1x1_reduce_channel(x)

        # 分别对 x_new 进行 7x7 的卷积操作
        x_7_c = self.c_layer_7x7(x_new)
        # 分别对 x_new 进行 7x1 的卷积操作
        x_7_v = self.v_layer_7x1(x_new)
        # 分别对 x_new 进行 1x7 的卷积操作
        x_7_q = self.q_layer_1x7(x_new)
        # 将三个卷积结果相加
        x_7 = x_7_c + x_7_v + x_7_q

        # 分别对 x_7 进行 5x5 的卷积操作
        x_5_c = self.c_layer_5x5(x_7)
        # 分别对 x_7 进行 5x1 的卷积操作
        x_5_v = self.v_layer_5x1(x_7)
        # 分别对 x_7 进行 1x5 的卷积操作
        x_5_q = self.q_layer_1x5(x_7)
        # 将三个卷积结果相加
        x_5 = x_5_c + x_5_v + x_5_q

        # 分别对 x_5 进行 3x3 的卷积操作
        x_3_c = self.c_layer_3x3(x_5)
        # 分别对 x_5 进行 3x1 的卷积操作
        x_3_v = self.v_layer_3x1(x_5)
        # 分别对 x_5 进行 1x3 的卷积操作
        x_3_q = self.q_layer_1x3(x_5)
        # 将三个卷积结果相加
        x_3 = x_3_c + x_3_v + x_3_q

        # 对 x_3 进行通道扩展操作
        x_relation = self.conv1x1_return_channel(x_3)

        # 对 x_relation 进行批归一化操作
        x_relation = self.bn(x_relation)
        # 对 x_relation 进行激活函数 relu 操作
        x_relation = self.relu(x_relation)

        # 返回输入 x 与 x_relation 相加的结果
        return x + x_relation

# 构建包含 num_block 个 IntraCLBlock 实例的列表
def build_intraclblock_list(num_block):
    # 创建一个空的 nn.LayerList 对象
    IntraCLBlock_list = nn.LayerList()
    # 循环 num_block 次，向 IntraCLBlock_list 中添加 IntraCLBlock 实例
    for i in range(num_block):
        IntraCLBlock_list.append(IntraCLBlock())

    # 返回构建好的 IntraCLBlock_list
    return IntraCLBlock_list
```