# `.\PaddleOCR\test_tipc\supplementary\custom_op\test.py`

```py
# 导入 PaddlePaddle 深度学习框架相关模块
import paddle
import paddle.nn as nn
from paddle.vision.transforms import Compose, Normalize
from paddle.utils.cpp_extension import load
from paddle.inference import Config
from paddle.inference import create_predictor
import numpy as np

# 定义训练轮数和批量大小
EPOCH_NUM = 4
BATCH_SIZE = 64

# 使用 JIT 编译自定义操作
custom_ops = load(
    name="custom_jit_ops", sources=["custom_relu_op.cc", "custom_relu_op.cu"])

# 定义 LeNet 网络结构
class LeNet(nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2D(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2D(
            in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = custom_ops.custom_relu(x)
        x = self.max_pool1(x)
        x = custom_ops.custom_relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = custom_ops.custom_relu(x)
        x = self.linear2(x)
        x = custom_ops.custom_relu(x)
        x = self.linear3(x)
        return x

# 设置设备为 GPU
paddle.set_device("gpu")

# 创建 LeNet 模型实例、损失函数和优化器
net = LeNet()
loss_fn = nn.CrossEntropyLoss()
opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# 数据加载器
transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format='CHW')])
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
train_loader = paddle.io.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)

# 开始训练
# 遍历训练轮次的循环，范围是从0到EPOCH_NUM-1
for epoch_id in range(EPOCH_NUM):
    # 遍历训练数据加载器中的每个批次，获取图像数据和标签数据
    for batch_id, (image, label) in enumerate(train_loader()):
        # 将图像数据输入神经网络，获取输出结果
        out = net(image)
        # 计算输出结果和标签数据之间的损失
        loss = loss_fn(out, label)
        # 反向传播计算梯度
        loss.backward()

        # 每300个批次打印一次当前轮次和批次号以及损失值的平均值
        if batch_id % 300 == 0:
            print("Epoch {} batch {}: loss = {}".format(epoch_id, batch_id,
                                                        np.mean(loss.numpy())))

        # 根据梯度更新优化器的参数
        opt.step()
        # 清空梯度，为下一次迭代做准备
        opt.clear_grad()
```