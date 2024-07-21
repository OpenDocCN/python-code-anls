# `.\pytorch\test\mobile\model_test\tensor_ops.py`

```
import torch

# 定义一个名为 TensorOpsModule 的类，继承自 torch.nn.Module
class TensorOpsModule(torch.nn.Module):
    # 定义类的前向传播方法
    def forward(self):
        # 调用类内部的 tensor_general_ops 方法，并返回其结果
        return self.tensor_general_ops()
    def tensor_general_ops(self):
        # 创建一个形状为(4,)的张量，其值从标准正态分布中抽取
        a = torch.randn(4)
        # 创建一个标量张量，值为1.5
        b = torch.tensor([1.5])
        # 创建一个形状为(2,)的全1张量
        x = torch.ones((2,))
        # 创建一个复数张量，形状为(4,)，数据类型为复数浮点数
        c = torch.randn(4, dtype=torch.cfloat)
        # 创建一个形状为(4, 4, 4, 4)的随机张量
        w = torch.rand(4, 4, 4, 4)
        # 创建一个形状为(4, 4, 4, 4)的随机张量
        v = torch.rand(4, 4, 4, 4)
        # 返回以下操作结果的数量
        return len(
            # 判断张量a是否为复数类型
            torch.is_complex(a),
            # 判断张量a是否为共轭类型
            torch.is_conj(a),
            # 判断张量a是否为浮点数类型
            torch.is_floating_point(a),
            # 判断张量b是否非零
            torch.is_nonzero(b),
            # 返回张量a的元素总数
            torch.numel(a),
            # 使用x的设备创建一个新张量，填充值为3.141592
            x.new_full((3, 4), 3.141592),
            # 使用x的设备创建一个形状为空的新张量
            x.new_empty((2, 3)),
            # 使用x的设备创建一个形状为(2, 3)且全1的新张量
            x.new_ones((2, 3)),
            # 使用x的设备创建一个形状为(2, 3)且全0的新张量
            x.new_zeros((2, 3)),
            # 判断张量x是否在CUDA设备上
            x.is_cuda,
            # 判断张量x是否是量化的
            x.is_quantized,
            # 判断张量x是否是元张量
            x.is_meta,
            # 返回张量x所在设备
            x.device,
            # 返回张量x的维度数
            x.dim(),
            # 返回张量c的实部
            c.real,
            # 返回张量c的虚部
            c.imag,
            # 克隆张量x并返回新张量
            x.clone(),
            # 返回一个连续的张量，确保其存储方式连续
            w.contiguous(),
            # 返回一个按照指定内存格式的连续张量
            w.contiguous(memory_format=torch.channels_last),
            # 将张量v的值拷贝到张量w中
            w.copy_(v),
            # 将张量w所有元素复制为1
            w.copy_(1),
            # 将张量w所有元素复制为0.5
            w.copy_(0.5),
            # 将张量x移动到CPU上
            x.cpu(),
            # 返回张量x的密集维度数
            x.dense_dim(),
            # 将张量w的对角线元素填充为0
            w.fill_diagonal_(0),
            # 返回张量元素的字节大小
            w.element_size(),
            # 对张量w进行指数操作
            w.exponential_(),
            # 将张量w所有元素填充为0
            w.fill_(0),
            # 对张量w进行几何操作
            w.geometric_(0.5),
            # 在张量a的索引0和2位置填充值为1
            a.index_fill(0, torch.tensor([0, 2]), 1),
            # 在张量a的最大值位置填充值为1.0
            a.index_put_([torch.argmax(a)], torch.tensor(1.0)),
            # 在张量a的最大值位置填充值为1.0
            a.index_put([torch.argmax(a)], torch.tensor(1.0)),
            # 判断张量w是否是连续的
            w.is_contiguous(),
            # 判断张量c是否为复数类型
            c.is_complex(),
            # 判断张量w是否为共轭类型
            w.is_conj(),
            # 判断张量w是否为浮点数类型
            w.is_floating_point(),
            # 判断张量w是否为叶子节点
            w.is_leaf,
            # 判断张量w是否被固定在内存中
            w.is_pinned(),
            # 判断张量w是否被设置为给定张量w
            w.is_set_to(w),
            # 判断张量w是否是合并的
            w.is_coalesced(),
            # 合并张量w并返回新张量
            w.coalesce(),
            # 判断张量w是否为有符号类型
            w.is_signed(),
            # 判断张量w是否为稀疏张量
            w.is_sparse,
            # 返回张量[1]的标量值
            torch.tensor([1]).item(),
            # 对张量x进行对数正态操作
            x.log_normal_(),
            # 将张量x进行复制并返回新张量
            x.clone(),
            # 对张量a进行最小值裁剪操作
            a.clamp_(0),
            # 返回对张量a进行最小值裁剪操作后的新张量
            a.clamp(0),
            # 返回对张量a进行最小值裁剪操作后的新张量
            a.clamp_min(0),
            # 对张量a进行hard sigmoid操作
            a.hardsigmoid_(),
            # 返回对张量a进行hard sigmoid操作后的新张量
            a.hardsigmoid(),
            # 对张量a进行hard swish操作
            a.hardswish_(),
            # 返回对张量a进行hard swish操作后的新张量
            a.hardswish(),
            # 对张量a进行hard tanh操作
            a.hardtanh_(),
            # 返回对张量a进行hard tanh操作后的新张量
            a.hardtanh(),
            # 对张量a进行leaky ReLU操作
            a.leaky_relu_(),
            # 返回对张量a进行leaky ReLU操作后的新张量
            a.leaky_relu(),
            # 对张量a进行ReLU操作
            a.relu_(),
            # 返回对张量a进行ReLU操作后的新张量
            a.relu(),
            # 将张量a调整为与a相同形状
            a.resize_as_(a),
            # 将张量a转换为与a相同类型的张量
            a.type_as(a),
            # 返回张量a的形状作为张量
            a._shape_as_tensor(),
            # 设置张量a是否需要梯度
            a.requires_grad_(False),
        )
class TensorCreationOpsModule(torch.nn.Module):
    # 定义一个 PyTorch 模块，处理张量的创建操作
    def forward(self):
        # 前向传播函数，调用 tensor_creation_ops 方法
        return self.tensor_creation_ops()

    def tensor_creation_ops(self):
        # 创建一个二维整数张量
        i = torch.tensor([[0, 1, 1], [2, 0, 2]])
        # 创建一个一维浮点数张量
        v = torch.tensor([3, 4, 5], dtype=torch.float32)
        # 创建一个一维实数张量
        real = torch.tensor([1, 2], dtype=torch.float32)
        # 创建一个一维虚数张量
        imag = torch.tensor([3, 4], dtype=torch.float32)
        # 创建一个一维张量
        inp = torch.tensor([-1.5, 0.0, 2.0])
        # 创建一个一维张量，包含一个元素
        values = torch.tensor([0.5])
        # 对二维张量进行通道量化
        quantized = torch.quantize_per_channel(
            torch.tensor([[-1.0, 0.0], [1.0, 2.0]]),
            torch.tensor([0.1, 0.01]),
            torch.tensor([10, 0]),
            0,
            torch.quint8,
        )
        # 返回多个张量的长度
        return len(
            torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]),
            # torch.sparse_coo_tensor(i, v, [2, 3]), # not work for iOS
            torch.as_tensor([1, 2, 3]),
            torch.as_strided(torch.randn(3, 3), (2, 2), (1, 2)),
            torch.zeros(2, 3),
            torch.zeros((2, 3)),
            torch.zeros([2, 3], out=i),
            torch.zeros(5),
            torch.zeros_like(torch.empty(2, 3)),
            torch.ones(2, 3),
            torch.ones((2, 3)),
            torch.ones([2, 3]),
            torch.ones(5),
            torch.ones_like(torch.empty(2, 3)),
            torch.arange(5),
            torch.arange(1, 4),
            torch.arange(1, 2.5, 0.5),
            torch.range(1, 4),  # torch.range() 已弃用
            torch.range(1, 4, 0.5),  # torch.range() 已弃用
            torch.linspace(3.0, 3.0, steps=1),
            torch.logspace(start=2, end=2, steps=1, base=2.0),
            torch.eye(3),
            torch.empty(2, 3),
            torch.empty_like(torch.empty(2, 3), dtype=torch.int64),
            torch.empty_strided((2, 3), (1, 2)),
            torch.full((2, 3), 3.141592),
            torch.full_like(torch.full((2, 3), 3.141592), 2.71828),
            torch.quantize_per_tensor(
                torch.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch.quint8
            ),
            torch.dequantize(quantized),
            torch.complex(real, imag),
            torch.polar(real, imag),
            torch.heaviside(inp, values),
        )
    # 定义一个方法，包含多种张量索引和操作示例
    def tensor_indexing_ops(self):
        # 创建一个大小为 (2, 4) 的随机张量 x
        x = torch.randn(2, 4)
        # 创建一个大小为 (4, 4) 的随机张量 y
        y = torch.randn(4, 4)
        # 创建一个指定索引的张量 t
        t = torch.tensor([[0, 0], [1, 0]])
        # 根据阈值条件生成一个布尔掩码 mask
        mask = x.ge(0.5)
        # 定义索引 i 作为一个列表
        i = [0, 1]
        # 返回下列张量操作的数量
        return len(
            # 在行维度上连接三个 x 张量
            torch.cat((x, x, x), 0),
            # 错误示例，应为 torch.cat
            torch.concat((x, x, x), 0),
            # 对 x 张量进行共轭运算
            torch.conj(x),
            # 将 x 张量按照指定维度分块
            torch.chunk(x, 2),
            # 在第三维度上拆分张量为指定索引 i
            torch.dsplit(torch.randn(2, 2, 4), i),
            # 水平堆叠两个 x 张量
            torch.column_stack((x, x)),
            # 深度堆叠两个 x 张量
            torch.dstack((x, x)),
            # 根据索引 t 在 x 张量中进行索引操作
            torch.gather(x, 0, t),
            # 按照索引 i 在列维度上拆分张量 x
            torch.hsplit(x, i),
            # 水平堆叠两个 x 张量
            torch.hstack((x, x)),
            # 根据指定维度和索引选择张量 x 的子集
            torch.index_select(x, 0, torch.tensor([0, 1])),
            # 使用索引张量 t 在 x 张量上进行索引操作
            x.index(t),
            # 根据掩码 mask 选择张量 x 的子集
            torch.masked_select(x, mask),
            # 将 x 张量的维度 1 移动到维度 0
            torch.movedim(x, 1, 0),
            # 将 x 张量的维度 1 移动到维度 0
            torch.moveaxis(x, 1, 0),
            # 在维度 0 上缩小张量 x 的范围为 [0, 2)
            torch.narrow(x, 0, 0, 2),
            # 返回张量 x 中非零元素的索引
            torch.nonzero(x),
            # 按照指定的轴顺序排列张量 x
            torch.permute(x, (0, 1)),
            # 将张量 x 重塑为一维张量
            torch.reshape(x, (-1,)),
            # 垂直堆叠两个 x 张量
            torch.row_stack((x, x)),
            # 返回张量 x 在指定维度和索引位置的子张量
            torch.select(x, 0, 0),
            # 根据索引 t 和值 x 在张量 x 上进行扩展操作
            torch.scatter(x, 0, t, x),
            # 在张量 x 上进行就地扩展操作
            x.scatter(0, t, x.clone()),
            # 对 y 张量的对角线进行扩展操作
            torch.diagonal_scatter(y, torch.ones(4)),
            # 在 y 张量上进行选择扩展操作
            torch.select_scatter(y, torch.ones(4), 0, 0),
            # 在张量 x 上进行切片扩展操作
            torch.slice_scatter(x, x),
            # 在张量 x 上进行就地累加扩展操作
            torch.scatter_add(x, 0, t, x),
            # 在张量 x 上进行就地索引扩展操作
            x.scatter_(0, t, y),
            # 在张量 x 上进行就地累加索引扩展操作
            x.scatter_add_(0, t, y),
            # 错误示例，应为 torch.scatter_reduce
            # torch.scatter_reduce(x, 0, t, reduce="sum"),
            # 在指定维度上拆分张量 x
            torch.split(x, 1),
            # 去除 x 张量的尺寸为 1 的维度
            torch.squeeze(x, 0),
            # 沿着新创建的维度堆叠两个 x 张量
            torch.stack([x, x]),
            # 交换张量 x 的维度 0 和维度 1
            torch.swapaxes(x, 0, 1),
            # 交换张量 x 的维度 0 和维度 1
            torch.swapdims(x, 0, 1),
            # 对张量 x 进行转置操作
            torch.t(x),
            # 根据索引 t 从张量 x 中获取元素
            torch.take(x, t),
            # 根据最大值的索引从张量 x 中获取元素
            torch.take_along_dim(x, torch.argmax(x)),
            # 将张量 x 按照指定维度进行分割
            torch.tensor_split(x, 1),
            # 根据给定的分割点在张量 x 上进行分割
            torch.tensor_split(x, [0, 1]),
            # 对张量 x 进行指定尺寸的重复操作
            torch.tile(x, (2, 2)),
            # 将张量 x 的维度 0 和维度 1 进行转置操作
            torch.transpose(x, 0, 1),
            # 解绑张量 x，返回一个元组
            torch.unbind(x),
            # 在张量 x 的尺寸为 -1 的维度上添加一个尺寸为 1 的维度
            torch.unsqueeze(x, -1),
            # 按照索引 i 在行维度上拆分张量 x
            torch.vsplit(x, i),
            # 垂直堆叠两个 x 张量
            torch.vstack((x, x)),
            # 返回张量 x 的非零元素的索引
            torch.where(x),
            # 根据条件返回张量 t 或 0 的值
            torch.where(t > 0, t, 0),
            # 根据条件返回张量 t 或 t 的值
            torch.where(t > 0, t, t),
        )
# 定义一个名为 TensorTypingOpsModule 的类，继承自 torch.nn.Module
class TensorTypingOpsModule(torch.nn.Module):
    
    # 定义类的前向传播方法
    def forward(self):
        # 调用类的 tensor_typing_ops 方法并返回结果
        return self.tensor_typing_ops()

    # 定义类的 tensor_typing_ops 方法
    def tensor_typing_ops(self):
        # 创建一个形状为 (1, 3, 4, 4) 的张量 x，元素为随机数
        x = torch.randn(1, 3, 4, 4)
        # 返回后续操作列表的长度
        return len(
            # 将张量 x 转换为不同数据类型的副本，以下操作每个都会返回一个新的张量
            x.to(torch.float),                  # 转换为 float 类型
            x.to(torch.double),                 # 转换为 double 类型
            x.to(torch.cfloat),                 # 转换为复数 float 类型
            x.to(torch.cdouble),                # 转换为复数 double 类型
            x.to(torch.half),                   # 转换为半精度类型
            x.to(torch.bfloat16),               # 转换为 bfloat16 类型
            x.to(torch.uint8),                  # 转换为 uint8 类型
            x.to(torch.int8),                   # 转换为 int8 类型
            x.to(torch.short),                  # 转换为 short 类型
            x.to(torch.int),                    # 转换为 int 类型
            x.to(torch.long),                   # 转换为 long 类型
            x.to(torch.bool),                   # 转换为 bool 类型
            x.to(torch.device("cpu")),          # 将张量移到 CPU 上
            x.to(device="cpu", dtype=torch.float),  # 将张量移到 CPU 并转换为 float 类型
            x.to(memory_format=torch.channels_last),  # 将张量转换为通道优先的内存格式
        )


# 定义一个名为 TensorViewOpsModule 的类，继承自 torch.nn.Module
class TensorViewOpsModule(torch.nn.Module):
    
    # 定义类的前向传播方法
    def forward(self):
        # 调用类的 tensor_view_ops 方法并返回结果
        return self.tensor_view_ops()

    # 定义类的 tensor_view_ops 方法
    def tensor_view_ops(self):
        # 创建两个形状分别为 (4, 4, 1) 和 (4, 4, 2) 的张量 x 和 y，元素为随机数
        x = torch.randn(4, 4, 1)
        y = torch.randn(4, 4, 2)
        # 返回后续操作列表的长度
        return len(
            x[0, 2:],                       # 切片操作，获取 x 的部分内容
            x.detach(),                     # 返回 x 的副本并且不需要梯度
            x.detach_(),                    # 将 x 原地去除梯度
            x.diagonal(),                   # 返回 x 的对角线元素
            x.expand(-1, -1, 3),            # 扩展 x 的尺寸
            x.expand_as(y),                 # 将 x 扩展为与 y 相同的尺寸
            x.select(0, 1),                 # 在第 0 维度选择索引为 1 的切片
            x.unflatten(1, (2, 2)),         # 在第 1 维度上重新组织 x 的形状
            x.unfold(1, 2, 2),              # 在第 1 维度上展开 x
            x.view(16),                     # 将 x 转换为一维张量
            x.view_as(torch.randn(16)),     # 将 x 转换为与给定张量相同形状的张量
        )
```