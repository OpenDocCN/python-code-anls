# `.\lucidrains\egnn-pytorch\tests\test_equivariance.py`

```py
import torch  # 导入PyTorch库

from egnn_pytorch import EGNN, EGNN_Sparse  # 导入EGNN和EGNN_Sparse类
from egnn_pytorch.utils import rot  # 导入rot函数

torch.set_default_dtype(torch.float64)  # 设置PyTorch默认数据类型为float64

def test_egnn_equivariance():  # 定义测试函数test_egnn_equivariance
    layer = EGNN(dim=512, edge_dim=4)  # 创建EGNN层对象，设置维度和边维度

    R = rot(*torch.rand(3))  # 生成随机旋转矩阵R
    T = torch.randn(1, 1, 3)  # 生成随机平移向量T

    feats = torch.randn(1, 16, 512)  # 生成随机特征张量
    coors = torch.randn(1, 16, 3)  # 生成随机坐标张量
    edges = torch.randn(1, 16, 16, 4)  # 生成随机边张量
    mask = torch.ones(1, 16).bool()  # 生成全为True的掩码张量

    # 缓存前两个节点的特征
    node1 = feats[:, 0, :]  # 获取第一个节点的特征
    node2 = feats[:, 1, :]  # 获取第二个节点的特征

    # 交换第一个和第二个节点的位置
    feats_permuted_row_wise = feats.clone().detach()  # 克隆特征张量
    feats_permuted_row_wise[:, 0, :] = node2  # 将第一个节点的特征替换为第二个节点的特征
    feats_permuted_row_wise[:, 1, :] = node1  # 将第二个节点的特征替换为第一个节点的特征

    feats1, coors1 = layer(feats, coors @ R + T, edges, mask=mask)  # 使用EGNN层进行前向传播
    feats2, coors2 = layer(feats, coors, edges, mask=mask)  # 使用EGNN层进行前向传播
    feats3, coors3 = layer(feats_permuted_row_wise, coors, edges, mask=mask)  # 使用EGNN层进行前向传播

    assert torch.allclose(feats1, feats2, atol=1e-6), 'type 0 features are invariant'  # 断言特征1和特征2在误差范围内相等
    assert torch.allclose(coors1, (coors2 @ R + T), atol=1e-6), 'type 1 features are equivariant'  # 断言坐标1和坐标2在误差范围内相等
    assert not torch.allclose(feats1, feats3, atol=1e-6), 'layer must be equivariant to permutations of node order'  # 断言特征1和特征3不在误差范围内相等

def test_higher_dimension():  # 定义测试函数test_higher_dimension
    layer = EGNN(dim=512, edge_dim=4)  # 创建EGNN层对象，设置维度和边维度

    feats = torch.randn(1, 16, 512)  # 生成随机特征张量
    coors = torch.randn(1, 16, 5)  # 生成随机坐标张量
    edges = torch.randn(1, 16, 16, 4)  # 生成随机边张量
    mask = torch.ones(1, 16).bool()  # 生成全为True的掩码张量

    feats, coors = layer(feats, coors, edges, mask=mask)  # 使用EGNN层进行前向传播
    assert True  # 断言为True

def test_egnn_equivariance_with_nearest_neighbors():  # 定义测试函数test_egnn_equivariance_with_nearest_neighbors
    layer = EGNN(dim=512, edge_dim=1, num_nearest_neighbors=8)  # 创建EGNN层对象，设置维度、边维度和最近邻节点数

    R = rot(*torch.rand(3))  # 生成随机旋转矩阵R
    T = torch.randn(1, 1, 3)  # 生成随机平移向量T

    feats = torch.randn(1, 256, 512)  # 生成随机特征张量
    coors = torch.randn(1, 256, 3)  # 生成随机坐标张量
    edges = torch.randn(1, 256, 256, 1)  # 生成随机边张量
    mask = torch.ones(1, 256).bool()  # 生成全为True的掩码张量

    # 缓存前两个节点的特征
    node1 = feats[:, 0, :]  # 获取第一个节点的特征
    node2 = feats[:, 1, :]  # 获取第二个节点的特征

    # 交换第一个和第二个节点的位置
    feats_permuted_row_wise = feats.clone().detach()  # 克隆特征张量
    feats_permuted_row_wise[:, 0, :] = node2  # 将第一个节点的特征替换为第二个节点的特征
    feats_permuted_row_wise[:, 1, :] = node1  # 将第二个节点的特征替换为第一个节点的特征

    feats1, coors1 = layer(feats, coors @ R + T, edges, mask=mask)  # 使用EGNN层进行前向传播
    feats2, coors2 = layer(feats, coors, edges, mask=mask)  # 使用EGNN层进行前向传播
    feats3, coors3 = layer(feats_permuted_row_wise, coors, edges, mask=mask)  # 使用EGNN层进行前向传播

    assert torch.allclose(feats1, feats2, atol=1e-6), 'type 0 features are invariant'  # 断言特征1和特征2在误差范围内相等
    assert torch.allclose(coors1, (coors2 @ R + T), atol=1e-6), 'type 1 features are equivariant'  # 断言坐标1和坐标2在误差范围内相等
    assert not torch.allclose(feats1, feats3, atol=1e-6), 'layer must be equivariant to permutations of node order'  # 断言特征1和特征3不在误差范围内相等

def test_egnn_equivariance_with_coord_norm():  # 定义测试函数test_egnn_equivariance_with_coord_norm
    layer = EGNN(dim=512, edge_dim=1, num_nearest_neighbors=8, norm_coors=True)  # 创建EGNN层对象，设置维度、边维度、最近邻节点数和是否对坐标进行归一化

    R = rot(*torch.rand(3))  # 生成随机旋转矩阵R
    T = torch.randn(1, 1, 3)  # 生成随机平移向量T

    feats = torch.randn(1, 256, 512)  # 生成随机特征张量
    coors = torch.randn(1, 256, 3)  # 生成随机坐标张量
    edges = torch.randn(1, 256, 256, 1)  # 生成随机边张量
    mask = torch.ones(1, 256).bool()  # 生成全为True的掩码张量

    # 缓存前两个节点的特征
    node1 = feats[:, 0, :]  # 获取第一个节点的特征
    node2 = feats[:, 1, :]  # 获取第二个节点的特征

    # 交换第一个和第二个节点的位置
    feats_permuted_row_wise = feats.clone().detach()  # 克隆特征张量
    feats_permuted_row_wise[:, 0, :] = node2  # 将第一个节点的特征替换为第二个节点的特征
    feats_permuted_row_wise[:, 1, :] = node1  # 将第二个节点的特征替换为第一个节点的特征

    feats1, coors1 = layer(feats, coors @ R + T, edges, mask=mask)  # 使用EGNN层进行前向传播
    feats2, coors2 = layer(feats, coors, edges, mask=mask)  # 使用EGNN层进行前向传播
    feats3, coors3 = layer(feats_permuted_row_wise, coors, edges, mask=mask)  # 使用EGNN层进行前向传播

    assert torch.allclose(feats1, feats2, atol=1e-6), 'type 0 features are invariant'  # 断言特征1和特征2在误差范围内相等
    assert torch.allclose(coors1, (coors2 @ R + T), atol=1e-6), 'type 1 features are equivariant'  # 断言坐标1和坐标2在误差范围内相等
    assert not torch.allclose(feats1, feats3, atol=1e-6), 'layer must be equivariant to permutations of node order'  # 断言特征1和特征3不在误差范围内相等

def test_egnn_sparse_equivariance():  # 定义测试函数test_egnn_sparse_equivariance
    layer = EGNN_Sparse(feats_dim=1, m_dim=16, fourier_features=4)  # 创建稀疏EGNN层对象，设置特征维度、消息维度和傅立叶特征数

    R = rot(*torch.rand(3))  # 生成随机旋转矩阵R
    T = torch.randn(1, 1, 3)  # 生成随机平移向量T
    apply_action = lambda t: (t @ R + T).squeeze()  # 定义应用旋转和平移的操作函数
    # 生成一个大小为16x1的随机张量，表示节点的特征
    feats = torch.randn(16, 1)
    # 生成一个大小为16x3的随机张量，表示节点的坐标
    coors = torch.randn(16, 3)
    # 生成一个大小为2x20的随机整数张量，表示边的索引
    edge_idxs = (torch.rand(2, 20) * 16).long()

    # 缓存第一个和第二个节点的特征
    node1 = feats[0, :]
    node2 = feats[1, :]

    # 交换第一个和第二个节点的位置，生成一个新的特征张量
    feats_permuted_row_wise = feats.clone().detach()
    feats_permuted_row_wise[0, :] = node2
    feats_permuted_row_wise[1, :] = node1

    # 将节点的坐标和特征拼接在一起，形成输入张量x1
    x1 = torch.cat([coors, feats], dim=-1)
    # 将节点的坐标和经过apply_action函数处理后的特征拼接在一起，形成输入张量x2
    x2 = torch.cat([apply_action(coors), feats], dim=-1)
    # 将节点的坐标和交换节点顺序后的特征拼接在一起，形成输入张量x3
    x3 = torch.cat([apply_action(coors), feats_permuted_row_wise], dim=-1)

    # 使用layer函数对输入张量x1进行处理，得到输出out1
    out1 = layer(x=x1, edge_index=edge_idxs)
    # 使用layer函数对输入张量x2进行处理，得到输出out2
    out2 = layer(x=x2, edge_index=edge_idxs)
    # 使用layer函数对输入张量x3进行处理，得到输出out3
    out3 = layer(x=x3, edge_index=edge_idxs)

    # 从out1中分离出特征和坐标
    feats1, coors1 = out1[:, 3:], out1[:, :3]
    # 从out2中分离出特征和坐标
    feats2, coors2 = out2[:, 3:], out2[:, :3]
    # 从out3中分离出特征和坐标
    feats3, coors3 = out3[:, 3:], out3[:, :3]

    # 打印feats1和feats2之间的差异
    print(feats1 - feats2)
    # 打印apply_action(coors1)和coors2之间的差异
    print(apply_action(coors1) - coors2)
    # 断言feats1和feats2必须非常接近，否则抛出异常
    assert torch.allclose(feats1, feats2), 'features must be invariant'
    # 断言apply_action(coors1)和coors2必须非常接近，否则抛出异常
    assert torch.allclose(apply_action(coors1), coors2), 'coordinates must be equivariant'
    # 断言feats1和feats3不能非常接近，否则抛出异常
    assert not torch.allclose(feats1, feats3, atol=1e-6), 'layer must be equivariant to permutations of node order'
# 定义一个测试函数，用于测试几何等效性
def test_geom_equivalence():
    # 创建一个 EGNN_Sparse 层对象，设置特征维度为128，边属性维度为4，m维度为16，傅立叶特征为4
    layer = EGNN_Sparse(feats_dim=128,
                        edge_attr_dim=4,
                        m_dim=16,
                        fourier_features=4)

    # 生成一个大小为16x128的随机特征张量
    feats = torch.randn(16, 128)
    # 生成一个大小为16x3的随机坐标张量
    coors = torch.randn(16, 3)
    # 将坐标和特征张量在最后一个维度上拼接起来
    x = torch.cat([coors, feats], dim=-1)
    # 生成一个2x20的随机整数张量，用于表示边的索引
    edge_idxs = (torch.rand(2, 20) * 16).long()
    # 生成一个大小为16x16x4的随机边属性张量
    edges_attrs = torch.randn(16, 16, 4)
    # 根据边索引从边属性张量中取出对应的边属性
    edges_attrs = edges_attrs[edge_idxs[0], edge_idxs[1]]

    # 断言通过 EGNN_Sparse 层的前向传播后输出的形状与输入张量 x 的形状相同
    assert layer.forward(x, edge_idxs, edge_attr=edges_attrs).shape == x.shape
```