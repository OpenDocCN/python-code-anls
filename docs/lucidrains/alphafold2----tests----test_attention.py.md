# `.\lucidrains\alphafold2\tests\test_attention.py`

```
import torch
from torch import nn
from einops import repeat

from alphafold2_pytorch.alphafold2 import Alphafold2
from alphafold2_pytorch.utils import *

# 定义测试函数 test_main
def test_main():
    # 创建 Alphafold2 模型对象
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32
    )

    # 生成随机序列数据和多序列比对数据
    seq = torch.randint(0, 21, (2, 128))
    msa = torch.randint(0, 21, (2, 5, 128))
    mask = torch.ones_like(seq).bool()
    msa_mask = torch.ones_like(msa).bool()

    # 使用模型进行预测
    distogram = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )
    # 断言测试结果为真
    assert True

# 定义测试函数 test_no_msa
def test_no_msa():
    # 创建 Alphafold2 模型对象
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32
    )

    # 生成随机序列数据和掩码
    seq = torch.randint(0, 21, (2, 128))
    mask = torch.ones_like(seq).bool()

    # 使用模型进行预测
    distogram = model(
        seq,
        mask = mask
    )
    # 断言测试结果为真
    assert True

# 定义测试函数 test_anglegrams
def test_anglegrams():
    # 创建 Alphafold2 模型对象
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_angles = True
    )

    # 生成随机序列数据、多序列比对数据和掩码
    seq = torch.randint(0, 21, (2, 128))
    msa = torch.randint(0, 21, (2, 5, 128))
    mask = torch.ones_like(seq).bool()
    msa_mask = torch.ones_like(msa).bool()

    # 使用模型进行预测
    ret = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )
    # 断言测试结果为真
    assert True

# 定义测试函数 test_templates
def test_templates():
    # 创建 Alphafold2 模型对象
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        templates_dim = 32,
        templates_angles_feats_dim = 32
    )

    # 生成随机序列数据、多序列比对数据和掩码
    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()
    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    # 生成随机模板特征数据、模板角度数据和模板掩码
    templates_feats = torch.randn(2, 3, 16, 16, 32)
    templates_angles = torch.randn(2, 3, 16, 32)
    templates_mask = torch.ones(2, 3, 16).bool()

    # 使用模型进行预测
    distogram = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask,
        templates_feats = templates_feats,
        templates_angles = templates_angles,
        templates_mask = templates_mask
    )
    # 断言测试结果为真
    assert True

# 定义测试函数 test_extra_msa
def test_extra_msa():
    # 创建 Alphafold2 模型对象
    model = Alphafold2(
        dim = 128,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_coords = True
    )

    # 生成随机序列数据、多序列比对数据和掩码
    seq = torch.randint(0, 21, (2, 4))
    mask = torch.ones_like(seq).bool()
    msa = torch.randint(0, 21, (2, 5, 4))
    msa_mask = torch.ones_like(msa).bool()

    # 生成额外的多序列比对数据和掩码
    extra_msa = torch.randint(0, 21, (2, 5, 4))
    extra_msa_mask = torch.ones_like(extra_msa).bool()

    # 使用模型进行预测
    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask,
        extra_msa = extra_msa,
        extra_msa_mask = extra_msa_mask
    )
    # 断言测试结果为真
    assert True

# 定义测试函数 test_embeddings
def test_embeddings():
    # 创建 Alphafold2 模型对象
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32
    )

    # 生成随机序列数据、掩码和嵌入数据
    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()
    embedds = torch.randn(2, 1, 16, 1280)

    # 使用模型进行预测（不带掩码）
    distogram = model(
        seq,
        mask = mask,
        embedds = embedds,
        msa_mask = None
    )
    
    # 生成嵌入数据的掩码
    embedds_mask = torch.ones_like(embedds[..., -1]).bool()

    # 使用模型进行预测（带掩码）
    distogram = model(
        seq,
        mask = mask,
        embedds = embedds,
        msa_mask = embedds_mask
    )
    # 断言测试结果为真
    assert True

# 定义测试函数 test_coords
def test_coords():
    # 创建 Alphafold2 模型对象
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
    )

    # 生成随机序列数据、多序列比对数据和掩码
    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()
    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    # 使用模型进行预测
    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    # 断言输出坐标的形状为 (2, 16, 3)
    assert coords.shape == (2, 16, 3), 'must output coordinates'

# 定义测试函数 test_coords_backbone_with_cbeta
def test_coords_backbone_with_cbeta():
    # 创建 Alphafold2 模型对象
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
    )

    # 生成随机序列数据、多序列比对数据和掩码
    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()
    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    # 使用模型进行预测
    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    # 断言输出坐标的形状为 (2, 16, 3)
    assert coords.shape == (2, 16, 3), 'must output coordinates'

# 定义测试函数 test_coords_all_atoms
def test_coords_all_atoms():
    # 创建 Alphafold2 模型对象
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
    )

    # 生成随机序列数据、多序列比对数据和掩码
    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()
    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    # 使用模型进行预测
    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    # 断言输出坐标的形状为 (2, 16, 3)
    assert coords.shape == (2, 16, 3), 'must output coordinates'

# 定义测试函数 test_mds
def test_mds():
    # 创建 Alphafold2 模型对象
    model = Alphafold2(
        dim = 32,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
    )

    # 生成随机序列数据、多序列比对数据和掩码
    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()
    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    # 使用模型进行预测
    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    # 断言输出坐标的形状为 (2, 16, 3)
    assert coords.shape == (2, 16, 3), 'must output coordinates'

# 定义测试函数 test_edges_to_equivariant_network
def test_edges_to_equivariant_network():
    # 创建 Alphafold2 模型对象
    model = Alphafold2(
        dim = 32,
        depth = 1,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
        predict_angles = True
    )

    # 生成随机序列数据、多序列比对数据和掩码
    seq = torch.randint(0, 21, (2, 32))
    mask = torch.ones_like(seq).bool()
    msa = torch.randint(0, 21, (2, 5, 32))
    msa_mask = torch.ones_like(msa).bool()

    # 使用模型进行预测
    coords, confidences = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask,
        return_confidence = True
    )
    # 断言测试结果为真
    assert True, 'should run without errors'

# 定义测试函数 test_coords_backwards
def test_coords_backwards():
    # 创建 Alphafold2 模型对象
    model = Alphafold2(
        dim = 256,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
        structure_module_depth = 1,
        structure_module_heads = 1,
        structure_module_dim_head = 1,
    )

    # 生成随机序列数据、多序列比对数据和掩码
    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()
    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    # 使用模型进行预测
    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

    # 反向传播
    coords.sum().backward()
    assert True, 'must be able to go backwards through MDS and center distogram'

# 定义测试函数 test_confidence
def test_confidence():
    # 创建 Alphafold2 模型对象
    model = Alphafold2(
        dim = 256,
        depth = 1,
        heads = 2,
        dim_head = 32,
        predict_coords = True
    )

    # 生成随机序列数据、多序列比对数据和掩码
    seq = torch.randint(0, 21, (2, 16))
    mask = torch.ones_like(seq).bool()
    msa = torch.randint(0, 21, (2, 5, 16))
    msa_mask = torch.ones_like(msa).bool()

    # 使用模型进行预测
    coords, confidences = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask,
        return_confidence = True
    )
    
    # 断言坐标和置信度的形状相同
    assert coords.shape[:-1] == confidences.shape[:-1]

# 定义测试函数 test_recycling
def test_recycling():
    # 创建 Alphafold2 模型对象
    model = Alphafold2(
        dim = 128,
        depth = 2,
        heads = 2,
        dim_head = 32,
        predict_coords = True,
    )

    # 生成随机序列数据、多序列比对数据和掩码
    seq = torch.randint(0, 21, (2, 4))
    mask = torch.ones_like(seq).bool()
    msa = torch.randint(0, 21, (2, 5, 4))
    msa_mask = torch.ones_like(msa).bool()

    # 生成额外的多序列比对数据和掩码
    extra_msa = torch.randint(0, 21, (2, 5, 4))
    extra_msa_mask = torch.ones_like(extra_msa).bool()
    # 调用模型，传入序列、多序列比对、掩码、多序列比对掩码、额外多序列比对、额外多序列比对掩码等参数，并返回坐标和结果
    coords, ret = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask,
        extra_msa = extra_msa,
        extra_msa_mask = extra_msa_mask,
        return_aux_logits = True,  # 返回辅助日志
        return_recyclables = True  # 返回可回收的数据
    )

    # 调用模型，传入序列、多序列比对、掩码、多序列比对掩码、额外多序列比对、额外多序列比对掩码、可回收的数据等参数，并返回坐标和结果
    coords, ret = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask,
        extra_msa = extra_msa,
        extra_msa_mask = extra_msa_mask,
        recyclables = ret.recyclables,  # 使用上一个调用返回的可回收数据
        return_aux_logits = True,  # 返回辅助日志
        return_recyclables = True  # 返回可回收的数据
    )

    # 断言，确保条件为真，否则会引发异常
    assert True
```