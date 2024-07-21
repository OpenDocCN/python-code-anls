# `.\pytorch\torch\distributed\_tensor\ops\basic_strategy.py`

```py
import itertools
from dataclasses import dataclass
from typing import List, Set, Tuple

from torch.distributed._tensor._op_schema import OpStrategy, PlacementStrategy
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.device_mesh import DeviceMesh

@dataclass
class EinsumDims:
    contracting_dims: List[str]
    batch_dims: List[str]
    lhs_out_only_dims: List[str]
    rhs_out_only_dims: List[str]

    @classmethod
    def parse_equation(cls, equation: str) -> Tuple[List[str], str]:
        # 解析 einsum 方程并提取参数规范
        """
        Parse the einsum equation str to input dim chars and output dim char
        """
        inputs, outputs = equation.split("->")
        input_dims, output_dims = inputs.split(","), outputs.split(",")

        # NOTE: only support at most two inputs, and single output
        # 在必要时扩展以支持更多输入
        assert len(input_dims) <= 2, "Only support at most two inputs"
        assert len(output_dims) == 1, "Only support single output"
        output_dim = output_dims[0]
        return input_dims, output_dim

    @classmethod
    def parse_dims(cls, input_dims: List[str], output_dim: str) -> "EinsumDims":
        # 解析维度并提取左右手边的收缩、批处理和自由维度
        """
        Parse the dims and extract the contracting, batch, and free dimensions
        for the left and right hand sides.
        """
        dim_char_set: Set[str] = set()
        for input_dim in input_dims:
            dim_char_set.update(input_dim)

        # get a determinisitc order of all dim chars
        all_dim_chars = sorted(dim_char_set)

        # parse input and output dimensions
        lhs_out_only_dims, rhs_out_only_dims = [], []
        batch_dims, contracting_dims = [], []

        for dim_char in all_dim_chars:
            if dim_char not in output_dim:
                contracting_dims.append(dim_char)
            else:
                is_batch_dim = True
                for input_dim in input_dims:
                    is_batch_dim = is_batch_dim and dim_char in input_dim

                if is_batch_dim:
                    batch_dims.append(dim_char)
                else:
                    assert (
                        len(input_dims) == 2
                    ), "free dimension only supported for two inputs!"
                    lhs, rhs = input_dims
                    if dim_char in lhs:
                        lhs_out_only_dims.append(dim_char)
                    elif dim_char in rhs:
                        rhs_out_only_dims.append(dim_char)
                    else:
                        raise RuntimeError("Invalid dimension character")

        return cls(
            contracting_dims=contracting_dims,
            batch_dims=batch_dims,
            lhs_out_only_dims=lhs_out_only_dims,
            rhs_out_only_dims=rhs_out_only_dims,
        )

def gen_einsum_strategies(
    equation: str,
    mesh: DeviceMesh,
    *,
    linearity: bool = False,


# 定义一个名为 linearity 的变量，类型为布尔型，默认值为 False
    ) -> OpStrategy:
    """
    Generate a strategy list for the ops that follow einsum style notation.
    """
    # 解析 einop 方程并提取维度信息
    input_dims, output_dim = EinsumDims.parse_equation(equation)
    # 解析输入维度和输出维度，生成 EinsumDims 对象
    edims = EinsumDims.parse_dims(input_dims, output_dim)

    all_mesh_dim_strategies = []

    # 为每个网格维度生成策略
    for mesh_dim in range(mesh.ndim):
        mesh_dim_strategies = []

        # placement list 存储 [输出, 输入1, 输入2, ...] 的放置方式
        # 首先总是复制所有输入和输出
        placement_list: List[Placement] = [Replicate()] * (len(input_dims) + 1)
        mesh_dim_strategies.append(placement_list)

        if mesh.size(mesh_dim) <= 1:
            # 对于尺寸为1的网格维度，仅有复制策略
            # TODO: 查看这对于子网格情况是否有效
            continue

        # 分割批处理维度
        for batch_dim in edims.batch_dims:
            output_batch_dim = output_dim.index(batch_dim)
            placement_list = [Shard(output_batch_dim)]
            for input_dim in input_dims:
                input_batch_dim = input_dim.index(batch_dim)
                placement_list.append(Shard(input_batch_dim))
            mesh_dim_strategies.append(placement_list)

        # 分割收缩维度
        for contracting_dim in edims.contracting_dims:
            placement_list = [Partial()]
            for input_dim in input_dims:
                input_contracting_dim = input_dim.index(contracting_dim)
                placement_list.append(Shard(input_contracting_dim))
            mesh_dim_strategies.append(placement_list)

        # 分割左边的自由维度
        for lhs_dim in edims.lhs_out_only_dims:
            lhs_free_dim = output_dim.index(lhs_dim)
            # 这意味着分割左边的输入和输出
            # 例如 S(0), R -> S(0)
            lhs_placement_list: List[Placement] = [
                Shard(lhs_free_dim),
                Shard(lhs_free_dim),
                Replicate(),
            ]
            mesh_dim_strategies.append(lhs_placement_list)

        # 分割右边的自由维度
        for rhs_dim in edims.rhs_out_only_dims:
            rhs_free_dim = output_dim.index(rhs_dim)
            rhs_placement_list: List[Placement] = [
                Shard(rhs_free_dim),
                Replicate(),
                Shard(rhs_free_dim),
            ]
            mesh_dim_strategies.append(rhs_placement_list)

        # 线性策略
        if linearity:
            linearity_placement_list: List[Placement] = [Partial()]
            for input_dim in input_dims:
                linearity_placement_list.append(Partial())
            mesh_dim_strategies.append(linearity_placement_list)

        all_mesh_dim_strategies.append(mesh_dim_strategies)

    # 为整个网格生成策略组合
    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    # TODO: 过滤掉无效的策略，在这一点上我们生成
    # 存储所有可能的策略的列表，这些策略未考虑张量维度是否可以分片，
    # 需要基于实际张量形状来过滤掉无效的策略
    # （例如对于Shard，张量维度大小必须大于网格大小）
    all_strategies = []

    # 遍历所有的策略组合
    for strategy_comb in strategy_combs:
        spec_list = []
        
        # 遍历每个策略组合中的规格参数
        for specs in zip(*strategy_comb):
            # 根据规格参数创建DTensorSpec对象，并加入spec_list中
            spec_list.append(DTensorSpec(mesh, tuple(specs)))
        
        # 创建PlacementStrategy对象，使用第一个规格参数作为输出规格，其余作为输入规格
        strat = PlacementStrategy(output_specs=spec_list[0], input_specs=spec_list[1:])
        
        # 将创建好的PlacementStrategy对象加入all_strategies列表中
        all_strategies.append(strat)

    # 返回OpStrategy对象，其中包含所有生成的策略
    return OpStrategy(all_strategies)
```