# `.\models\esm\openfold_utils\protein.py`

```py
# 版权声明和许可协议
# Protein 数据类型

# 导入必要的库和模块
import dataclasses
import re
import string
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from . import residue_constants

# 定义类型别名
FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # 是一个嵌套字典
PICO_TO_ANGSTROM = 0.01

# 定义一个名为 Protein 的数据类
@dataclasses.dataclass(frozen=True)
class Protein:
    # 原子在埃斯特朗单位下的笛卡尔坐标。原子类型对应 residue_constants.atom_types，前三个为 N, CA, CB。
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]
    
    # 每个残基的氨基酸类型，表示为 0 到 20 之间的整数，其中 20 代表 'X'。
    aatype: np.ndarray  # [num_res]
    
    # 二进制浮点掩码，指示特定原子是否存在。如果原子存在，则为 1.0，否则为 0.0。这应该用于损失掩码。
    atom_mask: np.ndarray  # [num_res, num_atom_type]
    
    # 在 PDB 中使用的残基索引。它不一定连续或以 0 为起始索引。
    residue_index: np.ndarray  # [num_res]
    
    # 每个残基的 B 因子或温度因子（以平方埃单位表示），表示残基与其真实平均值的偏差。
    b_factors: np.ndarray  # [num_res, num_atom_type]
    
    # 多链预测的链索引
    chain_index: Optional[np.ndarray] = None
    
    # 关于蛋白质的可选备注。在输出 PDB 文件中作为注释包含
    remark: Optional[str] = None
    
    # 用于生成该蛋白质的模板（仅用于预测）
    parents: Optional[Sequence[str]] = None
    
    # 每个父链对应的链索引
    parents_chain_index: Optional[Sequence[int]] = None

# 将 ProteinNet 字符串转换为 Protein 对象
def from_proteinnet_string(proteinnet_str: str) -> Protein:
    # 正则表达式匹配标签
    tag_re = r"(\[[A-Z]+\]\n)"
    tags: List[str] = [tag.strip() for tag in re.split(tag_re, proteinnet_str) if len(tag) > 0]
    # 按标签分组
    groups: Iterator[Tuple[str, List[str]]] = zip(tags[0::2], [l.split("\n") for l in tags[1::2]])
    
    # 标准原子类型
    atoms: List[str] = ["N", "CA", "C"]
    aatype = None
    atom_positions = None
    atom_mask = None
    # 遍历给定的 groups 列表
    for g in groups:
        # 如果 g 列表的第一个元素为 "[PRIMARY]"
        if "[PRIMARY]" == g[0]:
            # 获取 g 列表的第二个元素的第一个元素，并去除两端空白字符
            seq = g[1][0].strip()
            # 遍历序列中的每个字符
            for i in range(len(seq)):
                # 如果字符不在 residue_constants.restypes 中
                if seq[i] not in residue_constants.restypes:
                    # 将字符替换为 "X"，字符串是不可变的，所以需要重新创建字符串
                    seq[i] = "X"  # FIXME: strings are immutable
            # 创建一个包含每个氨基酸类型的 numpy 数组
            aatype = np.array(
                [residue_constants.restype_order.get(res_symbol, residue_constants.restype_num) for res_symbol in seq]
            )
        # 如果 g 列表的第一个元素为 "[TERTIARY]"
        elif "[TERTIARY]" == g[0]:
            # 创建一个空的 2D 列表 tertiary 
            tertiary: List[List[float]] = []
            # 遍历三个轴（axis）
            for axis in range(3):
                # 将 g 列表中第二个元素根据空格分割为列表，然后将列表中的每个元素转换为浮点数，添加到 tertiary 中
                tertiary.append(list(map(float, g[1][axis].split())))
            # 将 tertiary 转换为 numpy 数组
            tertiary_np = np.array(tertiary)
            # 创建一个形状为 (len(tertiary[0])//3, residue_constants.atom_type_num, 3) 的零数组，并转换为浮点数
            atom_positions = np.zeros((len(tertiary[0]) // 3, residue_constants.atom_type_num, 3)).astype(np.float32)
            # 遍历 atoms 列表中的原子
            for i, atom in enumerate(atoms):
                # 重新排列 tertiary_np，并将结果赋值给 atom_positions 中对应原子的位置
                atom_positions[:, residue_constants.atom_order[atom], :] = np.transpose(tertiary_np[:, i::3])
            # 将 atom_positions 转换单位从皮秒到埃
            atom_positions *= PICO_TO_ANGSTROM
        # 如果 g 列表的第一个元素为 "[MASK]"
        elif "[MASK]" == g[0]:
            # 将 g 列表中第二个元素的第一个元素根据字典映射转换为二进制数组
            mask = np.array(list(map({"-": 0, "+": 1}.get, g[1][0].strip())))
            # 创建一个形状为 (len(mask), residue_constants.atom_type_num) 的零数组
            atom_mask = np.zeros(
                (
                    len(mask),
                    residue_constants.atom_type_num,
                )
            ).astype(np.float32)
            # 遍历 atoms 列表中的原子
            for i, atom in enumerate(atoms):
                # 对应原子位置设置为 1
                atom_mask[:, residue_constants.atom_order[atom]] = 1
            # 将 atom_mask 乘以 mask，再与 None 填充的列进行相乘
            atom_mask *= mask[..., None]

    # 断言 aatype 不为空
    assert aatype is not None

    # 返回一个 Protein 对象，包含原子位置、原子掩码、氨基酸类型、残基索引和 B 因子
    return Protein(
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        aatype=aatype,
        residue_index=np.arange(len(aatype)),
        b_factors=None,
    )
# 获取给定蛋白质的 PDB 头信息列表
def get_pdb_headers(prot: Protein, chain_id: int = 0) -> List[str]:
    # 初始化 PDB 头信息列表
    pdb_headers: List[str] = []

    # 获取蛋白质的 REMARK 信息并添加到 PDB 头信息列表中
    remark = prot.remark
    if remark is not None:
        pdb_headers.append(f"REMARK {remark}")

    # 获取蛋白质的父级信息和链索引
    parents = prot.parents
    parents_chain_index = prot.parents_chain_index
    # 若父级信息和链索引均不为 None
    if parents is not None and parents_chain_index is not None:
        # 根据指定链索引筛选对应的父级信息
        parents = [p for i, p in zip(parents_chain_index, parents) if i == chain_id]
    
    # 若父级信息为 None 或为空列表，将字符串 "N/A" 添加到父级信息列表中
    if parents is None or len(parents) == 0:
        parents = ["N/A"]
    
    # 将父级信息列表转换为字符串并添加到 PDB 头信息列表中
    pdb_headers.append(f"PARENT {' '.join(parents)}")

    return pdb_headers


# 将 PDB 头信息添加到现有 PDB 字符串中，用于多链循环利用
def add_pdb_headers(prot: Protein, pdb_str: str) -> str:
    """Add pdb headers to an existing PDB string. Useful during multi-chain
    recycling
    """
    # 初始化输出 PDB 行列表
    out_pdb_lines: List[str] = []
    # 将输入的 PDB 字符串按换行符拆分为行列表
    lines = pdb_str.split("\n")

    # 获取蛋白质的 REMARK 信息并添加到输出 PDB 行列表中
    remark = prot.remark
    if remark is not None:
        out_pdb_lines.append(f"REMARK {remark}")

    # 初始化各链父级信息列表
    parents_per_chain: List[List[str]]
    # 若蛋白质的父级信息不为 None 且长度大于 0
    if prot.parents is not None and len(prot.parents) > 0:
        parents_per_chain = []
        # 若蛋白质的父级链索引不为 None
        if prot.parents_chain_index is not None:
            # 创建父级信息字典，按照链索引分类
            parent_dict: Dict[str, List[str]] = {}
            for p, i in zip(prot.parents, prot.parents_chain_index):
                parent_dict.setdefault(str(i), [])
                parent_dict[str(i)].append(p)

            # 获取父级信息字典中最大的链索引
            max_idx = max([int(chain_idx) for chain_idx in parent_dict])
            # 遍历链索引范围，获取每条链的父级信息并添加到父级信息列表中
            for i in range(max_idx + 1):
                chain_parents = parent_dict.get(str(i), ["N/A"])
                parents_per_chain.append(chain_parents)
        else:
            parents_per_chain.append(list(prot.parents))
    else:
        parents_per_chain = [["N/A"]]

    # 定义生成父级信息行的函数
    def make_parent_line(p: Sequence[str]) -> str:
        return f"PARENT {' '.join(p)}"

    # 添加第一条链的父级信息到输出 PDB 行列表中
    out_pdb_lines.append(make_parent_line(parents_per_chain[0])

    # 初始化链计数器
    chain_counter = 0
    # 遍历 PDB 行列表
    for i, l in enumerate(lines):
        # 若行中不包含 "PARENT" 和 "REMARK" 关键词，将行添加到输出 PDB 行列表中
        if "PARENT" not in l and "REMARK" not in l:
            out_pdb_lines.append(l)
        # 若行中包含 "TER" 并且下一行不包含 "END"
        if "TER" in l and "END" not in lines[i + 1]:
            chain_counter += 1
            # 若链计数器未超过父级信息列表长度，获取相应链的父级信息
            if not chain_counter >= len(parents_per_chain):
                chain_parents = parents_per_chain[chain_counter]
            else:
                chain_parents = ["N/A"]

            # 添加链的父级信息到输出 PDB 行列表中
            out_pdb_lines.append(make_parent_line(chain_parents))

    return "\n".join(out_pdb_lines)


# 将 `Protein` 实例转换为 PDB 字符串
def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    # 初始化氨基酸类型列表和字典
    restypes = residue_constants.restypes + ["X"]

    def res_1to3(r: int) -> str:
        return residue_constants.restype_1to3.get(restypes[r], "UNK")

    # 初始化原子类型列表
    atom_types = residue_constants.atom_types

    # 初始化 PDB 行列表
    pdb_lines: List[str] = []

    # 获取蛋白质的原子遮罩、氨基酸类型、原子位置、残基索引、B 因子和链索引
    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors
    chain_index = prot.chain_index
    # 如果任何一个原子的氨基酸类型大于残基常数的数量，引发值错误
    if np.any(aatype > residue_constants.restype_num):
        raise ValueError("Invalid aatypes.")
    
    # 获取蛋白质的 PDB 头部信息
    headers = get_pdb_headers(prot)
    # 如果头部信息的长度大于 0，则将其添加到 pdb_lines 列表中
    if len(headers) > 0:
        pdb_lines.extend(headers)
    
    # 获取氨基酸类型数组的长度
    n = aatype.shape[0]
    # 设置原子索引为 1
    atom_index = 1
    # 设置前一链的索引为 0
    prev_chain_index = 0
    # 设置链标签为大写英文字母序列
    chain_tags = string.ascii_uppercase
    # 初始化链标签为 None
    chain_tag = None
    
    # 添加所有原子位点
    # 遍历氨基酸类型数组的每个元素
    for i in range(n):
        # 获取三字母表示的氨基酸名称
        res_name_3 = res_1to3(aatype[i])
        # 遍历原子类型、原子位置、原子掩码和 B 因子
        for atom_name, pos, mask, b_factor in zip(atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            # 如果掩码小于 0.5，则跳过本次循环
            if mask < 0.5:
                continue
    
            # 设置记录类型为 "ATOM"
            record_type = "ATOM"
            # 如果原子名称长度为 4，则使用原子名称；否则，在原子名称前加一个空格
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            # 设置可选位置为空字符串
            alt_loc = ""
            # 设置插入代码为空字符串
            insertion_code = ""
            # 设置占有率为 1.00
            occupancy = 1.00
            # 设置元素为原子名称的首字母
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            # 设置电荷为空字符串
            charge = ""
    
            # 将链标签设为 "A"
            chain_tag = "A"
            # 如果链索引不为空，则将链标签设为相应的大写字母
            if chain_index is not None:
                chain_tag = chain_tags[chain_index[i]]
    
            # PDB 是列格式的，每个空格都很重要！
            # 生成原子行
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_tag:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            # 将原子行添加到 pdb_lines 列表中
            pdb_lines.append(atom_line)
            # 增加原子索引
            atom_index += 1
    
        # 设置是否终止标志为当前是否是最后一个氨基酸
        should_terminate = i == n - 1
        # 如果链索引不为空
        if chain_index is not None:
            # 如果当前不是最后一个氨基酸且下一个氨基酸的链索引与前一个不同，则应终止
            if i != n - 1 and chain_index[i + 1] != prev_chain_index:
                should_terminate = True
                prev_chain_index = chain_index[i + 1]
    
        # 如果应该终止
        if should_terminate:
            # 设置链结束符号
            chain_end = "TER"
            # 生成链终止行
            chain_termination_line = (
                f"{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[i]):>3} {chain_tag:>1}{residue_index[i]:>4}"
            )
            # 将链终止行添加到 pdb_lines 列表中
            pdb_lines.append(chain_termination_line)
            # 增加原子索引
            atom_index += 1
    
            # 如果当前不是最后一个氨基酸
            if i != n - 1:
                # 在开始每条新链时，将头部信息添加到 pdb_lines 列表中
                pdb_lines.extend(get_pdb_headers(prot, prev_chain_index))
    
    # 添加 "END" 行
    pdb_lines.append("END")
    # 添加空行
    pdb_lines.append("")
    # 将 pdb_lines 列表中的元素用换行符连接成字符串并返回
    return "\n".join(pdb_lines)
# 计算理想的原子掩模，返回一个布尔数组
def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.
    `Protein.atom_mask` typically is defined according to the atoms that are reported in the PDB. This function
    computes a mask according to heavy atoms that should be present in the given sequence of amino acids.
    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.
    Returns:
      An ideal atom mask.
    """
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]

# 从预测结果组装蛋白质
def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    chain_index: Optional[np.ndarray] = None,
    remark: Optional[str] = None,
    parents: Optional[Sequence[str]] = None,
    parents_chain_index: Optional[Sequence[int]] = None,
) -> Protein:
    """Assembles a protein from a prediction.
    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.
      chain_index: (Optional) Chain indices for multi-chain predictions
      remark: (Optional) Remark about the prediction
      parents: (Optional) List of template names
    Returns:
      A protein instance.
    """
    return Protein(
        aatype=features["aatype"],
        atom_positions=result["final_atom_positions"],
        atom_mask=result["final_atom_mask"],
        residue_index=features["residue_index"] + 1,
        b_factors=b_factors if b_factors is not None else np.zeros_like(result["final_atom_mask"]),
        chain_index=chain_index,
        remark=remark,
        parents=parents,
        parents_chain_index=parents_chain_index,
    )
```