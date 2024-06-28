# `.\models\esm\openfold_utils\protein.py`

```py
# 导入必要的模块和库
import dataclasses  # 用于创建不可变数据类
import re  # 用于正则表达式操作
import string  # 包含字符串相关的常量和函数
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple  # 导入类型提示相关的声明

import numpy as np  # 数组操作库

from . import residue_constants  # 导入本地模块 residue_constants

FeatureDict = Mapping[str, np.ndarray]  # 定义 FeatureDict 类型别名，表示一个字符串到 NumPy 数组的映射
ModelOutput = Mapping[str, Any]  # 定义 ModelOutput 类型别名，表示一个字符串到任意类型的映射，通常是嵌套字典

PICO_TO_ANGSTROM = 0.01  # 定义常量 PICO_TO_ANGSTROM，用于从皮科米转换为埃

@dataclasses.dataclass(frozen=True)
class Protein:
    """蛋白质结构的表示类。"""

    # 原子的笛卡尔坐标，单位为埃，atom_types 对应 residue_constants.atom_types
    atom_positions: np.ndarray  # 形状为 [num_res, num_atom_type, 3]

    # 每个残基的氨基酸类型，表示为 0 到 20 之间的整数，其中 20 表示 'X'
    aatype: np.ndarray  # 形状为 [num_res]

    # 二进制浮点掩码，指示特定原子的存在性。如果原子存在则为 1.0，否则为 0.0，用于损失掩码
    atom_mask: np.ndarray  # 形状为 [num_res, num_atom_type]

    # 残基在 PDB 中的索引。不一定连续或从零开始索引
    residue_index: np.ndarray  # 形状为 [num_res]

    # 残基的 B 因子或温度因子（单位为平方埃），表示残基与其基本真实均值之间的偏移量
    b_factors: np.ndarray  # 形状为 [num_res, num_atom_type]

    # 多链预测中的链索引
    chain_index: Optional[np.ndarray] = None  # 可选的链索引数组，形状为 [num_res]

    # 关于蛋白质的可选备注，将包含在输出 PDB 文件的注释中
    remark: Optional[str] = None  # 可选的字符串类型的备注信息

    # 用于生成此蛋白质的模板（仅限预测）
    parents: Optional[Sequence[str]] = None  # 可选的字符串序列，表示用于生成蛋白质的模板列表

    # 每个父模板对应的链索引
    parents_chain_index: Optional[Sequence[int]] = None  # 可选的整数序列，表示每个父模板对应的链索引

def from_proteinnet_string(proteinnet_str: str) -> Protein:
    # 匹配标签的正则表达式，如 [XXXX]\n
    tag_re = r"(\[[A-Z]+\]\n)"
    # 使用正则表达式分割蛋白质字符串，得到标签列表
    tags: List[str] = [tag.strip() for tag in re.split(tag_re, proteinnet_str) if len(tag) > 0]
    # 将标签分成组，每个组包含一个标签和相应的数据行列表
    groups: Iterator[Tuple[str, List[str]]] = zip(tags[0::2], [l.split("\n") for l in tags[1::2]])

    atoms: List[str] = ["N", "CA", "C"]  # 原子类型列表，包括 N、CA、C
    aatype = None  # 初始化氨基酸类型变量为 None
    atom_positions = None  # 初始化原子位置变量为 None
    atom_mask = None  # 初始化原子掩码变量为 None
    # 遍历给定的groups列表
    for g in groups:
        # 检查当前组是否为主要结构信息
        if "[PRIMARY]" == g[0]:
            # 提取序列信息并去除首尾空格
            seq = g[1][0].strip()
            # 对序列中每个字符进行检查，如果不在restypes中，则替换为"X"
            for i in range(len(seq)):
                if seq[i] not in residue_constants.restypes:
                    seq[i] = "X"  # FIXME: 字符串是不可变的
            # 根据序列中的氨基酸符号获取其对应的编号，形成NumPy数组
            aatype = np.array(
                [residue_constants.restype_order.get(res_symbol, residue_constants.restype_num) for res_symbol in seq]
            )
        # 检查当前组是否为三维结构信息
        elif "[TERTIARY]" == g[0]:
            # 初始化一个空的三维结构列表
            tertiary: List[List[float]] = []
            # 逐个轴解析三维结构信息并转换为浮点数列表
            for axis in range(3):
                tertiary.append(list(map(float, g[1][axis].split())))
            # 将解析后的三维结构信息转换为NumPy数组
            tertiary_np = np.array(tertiary)
            # 初始化原子位置数组，用于存储原子的坐标信息
            atom_positions = np.zeros((len(tertiary[0]) // 3, residue_constants.atom_type_num, 3)).astype(np.float32)
            # 根据原子顺序和三维结构信息填充原子位置数组
            for i, atom in enumerate(atoms):
                atom_positions[:, residue_constants.atom_order[atom], :] = np.transpose(tertiary_np[:, i::3])
            # 将位置从皮科秒转换为埃
            atom_positions *= PICO_TO_ANGSTROM
        # 检查当前组是否为掩码信息
        elif "[MASK]" == g[0]:
            # 解析掩码信息，将"-"映射为0，将"+"映射为1，存储为NumPy数组
            mask = np.array(list(map({"-": 0, "+": 1}.get, g[1][0].strip())))
            # 初始化原子掩码数组，用于表示原子是否被掩盖
            atom_mask = np.zeros(
                (
                    len(mask),
                    residue_constants.atom_type_num,
                )
            ).astype(np.float32)
            # 根据原子顺序填充原子掩码数组
            for i, atom in enumerate(atoms):
                atom_mask[:, residue_constants.atom_order[atom]] = 1
            # 将掩码数组应用到原子掩码数组上
            atom_mask *= mask[..., None]

    # 断言确保aatype不为空
    assert aatype is not None

    # 返回一个Protein对象，包括原子位置、原子掩码、氨基酸类型、残基索引和B因子信息
    return Protein(
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        aatype=aatype,
        residue_index=np.arange(len(aatype)),
        b_factors=None,
    )
def get_pdb_headers(prot: Protein, chain_id: int = 0) -> List[str]:
    pdb_headers: List[str] = []  # 初始化一个空列表，用于存储 PDB 头部信息

    remark = prot.remark  # 获取蛋白质对象的备注信息
    if remark is not None:  # 如果存在备注信息
        pdb_headers.append(f"REMARK {remark}")  # 将 REMARK 记录添加到 pdb_headers 中

    parents = prot.parents  # 获取蛋白质对象的父对象列表
    parents_chain_index = prot.parents_chain_index  # 获取父对象对应的链索引列表
    if parents is not None and parents_chain_index is not None:  # 如果父对象列表和链索引列表都不为空
        parents = [p for i, p in zip(parents_chain_index, parents) if i == chain_id]  # 筛选出指定链索引的父对象列表

    if parents is None or len(parents) == 0:  # 如果父对象列表为空
        parents = ["N/A"]  # 使用字符串 "N/A" 作为默认父对象

    pdb_headers.append(f"PARENT {' '.join(parents)}")  # 将格式化的 PARENT 记录添加到 pdb_headers 中

    return pdb_headers  # 返回包含 PDB 头部信息的列表


def add_pdb_headers(prot: Protein, pdb_str: str) -> str:
    """Add pdb headers to an existing PDB string. Useful during multi-chain
    recycling
    """
    out_pdb_lines: List[str] = []  # 初始化一个空列表，用于存储输出的 PDB 行

    lines = pdb_str.split("\n")  # 将输入的 PDB 字符串按行拆分成列表

    remark = prot.remark  # 获取蛋白质对象的备注信息
    if remark is not None:  # 如果存在备注信息
        out_pdb_lines.append(f"REMARK {remark}")  # 将 REMARK 记录添加到输出列表中

    parents_per_chain: List[List[str]]  # 声明一个二维列表，用于存储每条链的父对象列表
    if prot.parents is not None and len(prot.parents) > 0:  # 如果存在父对象列表且不为空
        parents_per_chain = []  # 初始化空的链列表
        if prot.parents_chain_index is not None:  # 如果存在链索引列表
            parent_dict: Dict[str, List[str]] = {}  # 创建一个字典，用于按链索引存储父对象
            for p, i in zip(prot.parents, prot.parents_chain_index):
                parent_dict.setdefault(str(i), [])  # 如果索引不存在则创建新列表，存在则不变
                parent_dict[str(i)].append(p)  # 将父对象添加到对应索引的列表中

            max_idx = max([int(chain_idx) for chain_idx in parent_dict])  # 获取最大的链索引
            for i in range(max_idx + 1):  # 遍历每个可能的链索引
                chain_parents = parent_dict.get(str(i), ["N/A"])  # 获取链索引对应的父对象列表或默认为 ["N/A"]
                parents_per_chain.append(chain_parents)  # 将该链的父对象列表添加到父对象列表中
        else:
            parents_per_chain.append(list(prot.parents))  # 如果没有链索引列表，则将整个父对象列表作为单链的父对象列表
    else:
        parents_per_chain = [["N/A"]]  # 如果不存在父对象列表，则将默认父对象列表作为单链的父对象列表

    def make_parent_line(p: Sequence[str]) -> str:  # 定义生成 PARENT 记录行的函数
        return f"PARENT {' '.join(p)}"  # 返回格式化的 PARENT 记录行

    out_pdb_lines.append(make_parent_line(parents_per_chain[0]))  # 将第一条链的 PARENT 记录行添加到输出列表中

    chain_counter = 0  # 初始化链计数器
    for i, l in enumerate(lines):  # 遍历输入的 PDB 行
        if "PARENT" not in l and "REMARK" not in l:  # 如果当前行不包含 PARENT 或 REMARK 记录
            out_pdb_lines.append(l)  # 将当前行添加到输出列表中
        if "TER" in l and "END" not in lines[i + 1]:  # 如果当前行包含 TER 记录且下一行不包含 END 记录
            chain_counter += 1  # 链计数器加一
            if not chain_counter >= len(parents_per_chain):  # 如果链计数器小于等于父对象列表的长度
                chain_parents = parents_per_chain[chain_counter]  # 获取下一条链的父对象列表
            else:
                chain_parents = ["N/A"]  # 否则使用默认的父对象列表

            out_pdb_lines.append(make_parent_line(chain_parents))  # 将下一条链的 PARENT 记录行添加到输出列表中

    return "\n".join(out_pdb_lines)  # 返回连接成字符串的输出 PDB 行


def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = residue_constants.restypes + ["X"]  # 将氨基酸类型和额外的 "X" 添加到 restypes 中

    def res_1to3(r: int) -> str:  # 定义从氨基酸单字母码到三字母码的转换函数
        return residue_constants.restype_1to3.get(restypes[r], "UNK")  # 返回单字母码对应的三字母码或 "UNK"

    atom_types = residue_constants.atom_types  # 获取原子类型常量

    pdb_lines: List[str] = []  # 初始化一个空列表，用于存储 PDB 行

    atom_mask = prot.atom_mask  # 获取蛋白质对象的原子掩码
    aatype = prot.aatype  # 获取蛋白质对象的氨基酸类型
    atom_positions = prot.atom_positions  # 获取蛋白质对象的原子位置
    residue_index = prot.residue_index.astype(np.int32)  # 获取蛋白质对象的残基索引，并转换为整数类型
    b_factors = prot.b_factors  # 获取蛋白质对象的 B 因子
    chain_index = prot.chain_index  # 获取蛋白质对象的链索引
    # 检查 aatype 中是否存在大于 residue_constants.restype_num 的任何值
    if np.any(aatype > residue_constants.restype_num):
        # 如果存在，则抛出值错误异常
        raise ValueError("Invalid aatypes.")

    # 获取蛋白质结构的 PDB 文件头信息
    headers = get_pdb_headers(prot)
    # 如果存在头信息，则将其加入到 pdb_lines 中
    if len(headers) > 0:
        pdb_lines.extend(headers)

    # 获取 aatype 数组的长度
    n = aatype.shape[0]
    atom_index = 1  # 初始化原子索引为 1
    prev_chain_index = 0  # 初始化前一个链的索引为 0
    chain_tags = string.ascii_uppercase  # 获取大写字母序列作为链标识符
    chain_tag = None  # 初始化链标识符为 None

    # 添加所有原子位置信息
    # 遍历每个残基
    for i in range(n):
        # 获取残基的三字母缩写
        res_name_3 = res_1to3(aatype[i])
        # 遍历每个原子的类型、位置、掩码、B因子
        for atom_name, pos, mask, b_factor in zip(atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            # 如果掩码小于 0.5，则跳过当前原子
            if mask < 0.5:
                continue

            record_type = "ATOM"  # 记录类型为 "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"  # 原子名称
            alt_loc = ""  # 替代位置标识为空字符串
            insertion_code = ""  # 插入代码为空字符串
            occupancy = 1.00  # 占用率设置为 1.00
            element = atom_name[0]  # 元素类型，蛋白质仅支持 C, N, O, S
            charge = ""  # 电荷为空字符串

            chain_tag = "A"  # 默认链标识符为 "A"
            # 如果提供了链索引，则使用对应的大写字母作为链标识符
            if chain_index is not None:
                chain_tag = chain_tags[chain_index[i]]

            # 构建 PDB 文件中的原子行信息
            # 注意每个字段的空格分隔是必要的
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_tag:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            # 将原子行信息添加到 pdb_lines 列表中
            pdb_lines.append(atom_line)
            atom_index += 1

        should_terminate = i == n - 1  # 判断是否是最后一个残基
        if chain_index is not None:
            # 如果存在链索引，并且下一个残基的链索引与当前不同，则应终止当前链
            if i != n - 1 and chain_index[i + 1] != prev_chain_index:
                should_terminate = True
                prev_chain_index = chain_index[i + 1]

        if should_terminate:
            # 结束当前链的标识符为 "TER"
            chain_end = "TER"
            # 构建链终止行信息并添加到 pdb_lines 列表中
            chain_termination_line = (
                f"{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[i]):>3} {chain_tag:>1}{residue_index[i]:>4}"
            )
            pdb_lines.append(chain_termination_line)
            atom_index += 1

            # 如果不是最后一个残基，则添加下一个链的头信息到 pdb_lines 列表中
            if i != n - 1:
                # 这里的名称“prev”有误导性，它在每个新链的开头发生。
                pdb_lines.extend(get_pdb_headers(prot, prev_chain_index))

    # 添加 PDB 文件的结束标记和空行
    pdb_lines.append("END")
    pdb_lines.append("")
    # 将 pdb_lines 列表中的所有行连接成一个字符串并返回
    return "\n".join(pdb_lines)
# 根据给定的蛋白质对象计算一个理想的原子掩码
def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are reported in the PDB. This function
    computes a mask according to heavy atoms that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    # 返回与给定氨基酸序列中标准原子掩码对应的掩码
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


# 从预测结果中组装一个蛋白质对象
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
    # 创建一个 Protein 对象并返回，使用给定的特征和模型输出来设置其属性
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