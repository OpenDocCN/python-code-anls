# `.\models\esm\openfold_utils\rigid_utils.py`

```
# 版权声明和许可证信息

from __future__ import annotations # 导入未来的注释语法

from functools import lru_cache  # 导入lru_cache函数
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple  # 导入类型提示相关模块

import numpy as np  # 导入numpy模块
import torch  # 导入torch模块


def rot_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # 定义函数rot_matmul，参数为两个torch.Tensor类型，返回值为torch.Tensor类型
    """
    对两个旋转矩阵张量进行矩阵乘法。手动编写以避免AMP降级。

    Args:
        a: [*, 3, 3] 左乘数
        b: [*, 3, 3] 右乘数
    Returns:
        乘积ab
    """

    def row_mul(i: int) -> torch.Tensor:  # 定义内部函数row_mul，参数为整数，返回值为torch.Tensor类型
        return torch.stack(  # 使用torch.stack函数对张量进行拼接
            [
                a[..., i, 0] * b[..., 0, 0] + a[..., i, 1] * b[..., 1, 0] + a[..., i, 2] * b[..., 2, 0],
                a[..., i, 0] * b[..., 0, 1] + a[..., i, 1] * b[..., 1, 1] + a[..., i, 2] * b[..., 2, 1],
                a[..., i, 0] * b[..., 0, 2] + a[..., i, 1] * b[..., 1, 2] + a[..., i, 2] * b[..., 2, 2],
            ],
            dim=-1,
        )

    return torch.stack(  # 使用torch.stack函数对张量进行拼接
        [
            row_mul(0),
            row_mul(1),
            row_mul(2),
        ],
        dim=-2,
    )


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # 定义函数rot_vec_mul，参数为两个torch.Tensor类型，返回值为torch.Tensor类型
    """
    将旋转应用于向量。手动编写以避免AMP降级。

    Args:
        r: [*, 3, 3] 旋转矩阵
        t: [*, 3] 坐标张量
    Returns:
        [*, 3] 旋转后的坐标
    """
    x, y, z = torch.unbind(t, dim=-1)  # 使用torch.unbind解绑张量
    return torch.stack(  # 使用torch.stack函数对张量进行拼接
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


@lru_cache(maxsize=None)  # 使用lru_cache进行函数结果缓存
def identity_rot_mats(
    batch_dims: Tuple[int, ...],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:  # 定义函数identity_rot_mats，参数为元组，返回值为torch.Tensor类型
    rots = torch.eye(3, dtype=dtype, device=device, requires_grad=requires_grad)  # 创建单位矩阵张量
    rots = rots.view(*((1,) * len(batch_dims)), 3, 3)  # 调整张量形状
    rots = rots.expand(*batch_dims, -1, -1)  # 根据扩展的维度进行张量扩展
    rots = rots.contiguous()  # 返回张量的连续版本

    return rots  # 返回结果张量


@lru_cache(maxsize=None)  # 使用lru_cache进行函数结果缓存
def identity_trans(
    batch_dims: Tuple[int, ...],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
# 返回一个全零的3维张量，形状为 (*batch_dims, 3)，数据类型为 dtype，存储设备为 device，是否需要梯度为 requires_grad
def identity_trans(*batch_dims: int, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None, requires_grad: bool = False) -> torch.Tensor:
    trans = torch.zeros((*batch_dims, 3), dtype=dtype, device=device, requires_grad=requires_grad)
    return trans

# 基于输入参数 batch_dims 创建一个全零的四维张量，形状为 (*batch_dims, 4)，数据类型为 dtype，存储设备为 device，是否需要梯度为 requires_grad
@lru_cache(maxsize=None)
def identity_quats(batch_dims: Tuple[int, ...], dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None, requires_grad: bool = True) -> torch.Tensor:
    quat = torch.zeros((*batch_dims, 4), dtype=dtype, device=device, requires_grad=requires_grad)

    # 使用 torch.no_grad() 包装，为张量的第一列赋值为 1
    with torch.no_grad():
        quat[..., 0] = 1

    return quat

# 用于四元数元素的列表，包括 "a", "b", "c", "d"
_quat_elements: List[str] = ["a", "b", "c", "d"]

# 用于创建列表的元素键，通过两层嵌套循环对四元数元素列表进行排列组合
_qtr_keys: List[str] = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]

# 将元素键和它们在列表中的索引形成字典，对应字典中 key: "aa" 的值为 0, key: "ab" 为 1, 以此类推
_qtr_ind_dict: Dict[str, int] = {key: ind for ind, key in enumerate(_qtr_keys)}

# 将键值对列表 pairs 转换为 numpy 数组
def _to_mat(pairs: List[Tuple[str, int]]) -> np.ndarray:
    mat = np.zeros((4, 4))
    for key, value in pairs:
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value

    return mat

# 创建一个全零的四维数组 _QTR_MAT，形状为 (4, 4, 3, 3)，并为它的不同元素位置赋值
_QTR_MAT = np.zeros((4, 4, 3, 3))
_QTR_MAT[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
# 更多类似的赋值操作...

# 将四元数转换为旋转矩阵
def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a rotation matrix.

    Args:
        quat: [*, 4] quaternions
    Returns:
        [*, 3, 3] rotation matrices
    """
    # [*, 4, 4]
    quat = quat[..., None] * quat[..., None, :]

    # [4, 4, 3, 3]
    mat = _get_quat("_QTR_MAT", dtype=quat.dtype, device=quat.device)

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat

    # [*, 3, 3]
    return torch.sum(quat, dim=(-3, -4))

# 将旋转矩阵转换为四元数
def rot_to_quat(rot: torch.Tensor) -> torch.Tensor:
    if rot.shape[-2:] != (3, 3):
        raise ValueError("Input rotation is incorrectly shaped")

    # 通过矩阵运算得到四元数
    # 更多与矩阵运算有关的代码...
    return vectors[..., -1]

# 创建一个全零的四维数组 _QUAT_MULTIPLY，形状为 (4, 4, 4)
_QUAT_MULTIPLY = np.zeros((4, 4, 4))
# 定义一个3x4x4的多维数组，用于存储四元数相乘的结果
_QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]

# 定义一个3x4x4的多维数组，用于存储四元数相乘的结果
_QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]

# 定义一个3x4x4的多维数组，用于存储四元数相乘的结果
_QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]]

# 定义一个3x4x4的多维数组，用于存储四元数相乘的结果
_QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]

# 从_QUAT_MULTIPLY中选取一部分，用于纯向量四元数的相乘
_QUAT_MULTIPLY_BY_VEC = _QUAT_MULTIPLY[:, 1:, :]

# 存储缓存的四元数相关数据的字典，包括"_QTR_MAT", "_QUAT_MULTIPLY", "_QUAT_MULTIPLY_BY_VEC"
_CACHED_QUATS: Dict[str, np.ndarray] = {
    "_QTR_MAT": _QTR_MAT,
    "_QUAT_MULTIPLY": _QUAT_MULTIPLY,
    "_QUAT_MULTIPLY_BY_VEC": _QUAT_MULTIPLY_BY_VEC,
}

# 使用LRU缓存装饰器装饰的函数，用于获取缓存的四元数数据
@lru_cache(maxsize=None)
def _get_quat(quat_key: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(_CACHED_QUATS[quat_key], dtype=dtype, device=device)

# 将两个四元数相乘的函数
def quat_multiply(quat1: torch.Tensor, quat2: torch.Tensor) -> torch.Tensor:
    """Multiply a quaternion by another quaternion."""
    mat = _get_quat("_QUAT_MULTIPLY", dtype=quat1.dtype, device=quat1.device)
    reshaped_mat = mat.view((1,) * len(quat1.shape[:-1]) + mat.shape)
    return torch.sum(reshaped_mat * quat1[..., :, None, None] * quat2[..., None, :, None], dim=(-3, -2))

# 将四元数与纯向量四元数相乘的函数
def quat_multiply_by_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Multiply a quaternion by a pure-vector quaternion."""
    mat = _get_quat("_QUAT_MULTIPLY_BY_VEC", dtype=quat.dtype, device=quat.device)
    reshaped_mat = mat.view((1,) * len(quat.shape[:-1]) + mat.shape)
    return torch.sum(reshaped_mat * quat[..., :, None, None] * vec[..., None, :, None], dim=(-3, -2))

# 对旋转矩阵进行转置操作的函数
def invert_rot_mat(rot_mat: torch.Tensor) -> torch.Tensor:
    return rot_mat.transpose(-1, -2)

# 对四元数进行取逆操作的函数
def invert_quat(quat: torch.Tensor) -> torch.Tensor:
    quat_prime = quat.clone()
    quat_prime[..., 1:] *= -1
    inv = quat_prime / torch.sum(quat**2, dim=-1, keepdim=True)
    return inv

# 表示3D旋转的类，可以用旋转矩阵或四元数进行初始化
class Rotation:
    """
    A 3D rotation. Depending on how the object is initialized, the rotation is represented by either a rotation matrix
    or a quaternion, though both formats are made available by helper functions. To simplify gradient computation, the
    underlying format of the rotation cannot be changed in-place. Like Rigid, the class is designed to mimic the
    behavior of a torch Tensor, almost as if each Rotation object were a tensor of rotations, in one format or another.
    """

    def __init__(
        self,
        rot_mats: Optional[torch.Tensor] = None,
        quats: Optional[torch.Tensor] = None,
        normalize_quats: bool = True,
    ):
        """
        Args:
            rot_mats:
                A [*, 3, 3] rotation matrix tensor. Mutually exclusive with quats
            quats:
                A [*, 4] quaternion. Mutually exclusive with rot_mats. If normalize_quats is not True, must be a unit
                quaternion
            normalize_quats:
                If quats is specified, whether to normalize quats
        """
        # 检查输入参数，确保只有一个输入参数被指定
        if (rot_mats is None and quats is None) or (rot_mats is not None and quats is not None):
            raise ValueError("Exactly one input argument must be specified")

        # 检查旋转矩阵或四元数的形状是否正确
        if (rot_mats is not None and rot_mats.shape[-2:] != (3, 3)) or (quats is not None and quats.shape[-1] != 4):
            raise ValueError("Incorrectly shaped rotation matrix or quaternion")

        # 强制转换为全精度的数据类型
        if quats is not None:
            quats = quats.to(dtype=torch.float32)
        if rot_mats is not None:
            rot_mats = rot_mats.to(dtype=torch.float32)

        # 如果指定了四元数并需要规范化，则进行四元数的规范化
        if quats is not None and normalize_quats:
            quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)

        # 将旋转矩阵和四元数赋值给私有成员变量
        self._rot_mats = rot_mats
        self._quats = quats

    @staticmethod
    def identity(
        shape,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
        fmt: str = "quat",
    ) -> Rotation:
        """
        Returns an identity Rotation.

        Args:
            shape:
                The "shape" of the resulting Rotation object. See documentation for the shape property
            dtype:
                The torch dtype for the rotation
            device:
                The torch device for the new rotation
            requires_grad:
                Whether the underlying tensors in the new rotation object should require gradient computation
            fmt:
                One of "quat" or "rot_mat". Determines the underlying format of the new object's rotation
        Returns:
            A new identity rotation
        """
        # 返回一个单位旋转矩阵或四元数
        if fmt == "rot_mat":
            rot_mats = identity_rot_mats(
                shape,
                dtype,
                device,
                requires_grad,
            )
            return Rotation(rot_mats=rot_mats, quats=None)
        elif fmt == "quat":
            quats = identity_quats(shape, dtype, device, requires_grad)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError(f"Invalid format: f{fmt}")

    # Magic methods
    def __getitem__(self, index: Any) -> Rotation:
        """
        允许使用类似于 torch 的索引方式在旋转对象的虚拟形状上进行索引。参见形状属性的文档。

        Args:
            index:
                一个 torch 索引。例如 (1, 3, 2)，或 (slice(None,))
        Returns:
            索引后的旋转对象
        """
        if type(index) != tuple:
            index = (index,)

        # 如果旋转矩阵不为 None
        if self._rot_mats is not None:
            # 从旋转矩阵中取出对应索引的部分，形成新的旋转矩阵
            rot_mats = self._rot_mats[index + (slice(None), slice(None))]
            return Rotation(rot_mats=rot_mats)
        # 如果四元数不为 None
        elif self._quats is not None:
            # 从四元数中取出对应索引的部分，形成新的四元数
            quats = self._quats[index + (slice(None),)]
            return Rotation(quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def __mul__(self, right: torch.Tensor) -> Rotation:
        """
        旋转对象与张量的逐点左乘。可用于例如对旋转进行掩码。

        Args:
            right:
                张量乘数
        Returns:
            乘积
        """
        if not (isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        # 如果旋转矩阵不为 None
        if self._rot_mats is not None:
            # 旋转矩阵逐点与张量 right 相乘，得到新的旋转矩阵
            rot_mats = self._rot_mats * right[..., None, None]
            return Rotation(rot_mats=rot_mats, quats=None)
        # 如果四元数不为 None
        elif self._quats is not None:
            # 四元数逐点与张量 right 相乘，得到新的四元数
            quats = self._quats * right[..., None]
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def __rmul__(self, left: torch.Tensor) -> Rotation:
        """
        旋转对象与张量的逆点乘。

        Args:
            left:
                左乘数
        Returns:
            乘积
        """
        # 调用 __mul__ 方法，因为逆点乘和点乘的计算方式相同
        return self.__mul__(left)

    # 属性

    @property
    def shape(self) -> torch.Size:
        """
        返回旋转对象的虚拟形状。该形状定义为基础旋转矩阵或四元数的批处理维度。例如，如果旋转对象用一个 [10, 3, 3] 的旋转矩阵张量初始化，
        结果形状将为 [10]。

        Returns:
            旋转对象的虚拟形状
        """
        # 如果旋转矩阵不为 None
        if self._rot_mats is not None:
            return self._rot_mats.shape[:-2]
        # 如果四元数不为 None
        elif self._quats is not None:
            return self._quats.shape[:-1]
        else:
            raise ValueError("Both rotations are None")

    @property
    # 返回底层旋转的数据类型
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the underlying rotation.

        Returns:
            The dtype of the underlying rotation
        """
        # 如果旋转矩阵不为 None，则返回其数据类型
        if self._rot_mats is not None:
            return self._rot_mats.dtype
        # 如果四元数不为 None，则返回其数据类型
        elif self._quats is not None:
            return self._quats.dtype
        # 如果旋转矩阵和四元数都为 None，则抛出 ValueError
        else:
            raise ValueError("Both rotations are None")

    # 返回底层旋转的设备
    @property
    def device(self) -> torch.device:
        """
        The device of the underlying rotation

        Returns:
            The device of the underlying rotation
        """
        # 如果旋转矩阵不为 None，则返回其设备信息
        if self._rot_mats is not None:
            return self._rot_mats.device
        # 如果四元数不为 None，则返回其设备信息
        elif self._quats is not None:
            return self._quats.device
        # 如果旋转矩阵和四元数都为 None，则抛出 ValueError
        else:
            raise ValueError("Both rotations are None")

    # 返回底层旋转的 requires_grad 属性
    @property
    def requires_grad(self) -> bool:
        """
        Returns the requires_grad property of the underlying rotation

        Returns:
            The requires_grad property of the underlying tensor
        """
        # 如果旋转矩阵不为 None，则返回其 requires_grad 属性
        if self._rot_mats is not None:
            return self._rot_mats.requires_grad
        # 如果四元数不为 None，则返回其 requires_grad 属性
        elif self._quats is not None:
            return self._quats.requires_grad
        # 如果旋转矩阵和四元数都为 None，则抛出 ValueError
        else:
            raise ValueError("Both rotations are None")

    # 返回底层旋转矩阵
    def get_rot_mats(self) -> torch.Tensor:
        """
        Returns the underlying rotation as a rotation matrix tensor.

        Returns:
            The rotation as a rotation matrix tensor
        """
        # 如果旋转矩阵不为 None，则直接返回它
        if self._rot_mats is not None:
            return self._rot_mats
        # 如果四元数不为 None，则将其转换为旋转矩阵并返回
        elif self._quats is not None:
            return quat_to_rot(self._quats)
        # 如果旋转矩阵和四元数都为 None，则抛出 ValueError
        else:
            raise ValueError("Both rotations are None")

    # 返回底层旋转的四元数表示
    def get_quats(self) -> torch.Tensor:
        """
        Returns the underlying rotation as a quaternion tensor.

        Depending on whether the Rotation was initialized with a quaternion, this function may call torch.linalg.eigh.

        Returns:
            The rotation as a quaternion tensor.
        """
        # 如果旋转矩阵不为 None，则将其转换为四元数并返回
        if self._rot_mats is not None:
            return rot_to_quat(self._rot_mats)
        # 如果四元数不为 None，则直接返回它
        elif self._quats is not None:
            return self._quats
        # 如果旋转矩阵和四元数都为 None，则抛出 ValueError
        else:
            raise ValueError("Both rotations are None")

    # 返回当前存储的旋转
    def get_cur_rot(self) -> torch.Tensor:
        """
        Return the underlying rotation in its current form

        Returns:
            The stored rotation
        """
        # 如果旋转矩阵不为 None，则直接返回它
        if self._rot_mats is not None:
            return self._rot_mats
        # 如果四元数不为 None，则直接返回它
        elif self._quats is not None:
            return self._quats
        # 如果旋转矩阵和四元数都为 None，则抛出 ValueError
        else:
            raise ValueError("Both rotations are None")

    # Rotation functions
    # 旋转函数（这里是个占位符，没有具体实现）
    def compose_q_update_vec(self, q_update_vec: torch.Tensor, normalize_quats: bool = True) -> Rotation:
        """
        Returns a new quaternion Rotation after updating the current object's underlying rotation with a quaternion
        update, formatted as a [*, 3] tensor whose final three columns represent x, y, z such that (1, x, y, z) is the
        desired (not necessarily unit) quaternion update.

        Args:
            q_update_vec:
                A [*, 3] quaternion update tensor
            normalize_quats:
                Whether to normalize the output quaternion
        Returns:
            An updated Rotation
        """
        # 获取当前对象的四元数
        quats = self.get_quats()
        # 计算新的四元数，通过当前四元数和四元数更新向量相乘
        new_quats = quats + quat_multiply_by_vec(quats, q_update_vec)
        # 返回一个新的 Rotation 对象，更新了四元数
        return Rotation(
            rot_mats=None,
            quats=new_quats,
            normalize_quats=normalize_quats,
        )

    def compose_r(self, r: Rotation) -> Rotation:
        """
        Compose the rotation matrices of the current Rotation object with those of another.

        Args:
            r:
                An update rotation object
        Returns:
            An updated rotation object
        """
        # 获取当前 Rotation 对象的旋转矩阵
        r1 = self.get_rot_mats()
        # 获取另一个 Rotation 对象的旋转矩阵
        r2 = r.get_rot_mats()
        # 计算新的旋转矩阵，通过矩阵相乘
        new_rot_mats = rot_matmul(r1, r2)
        # 返回一个新的 Rotation 对象，更新了旋转矩阵
        return Rotation(rot_mats=new_rot_mats, quats=None)

    def compose_q(self, r: Rotation, normalize_quats: bool = True) -> Rotation:
        """
        Compose the quaternions of the current Rotation object with those of another.

        Depending on whether either Rotation was initialized with quaternions, this function may call
        torch.linalg.eigh.

        Args:
            r:
                An update rotation object
        Returns:
            An updated rotation object
        """
        # 获取当前 Rotation 对象的四元数
        q1 = self.get_quats()
        # 获取另一个 Rotation 对象的四元数
        q2 = r.get_quats()
        # 计算新的四元数，通过四元数相乘
        new_quats = quat_multiply(q1, q2)
        # 返回一个新的 Rotation 对象，更新了四元数
        return Rotation(rot_mats=None, quats=new_quats, normalize_quats=normalize_quats)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Apply the current Rotation as a rotation matrix to a set of 3D coordinates.

        Args:
            pts:
                A [*, 3] set of points
        Returns:
            [*, 3] rotated points
        """
        # 获取当前 Rotation 对象的旋转矩阵
        rot_mats = self.get_rot_mats()
        # 将旋转矩阵应用到点集上，实现旋转
        return rot_vec_mul(rot_mats, pts)

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        The inverse of the apply() method.

        Args:
            pts:
                A [*, 3] set of points
        Returns:
            [*, 3] inverse-rotated points
        """
        # 获取当前 Rotation 对象的旋转矩阵
        rot_mats = self.get_rot_mats()
        # 计算旋转矩阵的逆矩阵
        inv_rot_mats = invert_rot_mat(rot_mats)
        # 将逆矩阵应用到点集上，实现逆向旋转
        return rot_vec_mul(inv_rot_mats, pts)
    # 返回当前旋转操作的逆操作
    def invert(self) -> Rotation:
        """
        Returns the inverse of the current Rotation.

        Returns:
            The inverse of the current Rotation
        """
        # 如果旋转矩阵不为空，则返回其逆矩阵所对应的旋转对象
        if self._rot_mats is not None:
            return Rotation(rot_mats=invert_rot_mat(self._rot_mats), quats=None)
        # 如果四元数不为空，则返回其逆四元数所对应的旋转对象
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=invert_quat(self._quats),
                normalize_quats=False,
            )
        else:
            # 如果旋转矩阵和四元数都为空，则抛出数值错误异常
            raise ValueError("Both rotations are None")

    # "Tensor" stuff

    # 对象在指定维度上添加一个大小为1的维度
    def unsqueeze(self, dim: int) -> Rotation:
        """
        Analogous to torch.unsqueeze. The dimension is relative to the shape of the Rotation object.

        Args:
            dim: A positive or negative dimension index.
        Returns:
            The unsqueezed Rotation.
        """
        # 如果给定维度超出了当前形状的维度范围，则抛出数值错误异常
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")

        # 如果旋转矩阵不为空，则在指定维度上添加一个大小为1的维度
        if self._rot_mats is not None:
            rot_mats = self._rot_mats.unsqueeze(dim if dim >= 0 else dim - 2)
            return Rotation(rot_mats=rot_mats, quats=None)
        # 如果四元数不为空，则在指定维度上添加一个大小为1的维度
        elif self._quats is not None:
            quats = self._quats.unsqueeze(dim if dim >= 0 else dim - 1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            # 如果旋转矩阵和四元数都为空，则抛出数值错误异常
            raise ValueError("Both rotations are None")

    # 沿着指定维度连接旋转对象
    @staticmethod
    def cat(rs: Sequence[Rotation], dim: int) -> Rotation:
        """
        Concatenates rotations along one of the batch dimensions. Analogous to torch.cat().

        Note that the output of this operation is always a rotation matrix, regardless of the format of input
        rotations.

        Args:
            rs:
                A list of rotation objects
            dim:
                The dimension along which the rotations should be concatenated
        Returns:
            A concatenated Rotation object in rotation matrix format
        """
        # 沿着指定维度连接传入的旋转对象并返回旋转矩阵格式的旋转对象
        rot_mats = torch.cat(
            [r.get_rot_mats() for r in rs],
            dim=dim if dim >= 0 else dim - 2,
        )

        return Rotation(rot_mats=rot_mats, quats=None)
    def map_tensor_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rotation:
        """
        Apply a Tensor -> Tensor function to underlying rotation tensors, mapping over the rotation dimension(s). Can
        be used e.g. to sum out a one-hot batch dimension.

        Args:
            fn:
                A Tensor -> Tensor function to be mapped over the Rotation
        Returns:
            The transformed Rotation object
        """
        # 检查是否存在旋转矩阵，如果存在，则对旋转矩阵应用fn函数并返回新的Rotation对象
        if self._rot_mats is not None:
            # 重新调整旋转矩阵的形状以便进行映射处理
            rot_mats = self._rot_mats.view(self._rot_mats.shape[:-2] + (9,))
            # 对旋转矩阵中的每个张量应用fn函数，然后将结果堆叠成新的张量
            rot_mats = torch.stack(list(map(fn, torch.unbind(rot_mats, dim=-1))), dim=-1)
            # 重新调整旋转矩阵的形状
            rot_mats = rot_mats.view(rot_mats.shape[:-1] + (3, 3))
            # 返回新的Rotation对象
            return Rotation(rot_mats=rot_mats, quats=None)
        # 检查是否存在四元数，如果存在，则对四元数应用fn函数并返回新的Rotation对象
        elif self._quats is not None:
            # 对四元数中的每个张量应用fn函数，然后将结果堆叠成新的张量
            quats = torch.stack(list(map(fn, torch.unbind(self._quats, dim=-1))), dim=-1)
            # 返回新的Rotation对象
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            # 如果旋转矩阵和四元数都不存在，则引发数值错误
            raise ValueError("Both rotations are None")

    def cuda(self) -> Rotation:
        """
        Analogous to the cuda() method of torch Tensors

        Returns:
            A copy of the Rotation in CUDA memory
        """
        # 检查是否存在旋转矩阵，如果存在，则将旋转矩阵复制到CUDA内存并返回新的Rotation对象
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats.cuda(), quats=None)
        # 检查是否存在四元数，如果存在，则将四元数复制到CUDA内存并返回新的Rotation对象
        elif self._quats is not None:
            return Rotation(rot_mats=None, quats=self._quats.cuda(), normalize_quats=False)
        else:
            # 如果旋转矩阵和四元数都不存在，则引发数值错误
            raise ValueError("Both rotations are None")

    def to(self, device: Optional[torch.device], dtype: Optional[torch.dtype]) -> Rotation:
        """
        Analogous to the to() method of torch Tensors

        Args:
            device:
                A torch device
            dtype:
                A torch dtype
        Returns:
            A copy of the Rotation using the new device and dtype
        """
        # 检查是否存在旋转矩阵，如果存在，则将旋转矩阵复制到新的设备并使用新的数据类型，然后返回新的Rotation对象
        if self._rot_mats is not None:
            return Rotation(
                rot_mats=self._rot_mats.to(device=device, dtype=dtype),
                quats=None,
            )
        # 检查是否存在四元数，如果存在，则将四元数复制到新的设备并使用新的数据类型，然后返回新的Rotation对象
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=self._quats.to(device=device, dtype=dtype),
                normalize_quats=False,
            )
        else:
            # 如果旋转矩阵和四元数都不存在，则引发数值错误
            raise ValueError("Both rotations are None")
    def detach(self) -> Rotation:
        """
        返回一个 Rotation 的副本，其底层 Tensor 已从其 torch 图中分离出来。

        Returns:
            底层 Tensor 已从其 torch 图中分离出来的 Rotation 的副本
        """
        # 如果旋转矩阵不为 None，则返回一个新的 Rotation 对象，其旋转矩阵已经被分离
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats.detach(), quats=None)
        # 如果四元数不为 None，则返回一个新的 Rotation 对象，其四元数已经被分离
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=self._quats.detach(),
                normalize_quats=False,
            )
        # 如果旋转矩阵和四元数都为 None，则引发 ValueError
        else:
            raise ValueError("Both rotations are None")
class Rigid:
    """
    A class representing a rigid transformation. Little more than a wrapper around two objects: a Rotation object and a
    [*, 3] translation Designed to behave approximately like a single torch tensor with the shape of the shared batch
    dimensions of its component parts.
    """

    def __init__(self, rots: Optional[Rotation], trans: Optional[torch.Tensor]):
        """
        Args:
            rots: A [*, 3, 3] rotation tensor
            trans: A corresponding [*, 3] translation tensor
        """
        # (we need device, dtype, etc. from at least one input)

        # Determine the batch dimensions, data type, device, and gradient requirement based on the input arguments
        batch_dims, dtype, device, requires_grad = None, None, None, None
        if trans is not None:
            batch_dims = trans.shape[:-1]
            dtype = trans.dtype
            device = trans.device
            requires_grad = trans.requires_grad
        elif rots is not None:
            batch_dims = rots.shape
            dtype = rots.dtype
            device = rots.device
            requires_grad = rots.requires_grad
        else:
            raise ValueError("At least one input argument must be specified")

        # If rotation object is not provided, create an identity rotation with specified properties
        if rots is None:
            rots = Rotation.identity(
                batch_dims,
                dtype,
                device,
                requires_grad,
            )
        # If translation tensor is not provided, create an identity translation with specified properties
        elif trans is None:
            trans = identity_trans(
                batch_dims,
                dtype,
                device,
                requires_grad,
            )

        # Ensure that both rotation and translation objects are not None
        assert rots is not None
        assert trans is not None

        # Check if the shapes and devices of rotation and translation are compatible
        if (rots.shape != trans.shape[:-1]) or (rots.device != trans.device):
            raise ValueError("Rots and trans incompatible")

        # Force full precision by converting the translation tensor to dtype=torch.float32
        trans = trans.to(dtype=torch.float32)

        # Set the rotation and translation objects as instance variables
        self._rots = rots
        self._trans = trans

    @staticmethod
    def identity(
        shape: Tuple[int, ...],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
        fmt: str = "quat",
    ) -> Rigid:
        """
        Constructs an identity transformation.

        Args:
            shape:
                The desired shape
            dtype:
                The dtype of both internal tensors
            device:
                The device of both internal tensors
            requires_grad:
                Whether grad should be enabled for the internal tensors
        Returns:
            The identity transformation
        """
        # Create an identity transformation using an identity rotation and translation
        return Rigid(
            Rotation.identity(shape, dtype, device, requires_grad, fmt=fmt),
            identity_trans(shape, dtype, device, requires_grad),
        )
    def __getitem__(self, index: Any) -> Rigid:
        """
        使用类似 PyTorch 的索引对仿射变换进行索引。索引应用于旋转和平移的共享维度。

        例如::

            r = Rotation(rot_mats=torch.rand(10, 10, 3, 3), quats=None) t = Rigid(r, torch.rand(10, 10, 3)) indexed =
            t[3, 4:6] assert(indexed.shape == (2,)) assert(indexed.get_rots().shape == (2,))
            assert(indexed.get_trans().shape == (2, 3))

        Args:
            index: 标准的 torch 张量索引。例如 8, (10, None, 3), 或 (3, slice(0, 1, None))
        Returns:
            索引后的张量
        """
        if type(index) != tuple:
            index = (index,)

        return Rigid(
            self._rots[index],
            self._trans[index + (slice(None),)],
        )

    def __mul__(self, right: torch.Tensor) -> Rigid:
        """
        以点乘方式将变换与张量相乘。可用于例如对刚体进行遮罩。

        Args:
            right: 张量乘数
        Returns:
            乘积
        """
        if not (isinstance(right, torch.Tensor)):
            raise TypeError("另一个乘数必须是 Tensor 类型")

        new_rots = self._rots * right
        new_trans = self._trans * right[..., None]

        return Rigid(new_rots, new_trans)

    def __rmul__(self, left: torch.Tensor) -> Rigid:
        """
        将变换与张量以点乘方式相乘（反向）。

        Args:
            left: 左侧乘数
        Returns:
            乘积
        """
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        """
        返回旋转和平移的共享维度的形状。

        Returns:
            变换的形状
        """
        return self._trans.shape[:-1]

    @property
    def device(self) -> torch.device:
        """
        返回 Rigid 张量所在的设备。

        Returns:
            Rigid 张量所在的设备
        """
        return self._trans.device

    def get_rots(self) -> Rotation:
        """
        旋转获取器。

        Returns:
            旋转对象
        """
        return self._rots

    def get_trans(self) -> torch.Tensor:
        """
        平移获取器。

        Returns:
            存储的平移
        """
        return self._trans
    def compose_q_update_vec(self, q_update_vec: torch.Tensor) -> Rigid:
        """
        Composes the transformation with a quaternion update vector of shape [*, 6], where the final 6 columns
        represent the x, y, and z values of a quaternion of form (1, x, y, z) followed by a 3D translation.

        Args:
            q_vec: The quaternion update vector.
        Returns:
            The composed transformation.
        """
        # 将输入的 quaternion update vector 分为旋转和平移部分
        q_vec, t_vec = q_update_vec[..., :3], q_update_vec[..., 3:]
        # 通过旋转部分更新旋转
        new_rots = self._rots.compose_q_update_vec(q_vec)

        # 通过旋转对平移部分进行更新
        trans_update = self._rots.apply(t_vec)
        new_translation = self._trans + trans_update

        return Rigid(new_rots, new_translation)

    def compose(self, r: Rigid) -> Rigid:
        """
        Composes the current rigid object with another.

        Args:
            r:
                Another Rigid object
        Returns:
            The composition of the two transformations
        """
        # 合成当前的刚体变换和另一个刚体变换
        new_rot = self._rots.compose_r(r._rots)
        new_trans = self._rots.apply(r._trans) + self._trans
        return Rigid(new_rot, new_trans)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Applies the transformation to a coordinate tensor.

        Args:
            pts: A [*, 3] coordinate tensor.
        Returns:
            The transformed points.
        """
        # 将刚体变换应用于坐标张量
        rotated = self._rots.apply(pts)
        return rotated + self._trans

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse of the transformation to a coordinate tensor.

        Args:
            pts: A [*, 3] coordinate tensor
        Returns:
            The transformed points.
        """
        # 对坐标张量应用刚体变换的逆转
        pts = pts - self._trans
        return self._rots.invert_apply(pts)

    def invert(self) -> Rigid:
        """
        Inverts the transformation.

        Returns:
            The inverse transformation.
        """
        # 计算刚体变换的逆转
        rot_inv = self._rots.invert()
        trn_inv = rot_inv.apply(self._trans)

        return Rigid(rot_inv, -1 * trn_inv)

    def map_tensor_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rigid:
        """
        Apply a Tensor -> Tensor function to underlying translation and rotation tensors, mapping over the
        translation/rotation dimensions respectively.

        Args:
            fn:
                A Tensor -> Tensor function to be mapped over the Rigid
        Returns:
            The transformed Rigid object
        """
        # 将给定的函数应用于底层平移和旋转张量，分别映射到平移/旋转维度
        new_rots = self._rots.map_tensor_fn(fn)
        new_trans = torch.stack(list(map(fn, torch.unbind(self._trans, dim=-1))), dim=-1)

        return Rigid(new_rots, new_trans)
    def to_tensor_4x4(self) -> torch.Tensor:
        """
        Converts a transformation to a homogenous transformation tensor.

        Returns:
            A [*, 4, 4] homogenous transformation tensor
        """
        # 创建一个全零的张量，形状为[*self.shape, 4, 4]
        tensor = self._trans.new_zeros((*self.shape, 4, 4))
        # 将旋转部分的旋转矩阵赋值给张量的前三行前三列
        tensor[..., :3, :3] = self._rots.get_rot_mats()
        # 将平移向量赋值给张量的前三行第四列
        tensor[..., :3, 3] = self._trans
        # 设置张量的最后一个元素为1
        tensor[..., 3, 3] = 1
        # 返回转换后的张量
        return tensor

    @staticmethod
    def from_tensor_4x4(t: torch.Tensor) -> Rigid:
        """
        Constructs a transformation from a homogenous transformation tensor.

        Args:
            t: [*, 4, 4] homogenous transformation tensor
        Returns:
            T object with shape [*]
        """
        # 检查输入张量的形状是否正确
        if t.shape[-2:] != (4, 4):
            raise ValueError("Incorrectly shaped input tensor")

        # 从输入张量的旋转矩阵部分构造旋转对象
        rots = Rotation(rot_mats=t[..., :3, :3], quats=None)
        # 获取输入张量的平移向量部分
        trans = t[..., :3, 3]

        # 构造刚体变换对象并返回
        return Rigid(rots, trans)

    def to_tensor_7(self) -> torch.Tensor:
        """
        Converts a transformation to a tensor with 7 final columns, four for the quaternion followed by three for the
        translation.

        Returns:
            A [*, 7] tensor representation of the transformation
        """
        # 创建一个全零的张量，形状为[*self.shape, 7]
        tensor = self._trans.new_zeros((*self.shape, 7))
        # 将旋转部分的四元数赋值给张量的前四列
        tensor[..., :4] = self._rots.get_quats()
        # 将平移向量赋值给张量的后三列
        tensor[..., 4:] = self._trans

        return tensor

    @staticmethod
    def from_tensor_7(t: torch.Tensor, normalize_quats: bool = False) -> Rigid:
        # 检查输入张量的形状是否正确
        if t.shape[-1] != 7:
            raise ValueError("Incorrectly shaped input tensor")

        # 获取输入张量的四元数部分和平移向量部分
        quats, trans = t[..., :4], t[..., 4:]

        # 从四元数构造旋转对象
        rots = Rotation(rot_mats=None, quats=quats, normalize_quats=normalize_quats)

        # 构造刚体变换对象并返回
        return Rigid(rots, trans)

    @staticmethod
    def from_3_points(
        p_neg_x_axis: torch.Tensor, origin: torch.Tensor, p_xy_plane: torch.Tensor, eps: float = 1e-8
    ) -> Rigid:
        """
        Implements algorithm 21. Constructs transformations from sets of 3 points using the Gram-Schmidt algorithm.

        Args:
            p_neg_x_axis: [*, 3] coordinates
            origin: [*, 3] coordinates used as frame origins
            p_xy_plane: [*, 3] coordinates
            eps: Small epsilon value
        Returns:
            A transformation object of shape [*]
        """
        # 根据最后一个维度对 p_neg_x_axis 进行分解
        p_neg_x_axis_unbound = torch.unbind(p_neg_x_axis, dim=-1)
        # 根据最后一个维度对 origin 进行分解
        origin_unbound = torch.unbind(origin, dim=-1)
        # 根据最后一个维度对 p_xy_plane 进行分解
        p_xy_plane_unbound = torch.unbind(p_xy_plane, dim=-1)

        # 计算 e0 向量
        e0 = [c1 - c2 for c1, c2 in zip(origin_unbound, p_neg_x_axis_unbound)]
        # 计算 e1 向量
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane_unbound, origin_unbound)]

        # 计算 e0 的模，避免除零错误
        denom = torch.sqrt(sum(c * c for c in e0) + eps * torch.ones_like(e0[0]))
        # 归一化 e0 向量
        e0 = [c / denom for c in e0]
        # 计算 e0 和 e1 的点积
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        # 计算 e1 向量
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        # 计算 e1 的模
        denom = torch.sqrt(sum((c * c for c in e1)) + eps * torch.ones_like(e1[0]))
        # 归一化 e1 向量
        e1 = [c / denom for c in e1]
        # 计算 e2 向量
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        # 将 e0、e1、e2 向量合并成 rots 矩阵
        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        # 将 rots 重塑成 3x3 的矩阵形式
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        # 创建 Rotation 对象
        rot_obj = Rotation(rot_mats=rots, quats=None)

        # 返回 Rigid 对象
        return Rigid(rot_obj, torch.stack(origin_unbound, dim=-1))

    def unsqueeze(self, dim: int) -> Rigid:
        """
        Analogous to torch.unsqueeze. The dimension is relative to the shared dimensions of the rotation/translation.

        Args:
            dim: A positive or negative dimension index.
        Returns:
            The unsqueezed transformation.
        """
        # 如果指定的维度大于共享维度的数量，抛出 ValueError
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        # 对旋转矩阵进行 unsqueeze 操作
        rots = self._rots.unsqueeze(dim)
        # 对平移矢量进行 unsqueeze 操作
        trans = self._trans.unsqueeze(dim if dim >= 0 else dim - 1)

        # 返回新的 Rigid 对象
        return Rigid(rots, trans)

    @staticmethod
    def cat(ts: Sequence[Rigid], dim: int) -> Rigid:
        """
        Concatenates transformations along a new dimension.

        Args:
            ts:
                A list of T objects
            dim:
                The dimension along which the transformations should be concatenated
        Returns:
            A concatenated transformation object
        """
        # 将多个 ts 对象的旋转矩阵按指定维度进行拼接
        rots = Rotation.cat([t._rots for t in ts], dim)
        # 将多个 ts 对象的平移矢量按指定维度进行拼接
        trans = torch.cat([t._trans for t in ts], dim=dim if dim >= 0 else dim - 1)

        # 返回新的 Rigid 对象
        return Rigid(rots, trans)
    # 将一个 Rotation -> Rotation 的函数应用于存储的旋转对象
    def apply_rot_fn(self, fn: Callable[[Rotation], Rotation]) -> Rigid:
        """
        Applies a Rotation -> Rotation function to the stored rotation object.

        Args:
            fn: A function of type Rotation -> Rotation
        Returns:
            A transformation object with a transformed rotation.
        """
        return Rigid(fn(self._rots), self._trans)

    # 将一个 Tensor -> Tensor 的函数应用于存储的平移
    def apply_trans_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rigid:
        """
        Applies a Tensor -> Tensor function to the stored translation.

        Args:
            fn:
                A function of type Tensor -> Tensor to be applied to the translation
        Returns:
            A transformation object with a transformed translation.
        """
        return Rigid(self._rots, fn(self._trans))

    # 将平移按照一个常数因子进行缩放
    def scale_translation(self, trans_scale_factor: float) -> Rigid:
        """
        Scales the translation by a constant factor.

        Args:
            trans_scale_factor:
                The constant factor
        Returns:
            A transformation object with a scaled translation.
        """
        return self.apply_trans_fn(lambda t: t * trans_scale_factor)

    # 分离底层旋转对象的梯度
    def stop_rot_gradient(self) -> Rigid:
        """
        Detaches the underlying rotation object

        Returns:
            A transformation object with detached rotations
        """
        return self.apply_rot_fn(lambda r: r.detach())

    # 从参考点创建变换
    @staticmethod
    def make_transform_from_reference(
        n_xyz: torch.Tensor, ca_xyz: torch.Tensor, c_xyz: torch.Tensor, eps: float = 1e-20
    ) -> Rigid:
        """
        从参考坐标返回一个变换对象。

        注意，该方法不处理对称性。如果您以非标准方式提供原子位置，氮原子将不会位于[-0.527250，1.359329，0.0]，而是位于[-0.527250，-1.359329，0.0]。您需要在代码中处理这种情况。

        Args:
            n_xyz: 一个形状为[*, 3]的张量，表示氮原子的xyz坐标。
            ca_xyz: 一个形状为[*, 3]的张量，表示α-碳原子的xyz坐标。
            c_xyz: 一个形状为[*, 3]的张量，表示碳原子的xyz坐标。
        Returns:
            一个变换对象。将平移和旋转应用于参考骨架后，坐标将近似等于输入坐标。
        """
        # 计算平移向量
        translation = -1 * ca_xyz
        # 应用平移向量到氮原子坐标
        n_xyz = n_xyz + translation
        # 应用平移向量到碳原子坐标
        c_xyz = c_xyz + translation

        # 计算旋转矩阵的第一部分
        c_x, c_y, c_z = [c_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + c_x**2 + c_y**2)
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm

        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1

        # 计算旋转矩阵的第二部分
        norm = torch.sqrt(eps + c_x**2 + c_y**2 + c_z**2)
        sin_c2 = c_z / norm
        cos_c2 = torch.sqrt(c_x**2 + c_y**2) / norm

        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c2_rots[..., 2, 0] = -1 * sin_c2
        c2_rots[..., 2, 2] = cos_c2

        # 计算总的旋转矩阵
        c_rots = rot_matmul(c2_rots, c1_rots)
        n_xyz = rot_vec_mul(c_rots, n_xyz)

        # 计算氮原子的旋转矩阵
        _, n_y, n_z = [n_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + n_y**2 + n_z**2)
        sin_n = -n_z / norm
        cos_n = n_y / norm

        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n

        # 计算最终的旋转矩阵
        rots = rot_matmul(n_rots, c_rots)

        # 转置旋转矩阵
        rots = rots.transpose(-1, -2)
        # 计算逆平移向量
        translation = -1 * translation

        # 创建旋转对象
        rot_obj = Rotation(rot_mats=rots, quats=None)

        # 返回刚体变换对象
        return Rigid(rot_obj, translation)

    def cuda(self) -> Rigid:
        """
        将变换对象移到GPU内存中

        Returns:
            在GPU上的变换版本
        """
        # 将旋转矩阵和平移向量移动到GPU
        return Rigid(self._rots.cuda(), self._trans.cuda())
```