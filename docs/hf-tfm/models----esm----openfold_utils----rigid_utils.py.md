# `.\models\esm\openfold_utils\rigid_utils.py`

```
# 引入必要的模块和库
from functools import lru_cache  # 导入 functools 库中的 lru_cache 装饰器
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple  # 引入类型提示

import numpy as np  # 导入 numpy 库，用于数值计算
import torch  # 导入 PyTorch 库，用于张量操作


def rot_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    执行两个旋转矩阵张量的矩阵乘法。手动编写以避免 AMP 下转换。

    Args:
        a: [*, 3, 3] 左乘数
        b: [*, 3, 3] 右乘数
    Returns:
        乘积 ab
    """

    def row_mul(i: int) -> torch.Tensor:
        # 计算矩阵乘法的每行结果
        return torch.stack(
            [
                a[..., i, 0] * b[..., 0, 0] + a[..., i, 1] * b[..., 1, 0] + a[..., i, 2] * b[..., 2, 0],
                a[..., i, 0] * b[..., 0, 1] + a[..., i, 1] * b[..., 1, 1] + a[..., i, 2] * b[..., 2, 1],
                a[..., i, 0] * b[..., 0, 2] + a[..., i, 1] * b[..., 1, 2] + a[..., i, 2] * b[..., 2, 2],
            ],
            dim=-1,
        )

    # 按行堆叠计算结果，形成最终的矩阵乘法结果
    return torch.stack(
        [
            row_mul(0),
            row_mul(1),
            row_mul(2),
        ],
        dim=-2,
    )


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    对向量施加旋转。手动编写以避免 AMP 下转换。

    Args:
        r: [*, 3, 3] 旋转矩阵
        t: [*, 3] 坐标张量
    Returns:
        [*, 3] 旋转后的坐标
    """
    x, y, z = torch.unbind(t, dim=-1)
    # 计算旋转后的坐标
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


@lru_cache(maxsize=None)
def identity_rot_mats(
    batch_dims: Tuple[int, ...],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    """
    返回指定批次维度下的单位旋转矩阵张量。

    Args:
        batch_dims: 批次维度的元组
        dtype: 张量数据类型，默认为 None
        device: 张量的设备，默认为 None
        requires_grad: 是否需要梯度，默认为 True
    Returns:
        torch.Tensor: 单位旋转矩阵张量
    """
    # 创建单位矩阵，并根据指定的批次维度进行形状调整和扩展
    rots = torch.eye(3, dtype=dtype, device=device, requires_grad=requires_grad)
    rots = rots.view(*((1,) * len(batch_dims)), 3, 3)
    rots = rots.expand(*batch_dims, -1, -1)
    rots = rots.contiguous()

    return rots


@lru_cache(maxsize=None)
def identity_trans(
    batch_dims: Tuple[int, ...],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    """
    返回指定批次维度下的单位平移张量。

    Args:
        batch_dims: 批次维度的元组
        dtype: 张量数据类型，默认为 None
        device: 张量的设备，默认为 None
        requires_grad: 是否需要梯度，默认为 True
    Returns:
        torch.Tensor: 单位平移张量
    """
    # 创建单位平移张量，并根据指定的批次维度进行形状调整和扩展
    trans = torch.zeros(3, dtype=dtype, device=device, requires_grad=requires_grad)
    trans = trans.view(*((1,) * len(batch_dims)), 3)
    trans = trans.expand(*batch_dims, -1)
    trans = trans.contiguous()

    return trans
# 定义一个函数，返回一个零填充的三维张量
def identity_quats(
    batch_dims: Tuple[int, ...],  # 批量维度，指定生成张量的形状
    dtype: Optional[torch.dtype] = None,  # 数据类型，默认为None
    device: Optional[torch.device] = None,  # 设备，默认为None
    requires_grad: bool = True,  # 是否需要梯度，默认为True
) -> torch.Tensor:  # 返回一个torch.Tensor对象
    trans = torch.zeros((*batch_dims, 3), dtype=dtype, device=device, requires_grad=requires_grad)
    return trans


# 使用LRU缓存装饰器包装的函数，生成批量维度的单位四元数
@lru_cache(maxsize=None)
def identity_quats(
    batch_dims: Tuple[int, ...],  # 批量维度，指定生成张量的形状
    dtype: Optional[torch.dtype] = None,  # 数据类型，默认为None
    device: Optional[torch.device] = None,  # 设备，默认为None
    requires_grad: bool = True,  # 是否需要梯度，默认为True
) -> torch.Tensor:  # 返回一个torch.Tensor对象
    quat = torch.zeros((*batch_dims, 4), dtype=dtype, device=device, requires_grad=requires_grad)

    # 使用torch.no_grad()上下文管理器，设置四元数的第一维度为1
    with torch.no_grad():
        quat[..., 0] = 1

    return quat


# 定义四元数元素列表
_quat_elements: List[str] = ["a", "b", "c", "d"]
# 生成四元数键的列表
_qtr_keys: List[str] = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
# 生成四元数键到索引的字典
_qtr_ind_dict: Dict[str, int] = {key: ind for ind, key in enumerate(_qtr_keys)}


# 定义一个将键值对列表转换为numpy数组的函数
def _to_mat(pairs: List[Tuple[str, int]]) -> np.ndarray:
    mat = np.zeros((4, 4))
    for key, value in pairs:
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value

    return mat


# 初始化一个形状为(4, 4, 3, 3)的四元数转换矩阵数组
_QTR_MAT = np.zeros((4, 4, 3, 3))
# 填充_QTR_MAT数组中的每个元素，使用_to_mat函数处理键值对列表
_QTR_MAT[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_QTR_MAT[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_QTR_MAT[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_QTR_MAT[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_QTR_MAT[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_QTR_MAT[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_QTR_MAT[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_QTR_MAT[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_QTR_MAT[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])


def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a rotation matrix.

    Args:
        quat: [*, 4] quaternions
    Returns:
        [*, 3, 3] rotation matrices
    """
    # [*, 4, 4] 扩展四元数的维度，用于矩阵乘法
    quat = quat[..., None] * quat[..., None, :]

    # [4, 4, 3, 3] 获取四元数转换矩阵
    mat = _get_quat("_QTR_MAT", dtype=quat.dtype, device=quat.device)

    # [*, 4, 4, 3, 3] 扩展_QTR_MAT数组的维度，用于矩阵乘法
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat

    # [*, 3, 3] 沿着指定维度求和，得到旋转矩阵
    return torch.sum(quat, dim=(-3, -4))


def rot_to_quat(rot: torch.Tensor) -> torch.Tensor:
    if rot.shape[-2:] != (3, 3):
        raise ValueError("Input rotation is incorrectly shaped")

    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = [[rot[..., i, j] for j in range(3)] for i in range(3)]

    k = [
        [
            xx + yy + zz,
            zy - yz,
            xz - zx,
            yx - xy,
        ],
        [
            zy - yz,
            xx - yy - zz,
            xy + yx,
            xz + zx,
        ],
        [
            xz - zx,
            xy + yx,
            yy - xx - zz,
            yz + zy,
        ],
        [
            yx - xy,
            xz + zx,
            yz + zy,
            zz - xx - yy,
        ],
    ]

    # 计算特征值和特征向量，返回最后一个特征向量作为四元数
    _, vectors = torch.linalg.eigh((1.0 / 3.0) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2))
    return vectors[..., -1]


# 初始化一个形状为(4, 4, 4)的四元数乘法表
_QUAT_MULTIPLY = np.zeros((4, 4, 4))
# 定义一个4x4的数组，用于执行四元数相乘操作的矩阵表示
_QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]

# 定义一个4x4的数组，用于执行四元数相乘操作的矩阵表示
_QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]

# 定义一个4x4的数组，用于执行四元数相乘操作的矩阵表示
_QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]]

# 定义一个4x4的数组，用于执行四元数相乘操作的矩阵表示
_QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]

# 从_QUAT_MULTIPLY中选取索引为1到末尾的切片，这是用于纯向量四元数相乘的子矩阵
_QUAT_MULTIPLY_BY_VEC = _QUAT_MULTIPLY[:, 1:, :]

# 初始化一个字典，包含缓存的四元数相关矩阵和矩阵切片
_CACHED_QUATS: Dict[str, np.ndarray] = {
    "_QTR_MAT": _QTR_MAT,
    "_QUAT_MULTIPLY": _QUAT_MULTIPLY,
    "_QUAT_MULTIPLY_BY_VEC": _QUAT_MULTIPLY_BY_VEC,
}

# 使用LRU缓存装饰器，缓存_get_quat函数的结果，以加快多次调用时的速度
@lru_cache(maxsize=None)
def _get_quat(quat_key: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(_CACHED_QUATS[quat_key], dtype=dtype, device=device)

# 执行四元数相乘操作的函数，输入两个四元数，返回它们的乘积
def quat_multiply(quat1: torch.Tensor, quat2: torch.Tensor) -> torch.Tensor:
    """Multiply a quaternion by another quaternion."""
    # 获取用于四元数相乘的矩阵
    mat = _get_quat("_QUAT_MULTIPLY", dtype=quat1.dtype, device=quat1.device)
    # 将矩阵形状调整为与输入四元数匹配
    reshaped_mat = mat.view((1,) * len(quat1.shape[:-1]) + mat.shape)
    # 执行张量运算，计算四元数乘积
    return torch.sum(reshaped_mat * quat1[..., :, None, None] * quat2[..., None, :, None], dim=(-3, -2))

# 执行四元数与纯向量四元数相乘操作的函数
def quat_multiply_by_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Multiply a quaternion by a pure-vector quaternion."""
    # 获取用于纯向量四元数相乘的子矩阵
    mat = _get_quat("_QUAT_MULTIPLY_BY_VEC", dtype=quat.dtype, device=quat.device)
    # 将矩阵形状调整为与输入四元数匹配
    reshaped_mat = mat.view((1,) * len(quat.shape[:-1]) + mat.shape)
    # 执行张量运算，计算四元数与纯向量四元数的乘积
    return torch.sum(reshaped_mat * quat[..., :, None, None] * vec[..., None, :, None], dim=(-3, -2))

# 执行旋转矩阵转置操作的函数
def invert_rot_mat(rot_mat: torch.Tensor) -> torch.Tensor:
    return rot_mat.transpose(-1, -2)

# 执行四元数取逆操作的函数
def invert_quat(quat: torch.Tensor) -> torch.Tensor:
    # 创建四元数副本
    quat_prime = quat.clone()
    # 将四元数除了第一个元素外的其余元素取反
    quat_prime[..., 1:] *= -1
    # 计算四元数的逆
    inv = quat_prime / torch.sum(quat**2, dim=-1, keepdim=True)
    return inv

# 表示一个3D旋转的类，支持旋转矩阵和四元数两种格式
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
        # 检查参数的合法性，确保只有一个输入参数被指定
        if (rot_mats is None and quats is None) or (rot_mats is not None and quats is not None):
            raise ValueError("Exactly one input argument must be specified")

        # 检查旋转矩阵和四元数的形状是否正确
        if (rot_mats is not None and rot_mats.shape[-2:] != (3, 3)) or (quats is not None and quats.shape[-1] != 4):
            raise ValueError("Incorrectly shaped rotation matrix or quaternion")

        # 强制使用全精度（float32）
        if quats is not None:
            quats = quats.to(dtype=torch.float32)
        if rot_mats is not None:
            rot_mats = rot_mats.to(dtype=torch.float32)

        # 如果指定了四元数且需要归一化，则进行归一化处理
        if quats is not None and normalize_quats:
            quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)

        # 将旋转矩阵和四元数存储在对象的私有属性中
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
        # 根据指定的 fmt 参数创建一个身份旋转对象
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
        Allows torch-style indexing over the virtual shape of the rotation object. See documentation for the shape
        property.

        Args:
            index:
                A torch index. E.g. (1, 3, 2), or (slice(None,))
        Returns:
            The indexed rotation
        """
        # 如果索引不是元组，则转换为元组形式
        if type(index) != tuple:
            index = (index,)

        # 如果存储旋转矩阵的属性不为空
        if self._rot_mats is not None:
            # 使用索引获取部分旋转矩阵，并返回一个新的 Rotation 对象
            rot_mats = self._rot_mats[index + (slice(None), slice(None))]
            return Rotation(rot_mats=rot_mats)
        # 如果存储四元数的属性不为空
        elif self._quats is not None:
            # 使用索引获取部分四元数，并返回一个新的 Rotation 对象
            quats = self._quats[index + (slice(None),)]
            return Rotation(quats=quats, normalize_quats=False)
        else:
            # 如果旋转矩阵和四元数都为空，则抛出异常
            raise ValueError("Both rotations are None")

    def __mul__(self, right: torch.Tensor) -> Rotation:
        """
        Pointwise left multiplication of the rotation with a tensor. Can be used to e.g. mask the Rotation.

        Args:
            right:
                The tensor multiplicand
        Returns:
            The product
        """
        # 确保右乘数是一个 Tensor
        if not (isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        # 如果存储旋转矩阵的属性不为空
        if self._rot_mats is not None:
            # 对旋转矩阵逐点进行左乘操作，并返回一个新的 Rotation 对象
            rot_mats = self._rot_mats * right[..., None, None]
            return Rotation(rot_mats=rot_mats, quats=None)
        # 如果存储四元数的属性不为空
        elif self._quats is not None:
            # 对四元数逐点进行左乘操作，并返回一个新的 Rotation 对象
            quats = self._quats * right[..., None]
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            # 如果旋转矩阵和四元数都为空，则抛出异常
            raise ValueError("Both rotations are None")

    def __rmul__(self, left: torch.Tensor) -> Rotation:
        """
        Reverse pointwise multiplication of the rotation with a tensor.

        Args:
            left:
                The left multiplicand
        Returns:
            The product
        """
        # 右乘的逆操作，调用 __mul__ 方法实现
        return self.__mul__(left)

    # Properties

    @property
    def shape(self) -> torch.Size:
        """
        Returns the virtual shape of the rotation object. This shape is defined as the batch dimensions of the
        underlying rotation matrix or quaternion. If the Rotation was initialized with a [10, 3, 3] rotation matrix
        tensor, for example, the resulting shape would be [10].

        Returns:
            The virtual shape of the rotation object
        """
        # 如果存储旋转矩阵的属性不为空，则返回其形状的前两个维度
        if self._rot_mats is not None:
            return self._rot_mats.shape[:-2]
        # 如果存储四元数的属性不为空，则返回其形状的前一个维度
        elif self._quats is not None:
            return self._quats.shape[:-1]
        else:
            # 如果旋转矩阵和四元数都为空，则抛出异常
            raise ValueError("Both rotations are None")

    @property
    # 返回基础旋转的数据类型（dtype）
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the underlying rotation.

        Returns:
            The dtype of the underlying rotation
        """
        # 如果存储旋转矩阵不为空，则返回其数据类型
        if self._rot_mats is not None:
            return self._rot_mats.dtype
        # 如果存储四元数不为空，则返回其数据类型
        elif self._quats is not None:
            return self._quats.dtype
        # 如果旋转矩阵和四元数都为空，则抛出数值错误异常
        else:
            raise ValueError("Both rotations are None")

    @property
    # 返回基础旋转所在的设备（device）
    def device(self) -> torch.device:
        """
        The device of the underlying rotation

        Returns:
            The device of the underlying rotation
        """
        # 如果存储旋转矩阵不为空，则返回其所在设备
        if self._rot_mats is not None:
            return self._rot_mats.device
        # 如果存储四元数不为空，则返回其所在设备
        elif self._quats is not None:
            return self._quats.device
        # 如果旋转矩阵和四元数都为空，则抛出数值错误异常
        else:
            raise ValueError("Both rotations are None")

    @property
    # 返回基础旋转张量是否需要梯度计算（requires_grad）
    def requires_grad(self) -> bool:
        """
        Returns the requires_grad property of the underlying rotation

        Returns:
            The requires_grad property of the underlying tensor
        """
        # 如果存储旋转矩阵不为空，则返回其是否需要梯度计算的属性
        if self._rot_mats is not None:
            return self._rot_mats.requires_grad
        # 如果存储四元数不为空，则返回其是否需要梯度计算的属性
        elif self._quats is not None:
            return self._quats.requires_grad
        # 如果旋转矩阵和四元数都为空，则抛出数值错误异常
        else:
            raise ValueError("Both rotations are None")

    # 返回基础旋转矩阵张量
    def get_rot_mats(self) -> torch.Tensor:
        """
        Returns the underlying rotation as a rotation matrix tensor.

        Returns:
            The rotation as a rotation matrix tensor
        """
        # 如果存储旋转矩阵不为空，则直接返回其存储的旋转矩阵张量
        if self._rot_mats is not None:
            return self._rot_mats
        # 如果存储四元数不为空，则将四元数转换为旋转矩阵张量并返回
        elif self._quats is not None:
            return quat_to_rot(self._quats)
        # 如果旋转矩阵和四元数都为空，则抛出数值错误异常
        else:
            raise ValueError("Both rotations are None")

    # 返回基础四元数张量
    def get_quats(self) -> torch.Tensor:
        """
        Returns the underlying rotation as a quaternion tensor.

        Depending on whether the Rotation was initialized with a quaternion, this function may call torch.linalg.eigh.

        Returns:
            The rotation as a quaternion tensor.
        """
        # 如果存储旋转矩阵不为空，则将旋转矩阵转换为四元数张量并返回
        if self._rot_mats is not None:
            return rot_to_quat(self._rot_mats)
        # 如果存储四元数不为空，则直接返回其存储的四元数张量
        elif self._quats is not None:
            return self._quats
        # 如果旋转矩阵和四元数都为空，则抛出数值错误异常
        else:
            raise ValueError("Both rotations are None")

    # 返回当前存储的旋转数据
    def get_cur_rot(self) -> torch.Tensor:
        """
        Return the underlying rotation in its current form

        Returns:
            The stored rotation
        """
        # 如果存储旋转矩阵不为空，则返回其存储的旋转矩阵张量
        if self._rot_mats is not None:
            return self._rot_mats
        # 如果存储四元数不为空，则返回其存储的四元数张量
        elif self._quats is not None:
            return self._quats
        # 如果旋转矩阵和四元数都为空，则抛出数值错误异常
        else:
            raise ValueError("Both rotations are None")

    # 旋转函数
    # 定义一个方法，用于计算并返回一个新的四元数旋转对象，通过一个四元数更新向量更新当前对象的底层旋转。
    # 更新向量以 [*, 3] 张量格式表示，最后三列表示 x、y、z，使得 (1, x, y, z) 是期望的（不一定是单位）四元数更新。
    
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
        # 计算新的四元数，通过当前四元数和更新向量相乘
        new_quats = quats + quat_multiply_by_vec(quats, q_update_vec)
        # 返回一个新的 Rotation 对象，使用新的四元数，可以选择是否归一化
        return Rotation(
            rot_mats=None,
            quats=new_quats,
            normalize_quats=normalize_quats,
        )
    
    # 定义一个方法，用于将当前 Rotation 对象的旋转矩阵与另一个 Rotation 对象的旋转矩阵组合。
    # 返回一个包含组合后旋转矩阵的新 Rotation 对象。
    
    def compose_r(self, r: Rotation) -> Rotation:
        """
        Compose the rotation matrices of the current Rotation object with those of another.
    
        Args:
            r:
                An update rotation object
        Returns:
            An updated rotation object
        """
        # 获取当前对象和参数对象的旋转矩阵
        r1 = self.get_rot_mats()
        r2 = r.get_rot_mats()
        # 计算新的旋转矩阵，通过矩阵乘法将两个旋转矩阵相乘
        new_rot_mats = rot_matmul(r1, r2)
        # 返回一个新的 Rotation 对象，使用新的旋转矩阵
        return Rotation(rot_mats=new_rot_mats, quats=None)
    
    # 定义一个方法，用于将当前 Rotation 对象的四元数与另一个 Rotation 对象的四元数组合。
    # 返回一个包含组合后四元数的新 Rotation 对象。
    
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
        # 获取当前对象和参数对象的四元数
        q1 = self.get_quats()
        q2 = r.get_quats()
        # 计算新的四元数，通过四元数乘法将两个四元数相乘
        new_quats = quat_multiply(q1, q2)
        # 返回一个新的 Rotation 对象，使用新的四元数，可以选择是否归一化
        return Rotation(rot_mats=None, quats=new_quats, normalize_quats=normalize_quats)
    
    # 定义一个方法，将当前 Rotation 对象的旋转矩阵作为旋转矩阵应用到一组 3D 坐标上。
    # 返回一个 [*, 3] 形状的旋转后的点坐标集合。
    
    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Apply the current Rotation as a rotation matrix to a set of 3D coordinates.
    
        Args:
            pts:
                A [*, 3] set of points
        Returns:
            [*, 3] rotated points
        """
        # 获取当前对象的旋转矩阵
        rot_mats = self.get_rot_mats()
        # 将旋转矩阵应用到点集合上，返回旋转后的点坐标
        return rot_vec_mul(rot_mats, pts)
    
    # 定义一个方法，将当前 Rotation 对象的逆旋转矩阵作为旋转矩阵应用到一组 3D 坐标上。
    # 返回一个 [*, 3] 形状的逆旋转后的点坐标集合。
    
    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        The inverse of the apply() method.
    
        Args:
            pts:
                A [*, 3] set of points
        Returns:
            [*, 3] inverse-rotated points
        """
        # 获取当前对象的旋转矩阵
        rot_mats = self.get_rot_mats()
        # 计算旋转矩阵的逆矩阵
        inv_rot_mats = invert_rot_mat(rot_mats)
        # 将逆旋转矩阵应用到点集合上，返回逆旋转后的点坐标
        return rot_vec_mul(inv_rot_mats, pts)
    def invert(self) -> Rotation:
        """
        Returns the inverse of the current Rotation.

        Returns:
            The inverse of the current Rotation
        """
        # 如果旋转矩阵不为 None，则返回其逆矩阵对应的 Rotation 对象
        if self._rot_mats is not None:
            return Rotation(rot_mats=invert_rot_mat(self._rot_mats), quats=None)
        # 如果四元数不为 None，则返回其逆四元数对应的 Rotation 对象
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=invert_quat(self._quats),
                normalize_quats=False,
            )
        else:
            # 如果旋转矩阵和四元数都为 None，则抛出数值错误异常
            raise ValueError("Both rotations are None")

    # "Tensor" stuff

    def unsqueeze(self, dim: int) -> Rotation:
        """
        Analogous to torch.unsqueeze. The dimension is relative to the shape of the Rotation object.

        Args:
            dim: A positive or negative dimension index.
        Returns:
            The unsqueezed Rotation.
        """
        # 如果指定的维度超出了 Rotation 对象的形状范围，则抛出数值错误异常
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")

        # 如果有旋转矩阵，则按照指定维度对旋转矩阵进行 unsqueeze 操作
        if self._rot_mats is not None:
            rot_mats = self._rot_mats.unsqueeze(dim if dim >= 0 else dim - 2)
            return Rotation(rot_mats=rot_mats, quats=None)
        # 如果有四元数，则按照指定维度对四元数进行 unsqueeze 操作
        elif self._quats is not None:
            quats = self._quats.unsqueeze(dim if dim >= 0 else dim - 1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            # 如果旋转矩阵和四元数都为 None，则抛出数值错误异常
            raise ValueError("Both rotations are None")

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
        # 将输入的 Rotation 对象列表中的旋转矩阵沿指定维度进行拼接
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
        # 如果存在旋转矩阵 _rot_mats
        if self._rot_mats is not None:
            # 将 _rot_mats 的形状调整为去掉最后两个维度后再加上一个长度为 9 的维度的形状
            rot_mats = self._rot_mats.view(self._rot_mats.shape[:-2] + (9,))
            # 解绑 _rot_mats 的最后一个维度，并对每个解绑后的张量应用函数 fn，然后重新堆叠起来
            rot_mats = torch.stack(list(map(fn, torch.unbind(rot_mats, dim=-1))), dim=-1)
            # 将 rot_mats 的形状调整回去，去掉最后一个维度并加上一个 3x3 的形状
            rot_mats = rot_mats.view(rot_mats.shape[:-1] + (3, 3))
            # 返回一个新的 Rotation 对象，传入新的旋转矩阵 rot_mats 和 None 的 quats
            return Rotation(rot_mats=rot_mats, quats=None)
        # 如果存在四元数 _quats
        elif self._quats is not None:
            # 对 _quats 解绑最后一个维度，并对每个解绑后的张量应用函数 fn，然后重新堆叠起来
            quats = torch.stack(list(map(fn, torch.unbind(self._quats, dim=-1))), dim=-1)
            # 返回一个新的 Rotation 对象，传入 None 的 rot_mats 和新的 quats，以及不需要归一化的标志
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            # 如果 _rot_mats 和 _quats 都是 None，则抛出异常
            raise ValueError("Both rotations are None")

    def cuda(self) -> Rotation:
        """
        Analogous to the cuda() method of torch Tensors

        Returns:
            A copy of the Rotation in CUDA memory
        """
        # 如果存在旋转矩阵 _rot_mats
        if self._rot_mats is not None:
            # 将 _rot_mats 移动到 CUDA 内存，并返回一个新的 Rotation 对象
            return Rotation(rot_mats=self._rot_mats.cuda(), quats=None)
        # 如果存在四元数 _quats
        elif self._quats is not None:
            # 将 _quats 移动到 CUDA 内存，并返回一个新的 Rotation 对象，不需要归一化
            return Rotation(rot_mats=None, quats=self._quats.cuda(), normalize_quats=False)
        else:
            # 如果 _rot_mats 和 _quats 都是 None，则抛出异常
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
        # 如果存在旋转矩阵 _rot_mats
        if self._rot_mats is not None:
            # 将 _rot_mats 转移到指定的 device 和 dtype，并返回一个新的 Rotation 对象
            return Rotation(
                rot_mats=self._rot_mats.to(device=device, dtype=dtype),
                quats=None,
            )
        # 如果存在四元数 _quats
        elif self._quats is not None:
            # 将 _quats 转移到指定的 device 和 dtype，并返回一个新的 Rotation 对象，不需要归一化
            return Rotation(
                rot_mats=None,
                quats=self._quats.to(device=device, dtype=dtype),
                normalize_quats=False,
            )
        else:
            # 如果 _rot_mats 和 _quats 都是 None，则抛出异常
            raise ValueError("Both rotations are None")
    # 返回一个 Rotation 对象的副本，其中底层的 Tensor 已从其 torch 图中分离
    def detach(self) -> Rotation:
        """
        Returns a copy of the Rotation whose underlying Tensor has been detached from its torch graph.

        Returns:
            A copy of the Rotation whose underlying Tensor has been detached from its torch graph
        """
        # 如果 _rot_mats 不为 None，则返回一个新的 Rotation 对象，其 rot_mats 被分离
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats.detach(), quats=None)
        # 如果 _quats 不为 None，则返回一个新的 Rotation 对象，其 quats 被分离
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=self._quats.detach(),
                normalize_quats=False,
            )
        else:
            # 如果 _rot_mats 和 _quats 都是 None，则抛出数值错误异常
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
        # 根据输入参数确定 batch 维度、数据类型、设备和梯度设置
        batch_dims, dtype, device, requires_grad = None, None, None, None
        if trans is not None:
            batch_dims = trans.shape[:-1]  # 获取除最后一维外的所有维度，即 batch 维度
            dtype = trans.dtype  # 获取数据类型
            device = trans.device  # 获取设备
            requires_grad = trans.requires_grad  # 获取梯度需求设置
        elif rots is not None:
            batch_dims = rots.shape  # 获取 rots 的形状作为 batch 维度
            dtype = rots.dtype  # 获取数据类型
            device = rots.device  # 获取设备
            requires_grad = rots.requires_grad  # 获取梯度需求设置
        else:
            raise ValueError("At least one input argument must be specified")  # 抛出数值错误，至少需要指定一个输入参数

        # 如果 rots 为 None，则使用 identity 方法创建默认的 Rotation 对象
        if rots is None:
            rots = Rotation.identity(
                batch_dims,
                dtype,
                device,
                requires_grad,
            )
        # 如果 trans 为 None，则使用 identity_trans 函数创建默认的 translation tensor
        elif trans is None:
            trans = identity_trans(
                batch_dims,
                dtype,
                device,
                requires_grad,
            )

        assert rots is not None
        assert trans is not None

        # 检查 rots 和 trans 的形状和设备是否兼容
        if (rots.shape != trans.shape[:-1]) or (rots.device != trans.device):
            raise ValueError("Rots and trans incompatible")  # 抛出数值错误，rots 和 trans 不兼容

        # 强制将 trans 转换为 torch.float32 数据类型
        trans = trans.to(dtype=torch.float32)

        self._rots = rots  # 将 rots 赋值给对象的 _rots 属性
        self._trans = trans  # 将 trans 赋值给对象的 _trans 属性

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
        # 使用 Rotation.identity 和 identity_trans 函数创建 identity transformation 对象并返回
        return Rigid(
            Rotation.identity(shape, dtype, device, requires_grad, fmt=fmt),
            identity_trans(shape, dtype, device, requires_grad),
        )
    def __getitem__(self, index: Any) -> Rigid:
        """
        Indexes the affine transformation with PyTorch-style indices. The index is applied to the shared dimensions of
        both the rotation and the translation.

        E.g.::

            r = Rotation(rot_mats=torch.rand(10, 10, 3, 3), quats=None)
            t = Rigid(r, torch.rand(10, 10, 3))
            indexed = t[3, 4:6]
            assert(indexed.shape == (2,))
            assert(indexed.get_rots().shape == (2,))
            assert(indexed.get_trans().shape == (2, 3))

        Args:
            index: A standard torch tensor index. E.g. 8, (10, None, 3),
            or (3, slice(0, 1, None))
        Returns:
            The indexed tensor
        """
        # 如果索引不是元组，则转换为元组形式
        if type(index) != tuple:
            index = (index,)

        # 返回一个新的 Rigid 对象，通过索引获取对应的旋转矩阵和平移向量
        return Rigid(
            self._rots[index],  # 使用索引获取旋转矩阵的子集
            self._trans[index + (slice(None),)],  # 使用索引获取平移向量的子集
        )

    def __mul__(self, right: torch.Tensor) -> Rigid:
        """
        Pointwise left multiplication of the transformation with a tensor. Can be used to e.g. mask the Rigid.

        Args:
            right:
                The tensor multiplicand
        Returns:
            The product
        """
        # 如果 right 不是 torch.Tensor 类型，则抛出类型错误异常
        if not (isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        # 对旋转矩阵和平移向量分别进行点乘操作
        new_rots = self._rots * right  # 对旋转矩阵进行点乘
        new_trans = self._trans * right[..., None]  # 对平移向量进行点乘（在最后一个维度上扩展）

        # 返回一个新的 Rigid 对象，代表点乘后的结果
        return Rigid(new_rots, new_trans)

    def __rmul__(self, left: torch.Tensor) -> Rigid:
        """
        Reverse pointwise multiplication of the transformation with a tensor.

        Args:
            left:
                The left multiplicand
        Returns:
            The product
        """
        # 调用 __mul__ 方法进行反向点乘操作
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the shared dimensions of the rotation and the translation.

        Returns:
            The shape of the transformation
        """
        # 返回旋转矩阵和平移向量共享维度的形状（去掉最后一个维度）
        return self._trans.shape[:-1]

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the Rigid's tensors are located.

        Returns:
            The device on which the Rigid's tensors are located
        """
        # 返回平移向量所在的设备
        return self._trans.device

    def get_rots(self) -> Rotation:
        """
        Getter for the rotation.

        Returns:
            The rotation object
        """
        # 返回存储的旋转矩阵对象
        return self._rots

    def get_trans(self) -> torch.Tensor:
        """
        Getter for the translation.

        Returns:
            The stored translation
        """
        # 返回存储的平移向量
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
        # Extract quaternion update vector and translation vector
        q_vec, t_vec = q_update_vec[..., :3], q_update_vec[..., 3:]
        # Compose rotations with quaternion update vector
        new_rots = self._rots.compose_q_update_vec(q_vec)

        # Apply rotations to translation vector
        trans_update = self._rots.apply(t_vec)
        # Calculate new translation by adding current translation with applied rotations
        new_translation = self._trans + trans_update

        # Return composed transformation
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
        # Compose rotations of current object with rotations of another object
        new_rot = self._rots.compose_r(r._rots)
        # Apply rotations of current object to translation of another object and add current translation
        new_trans = self._rots.apply(r._trans) + self._trans
        # Return composed transformation
        return Rigid(new_rot, new_trans)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Applies the transformation to a coordinate tensor.

        Args:
            pts: A [*, 3] coordinate tensor.
        Returns:
            The transformed points.
        """
        # Apply rotations to the input points
        rotated = self._rots.apply(pts)
        # Add translation to the rotated points
        return rotated + self._trans

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse of the transformation to a coordinate tensor.

        Args:
            pts: A [*, 3] coordinate tensor
        Returns:
            The transformed points.
        """
        # Subtract translation from the input points
        pts = pts - self._trans
        # Apply inverse rotations to the translated points
        return self._rots.invert_apply(pts)

    def invert(self) -> Rigid:
        """
        Inverts the transformation.

        Returns:
            The inverse transformation.
        """
        # Invert rotations
        rot_inv = self._rots.invert()
        # Apply inverted rotations to current translation and negate it
        trn_inv = rot_inv.apply(self._trans)

        # Return inverted transformation
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
        # Apply function to rotation tensors
        new_rots = self._rots.map_tensor_fn(fn)
        # Map function over each dimension of translation tensor and stack results
        new_trans = torch.stack(list(map(fn, torch.unbind(self._trans, dim=-1))), dim=-1)

        # Return transformed Rigid object
        return Rigid(new_rots, new_trans)
    def to_tensor_4x4(self) -> torch.Tensor:
        """
        Converts a transformation to a homogenous transformation tensor.

        Returns:
            A [*, 4, 4] homogenous transformation tensor
        """
        # 创建一个与当前对象形状相同的全零张量，形状为 [*self.shape, 4, 4]
        tensor = self._trans.new_zeros((*self.shape, 4, 4))
        # 将旋转矩阵填充到张量的前三行前三列
        tensor[..., :3, :3] = self._rots.get_rot_mats()
        # 将平移矢量填充到张量的前三行最后一列
        tensor[..., :3, 3] = self._trans
        # 最后一个元素设为1，构成齐次变换张量
        tensor[..., 3, 3] = 1
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
        # 检查输入张量的形状是否为 [*, 4, 4]
        if t.shape[-2:] != (4, 4):
            raise ValueError("Incorrectly shaped input tensor")

        # 从输入张量中提取旋转矩阵部分
        rots = Rotation(rot_mats=t[..., :3, :3], quats=None)
        # 从输入张量中提取平移矢量部分
        trans = t[..., :3, 3]

        # 返回一个 Rigid 类对象，其中包含旋转矩阵和平移矢量
        return Rigid(rots, trans)

    def to_tensor_7(self) -> torch.Tensor:
        """
        Converts a transformation to a tensor with 7 final columns, four for the quaternion followed by three for the
        translation.

        Returns:
            A [*, 7] tensor representation of the transformation
        """
        # 创建一个与当前对象形状相同的全零张量，形状为 [*self.shape, 7]
        tensor = self._trans.new_zeros((*self.shape, 7))
        # 将四元数填充到张量的前四列
        tensor[..., :4] = self._rots.get_quats()
        # 将平移矢量填充到张量的后三列
        tensor[..., 4:] = self._trans

        return tensor

    @staticmethod
    def from_tensor_7(t: torch.Tensor, normalize_quats: bool = False) -> Rigid:
        # 检查输入张量的形状是否为 [*..., 7]
        if t.shape[-1] != 7:
            raise ValueError("Incorrectly shaped input tensor")

        # 从输入张量中提取四元数部分和平移矢量部分
        quats, trans = t[..., :4], t[..., 4:]

        # 根据四元数和是否需要标准化创建 Rotation 对象
        rots = Rotation(rot_mats=None, quats=quats, normalize_quats=normalize_quats)

        # 返回一个 Rigid 类对象，其中包含旋转部分和平移部分
        return Rigid(rots, trans)

    @staticmethod
    def from_3_points(
        p_neg_x_axis: torch.Tensor, origin: torch.Tensor, p_xy_plane: torch.Tensor, eps: float = 1e-8
    ):
        # 该方法未提供完整的实现，需要进一步的代码来完成
        pass
    ) -> Rigid:
        """
        Implements algorithm 21. Constructs transformations from sets of 3 points using the Gram-Schmidt algorithm.

        Args:
            p_neg_x_axis: [*, 3] coordinates
                Coordinates of points defining the negative x-axis direction
            origin: [*, 3] coordinates used as frame origins
                Coordinates of points defining the origin of the frame
            p_xy_plane: [*, 3] coordinates
                Coordinates of points defining the xy-plane orientation
            eps: Small epsilon value
                Small value added to avoid division by zero
        Returns:
            A transformation object of shape [*]
        """
        # Unbind tensors along the last dimension
        p_neg_x_axis_unbound = torch.unbind(p_neg_x_axis, dim=-1)
        origin_unbound = torch.unbind(origin, dim=-1)
        p_xy_plane_unbound = torch.unbind(p_xy_plane, dim=-1)

        # Calculate the first two orthonormal vectors e0 and e1 using Gram-Schmidt process
        e0 = [c1 - c2 for c1, c2 in zip(origin_unbound, p_neg_x_axis_unbound)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane_unbound, origin_unbound)]

        # Normalize e0
        denom = torch.sqrt(sum(c * c for c in e0) + eps * torch.ones_like(e0[0]))
        e0 = [c / denom for c in e0]

        # Calculate e1 orthogonal to e0 and normalize it
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps * torch.ones_like(e1[0]))
        e1 = [c / denom for c in e1]

        # Calculate the third orthonormal vector e2
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        # Stack e0, e1, and e2 to form rotation matrices and reshape into proper shape
        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        # Create a Rotation object using the calculated rotation matrices
        rot_obj = Rotation(rot_mats=rots, quats=None)

        # Return a Rigid transformation object with the rotation and origin vectors stacked
        return Rigid(rot_obj, torch.stack(origin_unbound, dim=-1))

    def unsqueeze(self, dim: int) -> Rigid:
        """
        Analogous to torch.unsqueeze. The dimension is relative to the shared dimensions of the rotation/translation.

        Args:
            dim: A positive or negative dimension index.
        Returns:
            The unsqueezed transformation.
        """
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        # Unsqueeze rotation matrices and translations along the specified dimension
        rots = self._rots.unsqueeze(dim)
        trans = self._trans.unsqueeze(dim if dim >= 0 else dim - 1)

        # Return a Rigid transformation object with unsqueezed rotation and translation tensors
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
        # Concatenate rotation matrices and translations along the specified dimension
        rots = Rotation.cat([t._rots for t in ts], dim)
        trans = torch.cat([t._trans for t in ts], dim=dim if dim >= 0 else dim - 1)

        # Return a Rigid transformation object with concatenated rotation and translation tensors
        return Rigid(rots, trans)
    def apply_rot_fn(self, fn: Callable[[Rotation], Rotation]) -> Rigid:
        """
        Applies a Rotation -> Rotation function to the stored rotation object.

        Args:
            fn: A function of type Rotation -> Rotation

        Returns:
            A transformation object with a transformed rotation.
        """
        return Rigid(fn(self._rots), self._trans)



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



    def stop_rot_gradient(self) -> Rigid:
        """
        Detaches the underlying rotation object

        Returns:
            A transformation object with detached rotations
        """
        return self.apply_rot_fn(lambda r: r.detach())



    @staticmethod
    def make_transform_from_reference(
        n_xyz: torch.Tensor, ca_xyz: torch.Tensor, c_xyz: torch.Tensor, eps: float = 1e-20
    ) -> Rigid:
        """
        Constructs a transformation object based on reference points.

        Args:
            n_xyz:
                Tensor representing N atom coordinates
            ca_xyz:
                Tensor representing C-alpha atom coordinates
            c_xyz:
                Tensor representing C atom coordinates
            eps:
                Small value to avoid division by zero (default: 1e-20)

        Returns:
            A transformation object initialized with the given reference points.
        """
    ) -> Rigid:
        """
        Returns a transformation object from reference coordinates.

        Note that this method does not take care of symmetries. If you provide the atom positions in the non-standard
        way, the N atom will end up not at [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You
        need to take care of such cases in your code.

        Args:
            n_xyz: A [*, 3] tensor of nitrogen xyz coordinates.
            ca_xyz: A [*, 3] tensor of carbon alpha xyz coordinates.
            c_xyz: A [*, 3] tensor of carbon xyz coordinates.
        Returns:
            A transformation object. After applying the translation and rotation to the reference backbone, the
            coordinates will approximately equal to the input coordinates.
        """
        # Calculate translation vector by negating carbon alpha coordinates
        translation = -1 * ca_xyz
        # Translate nitrogen and carbon coordinates accordingly
        n_xyz = n_xyz + translation
        c_xyz = c_xyz + translation

        # Extract x, y, z components of carbon coordinates
        c_x, c_y, c_z = [c_xyz[..., i] for i in range(3)]
        # Compute normalization factor with epsilon smoothing
        norm = torch.sqrt(eps + c_x**2 + c_y**2)
        # Calculate sine and cosine of the first rotation angle
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm

        # Initialize rotation matrices for the first rotation (around z-axis)
        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1

        # Compute normalization factor with epsilon smoothing
        norm = torch.sqrt(eps + c_x**2 + c_y**2 + c_z**2)
        # Calculate sine and cosine of the second rotation angle
        sin_c2 = c_z / norm
        cos_c2 = torch.sqrt(c_x**2 + c_y**2) / norm

        # Initialize rotation matrices for the second rotation (around x-axis)
        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c2_rots[..., 2, 0] = -1 * sin_c2
        c2_rots[..., 2, 2] = cos_c2

        # Combine the two rotation matrices
        c_rots = rot_matmul(c2_rots, c1_rots)
        # Rotate nitrogen coordinates using the combined rotation matrix
        n_xyz = rot_vec_mul(c_rots, n_xyz)

        # Extract y, z components of rotated nitrogen coordinates
        _, n_y, n_z = [n_xyz[..., i] for i in range(3)]
        # Compute normalization factor with epsilon smoothing
        norm = torch.sqrt(eps + n_y**2 + n_z**2)
        # Calculate sine and cosine of the final rotation angle
        sin_n = -n_z / norm
        cos_n = n_y / norm

        # Initialize rotation matrices for the final rotation (around y-axis)
        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n

        # Combine all rotations to get the final rotation matrix
        rots = rot_matmul(n_rots, c_rots)

        # Transpose the rotation matrix
        rots = rots.transpose(-1, -2)
        # Negate translation vector
        translation = -1 * translation

        # Create a Rotation object using the computed rotation matrix
        rot_obj = Rotation(rot_mats=rots, quats=None)

        # Return a Rigid object encapsulating the rotation and translation
        return Rigid(rot_obj, translation)

    def cuda(self) -> Rigid:
        """
        Moves the transformation object to GPU memory

        Returns:
            A version of the transformation on GPU
        """
        # Move rotation and translation tensors to GPU
        return Rigid(self._rots.cuda(), self._trans.cuda())
```