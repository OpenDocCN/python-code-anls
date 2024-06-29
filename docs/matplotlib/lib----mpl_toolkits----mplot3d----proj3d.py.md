# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\mplot3d\proj3d.py`

```
"""
Various transforms used for by the 3D code
"""

import numpy as np  # 导入 NumPy 库

from matplotlib import _api  # 导入 Matplotlib 的 _api 模块


def world_transformation(xmin, xmax,
                         ymin, ymax,
                         zmin, zmax, pb_aspect=None):
    """
    Produce a matrix that scales homogeneous coords in the specified ranges
    to [0, 1], or [0, pb_aspect[i]] if the plotbox aspect ratio is specified.
    """
    dx = xmax - xmin  # 计算 x 方向的范围
    dy = ymax - ymin  # 计算 y 方向的范围
    dz = zmax - zmin  # 计算 z 方向的范围
    if pb_aspect is not None:
        ax, ay, az = pb_aspect
        dx /= ax  # 根据 plotbox 的比例缩放 x 方向
        dy /= ay  # 根据 plotbox 的比例缩放 y 方向
        dz /= az  # 根据 plotbox 的比例缩放 z 方向

    return np.array([[1/dx, 0,    0,    -xmin/dx],  # 返回仿射变换矩阵
                     [0,    1/dy, 0,    -ymin/dy],
                     [0,    0,    1/dz, -zmin/dz],
                     [0,    0,    0,    1]])


@_api.deprecated("3.8")
def rotation_about_vector(v, angle):
    """
    Produce a rotation matrix for an angle in radians about a vector.
    """
    return _rotation_about_vector(v, angle)  # 调用 _rotation_about_vector 函数并返回其结果


def _rotation_about_vector(v, angle):
    """
    Produce a rotation matrix for an angle in radians about a vector.
    """
    vx, vy, vz = v / np.linalg.norm(v)  # 将向量 v 单位化
    s = np.sin(angle)  # 计算正弦值
    c = np.cos(angle)  # 计算余弦值
    t = 2*np.sin(angle/2)**2  # 更稳定的计算方法，比 t = 1-c 更精确

    R = np.array([
        [t*vx*vx + c,    t*vx*vy - vz*s, t*vx*vz + vy*s],  # 构造旋转矩阵
        [t*vy*vx + vz*s, t*vy*vy + c,    t*vy*vz - vx*s],
        [t*vz*vx - vy*s, t*vz*vy + vx*s, t*vz*vz + c]])

    return R  # 返回旋转矩阵 R


def _view_axes(E, R, V, roll):
    """
    Get the unit viewing axes in data coordinates.

    Parameters
    ----------
    E : 3-element numpy array
        The coordinates of the eye/camera.
    R : 3-element numpy array
        The coordinates of the center of the view box.
    V : 3-element numpy array
        Unit vector in the direction of the vertical axis.
    roll : float
        The roll angle in radians.

    Returns
    -------
    u : 3-element numpy array
        Unit vector pointing towards the right of the screen.
    v : 3-element numpy array
        Unit vector pointing towards the top of the screen.
    w : 3-element numpy array
        Unit vector pointing out of the screen.
    """
    w = (E - R)  # 计算观察方向单位向量
    w = w/np.linalg.norm(w)  # 将观察方向单位化
    u = np.cross(V, w)  # 计算屏幕右侧方向的单位向量
    u = u/np.linalg.norm(u)  # 将屏幕右侧方向单位化
    v = np.cross(w, u)  # 计算屏幕上方方向的单位向量，结果也是单位向量

    # 对于默认 roll=0，进行一些计算优化
    if roll != 0:
        # 正向旋转摄像机意味着反向旋转世界
        Rroll = _rotation_about_vector(w, -roll)
        u = np.dot(Rroll, u)  # 应用旋转矩阵到 u
        v = np.dot(Rroll, v)  # 应用旋转矩阵到 v
    return u, v, w  # 返回单位向量 u, v, w


def _view_transformation_uvw(u, v, w, E):
    """
    Return the view transformation matrix.

    Parameters
    ----------
    u : 3-element numpy array
        Unit vector pointing towards the right of the screen.
    v : 3-element numpy array
        Unit vector pointing towards the top of the screen.
    w : 3-element numpy array
        Unit vector pointing out of the screen.
    # 定义一个函数，接受一个参数 E，该参数是一个包含三个元素的 NumPy 数组，表示眼睛或相机的坐标。
    def transform_matrix(E):
        # 创建一个4x4的单位矩阵 Mr
        Mr = np.eye(4)
        # 创建一个4x4的单位矩阵 Mt
        Mt = np.eye(4)
        # 将 Mr 的前三行前三列部分替换为给定的向量 [u, v, w]
        Mr[:3, :3] = [u, v, w]
        # 将 Mt 的前三行最后一列设置为 -E，表示平移变换
        Mt[:3, -1] = -E
        # 计算变换矩阵 M，通过将 Mr 和 Mt 矩阵相乘得到
        M = np.dot(Mr, Mt)
        # 返回计算得到的变换矩阵 M
        return M
@_api.deprecated("3.8")
# 标记函数为已弃用，推荐使用 alternative 参数指定的函数
def view_transformation(E, R, V, roll):
    """
    Return the view transformation matrix.

    Parameters
    ----------
    E : 3-element numpy array
        The coordinates of the eye/camera.
    R : 3-element numpy array
        The coordinates of the center of the view box.
    V : 3-element numpy array
        Unit vector in the direction of the vertical axis.
    roll : float
        The roll angle in radians.
    """
    # 调用 _view_axes 函数获取视图坐标轴
    u, v, w = _view_axes(E, R, V, roll)
    # 调用 _view_transformation_uvw 函数计算视图变换矩阵
    M = _view_transformation_uvw(u, v, w, E)
    return M


@_api.deprecated("3.8")
# 标记函数为已弃用，推荐使用 alternative 参数指定的函数
def persp_transformation(zfront, zback, focal_length):
    return _persp_transformation(zfront, zback, focal_length)


def _persp_transformation(zfront, zback, focal_length):
    e = focal_length
    a = 1  # aspect ratio
    b = (zfront+zback)/(zfront-zback)
    c = -2*(zfront*zback)/(zfront-zback)
    # 创建透视投影矩阵
    proj_matrix = np.array([[e,   0,  0, 0],
                            [0, e/a,  0, 0],
                            [0,   0,  b, c],
                            [0,   0, -1, 0]])
    return proj_matrix


@_api.deprecated("3.8")
# 标记函数为已弃用，推荐使用 alternative 参数指定的函数
def ortho_transformation(zfront, zback):
    return _ortho_transformation(zfront, zback)


def _ortho_transformation(zfront, zback):
    # 注意：返回向量中的 w 分量为 (zback-zfront)，而不是 1
    a = -(zfront + zback)
    b = -(zfront - zback)
    # 创建正交投影矩阵
    proj_matrix = np.array([[2, 0,  0, 0],
                            [0, 2,  0, 0],
                            [0, 0, -2, 0],
                            [0, 0,  a, b]])
    return proj_matrix


def _proj_transform_vec(vec, M):
    # 对向量进行投影变换
    vecw = np.dot(M, vec)
    w = vecw[3]
    # 执行裁剪操作
    txs, tys, tzs = vecw[0]/w, vecw[1]/w, vecw[2]/w
    return txs, tys, tzs


def _proj_transform_vec_clip(vec, M):
    # 对向量进行投影变换
    vecw = np.dot(M, vec)
    w = vecw[3]
    # 执行裁剪操作
    txs, tys, tzs = vecw[0] / w, vecw[1] / w, vecw[2] / w
    # 执行裁剪测试
    tis = (0 <= vecw[0]) & (vecw[0] <= 1) & (0 <= vecw[1]) & (vecw[1] <= 1)
    if np.any(tis):
        tis = vecw[1] < 1
    return txs, tys, tzs, tis


def inv_transform(xs, ys, zs, invM):
    """
    Transform the points by the inverse of the projection matrix, *invM*.
    """
    # 构造向量并应用逆投影变换矩阵
    vec = _vec_pad_ones(xs, ys, zs)
    vecr = np.dot(invM, vec)
    if vecr.shape == (4,):
        vecr = vecr.reshape((4, 1))
    # 根据向量第四个元素进行归一化处理
    for i in range(vecr.shape[1]):
        if vecr[3][i] != 0:
            vecr[:, i] = vecr[:, i] / vecr[3][i]
    return vecr[0], vecr[1], vecr[2]


def _vec_pad_ones(xs, ys, zs):
    # 创建并返回带有额外 '1' 的向量数组
    return np.array([xs, ys, zs, np.ones_like(xs)])


def proj_transform(xs, ys, zs, M):
    """
    Transform the points by the projection matrix *M*.
    """
    # 创建带有额外 '1' 的向量数组并应用投影变换
    vec = _vec_pad_ones(xs, ys, zs)
    return _proj_transform_vec(vec, M)


transform = _api.deprecated(
    "3.8", obj_type="function", name="transform",
    alternative="proj_transform")(proj_transform)


def proj_transform_clip(xs, ys, zs, M):
    """
    Transform the points by the projection matrix
    and return the clipping result
    """
    # 创建带有额外 '1' 的向量数组并应用带裁剪的投影变换
    vec = _vec_pad_ones(xs, ys, zs)
    return _proj_transform_vec_clip(vec, M)
    # 返回 txs, tys, tzs, tis 四个变量的元组
    """
    # 调用 _vec_pad_ones 函数，将 xs, ys, zs 向量填充为齐次坐标向量 vec
    vec = _vec_pad_ones(xs, ys, zs)
    # 调用 _proj_transform_vec_clip 函数，将 vec 进行投影变换，使用矩阵 M
    # 返回投影变换后的结果
    return _proj_transform_vec_clip(vec, M)
# 使用装饰器标记该函数已弃用，自版本3.8起不建议使用
@_api.deprecated("3.8")
def proj_points(points, M):
    # 调用内部函数 _proj_points 处理投影变换后的点集合
    return _proj_points(points, M)


# 处理点集合的投影变换，返回变换后的点的列堆叠
def _proj_points(points, M):
    # 调用内部函数 _proj_trans_points 处理点的投影变换，然后进行列堆叠
    return np.column_stack(_proj_trans_points(points, M))


# 使用装饰器标记该函数已弃用，自版本3.8起不建议使用
@_api.deprecated("3.8")
def proj_trans_points(points, M):
    # 调用内部函数 _proj_trans_points 处理点的投影变换
    return _proj_trans_points(points, M)


# 处理点的投影变换，返回变换后的点集合
def _proj_trans_points(points, M):
    # 将点集合分解为 xs, ys, zs，然后调用 proj_transform 进行投影变换
    xs, ys, zs = zip(*points)
    return proj_transform(xs, ys, zs, M)


# 使用装饰器标记该函数已弃用，自版本3.8起不建议使用
@_api.deprecated("3.8")
def rot_x(V, alpha):
    # 计算旋转矩阵的 cos 和 sin 值
    cosa, sina = np.cos(alpha), np.sin(alpha)
    # 构造绕 x 轴旋转 alpha 角度的旋转矩阵 M1
    M1 = np.array([[1, 0, 0, 0],
                   [0, cosa, -sina, 0],
                   [0, sina, cosa, 0],
                   [0, 0, 0, 1]])
    # 返回向量 V 经过 M1 变换后的结果
    return np.dot(M1, V)
```