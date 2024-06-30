# `D:\src\scipysrc\scikit-learn\examples\applications\plot_tomography_l1_reconstruction.py`

```
# 导入必要的库
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, sparse

# 导入 Lasso 和 Ridge 模型类
from sklearn.linear_model import Lasso, Ridge


def _weights(x, dx=1, orig=0):
    # 将输入 x 展平为一维数组
    x = np.ravel(x)
    # 计算 x 对应的整数下限和权重
    floor_x = np.floor((x - orig) / dx).astype(np.int64)
    alpha = (x - orig - floor_x * dx) / dx
    return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))


def _generate_center_coordinates(l_x):
    # 生成一个 l_x x l_x 的网格坐标，以浮点数格式表示
    X, Y = np.mgrid[:l_x, :l_x].astype(np.float64)
    # 将中心坐标平移到图像中心
    center = l_x / 2.0
    X += 0.5 - center
    Y += 0.5 - center
    return X, Y


def build_projection_operator(l_x, n_dir):
    """Compute the tomography design matrix.

    Parameters
    ----------

    l_x : int
        图像数组的线性大小

    n_dir : int
        获取投影的角度数量

    Returns
    -------
    p : sparse matrix of shape (n_dir * l_x, l_x**2)
        稀疏矩阵，表示投影操作的设计矩阵
    """
    # 生成中心坐标 X, Y，用于后续旋转
    X, Y = _generate_center_coordinates(l_x)
    
    # 在0到π之间生成 n_dir 个角度，不包括终点π
    angles = np.linspace(0, np.pi, n_dir, endpoint=False)
    
    # 初始化空列表用于存储数据索引、权重和相机索引
    data_inds, weights, camera_inds = [], [], []
    
    # 创建一个包含 l_x**2 个元素的一维数组作为数据的展开索引
    data_unravel_indices = np.arange(l_x**2)
    
    # 将数据展开索引数组在水平方向重复一次，使其长度变为原来的两倍
    data_unravel_indices = np.hstack((data_unravel_indices, data_unravel_indices))
    
    # 遍历角度列表，依次进行旋转和权重计算
    for i, angle in enumerate(angles):
        # 计算经过角度旋转后的新坐标 Xrot
        Xrot = np.cos(angle) * X - np.sin(angle) * Y
        
        # 计算 Xrot 对应的权重和索引
        inds, w = _weights(Xrot, dx=1, orig=X.min())
        
        # 创建布尔掩码，限制索引在有效范围内
        mask = np.logical_and(inds >= 0, inds < l_x)
        
        # 将符合条件的权重和相机索引加入对应的列表中
        weights += list(w[mask])
        camera_inds += list(inds[mask] + i * l_x)
        data_inds += list(data_unravel_indices[mask])
    
    # 使用权重、相机索引和数据索引创建稀疏矩阵表示的投影算子
    proj_operator = sparse.coo_matrix((weights, (camera_inds, data_inds)))
    
    # 返回创建的投影算子
    return proj_operator
# 定义生成合成数据的函数
def generate_synthetic_data():
    """Synthetic binary data"""
    # 使用随机种子0创建随机数生成器
    rs = np.random.RandomState(0)
    # 设定数据点数目
    n_pts = 36
    # 创建网格
    x, y = np.ogrid[0:l, 0:l]
    # 创建外部掩码
    mask_outer = (x - l / 2.0) ** 2 + (y - l / 2.0) ** 2 < (l / 2.0) ** 2
    # 创建空白掩码
    mask = np.zeros((l, l))
    # 生成随机点并将其置为1
    points = l * rs.rand(2, n_pts)
    mask[(points[0]).astype(int), (points[1]).astype(int)] = 1
    # 对掩码进行高斯滤波
    mask = ndimage.gaussian_filter(mask, sigma=l / n_pts)
    # 通过逻辑操作获取最终结果
    res = np.logical_and(mask > mask.mean(), mask_outer)
    return np.logical_xor(res, ndimage.binary_erosion(res))


# 设定图像的大小
l = 128
# 构建投影操作器
proj_operator = build_projection_operator(l, l // 7)
# 生成合成数据
data = generate_synthetic_data()
# 对数据进行投影
proj = proj_operator @ data.ravel()[:, np.newaxis]
# 加入高斯噪声
proj += 0.15 * np.random.randn(*proj.shape)

# 使用L2（岭）惩罚进行重建
rgr_ridge = Ridge(alpha=0.2)
rgr_ridge.fit(proj_operator, proj.ravel())
rec_l2 = rgr_ridge.coef_.reshape(l, l)

# 使用L1（Lasso）惩罚进行重建
# 最佳的alpha值是通过交叉验证与LassoCV确定的
rgr_lasso = Lasso(alpha=0.001)
rgr_lasso.fit(proj_operator, proj.ravel())
rec_l1 = rgr_lasso.coef_.reshape(l, l)

# 绘制图像
plt.figure(figsize=(8, 3.3))
plt.subplot(131)
plt.imshow(data, cmap=plt.cm.gray, interpolation="nearest")
plt.axis("off")
plt.title("original image")
plt.subplot(132)
plt.imshow(rec_l2, cmap=plt.cm.gray, interpolation="nearest")
plt.title("L2 penalization")
plt.axis("off")
plt.subplot(133)
plt.imshow(rec_l1, cmap=plt.cm.gray, interpolation="nearest")
plt.title("L1 penalization")
plt.axis("off")

# 调整子图之间的空间和边界
plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)

# 显示图像
plt.show()
```