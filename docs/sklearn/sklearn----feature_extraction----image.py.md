# `D:\src\scipysrc\scikit-learn\sklearn\feature_extraction\image.py`

```
"""Utilities to extract features from images."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from itertools import product            # 导入 itertools 中的 product 函数，用于迭代生成笛卡尔积
from numbers import Integral, Number, Real  # 导入 numbers 模块中的整数、数字和实数类型

import numpy as np                       # 导入 NumPy 库，并简化命名为 np
from numpy.lib.stride_tricks import as_strided  # 导入 NumPy 中的 as_strided 函数，用于创建视图

from scipy import sparse                  # 导入 SciPy 库中的 sparse 子模块，用于稀疏矩阵操作

from ..base import BaseEstimator, TransformerMixin, _fit_context  # 从相对路径的 ..base 模块导入指定类和函数
from ..utils import check_array, check_random_state  # 从 ..utils 模块导入检查数组和随机状态检查函数
from ..utils._param_validation import Hidden, Interval, RealNotInt, validate_params  # 导入参数验证相关类和函数

__all__ = [                               # 定义 __all__ 列表，指定公开的模块成员
    "PatchExtractor",
    "extract_patches_2d",
    "grid_to_graph",
    "img_to_graph",
    "reconstruct_from_patches_2d",
]

###############################################################################
# From an image to a graph


def _make_edges_3d(n_x, n_y, n_z=1):
    """Returns a list of edges for a 3D image.

    Parameters
    ----------
    n_x : int
        The size of the grid in the x direction.
    n_y : int
        The size of the grid in the y direction.
    n_z : integer, default=1
        The size of the grid in the z direction, defaults to 1
    """
    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))  # 创建包含所有顶点的数组
    edges_deep = np.vstack((vertices[:, :, :-1].ravel(), vertices[:, :, 1:].ravel()))  # 创建深度方向上的边
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))  # 创建右向边
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))  # 创建下降方向上的边
    edges = np.hstack((edges_deep, edges_right, edges_down))  # 合并所有边为一个数组
    return edges  # 返回边数组


def _compute_gradient_3d(edges, img):
    _, n_y, n_z = img.shape  # 获取图像的形状
    gradient = np.abs(
        img[
            edges[0] // (n_y * n_z),   # 计算边起点在图像中的索引
            (edges[0] % (n_y * n_z)) // n_z,
            (edges[0] % (n_y * n_z)) % n_z,
        ]
        - img[
            edges[1] // (n_y * n_z),   # 计算边终点在图像中的索引
            (edges[1] % (n_y * n_z)) // n_z,
            (edges[1] % (n_y * n_z)) % n_z,
        ]
    )
    return gradient  # 返回梯度数组


# XXX: Why mask the image after computing the weights?


def _mask_edges_weights(mask, edges, weights=None):
    """Apply a mask to edges (weighted or not)"""
    inds = np.arange(mask.size)  # 创建索引数组
    inds = inds[mask.ravel()]   # 根据掩码压平数组并获取对应索引
    ind_mask = np.logical_and(np.isin(edges[0], inds), np.isin(edges[1], inds))  # 创建边的掩码
    edges = edges[:, ind_mask]  # 根据掩码筛选边
    if weights is not None:
        weights = weights[ind_mask]  # 如果有权重，也根据掩码筛选权重
    if len(edges.ravel()):
        maxval = edges.max()  # 计算最大边值
    else:
        maxval = 0
    order = np.searchsorted(np.flatnonzero(mask), np.arange(maxval + 1))  # 根据掩码排序边
    edges = order[edges]  # 应用排序后的边
    if weights is None:
        return edges  # 返回边
    else:
        return edges, weights  # 返回边和权重


def _to_graph(
    n_x, n_y, n_z, mask=None, img=None, return_as=sparse.coo_matrix, dtype=None
):
    """Auxiliary function for img_to_graph and grid_to_graph"""
    edges = _make_edges_3d(n_x, n_y, n_z)  # 生成图的边

    if dtype is None:  # 确定数据类型以避免覆盖输入的数据类型
        if img is None:
            dtype = int  # 如果没有图像，数据类型为整数
        else:
            dtype = img.dtype  # 否则使用图像的数据类型
    # 检查图像是否存在，如果存在则转换为至少是3维的数组
    if img is not None:
        img = np.atleast_3d(img)
        # 计算图像边缘和图像梯度权重
        weights = _compute_gradient_3d(edges, img)
        # 如果存在掩码，则应用掩码到边缘和权重上，并获取对角线数据
        if mask is not None:
            edges, weights = _mask_edges_weights(mask, edges, weights)
            diag = img.squeeze()[mask]
        else:
            # 否则获取图像的拉平数据作为对角线数据
            diag = img.ravel()
        # 获取对角线数据的大小
        n_voxels = diag.size
    else:
        # 如果图像不存在，检查是否存在掩码
        if mask is not None:
            # 将掩码转换为布尔类型，并将掩码应用到边缘上
            mask = mask.astype(dtype=bool, copy=False)
            edges = _mask_edges_weights(mask, edges)
            # 获取掩码的总数作为体素数
            n_voxels = np.sum(mask)
        else:
            # 否则计算总体素数
            n_voxels = n_x * n_y * n_z
        # 创建边缘权重为全1数组，并创建对角线为全1数组
        weights = np.ones(edges.shape[1], dtype=dtype)
        diag = np.ones(n_voxels, dtype=dtype)

    # 创建对角线索引数组
    diag_idx = np.arange(n_voxels)
    # 创建边缘的起点索引和终点索引数组
    i_idx = np.hstack((edges[0], edges[1]))
    j_idx = np.hstack((edges[1], edges[0]))
    # 使用稀疏矩阵创建图形，包括边缘权重和对角线数据
    graph = sparse.coo_matrix(
        (
            np.hstack((weights, weights, diag)),
            (np.hstack((i_idx, diag_idx)), np.hstack((j_idx, diag_idx))),
        ),
        (n_voxels, n_voxels),
        dtype=dtype,
    )
    # 如果要求返回格式为 ndarray，则将稀疏矩阵转换为 ndarray 返回
    if return_as is np.ndarray:
        return graph.toarray()
    # 否则直接返回稀疏矩阵
    return return_as(graph)
@validate_params(
    {
        "img": ["array-like"],  # 参数验证装饰器，验证img为array-like类型
        "mask": [None, np.ndarray],  # 参数验证装饰器，验证mask可以为None或者np.ndarray类型
        "return_as": [type],  # 参数验证装饰器，验证return_as为type类型
        "dtype": "no_validation",  # 参数验证装饰器，不对dtype进行验证，交由numpy处理
    },
    prefer_skip_nested_validation=True,  # 参数验证装饰器设置，优先跳过嵌套验证
)
def img_to_graph(img, *, mask=None, return_as=sparse.coo_matrix, dtype=None):
    """Graph of the pixel-to-pixel gradient connections.

    Edges are weighted with the gradient values.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    img : array-like of shape (height, width) or (height, width, channel)
        2D or 3D image.
    mask : ndarray of shape (height, width) or \
            (height, width, channel), dtype=bool, default=None
        An optional mask of the image, to consider only part of the
        pixels.
    return_as : np.ndarray or a sparse matrix class, \
            default=sparse.coo_matrix
        The class to use to build the returned adjacency matrix.
    dtype : dtype, default=None
        The data of the returned sparse matrix. By default it is the
        dtype of img.

    Returns
    -------
    graph : ndarray or a sparse matrix class
        The computed adjacency matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.feature_extraction.image import img_to_graph
    >>> img = np.array([[0, 0], [0, 1]])
    >>> img_to_graph(img, return_as=np.ndarray)
    array([[0, 0, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 0, 1],
           [0, 1, 1, 1]])
    """
    img = np.atleast_3d(img)  # 将img至少转换为3维数组
    n_x, n_y, n_z = img.shape  # 获取图像的维度信息
    return _to_graph(n_x, n_y, n_z, mask, img, return_as, dtype)  # 调用_to_graph函数生成图像的邻接矩阵


@validate_params(
    {
        "n_x": [Interval(Integral, left=1, right=None, closed="left")],  # 参数验证装饰器，验证n_x为大于等于1的整数
        "n_y": [Interval(Integral, left=1, right=None, closed="left")],  # 参数验证装饰器，验证n_y为大于等于1的整数
        "n_z": [Interval(Integral, left=1, right=None, closed="left")],  # 参数验证装饰器，验证n_z为大于等于1的整数
        "mask": [None, np.ndarray],  # 参数验证装饰器，验证mask可以为None或者np.ndarray类型
        "return_as": [type],  # 参数验证装饰器，验证return_as为type类型
        "dtype": "no_validation",  # 参数验证装饰器，不对dtype进行验证，交由numpy处理
    },
    prefer_skip_nested_validation=True,  # 参数验证装饰器设置，优先跳过嵌套验证
)
def grid_to_graph(
    n_x, n_y, n_z=1, *, mask=None, return_as=sparse.coo_matrix, dtype=int
):
    """Graph of the pixel-to-pixel connections.

    Edges exist if 2 voxels are connected.

    Parameters
    ----------
    n_x : int
        Dimension in x axis.
    n_y : int
        Dimension in y axis.
    n_z : int, default=1
        Dimension in z axis.
    mask : ndarray of shape (n_x, n_y, n_z), dtype=bool, default=None
        An optional mask of the image, to consider only part of the
        pixels.
    return_as : np.ndarray or a sparse matrix class, \
            default=sparse.coo_matrix
        The class to use to build the returned adjacency matrix.
    dtype : dtype, default=int
        The data of the returned sparse matrix. By default it is int.

    Returns
    -------
    graph : np.ndarray or a sparse matrix class
        The computed adjacency matrix.

    Examples
    --------
    """
    # 导入 NumPy 库，命名为 np
    import numpy as np
    # 从 sklearn.feature_extraction.image 模块导入 grid_to_graph 函数
    from sklearn.feature_extraction.image import grid_to_graph
    # 定义图像形状为 (4, 4, 1)
    shape_img = (4, 4, 1)
    # 创建一个形状与图像相同的布尔类型的零数组作为掩码
    mask = np.zeros(shape=shape_img, dtype=bool)
    # 将掩码中第二行第二列和第三行第三列的所有通道设置为 True
    mask[[1, 2], [1, 2], :] = True
    # 使用 grid_to_graph 函数生成基于掩码的图结构
    graph = grid_to_graph(*shape_img, mask=mask)
    # 打印生成的图结构
    print(graph)
    """
    调用 _to_graph 函数，返回图结构的表示形式
    """
    return _to_graph(n_x, n_y, n_z, mask=mask, return_as=return_as, dtype=dtype)
###############################################################################
# From an image to a set of small image patches

# 计算在图像中提取的小图像块的数量

def _compute_n_patches(i_h, i_w, p_h, p_w, max_patches=None):
    """Compute the number of patches that will be extracted in an image.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    i_h : int
        The image height
    i_w : int
        The image with
    p_h : int
        The height of a patch
    p_w : int
        The width of a patch
    max_patches : int or float, default=None
        The maximum number of patches to extract. If `max_patches` is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches. If `max_patches` is None, all possible patches are extracted.
    """
    # 计算可以从图像中提取的垂直和水平方向上的小图像块数量
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    all_patches = n_h * n_w

    # 根据 max_patches 参数确定最终返回的小图像块数量
    if max_patches:
        if isinstance(max_patches, (Integral)) and max_patches < all_patches:
            return max_patches
        elif isinstance(max_patches, (Integral)) and max_patches >= all_patches:
            return all_patches
        elif isinstance(max_patches, (Real)) and 0 < max_patches < 1:
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def _extract_patches(arr, patch_shape=8, extraction_step=1):
    """Extracts patches of any n-dimensional array in place using strides.

    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted

    patch_shape : int or tuple of length arr.ndim.default=8
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.

    extraction_step : int or tuple of length arr.ndim, default=1
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.


    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    # 根据输入的 patch_shape 参数确定每个维度上的小图像块的尺寸
    if isinstance(patch_shape, Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    # 检查 extraction_step 是否为数字类型，如果是，则转换为一个包含 arr_ndim 个元素的元组，每个元素都是 extraction_step 的值
    if isinstance(extraction_step, Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    # 获取 arr 的步幅信息
    patch_strides = arr.strides

    # 创建一个包含 arr 的每个维度的切片对象的元组，步幅为 extraction_step 中的每个值
    slices = tuple(slice(None, None, st) for st in extraction_step)

    # 使用切片 slices 对 arr 进行索引，获取索引后的数组的步幅信息
    indexing_strides = arr[slices].strides

    # 计算生成的补丁数组（patches）的形状
    patch_indices_shape = (
        (np.array(arr.shape) - np.array(patch_shape)) // np.array(extraction_step)
    ) + 1

    # 将 patch_indices_shape 和 patch_shape 组合成补丁数组 patches 的形状
    shape = tuple(list(patch_indices_shape) + list(patch_shape))

    # 将 indexing_strides 和 patch_strides 组合成补丁数组 patches 的步幅
    strides = tuple(list(indexing_strides) + list(patch_strides))

    # 使用 as_strided 函数创建具有给定形状和步幅的补丁数组 patches
    patches = as_strided(arr, shape=shape, strides=strides)

    # 返回生成的补丁数组 patches
    return patches
# 使用装饰器 @validate_params 对 extract_patches_2d 函数进行参数验证
@validate_params(
    {
        "image": [np.ndarray],  # 参数 image 应为 numpy 数组
        "patch_size": [tuple, list],  # 参数 patch_size 应为元组或列表
        "max_patches": [  # 参数 max_patches 应为以下类型之一：
            Interval(RealNotInt, 0, 1, closed="neither"),  # 实数区间 (0, 1)，不包括边界
            Interval(Integral, 1, None, closed="left"),    # 整数区间 [1, 无穷)
            None,  # 可以为 None
        ],
        "random_state": ["random_state"],  # 参数 random_state 应为 random_state 类型
    },
    prefer_skip_nested_validation=True,  # 首选跳过嵌套验证
)
def extract_patches_2d(image, patch_size, *, max_patches=None, random_state=None):
    """Reshape a 2D image into a collection of patches.

    The resulting patches are allocated in a dedicated array.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    image : ndarray of shape (image_height, image_width) or \
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    patch_size : tuple of int (patch_height, patch_width)
        The dimensions of one patch.

    max_patches : int or float, default=None
        The maximum number of patches to extract. If `max_patches` is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches. If `max_patches` is None it corresponds to the total number
        of patches that can be extracted.

    random_state : int, RandomState instance, default=None
        Determines the random number generator used for random sampling when
        `max_patches` is not None. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    patches : array of shape (n_patches, patch_height, patch_width) or \
        (n_patches, patch_height, patch_width, n_channels)
        The collection of patches extracted from the image, where `n_patches`
        is either `max_patches` or the total number of patches that can be
        extracted.

    Examples
    --------
    >>> from sklearn.datasets import load_sample_image
    >>> from sklearn.feature_extraction import image
    >>> # Use the array data from the first image in this dataset:
    >>> one_image = load_sample_image("china.jpg")
    >>> print('Image shape: {}'.format(one_image.shape))
    Image shape: (427, 640, 3)
    >>> patches = image.extract_patches_2d(one_image, (2, 2))
    >>> print('Patches shape: {}'.format(patches.shape))
    Patches shape: (272214, 2, 2, 3)
    >>> # Here are just two of these patches:
    >>> print(patches[1])
    [[[174 201 231]
      [174 201 231]]
     [[173 200 230]
      [173 200 230]]]
    >>> print(patches[800])
    [[[187 214 243]
      [188 215 244]]
     [[187 214 243]
      [188 215 244]]]
    """
    i_h, i_w = image.shape[:2]  # 获取图像的高度和宽度

    p_h, p_w = patch_size  # 获取补丁的高度和宽度

    if p_h > i_h:  # 如果补丁的高度大于图像的高度，则抛出 ValueError 异常
        raise ValueError(
            "Height of the patch should be less than the height of the image."
        )
    # 如果待提取的补丁宽度大于图像宽度，则引发值错误异常
    if p_w > i_w:
        raise ValueError(
            "Width of the patch should be less than the width of the image."
        )

    # 将输入图像转换为 NumPy 数组，允许多维数组
    image = check_array(image, allow_nd=True)
    # 将图像重塑为指定高度、宽度和通道数的形状
    image = image.reshape((i_h, i_w, -1))
    # 获取图像的通道数
    n_colors = image.shape[-1]

    # 从输入图像中提取补丁，以给定的形状和步长
    extracted_patches = _extract_patches(
        image, patch_shape=(p_h, p_w, n_colors), extraction_step=1
    )

    # 计算在图像上可能提取的补丁数目
    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    
    # 如果指定了最大补丁数目
    if max_patches:
        # 使用随机状态对象生成随机数发生器
        rng = check_random_state(random_state)
        # 随机生成行索引，确保补丁不超出图像边界
        i_s = rng.randint(i_h - p_h + 1, size=n_patches)
        # 随机生成列索引，确保补丁不超出图像边界
        j_s = rng.randint(i_w - p_w + 1, size=n_patches)
        # 从提取的补丁中选取随机的子集
        patches = extracted_patches[i_s, j_s, 0]
    else:
        # 否则，使用所有提取的补丁
        patches = extracted_patches

    # 将补丁重塑为指定形状
    patches = patches.reshape(-1, p_h, p_w, n_colors)
    
    # 如果补丁的最后一个维度表示颜色且只有一个通道，则去除该维度
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
        return patches
@validate_params(
    {"patches": [np.ndarray], "image_size": [tuple, Hidden(list)]},
    prefer_skip_nested_validation=True,
)
# 使用装饰器 @validate_params 对函数进行参数验证，确保 patches 是 ndarray 类型且 image_size 是 tuple 类型
def reconstruct_from_patches_2d(patches, image_size):
    """Reconstruct the image from all of its patches.

    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    patches : ndarray of shape (n_patches, patch_height, patch_width) or \
        (n_patches, patch_height, patch_width, n_channels)
        The complete set of patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.

    image_size : tuple of int (image_height, image_width) or \
        (image_height, image_width, n_channels)
        The size of the image that will be reconstructed.

    Returns
    -------
    image : ndarray of shape image_size
        The reconstructed image.

    Examples
    --------
    >>> from sklearn.datasets import load_sample_image
    >>> from sklearn.feature_extraction import image
    >>> one_image = load_sample_image("china.jpg")
    >>> print('Image shape: {}'.format(one_image.shape))
    Image shape: (427, 640, 3)
    >>> image_patches = image.extract_patches_2d(image=one_image, patch_size=(10, 10))
    >>> print('Patches shape: {}'.format(image_patches.shape))
    Patches shape: (263758, 10, 10, 3)
    >>> image_reconstructed = image.reconstruct_from_patches_2d(
    ...     patches=image_patches,
    ...     image_size=one_image.shape
    ... )
    >>> print(f"Reconstructed shape: {image_reconstructed.shape}")
    Reconstructed shape: (427, 640, 3)
    """
    # 提取图像大小的高度和宽度信息
    i_h, i_w = image_size[:2]
    # 提取补丁的高度和宽度信息
    p_h, p_w = patches.shape[1:3]
    # 创建一个全零的图像数组，用于重建图像
    img = np.zeros(image_size)
    
    # 计算补丁数组的维度
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    
    # 对每个补丁及其在图像中的位置进行迭代，从左到右，从上到下填充补丁，考虑重叠区域进行平均
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i : i + p_h, j : j + p_w] += p

    # 对重建图像的每个像素进行迭代
    for i in range(i_h):
        for j in range(i_w):
            # 按照重叠区域的大小进行除法，用于平均化像素值
            # XXX: 这是最有效的方式吗？在内存和 CPU 方面？
            img[i, j] /= float(min(i + 1, p_h, i_h - i) * min(j + 1, p_w, i_w - j))
    
    # 返回重建的图像
    return img
    max_patches : int or float, default=None
        最大要提取的每个图像的补丁数。如果 `max_patches` 是一个在 (0, 1) 范围内的浮点数，
        则表示总补丁数的比例。如果设置为 None，则提取所有可能的补丁。

    random_state : int, RandomState instance, default=None
        当 `max_patches` 不为 None 时，确定用于随机采样的随机数生成器。
        使用一个整数以确保随机性的确定性。
        参见 :term:`Glossary <random_state>`。

    See Also
    --------
    reconstruct_from_patches_2d : 从所有补丁重构图像的方法。

    Notes
    -----
    此估计器是无状态的，不需要拟合。然而，我们建议使用 :meth:`fit_transform` 而不是 :meth:`transform`，
    因为参数验证仅在 :meth:`fit` 中执行。

    Examples
    --------
    >>> from sklearn.datasets import load_sample_images
    >>> from sklearn.feature_extraction import image
    >>> # Use the array data from the second image in this dataset:
    >>> X = load_sample_images().images[1]
    >>> X = X[None, ...]
    >>> print(f"Image shape: {X.shape}")
    Image shape: (1, 427, 640, 3)
    >>> pe = image.PatchExtractor(patch_size=(10, 10))
    >>> pe_trans = pe.transform(X)
    >>> print(f"Patches shape: {pe_trans.shape}")
    Patches shape: (263758, 10, 10, 3)
    >>> X_reconstructed = image.reconstruct_from_patches_2d(pe_trans, X.shape[1:])
    >>> print(f"Reconstructed shape: {X_reconstructed.shape}")
    Reconstructed shape: (427, 640, 3)
    """

    _parameter_constraints: dict = {
        "patch_size": [tuple, None],
        "max_patches": [
            None,
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
        ],
        "random_state": ["random_state"],
    }

    def __init__(self, *, patch_size=None, max_patches=None, random_state=None):
        self.patch_size = patch_size  # 初始化补丁大小参数
        self.max_patches = max_patches  # 初始化最大补丁数参数
        self.random_state = random_state  # 初始化随机状态参数

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """仅验证估计器的参数。

        该方法允许：(i) 验证估计器的参数，并且 (ii) 与 scikit-learn 转换器 API 保持一致。

        Parameters
        ----------
        X : ndarray of shape (n_samples, image_height, image_width) or \
                (n_samples, image_height, image_width, n_channels)
            要从中提取补丁的图像数组。对于彩色图像，最后一个维度指定通道数：RGB 图像将有 `n_channels=3`。

        y : 忽略
            不使用，仅出于 API 一致性而存在。

        Returns
        -------
        self : object
            返回实例本身。
        """
        return self
    # 定义一个方法，用于将输入的图像样本 `X` 转换为补丁数据的矩阵

    """
    Transform the image samples in `X` into a matrix of patch data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, image_height, image_width) or \
            (n_samples, image_height, image_width, n_channels)
        图像数组，从中提取补丁。对于彩色图像，最后一个维度指定通道数：RGB 图像将有 `n_channels=3`。

    Returns
    -------
    patches : array of shape (n_patches, patch_height, patch_width) or \
            (n_patches, patch_height, patch_width, n_channels)
        从图像中提取的补丁集合，其中 `n_patches` 是 `n_samples * max_patches` 或可提取的总补丁数。
    """

    # 验证输入数据 `X`，确保其为二维数组或更高维度，且包含足够的样本和特征
    X = self._validate_data(
        X=X,
        ensure_2d=False,
        allow_nd=True,
        ensure_min_samples=1,
        ensure_min_features=1,
        reset=False,
    )

    # 检查随机状态并初始化
    random_state = check_random_state(self.random_state)

    # 获取图像数量、高度和宽度
    n_imgs, img_height, img_width = X.shape[:3]

    # 如果未指定补丁大小，则设定默认为图像高度和宽度的十分之一
    if self.patch_size is None:
        patch_size = img_height // 10, img_width // 10
    else:
        # 如果指定了补丁大小，确保其为长度为 2 的元组
        if len(self.patch_size) != 2:
            raise ValueError(
                "patch_size must be a tuple of two integers. Got"
                f" {self.patch_size} instead."
            )
        patch_size = self.patch_size

    # 再次获取图像数量、高度和宽度（因为可能在上面的 if/else 语句中已修改 `X`）
    n_imgs, img_height, img_width = X.shape[:3]

    # 将输入 `X` 重新整形为四维数组，最后一个维度是通道数
    X = np.reshape(X, (n_imgs, img_height, img_width, -1))

    # 获取图像的通道数
    n_channels = X.shape[-1]

    # 计算补丁数组的维度
    patch_height, patch_width = patch_size
    n_patches = _compute_n_patches(
        img_height, img_width, patch_height, patch_width, self.max_patches
    )
    patches_shape = (n_imgs * n_patches,) + patch_size
    if n_channels > 1:
        patches_shape += (n_channels,)

    # 创建一个空数组，用于存储提取的补丁
    patches = np.empty(patches_shape)

    # 遍历每张图像，提取补丁并存储到 `patches` 数组中
    for ii, image in enumerate(X):
        patches[ii * n_patches : (ii + 1) * n_patches] = extract_patches_2d(
            image,
            patch_size,
            max_patches=self.max_patches,
            random_state=random_state,
        )

    # 返回存储所有补丁的数组
    return patches
```