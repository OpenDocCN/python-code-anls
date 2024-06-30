# `D:\src\scipysrc\scikit-learn\sklearn\feature_extraction\tests\test_image.py`

```
# 导入所需的库和模块
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试
from scipy import ndimage  # 导入scipy的ndimage模块，用于图像处理
from scipy.sparse.csgraph import connected_components  # 导入scipy.sparse.csgraph模块的connected_components函数，用于计算连通组件

# 导入sklearn库中的图像特征提取相关模块和函数
from sklearn.feature_extraction.image import (
    PatchExtractor,  # 导入PatchExtractor类，用于从图像中提取图像块
    _extract_patches,  # 导入_extract_patches函数，用于从图像中提取图像块
    extract_patches_2d,  # 导入extract_patches_2d函数，用于从2维图像中提取图像块
    grid_to_graph,  # 导入grid_to_graph函数，用于生成图像网格到图的转换
    img_to_graph,  # 导入img_to_graph函数，用于生成图像到图的转换
    reconstruct_from_patches_2d,  # 导入reconstruct_from_patches_2d函数，用于从图像块重建图像
)


# 测试函数：验证img_to_graph函数的功能
def test_img_to_graph():
    # 创建一个4x4的网格矩阵，每个元素减去10
    x, y = np.mgrid[:4, :4] - 10
    # 使用img_to_graph函数生成x和y的梯度图
    grad_x = img_to_graph(x)
    grad_y = img_to_graph(y)
    # 断言两个梯度图的非零元素个数相等
    assert grad_x.nnz == grad_y.nnz
    # 使用NumPy的断言测试，验证梯度图中正元素相等
    np.testing.assert_array_equal(
        grad_x.data[grad_x.data > 0], grad_y.data[grad_y.data > 0]
    )


# 测试函数：验证img_to_graph函数在稀疏图像上的功能
def test_img_to_graph_sparse():
    # 创建一个2x3的零矩阵，将第一行第一列和全列最后一行设为True
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 0] = 1
    mask[:, 2] = 1
    # 创建一个2x3的零矩阵，设置部分元素的值
    x = np.zeros((2, 3))
    x[0, 0] = 1
    x[0, 2] = -1
    x[1, 2] = -2
    # 使用img_to_graph函数生成稀疏图形式的梯度图
    grad_x = img_to_graph(x, mask=mask).todense()
    # 预期的稀疏梯度图
    desired = np.array([[1, 0, 0], [0, -1, 1], [0, 1, -2]])
    # 使用NumPy的断言测试，验证生成的梯度图是否与预期一致
    np.testing.assert_array_equal(grad_x, desired)


# 测试函数：验证grid_to_graph函数的功能
def test_grid_to_graph():
    size = 2
    roi_size = 1
    # 创建一个2x2的零矩阵，设置部分元素为True
    mask = np.zeros((size, size), dtype=bool)
    mask[0:roi_size, 0:roi_size] = True
    mask[-roi_size:, -roi_size:] = True
    # 将mask转换为一维数组，并生成对应的图
    mask = mask.reshape(size**2)
    # 使用grid_to_graph函数生成图的邻接矩阵
    A = grid_to_graph(n_x=size, n_y=size, mask=mask, return_as=np.ndarray)
    # 断言连通组件的数量是否为2
    assert connected_components(A)[0] == 2

    # 创建一个2x3的零矩阵，设置部分元素为True
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 0] = 1
    mask[:, 2] = 1
    # 将mask展平为一维数组，并生成对应的图
    graph = grid_to_graph(2, 3, 1, mask=mask.ravel()).todense()
    # 预期的邻接矩阵
    desired = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
    # 使用NumPy的断言测试，验证生成的邻接矩阵是否与预期一致
    np.testing.assert_array_equal(graph, desired)

    # 创建一个全为1的大小为size*size的矩阵，并生成对应的图
    mask = np.ones((size, size), dtype=np.int16)
    A = grid_to_graph(n_x=size, n_y=size, n_z=size, mask=mask)
    # 断言连通组件的数量是否为1
    assert connected_components(A)[0] == 1

    # 创建一个大小为size*size的全为1的矩阵，并生成对应的图，指定图的数据类型为布尔型
    mask = np.ones((size, size))
    A = grid_to_graph(n_x=size, n_y=size, n_z=size, mask=mask, dtype=bool)
    # 断言生成的邻接矩阵的数据类型是否为布尔型
    assert A.dtype == bool
    # 创建一个大小为size*size的全为1的矩阵，并生成对应的图，指定图的数据类型为整型
    A = grid_to_graph(n_x=size, n_y=size, n_z=size, mask=mask, dtype=int)
    # 断言生成的邻接矩阵的数据类型是否为整型
    assert A.dtype == int
    # 创建一个大小为size*size的全为1的矩阵，并生成对应的图，指定图的数据类型为浮点型
    A = grid_to_graph(n_x=size, n_y=size, n_z=size, mask=mask, dtype=np.float64)
    # 断言生成的邻接矩阵的数据类型是否为浮点型
    assert A.dtype == np.float64


# 测试函数：验证connect_regions函数的功能
def test_connect_regions(raccoon_face_fxt):
    # 获取raccoon_face_fxt，并对其进行4倍下采样
    face = raccoon_face_fxt
    face = face[::4, ::4]
    # 遍历阈值列表，对face进行二值化处理
    for thr in (50, 150):
        mask = face > thr
        # 使用img_to_graph函数生成face的图
        graph = img_to_graph(face, mask=mask)
        # 断言使用ndimage.label函数标记后的连通组件数与使用connected_components函数计算得到的连通组件数相等
        assert ndimage.label(mask)[1] == connected_components(graph)[0]
def test_connect_regions_with_grid(raccoon_face_fxt):
    # 获取测试用的浣熊脸图像
    face = raccoon_face_fxt

    # 按4倍子采样以减少运行时间
    face = face[::4, ::4]

    # 创建一个布尔掩码，标记大于50的像素点
    mask = face > 50
    # 根据掩码创建图形网络
    graph = grid_to_graph(*face.shape, mask=mask)
    # 断言标记后的连通组件数量与预期相同
    assert ndimage.label(mask)[1] == connected_components(graph)[0]

    # 更新掩码，标记大于150的像素点
    mask = face > 150
    # 根据更新后的掩码创建图形网络，指定数据类型为None
    graph = grid_to_graph(*face.shape, mask=mask, dtype=None)
    # 再次断言标记后的连通组件数量与预期相同
    assert ndimage.label(mask)[1] == connected_components(graph)[0]


@pytest.fixture
def downsampled_face(raccoon_face_fxt):
    # 获取测试用的浣熊脸图像
    face = raccoon_face_fxt
    # 对图像进行2倍子采样并求和
    face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2, 1::2]
    # 转换图像数据类型为32位浮点数
    face = face.astype(np.float32)
    # 将图像数据除以16.0，得到下采样后的图像
    face /= 16.0
    return face


@pytest.fixture
def orange_face(downsampled_face):
    # 获取下采样后的浣熊脸图像
    face = downsampled_face
    # 创建一个形状为(face高, face宽, 3)的零数组，表示彩色图像
    face_color = np.zeros(face.shape + (3,))
    # 设置红色通道为256减去下采样后的浣熊脸图像
    face_color[:, :, 0] = 256 - face
    # 设置绿色通道为256减去下采样后的浣熊脸图像除以2
    face_color[:, :, 1] = 256 - face / 2
    # 设置蓝色通道为256减去下采样后的浣熊脸图像除以4
    face_color[:, :, 2] = 256 - face / 4
    return face_color


def _make_images(face):
    # 创建一个形状为(3, face高, face宽)的零数组，表示多张人脸图像
    images = np.zeros((3,) + face.shape)
    # 将输入的人脸图像复制到images数组的不同通道中
    images[0] = face
    images[1] = face + 1
    images[2] = face + 2
    return images


@pytest.fixture
def downsampled_face_collection(downsampled_face):
    # 获取下采样后的浣熊脸图像集合
    return _make_images(downsampled_face)


def test_extract_patches_all(downsampled_face):
    # 获取下采样后的浣熊脸图像
    face = downsampled_face
    # 获取图像的高度和宽度
    i_h, i_w = face.shape
    # 定义要提取的补丁大小
    p_h, p_w = 16, 16
    # 计算预期的补丁数量
    expected_n_patches = (i_h - p_h + 1) * (i_w - p_w + 1)
    # 提取大小为(p_h, p_w)的所有补丁
    patches = extract_patches_2d(face, (p_h, p_w))
    # 断言提取的补丁数组形状符合预期
    assert patches.shape == (expected_n_patches, p_h, p_w)


def test_extract_patches_all_color(orange_face):
    # 获取彩色的浣熊脸图像
    face = orange_face
    # 获取图像的高度和宽度
    i_h, i_w = face.shape[:2]
    # 定义要提取的补丁大小
    p_h, p_w = 16, 16
    # 计算预期的补丁数量
    expected_n_patches = (i_h - p_h + 1) * (i_w - p_w + 1)
    # 提取大小为(p_h, p_w)的所有补丁
    patches = extract_patches_2d(face, (p_h, p_w))
    # 断言提取的补丁数组形状符合预期
    assert patches.shape == (expected_n_patches, p_h, p_w, 3)


def test_extract_patches_all_rect(downsampled_face):
    # 获取下采样后的浣熊脸图像
    face = downsampled_face
    # 选择图像的子区域，仅使用第32到第97列的像素
    face = face[:, 32:97]
    # 获取图像的高度和宽度
    i_h, i_w = face.shape
    # 定义要提取的补丁大小
    p_h, p_w = 16, 12
    # 计算预期的补丁数量
    expected_n_patches = (i_h - p_h + 1) * (i_w - p_w + 1)

    # 提取大小为(p_h, p_w)的所有补丁
    patches = extract_patches_2d(face, (p_h, p_w))
    # 断言提取的补丁数组形状符合预期
    assert patches.shape == (expected_n_patches, p_h, p_w)


def test_extract_patches_max_patches(downsampled_face):
    # 获取下采样后的浣熊脸图像
    face = downsampled_face
    # 获取图像的高度和宽度
    i_h, i_w = face.shape
    # 定义要提取的补丁大小
    p_h, p_w = 16, 16

    # 提取最多100个大小为(p_h, p_w)的补丁
    patches = extract_patches_2d(face, (p_h, p_w), max_patches=100)
    # 断言提取的补丁数组形状符合预期
    assert patches.shape == (100, p_h, p_w)

    # 计算预期的补丁数量，约为总数的一半
    expected_n_patches = int(0.5 * (i_h - p_h + 1) * (i_w - p_w + 1))
    # 提取大小为(p_h, p_w)的补丁，数量为总数的一半
    patches = extract_patches_2d(face, (p_h, p_w), max_patches=0.5)
    # 断言提取的补丁数组形状符合预期
    assert patches.shape == (expected_n_patches, p_h, p_w)

    # 测试当max_patches参数为非正数时，是否引发值错误异常
    with pytest.raises(ValueError):
        extract_patches_2d(face, (p_h, p_w), max_patches=2.0)
    with pytest.raises(ValueError):
        extract_patches_2d(face, (p_h, p_w), max_patches=-1.0)


def test_extract_patch_same_size_image(downsampled_face):
    # 获取下采样后的浣熊脸图像
    face = downsampled_face
    # 请求与图像相同大小的补丁
    # 应该返回单个补丁，即图像本身
    patches = extract_patches_2d(face, face.shape, max_patches=2)
    # 断言确保补丁数组的第一维度大小为1，即确保只返回了单个补丁（即图像本身）
    assert patches.shape[0] == 1
# 定义一个测试函数，用于从降采样的人脸图像中提取小于最大补丁数量的补丁
def test_extract_patches_less_than_max_patches(downsampled_face):
    # 复制输入的降采样人脸图像
    face = downsampled_face
    # 获取图像的高度和宽度
    i_h, i_w = face.shape
    # 计算补丁的高度和宽度，为图像的3/4
    p_h, p_w = 3 * i_h // 4, 3 * i_w // 4
    # 计算期望的补丁数量，这是通过给定的公式计算得出的
    # this is 3185
    expected_n_patches = (i_h - p_h + 1) * (i_w - p_w + 1)

    # 使用给定的函数从图像中提取补丁
    patches = extract_patches_2d(face, (p_h, p_w), max_patches=4000)
    # 断言提取的补丁的形状符合预期的补丁数量
    assert patches.shape == (expected_n_patches, p_h, p_w)


# 定义一个测试函数，用于从降采样的人脸图像中完美重建补丁
def test_reconstruct_patches_perfect(downsampled_face):
    # 复制输入的降采样人脸图像
    face = downsampled_face
    # 定义补丁的高度和宽度
    p_h, p_w = 16, 16

    # 使用给定的函数从图像中提取补丁
    patches = extract_patches_2d(face, (p_h, p_w))
    # 使用给定的函数从补丁中完美重建原始人脸图像
    face_reconstructed = reconstruct_from_patches_2d(patches, face.shape)
    # 断言重建的人脸图像与原始人脸图像几乎相等
    np.testing.assert_array_almost_equal(face, face_reconstructed)


# 定义一个测试函数，用于从橙色人脸图像中完美重建补丁（彩色）
def test_reconstruct_patches_perfect_color(orange_face):
    # 复制输入的橙色人脸图像
    face = orange_face
    # 定义补丁的高度和宽度
    p_h, p_w = 16, 16

    # 使用给定的函数从图像中提取补丁
    patches = extract_patches_2d(face, (p_h, p_w))
    # 使用给定的函数从补丁中完美重建原始人脸图像
    face_reconstructed = reconstruct_from_patches_2d(patches, face.shape)
    # 断言重建的人脸图像与原始人脸图像几乎相等
    np.testing.assert_array_almost_equal(face, face_reconstructed)


# 定义一个测试函数，用于测试补丁提取器的适应性
def test_patch_extractor_fit(downsampled_face_collection):
    # 复制输入的降采样人脸集合
    faces = downsampled_face_collection
    # 创建补丁提取器对象，指定补丁的大小和最大补丁数
    extr = PatchExtractor(patch_size=(8, 8), max_patches=100, random_state=0)
    # 断言补丁提取器适合给定的人脸集合
    assert extr == extr.fit(faces)


# 定义一个测试函数，用于测试补丁提取器的最大补丁数限制
def test_patch_extractor_max_patches(downsampled_face_collection):
    # 复制输入的降采样人脸集合
    faces = downsampled_face_collection
    # 获取集合中每张人脸图像的高度和宽度
    i_h, i_w = faces.shape[1:3]
    # 定义补丁的高度和宽度
    p_h, p_w = 8, 8

    # 设置最大补丁数为100
    max_patches = 100
    # 计算预期的补丁数量，这是集合中每张图像补丁数的总和
    expected_n_patches = len(faces) * max_patches
    # 创建补丁提取器对象，指定补丁的大小、最大补丁数和随机种子
    extr = PatchExtractor(
        patch_size=(p_h, p_w), max_patches=max_patches, random_state=0
    )
    # 使用补丁提取器从人脸集合中提取补丁
    patches = extr.transform(faces)
    # 断言提取的补丁的形状符合预期的补丁数量
    assert patches.shape == (expected_n_patches, p_h, p_w)

    # 设置最大补丁数为0.5
    max_patches = 0.5
    # 计算预期的补丁数量，这是集合中每张图像补丁数的总和
    expected_n_patches = len(faces) * int(
        (i_h - p_h + 1) * (i_w - p_w + 1) * max_patches
    )
    # 创建补丁提取器对象，指定补丁的大小、最大补丁数和随机种子
    extr = PatchExtractor(
        patch_size=(p_h, p_w), max_patches=max_patches, random_state=0
    )
    # 使用补丁提取器从人脸集合中提取补丁
    patches = extr.transform(faces)
    # 断言提取的补丁的形状符合预期的补丁数量
    assert patches.shape == (expected_n_patches, p_h, p_w)


# 定义一个测试函数，用于测试补丁提取器的默认最大补丁数
def test_patch_extractor_max_patches_default(downsampled_face_collection):
    # 复制输入的降采样人脸集合
    faces = downsampled_face_collection
    # 创建补丁提取器对象，只指定最大补丁数
    extr = PatchExtractor(max_patches=100, random_state=0)
    # 使用补丁提取器从人脸集合中提取补丁
    patches = extr.transform(faces)
    # 断言提取的补丁的形状符合预期的补丁数量
    assert patches.shape == (len(faces) * 100, 19, 25)


# 定义一个测试函数，用于测试补丁提取器提取所有补丁的情况
def test_patch_extractor_all_patches(downsampled_face_collection):
    # 复制输入的降采样人脸集合
    faces = downsampled_face_collection
    # 获取集合中每张人脸图像的高度和宽度
    i_h, i_w = faces.shape[1:3]
    # 定义补丁的高度和宽度
    p_h, p_w = 8, 8
    # 计算预期的补丁数量，这是集合中每张图像补丁数的总和
    expected_n_patches = len(faces) * (i_h - p_h + 1) * (i_w - p_w + 1)
    # 创建补丁提取器对象，指定补丁的大小和随机种子
    extr = PatchExtractor(patch_size=(p_h, p_w), random_state=0)
    # 使用补丁提取器从人脸集合中提取补丁
    patches = extr.transform(faces)
    # 断言提取的补丁的形状符合预期的补丁数量
    assert patches.shape == (expected_n_patches, p_h, p_w)


# 定义一个测试函数，用于测试补丁提取器在彩色图像上的表现
def test_patch_extractor_color(orange_face):
    # 生成橙色人脸图像的多张版本（假设是多张图像）
    faces
def test_extract_patches_strided():
    # 定义一维图像形状列表
    image_shapes_1D = [(10,), (10,), (11,), (10,)]
    # 定义一维补丁大小列表
    patch_sizes_1D = [(1,), (2,), (3,), (8,)]
    # 定义一维补丁步长列表
    patch_steps_1D = [(1,), (1,), (4,), (2,)]

    # 定义预期的一维视图大小列表
    expected_views_1D = [(10,), (9,), (3,), (2,)]
    # 定义一维最后一个补丁列表
    last_patch_1D = [(10,), (8,), (8,), (2,)]

    # 定义二维图像形状列表
    image_shapes_2D = [(10, 20), (10, 20), (10, 20), (11, 20)]
    # 定义二维补丁大小列表
    patch_sizes_2D = [(2, 2), (10, 10), (10, 11), (6, 6)]
    # 定义二维补丁步长列表
    patch_steps_2D = [(5, 5), (3, 10), (3, 4), (4, 2)]

    # 定义预期的二维视图大小列表
    expected_views_2D = [(2, 4), (1, 2), (1, 3), (2, 8)]
    # 定义二维最后一个补丁列表
    last_patch_2D = [(5, 15), (0, 10), (0, 8), (4, 14)]

    # 定义三维图像形状列表
    image_shapes_3D = [(5, 4, 3), (3, 3, 3), (7, 8, 9), (7, 8, 9)]
    # 定义三维补丁大小列表
    patch_sizes_3D = [(2, 2, 3), (2, 2, 2), (1, 7, 3), (1, 3, 3)]
    # 定义三维补丁步长列表
    patch_steps_3D = [(1, 2, 10), (1, 1, 1), (2, 1, 3), (3, 3, 4)]

    # 定义预期的三维视图大小列表
    expected_views_3D = [(4, 2, 1), (2, 2, 2), (4, 2, 3), (3, 2, 2)]
    # 定义三维最后一个补丁列表
    last_patch_3D = [(3, 2, 0), (1, 1, 1), (6, 1, 6), (6, 3, 4)]

    # 将所有维度的图像形状合并为一个列表
    image_shapes = image_shapes_1D + image_shapes_2D + image_shapes_3D
    # 将所有维度的补丁大小合并为一个列表
    patch_sizes = patch_sizes_1D + patch_sizes_2D + patch_sizes_3D
    # 将所有维度的补丁步长合并为一个列表
    patch_steps = patch_steps_1D + patch_steps_2D + patch_steps_3D
    # 将所有维度的预期视图大小合并为一个列表
    expected_views = expected_views_1D + expected_views_2D + expected_views_3D
    # 将所有维度的最后一个补丁合并为一个列表
    last_patches = last_patch_1D + last_patch_2D + last_patch_3D

    # 遍历所有维度的图像形状、补丁大小、补丁步长、预期视图大小和最后一个补丁
    for image_shape, patch_size, patch_step, expected_view, last_patch in zip(
        image_shapes, patch_sizes, patch_steps, expected_views, last_patches
    ):
        # 创建一个测试用的图像数组
        image = np.arange(np.prod(image_shape)).reshape(image_shape)
        # 调用 _extract_patches 函数提取补丁
        patches = _extract_patches(
            image, patch_shape=patch_size, extraction_step=patch_step
        )

        # 获取图像形状的维度数
        ndim = len(image_shape)

        # 断言提取的补丁形状的前几维与预期视图大小一致
        assert patches.shape[:ndim] == expected_view
        # 计算最后一个补丁在原始图像中的切片范围
        last_patch_slices = tuple(
            slice(i, i + j, None) for i, j in zip(last_patch, patch_size)
        )
        # 断言提取的最后一个补丁与原始图像中相应切片的内容一致
        assert (
            patches[(-1, None, None) * ndim] == image[last_patch_slices].squeeze()
        ).all()


def test_extract_patches_square(downsampled_face):
    # 测试所有维度的补丁大小相同的情况
    face = downsampled_face
    i_h, i_w = face.shape
    p = 8
    expected_n_patches = ((i_h - p + 1), (i_w - p + 1))
    # 调用 _extract_patches 函数提取补丁
    patches = _extract_patches(face, patch_shape=p)
    # 断言提取的补丁形状符合预期
    assert patches.shape == (expected_n_patches[0], expected_n_patches[1], p, p)


def test_width_patch():
    # 补丁的宽度和高度应小于图像的对应维度
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with pytest.raises(ValueError):
        extract_patches_2d(x, (4, 1))
    with pytest.raises(ValueError):
        extract_patches_2d(x, (1, 4))


def test_patch_extractor_wrong_input(orange_face):
    """检查如果补丁大小不合法是否会引发错误信息。"""
    # 创建多张人脸图像作为测试数据
    faces = _make_images(orange_face)
    # 定义预期的错误信息
    err_msg = "patch_size must be a tuple of two integers"
    # 初始化 PatchExtractor 类，并传入非法的补丁大小参数
    extractor = PatchExtractor(patch_size=(8, 8, 8))
    # 断言调用 transform 方法时会引发 ValueError 错误，并且错误信息符合预期
    with pytest.raises(ValueError, match=err_msg):
        extractor.transform(faces)
```