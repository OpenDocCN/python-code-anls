# `D:\src\scipysrc\scikit-learn\examples\decomposition\plot_image_denoising.py`

```
# %%
# Generate distorted image
# ------------------------

import numpy as np

try:  # Scipy >= 1.10
    from scipy.datasets import face
except ImportError:
    from scipy.misc import face

# 从 Scipy 库中获取灰度的 raccoon 脸部图像
raccoon_face = face(gray=True)

# 将图像的数据类型从 uint8 转换为浮点型，并将像素值范围从 [0, 255] 映射到 [0, 1]
raccoon_face = raccoon_face / 255.0

# 对图像进行下采样，以提高处理速度
raccoon_face = (
    raccoon_face[::4, ::4]
    + raccoon_face[1::4, ::4]
    + raccoon_face[::4, 1::4]
    + raccoon_face[1::4, 1::4]
)
raccoon_face /= 4.0
height, width = raccoon_face.shape

# 在图像的右半部分添加噪声，以扭曲图像
print("Distorting image...")
distorted = raccoon_face.copy()
distorted[:, width // 2 :] += 0.075 * np.random.randn(height, width // 2)


# %%
# Display the distorted image
# ---------------------------

import matplotlib.pyplot as plt

def show_with_diff(image, reference, title):
    """Helper function to display denoising"""
    # 创建一个新的图像窗口，大小为 5x3.3 英寸
    plt.figure(figsize=(5, 3.3))
    
    # 在窗口中创建左右两个子图之一，显示原始图像
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation="nearest")
    plt.xticks(())
    plt.yticks(())
    
    # 在窗口中创建左右两个子图之二，显示重构图像与原始图像的差异
    difference = image - reference
    plt.subplot(1, 2, 2)
    plt.title("Difference (norm: %.2f)" % np.sqrt(np.sum(difference**2)))
    plt.imshow(
        difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr, interpolation="nearest"
    )
    plt.xticks(())
    plt.yticks(())
    
    # 设置整体图像的标题
    plt.suptitle(title, size=16)
    # 调整子图布局的参数，设置左边界、底边界、右边界、顶边界、水平间距和垂直间距
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)
# 显示带有差异的图像
show_with_diff(distorted, raccoon_face, "Distorted image")

# %%
# 提取参考补丁
# ----------------------------
from time import time
from sklearn.feature_extraction.image import extract_patches_2d

# 从图像左半部分提取所有参考补丁
print("Extracting reference patches...")
t0 = time()
patch_size = (7, 7)
data = extract_patches_2d(distorted[:, : width // 2], patch_size)
data = data.reshape(data.shape[0], -1)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
print(f"{data.shape[0]} patches extracted in %.2fs." % (time() - t0))

# %%
# 从参考补丁中学习字典
# -------------------------------------------
from sklearn.decomposition import MiniBatchDictionaryLearning

print("Learning the dictionary...")
t0 = time()
dico = MiniBatchDictionaryLearning(
    # 增加到 300 可以获得更高质量的结果，但会降低训练速度。
    n_components=50,
    batch_size=200,
    alpha=1.0,
    max_iter=10,
)
V = dico.fit(data).components_
dt = time() - t0
print(f"{dico.n_iter_} iterations / {dico.n_steps_} steps in {dt:.2f}.")

# 绘制学习到的字典中的前100个原子
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r, interpolation="nearest")
    plt.xticks(())
    plt.yticks(())
plt.suptitle(
    "Dictionary learned from face patches\n"
    + "Train time %.1fs on %d patches" % (dt, len(data)),
    fontsize=16,
)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

# %%
# 提取噪声补丁并使用字典重建它们
# ---------------------------------------------------------------
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

print("Extracting noisy patches... ")
t0 = time()
data = extract_patches_2d(distorted[:, width // 2 :], patch_size)
data = data.reshape(data.shape[0], -1)
intercept = np.mean(data, axis=0)
data -= intercept
print("done in %.2fs." % (time() - t0))

# 不同的变换算法及其参数
transform_algorithms = [
    ("Orthogonal Matching Pursuit\n1 atom", "omp", {"transform_n_nonzero_coefs": 1}),
    ("Orthogonal Matching Pursuit\n2 atoms", "omp", {"transform_n_nonzero_coefs": 2}),
    ("Least-angle regression\n4 atoms", "lars", {"transform_n_nonzero_coefs": 4}),
    ("Thresholding\n alpha=0.1", "threshold", {"transform_alpha": 0.1}),
]

reconstructions = {}
for title, transform_algorithm, kwargs in transform_algorithms:
    print(title + "...")
    reconstructions[title] = raccoon_face.copy()
    t0 = time()
    dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
    code = dico.transform(data)
    patches = np.dot(code, V)

    patches += intercept
    patches = patches.reshape(len(data), *patch_size)
    if transform_algorithm == "threshold":
        patches -= patches.min()
        patches /= patches.max()
    reconstructions[title][:, width // 2 :] = reconstruct_from_patches_2d(
        patches, (height, width // 2)
    )
    dt = time() - t0
    # 打印完成信息，显示操作所花费的时间，%.2fs 是格式化字符串，用 dt 替换
    print("done in %.2fs." % dt)
    # 调用 show_with_diff 函数展示重建的图像和原始图像的差异
    # 使用 reconstructions[title] 作为重建图像，raccoon_face 作为原始图像，
    # 并显示标题和操作时间（%.1fs 是格式化字符串，用 dt 替换）
    show_with_diff(reconstructions[title], raccoon_face, title + " (time: %.1fs)" % dt)
# 显示当前 matplotlib 图形。通常用于在绘图后显示图形。
plt.show()
```