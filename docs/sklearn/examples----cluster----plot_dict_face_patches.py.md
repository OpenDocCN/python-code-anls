# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_dict_face_patches.py`

```
# %%
# Load the data
# -------------

# 导入必要的数据集模块
from sklearn import datasets

# 从 Olivetti faces 数据集中获取人脸数据
faces = datasets.fetch_olivetti_faces()

# %%
# Learn the dictionary of images
# ------------------------------

# 导入必要的库
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d

# 输出学习字典的提示信息
print("Learning the dictionary... ")

# 设置随机种子
rng = np.random.RandomState(0)

# 初始化 MiniBatchKMeans 对象
kmeans = MiniBatchKMeans(n_clusters=81, random_state=rng, verbose=True, n_init=3)

# 设置图像块的尺寸
patch_size = (20, 20)

# 初始化数据缓冲区
buffer = []

# 记录开始时间
t0 = time.time()

# 在整个数据集上进行6次迭代的在线学习过程
index = 0
for _ in range(6):
    for img in faces.images:
        # 从图像中提取随机的图像块
        data = extract_patches_2d(img, patch_size, max_patches=50, random_state=rng)
        
        # 对提取的图像块进行形状调整，展平为一维数组
        data = np.reshape(data, (len(data), -1))
        
        # 将处理后的数据存入缓冲区
        buffer.append(data)
        
        # 每处理10个图像块，进行一次数据处理和部分拟合
        index += 1
        if index % 10 == 0:
            data = np.concatenate(buffer, axis=0)
            data -= np.mean(data, axis=0)
            data /= np.std(data, axis=0)
            kmeans.partial_fit(data)
            buffer = []
        
        # 每处理100个图像块，打印部分拟合的进度信息
        if index % 100 == 0:
            print("Partial fit of %4i out of %i" % (index, 6 * len(faces.images)))

# 计算并打印总共用时
dt = time.time() - t0
print("done in %.2fs." % dt)

# %%
# Plot the results
# ----------------

# 导入绘图库
import matplotlib.pyplot as plt

# 创建绘图窗口
plt.figure(figsize=(4.2, 4))

# 对每个聚类中心绘制对应的图像块
for i, patch in enumerate(kmeans.cluster_centers_):
    plt.subplot(9, 9, i + 1)
    plt.imshow(patch.reshape(patch_size), cmap=plt.cm.gray, interpolation="nearest")
    plt.xticks(())
    plt.yticks(())

# 设置标题和调整布局
plt.suptitle(
    "Patches of faces\nTrain time %.1fs on %d patches" % (dt, 8 * len(faces.images)),
    fontsize=16,
)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

# 展示绘图结果
plt.show()
```