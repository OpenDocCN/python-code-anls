# `.\PaddleOCR\test_tipc\supplementary\load_cifar.py`

```py
# 导入 pickle 库并重命名为 p
import pickle as p
# 导入 numpy 库并重命名为 np
import numpy as np
# 从 PIL 库中导入 Image 类
from PIL import Image

# 加载 CIFAR 数据集中的单个批次
def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    # 以二进制只读方式打开文件
    with open(filename, 'rb') as f:
        # 使用 pickle 加载文件数据，指定编码为 bytes
        datadict = p.load(f, encoding='bytes')
        # 从数据字典中取出图像数据 X 和标签数据 Y
        X = datadict[b'data']
        Y = datadict[b'fine_labels']
        # 尝试将图像数据 X 重塑为指定形状
        try:
            X = X.reshape(10000, 3, 32, 32)
        except:
            X = X.reshape(50000, 3, 32, 32)
        # 将标签数据 Y 转换为 numpy 数组
        Y = np.array(Y)
        # 打印标签数据 Y 的形状
        print(Y.shape)
        # 返回图像数据 X 和标签数据 Y
        return X, Y

# 主程序入口
if __name__ == "__main__":
    # 设置模式为训练模式
    mode = "train"
    # 载入 CIFAR 数据集中指定模式的图像数据和标签数据
    imgX, imgY = load_CIFAR_batch(f"./cifar-100-python/{mode}")
    # 以追加模式打开文件，将图像标签信息写入文件
    with open(f'./cifar-100-python/{mode}_imgs/img_label.txt', 'a+') as f:
        for i in range(imgY.shape[0]):
            f.write('img' + str(i) + ' ' + str(imgY[i]) + '\n')

    # 遍历图像数据 X
    for i in range(imgX.shape[0]):
        # 获取单个图像的 RGB 通道数据
        imgs = imgX[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        # 使用 PIL 创建图像对象
        i0 = Image.fromarray(img0)
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        # 合并 RGB 通道，创建完整图像
        img = Image.merge("RGB", (i0, i1, i2))
        # 设置图像文件名
        name = "img" + str(i) + ".png"
        # 保存图像文件
        img.save(f"./cifar-100-python/{mode}_imgs/" + name, "png")
    # 打印保存成功信息
    print("save successfully!")
```