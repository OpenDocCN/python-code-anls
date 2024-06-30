# `D:\src\scipysrc\scikit-learn\benchmarks\plot_tsne_mnist.py`

```
# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import os.path as op  # 导入 os.path 模块并用别名 op 表示

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 用于绘图
import numpy as np  # 导入 numpy 库用于数值计算

LOG_DIR = "mnist_tsne_output"  # 定义日志目录路径为 mnist_tsne_output

# 程序入口，判断是否处于主程序中
if __name__ == "__main__":
    # 创建命令行参数解析器，设置程序描述
    parser = argparse.ArgumentParser("Plot benchmark results for t-SNE")
    # 添加命令行参数 --labels，指定类型为字符串，默认为 mnist_original_labels_10000.npy
    parser.add_argument(
        "--labels",
        type=str,
        default=op.join(LOG_DIR, "mnist_original_labels_10000.npy"),
        help="1D integer numpy array for labels",
    )
    # 添加命令行参数 --embedding，指定类型为字符串，默认为 mnist_sklearn_TSNE_10000.npy
    parser.add_argument(
        "--embedding",
        type=str,
        default=op.join(LOG_DIR, "mnist_sklearn_TSNE_10000.npy"),
        help="2D float numpy array for embedded data",
    )
    # 解析命令行参数并将其存储在 args 变量中
    args = parser.parse_args()

    # 从文件中加载嵌入数据到 X 变量中
    X = np.load(args.embedding)
    # 从文件中加载标签数据到 y 变量中
    y = np.load(args.labels)

    # 遍历标签中的唯一值
    for i in np.unique(y):
        # 创建布尔掩码，用于选取具有相同标签 i 的数据点
        mask = y == i
        # 绘制散点图，展示具有相同标签 i 的数据点，设置透明度为 0.2，标签为整数 i
        plt.scatter(X[mask, 0], X[mask, 1], alpha=0.2, label=int(i))
    
    # 在最佳位置添加图例
    plt.legend(loc="best")
    # 显示绘制的图形
    plt.show()
```