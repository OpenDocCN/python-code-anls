# `D:\src\scipysrc\scipy\scipy\ndimage\utils\generate_label_testvectors.py`

```
import numpy as np  # 导入 NumPy 库，用于处理多维数组和矩阵运算
from scipy.ndimage import label  # 从 SciPy 库中导入 label 函数，用于图像标记


def generate_test_vecs(infile, strelfile, resultfile):
    "test label with different structuring element neighborhoods"
    # 定义内部函数 bitimage，将字符串列表转换为布尔型二维数组
    def bitimage(l):
        return np.array([[c for c in s] for s in l]) == '1'
    
    # 初始化数据列表，包含三个不同的数据集合
    data = [np.ones((7, 7)),  # 7x7 全 1 矩阵
            bitimage(["1110111",  # 自定义的布尔型二维数组
                      "1100011",
                      "1010101",
                      "0001000",
                      "1010101",
                      "1100011",
                      "1110111"]),
            bitimage(["1011101",  # 另一个自定义的布尔型二维数组
                      "0001000",
                      "1001001",
                      "1111111",
                      "1001001",
                      "0001000",
                      "1011101"])]
    
    # 初始化结构元素列表，包含不同的结构元素形状
    strels = [np.ones((3, 3)),  # 3x3 全 1 矩阵
              np.zeros((3, 3)),  # 3x3 全 0 矩阵
              bitimage(["010", "111", "010"]),  # 自定义的布尔型二维数组
              bitimage(["101", "010", "101"]),  # 自定义的布尔型二维数组
              bitimage(["100", "010", "001"]),  # 自定义的布尔型二维数组
              bitimage(["000", "111", "000"]),  # 自定义的布尔型二维数组
              bitimage(["110", "010", "011"]),  # 自定义的布尔型二维数组
              bitimage(["110", "111", "011"])]  # 自定义的布尔型二维数组
    
    # 扩展结构元素列表，包括每个结构元素的上下翻转
    strels = strels + [np.flipud(s) for s in strels]
    
    # 扩展结构元素列表，包括每个结构元素的旋转90度
    strels = strels + [np.rot90(s) for s in strels]
    
    # 去除重复的结构元素，并转换为3x3整数型数组列表
    strels = [np.fromstring(s, dtype=int).reshape((3, 3))
              for s in {t.astype(int).tobytes() for t in strels}]
    
    # 将数据集合垂直堆叠成输入数据
    inputs = np.vstack(data)
    
    # 对每个数据和结构元素的组合进行标记，得到结果数据，并垂直堆叠
    results = np.vstack([label(d, s)[0] for d in data for s in strels])
    
    # 将结构元素列表垂直堆叠，并保存到文件中
    strels = np.vstack(strels)
    np.savetxt(infile, inputs, fmt="%d")  # 将输入数据保存到文件
    np.savetxt(strelfile, strels, fmt="%d")  # 将结构元素保存到文件
    np.savetxt(resultfile, results, fmt="%d")  # 将结果数据保存到文件


generate_test_vecs("label_inputs.txt", "label_strels.txt", "label_results.txt")  # 调用函数生成测试向量，并保存到文件
```