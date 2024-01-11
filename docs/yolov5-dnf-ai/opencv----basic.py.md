# `yolov5-DNF\opencv\basic.py`

```
# 导入 OpenCV 库
import cv2
# 导入 NumPy 库
import numpy as np

# -----------蓝色通道值--------------
# 创建一个 300x300 的三通道图像，数据类型为无符号 8 位整数
blue = np.zeros((300, 300, 3), dtype=np.uint8)
# 将蓝色通道的所有像素值设为 255
blue[:, :, 0] = 255
# 打印蓝色通道图像的像素值
print("blue=\n", blue)
# 在窗口中显示蓝色通道图像
cv2.imshow("blue", blue)

# -----------绿色通道值--------------
# 创建一个 300x300 的三通道图像，数据类型为无符号 8 位整数
green = np.zeros((300, 300, 3), dtype=np.uint8)
# 将绿色通道的所有像素值设为 255
green[:, :, 1] = 255
# 打印绿色通道图像的像素值
print("green=\n", green)
# 在窗口中显示绿色通道图像
cv2.imshow("green", green)

# -----------红色通道值--------------
# 创建一个 300x300 的三通道图像，数据类型为无符号 8 位整数
red = np.zeros((300, 300, 3), dtype=np.uint8)
# 将红色通道的所有像素值设为 255
red[:, :, 2] = 255
# 打印红色通道图像的像素值
print("red=\n", red)
# 在窗口中显示红色通道图像
cv2.imshow("red", red)

# -----------释放窗口--------------
# 等待按键输入
cv2.waitKey()
# 关闭所有窗口
cv2.destroyAllWindows()
```