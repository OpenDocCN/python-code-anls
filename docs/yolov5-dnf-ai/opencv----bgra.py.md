# `yolov5-DNF\opencv\bgra.py`

```
# 导入 OpenCV 库
import cv2

# 读取名为 "lena.jpg" 的图像文件
img = cv2.imread("lena.jpg")

# 将图像从 BGR 转换为 BGRA 格式
bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

# 将 BGRA 图像拆分成单独的通道
b, g, r, a = cv2.split(bgra)

# 将 alpha 通道的所有像素值设置为 125
a[:, :] = 125

# 合并修改后的通道，得到新的 BGRA 图像
bgra125 = cv2.merge([b, g, r, a])

# 将 alpha 通道的所有像素值设置为 0
a[:, :] = 0

# 合并修改后的通道，得到新的 BGRA 图像
bgra0 = cv2.merge([b, g, r, a])

# 在窗口中显示原始图像
cv2.imshow("img", img)

# 在窗口中显示转换后的 BGRA 图像
cv2.imshow("bgra", bgra)

# 在窗口中显示 alpha 通道像素值为 125 的 BGRA 图像
cv2.imshow("bgra125", bgra125)

# 在窗口中显示 alpha 通道像素值为 0 的 BGRA 图像
cv2.imshow("bgra0", bgra0)

# 等待用户按下任意键
cv2.waitKey()

# 关闭所有窗口
cv2.destroyAllWindows()

# 将 BGRA 图像保存为 "bgra.png" 文件
cv2.imwrite("bgra.png", bgra)

# 将 alpha 通道像素值为 125 的 BGRA 图像保存为 "bgra125.png" 文件
cv2.imwrite("bgra125.png", bgra125)

# 将 alpha 通道像素值为 0 的 BGRA 图像保存为 "bgra0.png" 文件
cv2.imwrite("bgra0.png", bgra0)
```