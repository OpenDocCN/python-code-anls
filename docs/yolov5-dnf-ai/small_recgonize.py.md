# `yolov5-DNF\small_recgonize.py`

```
# 导入 OpenCV 库并重命名为 cv，导入 NumPy 库并重命名为 np
import cv2 as cv
import numpy as np

# 设置图像文件路径
img_path = "test/DNF.png"
# 读取图像文件
img = cv.imread(img_path)

# 定义函数 img_show 用于显示图像
def img_show(img):
    # 在窗口中显示图像
    cv.imshow("win", img)
    # 等待按键输入，参数为 0 表示一直等待
    cv.waitKey(0)
    # 关闭所有窗口
    cv.destroyAllWindows()

# 定义函数 current_door 用于识别当前门的位置
def current_door(img, stride = 17):
    # 从图像中裁剪出门的区域
    crop = img[45:65, 1107:1270, 0]
    # 打开注释可视化裁剪出的门的区域
    # img_show(crop)
    # 找到门区域中像素值最大的位置
    index = np.unravel_index(crop.argmax(), crop.shape)
    # 计算门的位置
    i = int((index[1] // stride) + 1)
    return i  # 返回的是在第几个房间

# 定义函数 next_door 用于识别下一个门的位置
def next_door(img):
    # 从文件中加载问号模板图像
    img_temp = np.load("问号模板.npy")
    # 打开注释可视化加载的问号模板图像
    # img_show(img_temp)
    # 从当前图像中裁剪出门的区域
    target = img[45:65, 1107:1270]
    # 使用模板匹配算法找到问号模板在门区域中的位置
    result = cv.matchTemplate(target, img_temp, cv.TM_SQDIFF_NORMED)
    # 对匹配结果进行归一化
    cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)
    # 找到匹配结果中的最小值和位置
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    next_door_id = 0
    if min_val < 1e-10:
        # 打开注释可视化匹配结果和匹配值
        # print(min_val, max_val, min_loc, max_loc)
        # 将最小匹配值转换为字符串
        strmin_val = str(min_val)
        theight, twidth = img_temp.shape[:2]
        # 计算下一个门的位置
        next_door_id = int(((min_loc[0] + 0.5 * twidth) // 18.11) + 1)
    return next_door_id

# 如果作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 打印当前门的位置
    print(current_door(img))
    # 打印下一个门的位置
    print(next_door(img))
    # 打开注释可视化裁剪出的门的区域
    # img_show(img[45:65, 1144:1162])
    # 将裁剪出的门的区域保存为问号模板文件
    # np.save("问号模板", img[45:65, 1144:1162])
```