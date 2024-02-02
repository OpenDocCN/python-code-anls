# `yolov5-DNF\skill_recgnize.py`

```py
import cv2 as cv  # 导入 OpenCV 库，重命名为 cv

# 计算图像中像素值大于 127 的比例
def score(img):
    counter = 0  # 初始化计数器
    for i in range(img.shape[0]):  # 遍历图像的行
        for j in range(img.shape[1]):  # 遍历图像的列
            if img[i,j] > 127:  # 如果像素值大于 127
                counter += 1  # 计数器加一
    return counter/(img.shape[0] * img.shape[1])  # 返回大于 127 的像素比例

# 显示图像
def img_show(img):
    cv.imshow("win", img)  # 在窗口中显示图像
    cv.waitKey(0)  # 等待按键按下
    cv.destroyAllWindows()  # 关闭所有窗口

# 计算技能框的高度和宽度
skill_height = int((793-733)/2)  # 技能框的高度
skill_width = int((750-538)/7)  # 技能框的宽度

# 技能框的位置信息
dict = {"A": (733+skill_height, 538), "S": (733+skill_height, 538+skill_width), "D": (733+skill_height, 538+2*skill_width),
        "F": (733+skill_height, 538+3*skill_width), "G": (733+skill_height, 538+4*skill_width),
        "H": (733+skill_height, 538+5*skill_width), "Q": (733, 538), "W": (733, 538+skill_width), "E": (733, 538+2*skill_width),
        "R": (733, 538+3*skill_width), "T": (733, 538+4*skill_width), "Y": (733, 538+5*skill_width)}

# 检测技能是否可用
def skill_rec(skill_name, img):
    if skill_name == "X":  # 如果技能名为 "X"
        return True  # 返回 True
    skill_img = img[dict[skill_name][0]: dict[skill_name][0]+skill_height,
                dict[skill_name][1]: dict[skill_name][1]+skill_width, 2]  # 获取技能框对应的图像
    if score(skill_img) > 0.1:  # 如果技能框中像素值大于 127 的比例大于 0.1
        return True  # 返回 True
    else:
        return False  # 返回 False

# 主函数
if __name__ == "__main__":
    img_path = "datasets/guiqi/test/20_93.jpg"  # 图像路径
    img = cv.imread(img_path)  # 读取图像
    print(skill_height, skill_width)  # 打印技能框的高度和宽度
    print(img.shape)  # 打印图像的形状
    skill_img = img[733: 793, 538:750, 2]  # 获取特定区域的图像
    img_show(skill_img)  # 显示图像

    # 获取特定技能框的图像
    skill_imgA = img[dict["A"][0]: dict["A"][0]+skill_height, dict["A"][1]: dict["A"][1]+skill_width, 2]
    skill_imgH= img[dict["H"][0]: dict["H"][0]+skill_height, dict["H"][1]: dict["H"][1]+skill_width, 2]
    skill_imgG= img[dict["G"][0]: dict["G"][0]+skill_height, dict["G"][1]: dict["G"][1]+skill_width, 2]
    skill_imgE= img[dict["E"][0]: dict["E"][0]+skill_height, dict["E"][1]: dict["E"][1]+skill_width, 2]
    skill_imgQ= img[dict["Q"][0]: dict["Q"][0]+skill_height, dict["Q"][1]: dict["Q"][1]+skill_width, 2]
    # 从原始图像中提取技能图像S的蓝色通道
    skill_imgS= img[dict["S"][0]: dict["S"][0]+skill_height, dict["S"][1]: dict["S"][1]+skill_width, 2]
    # 从原始图像中提取技能图像Y的蓝色通道
    skill_imgY= img[dict["Y"][0]: dict["Y"][0]+skill_height, dict["Y"][1]: dict["Y"][1]+skill_width, 2]
    # 从原始图像中提取技能图像D的蓝色通道
    skill_imgD = img[dict["D"][0]: dict["D"][0]+skill_height, dict["D"][1]: dict["D"][1]+skill_width, 2]
    # 从原始图像中提取技能图像F的蓝色通道
    skill_imgF = img[dict["F"][0]: dict["F"][0]+skill_height, dict["F"][1]: dict["F"][1]+skill_width, 2]
    # 从原始图像中提取技能图像W的蓝色通道
    skill_imgW = img[dict["W"][0]: dict["W"][0]+skill_height, dict["W"][1]: dict["W"][1]+skill_width, 2]
    # 从原始图像中提取技能图像R的蓝色通道
    skill_imgR = img[dict["R"][0]: dict["R"][0]+skill_height, dict["R"][1]: dict["R"][1]+skill_width, 2]

    # 打印技能图像A的蓝色通道的平均值
    # print("A", np.mean(skill_imgA))
    # 打印技能图像H的蓝色通道的平均值
    # print("H", np.mean(skill_imgH))
    # 打印技能图像G的蓝色通道的平均值
    # print("G", np.mean(skill_imgG))
    # 打印技能图像E的蓝色通道的平均值
    # print("E", np.mean(skill_imgE))
    # 打印技能图像Q的蓝色通道的平均值
    # print("Q", np.mean(skill_imgQ))
    # 打印技能图像S的蓝色通道的平均值
    # print("S", np.mean(skill_imgS))
    # 打印技能图像Y的蓝色通道的平均值
    # print("Y", np.mean(skill_imgY))

    # 打印技能图像A的蓝色通道的得分
    print("A", score(skill_imgA))
    # 打印技能图像Q的蓝色通道的得分
    print("Q", score(skill_imgQ))
    # 打印技能图像S的蓝色通道的得分
    print("S", score(skill_imgS))
    # 打印技能图像D的蓝色通道的得分
    print("D", score(skill_imgD))
    # 打印技能图像F的蓝色通道的得分
    print("F", score(skill_imgF))
    # 打印技能图像W的蓝色通道的得分
    print("W", score(skill_imgW))
    # 打印技能图像R的蓝色通道的得分
    print("R", score(skill_imgR))
    # 打印技能图像Y的蓝色通道的得分
    print("Y", score(skill_imgY))
    # 打印技能图像H的蓝色通道的得分
    print("H", score(skill_imgH))
    # 打印技能图像G的蓝色通道的得分
    print("G", score(skill_imgG))
    # 打印技能图像E的蓝色通道的得分
    print("E", score(skill_imgE))

    # 打印技能图像W的识别结果
    print(skill_rec("W", img))
```