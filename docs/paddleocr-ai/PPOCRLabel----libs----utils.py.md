# `.\PaddleOCR\PPOCRLabel\libs\utils.py`

```
# 版权声明，允许在遵守条件下自由使用软件
# 导入所需的库和模块
import hashlib
import os
import re
import sys
from math import sqrt

import cv2
import numpy as np
from PyQt5.QtCore import QRegExp, QT_VERSION_STR
from PyQt5.QtGui import QIcon, QRegExpValidator, QColor
from PyQt5.QtWidgets import QPushButton, QAction, QMenu
from libs.ustr import ustr

# 获取当前文件所在目录路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 获取图标文件路径
__iconpath__ = os.path.abspath(os.path.join(__dir__, '../resources/icons'))

# 定义函数，根据图标名称和大小创建 QIcon 对象
def newIcon(icon, iconSize=None):
    if iconSize is not None:
        return QIcon(QIcon(__iconpath__ + "/" + icon + ".png").pixmap(iconSize, iconSize))
    else:
        return QIcon(__iconpath__ + "/" + icon + ".png")

# 定义函数，创建带有文本和图标的 QPushButton 对象
def newButton(text, icon=None, slot=None):
    b = QPushButton(text)
    if icon is not None:
        b.setIcon(newIcon(icon))
    # 如果槽函数不为空，则将按钮的点击信号连接到槽函数
    if slot is not None:
        b.clicked.connect(slot)
    # 返回创建的按钮
    return b
# 创建一个新的动作，并设置回调函数、快捷键、图标等属性
def newAction(parent, text, slot=None, shortcut=None, icon=None,
              tip=None, checkable=False, enabled=True, iconSize=None):
    # 创建一个 QAction 对象，设置文本和父对象
    a = QAction(text, parent)
    # 如果有图标，则设置图标
    if icon is not None:
        # 如果有指定图标大小，则设置图标和大小
        if iconSize is not None:
            a.setIcon(newIcon(icon, iconSize))
        else:
            a.setIcon(newIcon(icon))
    # 如果有快捷键，则设置快捷键
    if shortcut is not None:
        # 如果快捷键是列表或元组，则设置多个快捷键
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    # 如果有提示信息，则设置提示信息和状态栏提示信息
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    # 如果有回调函数，则连接到 triggered 信号
    if slot is not None:
        a.triggered.connect(slot)
    # 如果是可选中的动作，则设置为可选中
    if checkable:
        a.setCheckable(True)
    # 设置动作是否可用
    a.setEnabled(enabled)
    return a


# 将动作添加到小部件中
def addActions(widget, actions):
    for action in actions:
        # 如果动作为 None，则添加分隔符
        if action is None:
            widget.addSeparator()
        # 如果动作是 QMenu 对象，则添加菜单
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        # 否则添加动作
        else:
            widget.addAction(action)


# 创建一个验证器，用于验证标签
def labelValidator():
    return QRegExpValidator(QRegExp(r'^[^ \t].+'), None)


# 定义一个结构体类，用于存储属性
class struct(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# 计算点 p 到原点的距离
def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


# 格式化快捷键文本
def fmtShortcut(text):
    mod, key = text.split('+', 1)
    return '<b>%s</b>+<b>%s</b>' % (mod, key)


# 根据文本生成颜色
def generateColorByText(text):
    s = ustr(text)
    hashCode = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
    r = int((hashCode / 255) % 255)
    g = int((hashCode / 65025) % 255)
    b = int((hashCode / 16581375) % 255)
    return QColor(r, g, b, 100)


# 检查是否有 QString 类型
def have_qstring():
    '''p3/qt5 get rid of QString wrapper as py3 has native unicode str type'''
    return not (sys.version_info.major >= 3 or QT_VERSION_STR.startswith('5.'))


# 对列表进行自然排序
def natural_sort(list, key=lambda s: s):
    """
    Sort the list into natural alphanumeric order.
    """
    # 定义一个函数，用于生成按照字母和数字排序的键值函数
    def get_alphanum_key_func(key):
        # 定义一个匿名函数，根据文本内容是数字还是字母进行转换
        convert = lambda text: int(text) if text.isdigit() else text
        # 返回一个 lambda 函数，根据传入的 key 函数生成按照字母和数字排序的键值函数
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    
    # 根据传入的 key 函数生成按照字母和数字排序的键值函数
    sort_key = get_alphanum_key_func(key)
    # 使用生成的按照字母和数字排序的键值函数对列表进行排序
    list.sort(key=sort_key)
# 根据给定的图像和四个点的坐标，进行旋转和裁剪操作
def get_rotate_crop_image(img, points):
    # 使用 Green's theory 判断顺时针还是逆时针
    # 作者：biyanhua
    d = 0.0
    for index in range(-1, 3):
        d += -0.5 * (points[index + 1][1] + points[index][1]) * (
                points[index + 1][0] - points[index][0])
    if d < 0:  # 逆时针
        tmp = np.array(points)
        points[1], points[3] = tmp[3], tmp[1]

    try:
        # 计算裁剪后图像的宽度和高度
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        # 定义标准的四个点坐标
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        # 获取透视变换矩阵
        M = cv2.getPerspectiveTransform(points, pts_std)
        # 进行透视变换
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        # 根据高宽比例判断是否需要旋转图像
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
    except Exception as e:
        print(e)


# 对给定的框进行填充，每个边填充 [pad] 个像素
def boxPad(box, imgShape, pad : int) -> np.array:
    """
    对框进行填充，每个边填充 [pad] 个像素
    """
    box = np.array(box, dtype=np.int32)
    # 对框的四个顶点进行填充操作
    box[0][0], box[0][1] = box[0][0] - pad, box[0][1] - pad
    box[1][0], box[1][1] = box[1][0] + pad, box[1][1] - pad
    box[2][0], box[2][1] = box[2][0] + pad, box[2][1] + pad
    box[3][0], box[3][1] = box[3][0] - pad, box[3][1] + pad
    h, w, _ = imgShape
    # 对填充后的框进行边界裁剪
    box[:,0] = np.clip(box[:,0], 0, w)
    box[:,1] = np.clip(box[:,1], 0, h)
    return box


# 根据合并的单元格填充列表中的空白
def expand_list(merged, html_list):
    '''
    根据合并的单元格填充列表中的空白
    '''
    # 解压元组 merged 中的四个值，分别赋值给 sr, er, sc, ec
    sr, er, sc, ec = merged
    # 遍历二维列表 html_list 的行索引范围从 sr 到 er-1
    for i in range(sr, er):
        # 遍历二维列表 html_list 的列索引范围从 sc 到 ec-1
        for j in range(sc, ec):
            # 将二维列表 html_list 中第 i 行第 j 列的元素设为 None
            html_list[i][j] = None
    # 将二维列表 html_list 中第 sr 行第 sc 列的元素设为空字符串
    html_list[sr][sc] = ''
    # 如果合并的列数大于 1，则在 html_list 中第 sr 行第 sc 列的元素后添加 colspan=合并的列数
    if ec - sc > 1:
        html_list[sr][sc] += " colspan={}".format(ec - sc)
    # 如果合并的行数大于 1，则在 html_list 中第 sr 行第 sc 列的元素后添加 rowspan=合并的行数
    if er - sr > 1:
        html_list[sr][sc] += " rowspan={}".format(er - sr)
    # 返回修改后的二维列表 html_list
    return html_list
# 将原始 HTML 转换为标签格式
def convert_token(html_list):
    '''
    Convert raw html to label format
    '''
    # 初始化标签列表，包含 <tbody> 标签
    token_list = ["<tbody>"]
    # 遍历原始 HTML 列表
    for row in html_list:
        # 添加 <tr> 标签
        token_list.append("<tr>")
        # 遍历每一行的列
        for col in row:
            # 如果列为空，则跳过
            if col == None:
                continue
            # 如果列为 'td'，则添加 <td> 和 </td> 标签
            elif col == 'td':
                token_list.extend(["<td>", "</td>"])
            else:
                # 如果列不为空，添加 <td 标签
                token_list.append("<td")
                # 如果列包含 'colspan' 属性，添加 colspan 属性
                if 'colspan' in col:
                    _, n = col.split('colspan=')
                    token_list.append(" colspan=\"{}\"".format(n[0]))
                # 如果列包含 'rowspan' 属性，添加 rowspan 属性
                if 'rowspan' in col:
                    _, n = col.split('rowspan=')
                    token_list.append(" rowspan=\"{}\"".format(n[0]))
                token_list.extend([">", "</td>"])
        # 添加 </tr> 标签
        token_list.append("</tr>")
    # 添加 </tbody> 标签
    token_list.append("</tbody>")
    # 返回标签列表
    return token_list


# 从预处理结构标签信息中重建 HTML
def rebuild_html_from_ppstructure_label(label_info):
        from html import escape
        # 复制 HTML 代码结构的标记
        html_code = label_info['html']['structure']['tokens'].copy()
        # 找到需要插入的位置
        to_insert = [
            i for i, tag in enumerate(html_code) if tag in ('<td>', '>')
        ]
        # 逆序遍历需要插入的位置和对应的单元格信息
        for i, cell in zip(to_insert[::-1], label_info['html']['cells'][::-1]):
            if cell['tokens']:
                # 对单元格中的标记进行转义处理
                cell = [
                    escape(token) if len(token) == 1 else token
                    for token in cell['tokens']
                ]
                cell = ''.join(cell)
                # 在指定位置插入单元格内容
                html_code.insert(i + 1, cell)
        # 将标记列表转换为字符串
        html_code = ''.join(html_code)
        # 添加 HTML 头部和尾部，构成完整的 HTML 代码
        html_code = '<html><body><table>{}</table></body></html>'.format(
            html_code)
        # 返回重建后的 HTML 代码
        return html_code


# 获取步骤信息
def stepsInfo(lang='en'):
    # 如果语言为中文
    if lang == 'ch':
        # 设置包含程序使用说明的消息
        msg = "1. 安装与运行：使用上述命令安装与运行程序。\n" \
              "2. 打开文件夹：在菜单栏点击 “文件” - 打开目录 选择待标记图片的文件夹.\n" \
              "3. 自动标注：点击 ”自动标注“，使用PPOCR超轻量模型对图片文件名前图片状态为 “X” 的图片进行自动标注。\n" \
              "4. 手动标注：点击 “矩形标注”（推荐直接在英文模式下点击键盘中的 “W”)，用户可对当前图片中模型未检出的部分进行手动" \
              "绘制标记框。点击键盘P，则使用四点标注模式（或点击“编辑” - “四点标注”），用户依次点击4个点后，双击左键表示标注完成。\n" \
              "5. 标记框绘制完成后，用户点击 “确认”，检测框会先被预分配一个 “待识别” 标签。\n" \
              "6. 重新识别：将图片中的所有检测画绘制/调整完成后，点击 “重新识别”，PPOCR模型会对当前图片中的**所有检测框**重新识别。\n" \
              "7. 内容更改：双击识别结果，对不准确的识别结果进行手动更改。\n" \
              "8. 保存：点击 “保存”，图片状态切换为 “√”，跳转至下一张。\n" \
              "9. 删除：点击 “删除图像”，图片将会被删除至回收站。\n" \
              "10. 标注结果：关闭应用程序或切换文件路径后，手动保存过的标签将会被存放在所打开图片文件夹下的" \
              "*Label.txt*中。在菜单栏点击 “PaddleOCR” - 保存识别结果后，会将此类图片的识别训练数据保存在*crop_img*文件夹下，" \
              "识别标签保存在*rec_gt.txt*中。\n"
    # 如果条件不满足，则返回以下提示信息
    else:
        # 提示用户操作步骤
        msg = "1. Build and launch using the instructions above.\n" \
              "2. Click 'Open Dir' in Menu/File to select the folder of the picture.\n" \
              "3. Click 'Auto recognition', use PPOCR model to automatically annotate images which marked with 'X' before the file name." \
              "4. Create Box:\n" \
              "4.1 Click 'Create RectBox' or press 'W' in English keyboard mode to draw a new rectangle detection box. Click and release left mouse to select a region to annotate the text area.\n" \
              "4.2 Press 'P' to enter four-point labeling mode which enables you to create any four-point shape by clicking four points with the left mouse button in succession and DOUBLE CLICK the left mouse as the signal of labeling completion.\n" \
              "5. After the marking frame is drawn, the user clicks 'OK', and the detection frame will be pre-assigned a TEMPORARY label.\n" \
              "6. Click re-Recognition, model will rewrite ALL recognition results in ALL detection box.\n" \
              "7. Double click the result in 'recognition result' list to manually change inaccurate recognition results.\n" \
              "8. Click 'Save', the image status will switch to '√',then the program automatically jump to the next.\n" \
              "9. Click 'Delete Image' and the image will be deleted to the recycle bin.\n" \
              "10. Labeling result: After closing the application or switching the file path, the manually saved label will be stored in *Label.txt* under the opened picture folder.\n" \
              "    Click PaddleOCR-Save Recognition Results in the menu bar, the recognition training data of such pictures will be saved in the *crop_img* folder, and the recognition label will be saved in *rec_gt.txt*.\n"
    
    # 返回提示信息
    return msg
# 定义一个函数，返回不同语言环境下的快捷键信息
def keysInfo(lang='en'):
    # 如果语言为中文
    if lang == 'ch':
        # 设置中文语言下的快捷键信息
        msg = "快捷键\t\t\t说明\n" \
              "———————————————————————\n" \
              "Ctrl + shift + R\t\t对当前图片的所有标记重新识别\n" \
              "W\t\t\t新建矩形框\n" \
              "Q\t\t\t新建四点框\n" \
              "Ctrl + E\t\t编辑所选框标签\n" \
              "Ctrl + R\t\t重新识别所选标记\n" \
              "Ctrl + C\t\t复制并粘贴选中的标记框\n" \
              "Ctrl + 鼠标左键\t\t多选标记框\n" \
              "Backspace\t\t删除所选框\n" \
              "Ctrl + V\t\t确认本张图片标记\n" \
              "Ctrl + Shift + d\t删除本张图片\n" \
              "D\t\t\t下一张图片\n" \
              "A\t\t\t上一张图片\n" \
              "Ctrl++\t\t\t缩小\n" \
              "Ctrl--\t\t\t放大\n" \
              "↑→↓←\t\t\t移动标记框\n" \
              "———————————————————————\n" \
              "注：Mac用户Command键替换上述Ctrl键"
    # 如果语言为英文或其他
    else:
        # 设置英文语言下的快捷键信息
        msg = "Shortcut Keys\t\tDescription\n" \
              "———————————————————————\n" \
              "Ctrl + shift + R\t\tRe-recognize all the labels\n" \
              "\t\t\tof the current image\n" \
              "\n" \
              "W\t\t\tCreate a rect box\n" \
              "Q\t\t\tCreate a four-points box\n" \
              "Ctrl + E\t\tEdit label of the selected box\n" \
              "Ctrl + R\t\tRe-recognize the selected box\n" \
              "Ctrl + C\t\tCopy and paste the selected\n" \
              "\t\t\tbox\n" \
              "\n" \
              "Ctrl + Left Mouse\tMulti select the label\n" \
              "Button\t\t\tbox\n" \
              "\n" \
              "Backspace\t\tDelete the selected box\n" \
              "Ctrl + V\t\tCheck image\n" \
              "Ctrl + Shift + d\tDelete image\n" \
              "D\t\t\tNext image\n" \
              "A\t\t\tPrevious image\n" \
              "Ctrl++\t\t\tZoom in\n" \
              "Ctrl--\t\t\tZoom out\n" \
              "↑→↓←\t\t\tMove selected box" \
              "———————————————————————\n" \
              "Notice:For Mac users, use the 'Command' key instead of the 'Ctrl' key"
    
    # 返回对应语言环境下的快捷键信息
    return msg
```