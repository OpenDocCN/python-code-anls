# `.\PaddleOCR\PPOCRLabel\libs\autoDialog.py`

```py
# 尝试导入 PyQt5 库中的模块，如果导入失败则导入 PyQt4 库中的模块
try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

# 导入时间、日期时间、JSON、OpenCV、NumPy 等模块
import time
import datetime
import json
import cv2
import numpy as np

# 从自定义的 libs.utils 模块中导入 newIcon 函数
from libs.utils import newIcon

# 定义 QDialogButtonBox 的别名为 BB
BB = QDialogButtonBox

# 定义一个继承自 QThread 的 Worker 类
class Worker(QThread):
    # 定义信号 progressBarValue，用于发送进度条数值
    progressBarValue = pyqtSignal(int)
    # 定义信号 listValue，用于发送字符串数值
    listValue = pyqtSignal(str)
    # 定义信号 endsignal，用于发送整数和字符串数值
    endsignal = pyqtSignal(int, str)
    # 初始化类属性 handle 为 0
    handle = 0

    # 定义 Worker 类的初始化方法，接受 ocr、mImgList、mainThread、model 参数
    def __init__(self, ocr, mImgList, mainThread, model):
        # 调用父类 QThread 的初始化方法
        super(Worker, self).__init__()
        # 初始化实例属性 ocr、mImgList、mainThread、model
        self.ocr = ocr
        self.mImgList = mImgList
        self.mainThread = mainThread
        self.model = model
        # 设置线程栈大小为 1024*1024 字节
        self.setStackSize(1024*1024)
    # 定义一个方法，用于执行OCR识别任务
    def run(self):
        try:
            # 初始化文件索引
            findex = 0
            # 遍历待处理的图片路径列表
            for Imgpath in self.mImgList:
                # 如果处理标志为0
                if self.handle == 0:
                    # 发射信号，通知界面显示当前处理的图片路径
                    self.listValue.emit(Imgpath)
                    # 如果模型为'paddle'
                    if self.model == 'paddle':
                        # 读取图片并获取其高度、宽度
                        h, w, _ = cv2.imdecode(np.fromfile(Imgpath, dtype=np.uint8), 1).shape
                        # 如果图片高度和宽度均大于32
                        if h > 32 and w > 32:
                            # 进行OCR识别，获取识别结果
                            self.result_dic = self.ocr.ocr(Imgpath, cls=True, det=True)[0]
                        else:
                            # 如果图片尺寸过小无法识别，则打印提示信息
                            print('The size of', Imgpath, 'is too small to be recognised')
                            self.result_dic = None

                    # 处理识别结果
                    if self.result_dic is None or len(self.result_dic) == 0:
                        # 如果识别结果为空，则打印提示信息
                        print('Can not recognise file', Imgpath)
                        pass
                    else:
                        # 处理识别结果，拼接识别文本、置信度、位置信息
                        strs = ''
                        for res in self.result_dic:
                            chars = res[1][0]
                            cond = res[1][1]
                            posi = res[0]
                            strs += "Transcription: " + chars + " Probability: " + str(cond) + \
                                    " Location: " + json.dumps(posi) +'\n'
                        # 发射信号，通知界面显示识别结果
                        self.listValue.emit(strs)
                        # 更新主线程的识别结果和文件路径
                        self.mainThread.result_dic = self.result_dic
                        self.mainThread.filePath = Imgpath
                        # 保存识别结果
                        self.mainThread.saveFile(mode='Auto')
                    # 更新进度条数值
                    findex += 1
                    self.progressBarValue.emit(findex)
                else:
                    # 如果处理标志不为0，则跳出循环
                    break
            # 发射结束信号
            self.endsignal.emit(0, "readAll")
            # 执行事件循环
            self.exec()
        except Exception as e:
            # 捕获异常并打印
            print(e)
            raise
class AutoDialog(QDialog):
    # 自动对话框类，继承自 QDialog

    def __init__(self, text="Enter object label", parent=None, ocr=None, mImgList=None, lenbar=0):
        # 初始化函数，接受文本、父窗口、OCR对象、图像列表和长度参数

        super(AutoDialog, self).__init__(parent)
        # 调用父类的初始化函数

        self.setFixedWidth(1000)
        # 设置对话框固定宽度为1000

        self.parent = parent
        # 设置父窗口

        self.ocr = ocr
        # 设置OCR对象

        self.mImgList = mImgList
        # 设置图像列表

        self.lender = lenbar
        # 设置长度参数

        self.pb = QProgressBar()
        # 创建进度条对象

        self.pb.setRange(0, self.lender)
        # 设置进度条范围为0到长度参数

        self.pb.setValue(0)
        # 设置进度条初始值为0

        layout = QVBoxLayout()
        # 创建垂直布局对象

        layout.addWidget(self.pb)
        # 将进度条添加到布局中

        self.model = 'paddle'
        # 设置模型为'paddle'

        self.listWidget = QListWidget(self)
        # 创建列表部件对象

        layout.addWidget(self.listWidget)
        # 将列表部件添加到布局中

        self.buttonBox = bb = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        # 创建按钮盒子对象，包含确定和取消按钮

        bb.button(BB.Ok).setIcon(newIcon('done'))
        # 设置确定按钮图标

        bb.button(BB.Cancel).setIcon(newIcon('undo'))
        # 设置取消按钮图标

        bb.accepted.connect(self.validate)
        # 连接确定按钮的信号到 validate 方法

        bb.rejected.connect(self.reject)
        # 连接取消按钮的信号到 reject 方法

        layout.addWidget(bb)
        # 将按钮盒子添加到布局中

        bb.button(BB.Ok).setEnabled(False)
        # 设置确定按钮不可用

        self.setLayout(layout)
        # 设置布局

        self.setWindowModality(Qt.ApplicationModal)
        # 设置窗口模态性为应用程序模态

        self.thread_1 = Worker(self.ocr, self.mImgList, self.parent, 'paddle')
        # 创建 Worker 线程对象

        self.thread_1.progressBarValue.connect(self.handleProgressBarSingal)
        # 连接 Worker 线程的进度条数值信号到 handleProgressBarSingal 方法

        self.thread_1.listValue.connect(self.handleListWidgetSingal)
        # 连接 Worker 线程的列表数值信号到 handleListWidgetSingal 方法

        self.thread_1.endsignal.connect(self.handleEndsignalSignal)
        # 连接 Worker 线程的结束信号到 handleEndsignalSignal 方法

        self.time_start = time.time()  # save start time
        # 记录开始时间

    def handleProgressBarSingal(self, i):
        # 处理进度条信号的方法，接受参数 i

        self.pb.setValue(i)
        # 设置进度条值为 i

        avg_time = (time.time() - self.time_start) / i
        # 计算平均时间

        time_left = str(datetime.timedelta(seconds=avg_time * (self.lender - i))).split(".")[0]
        # 计算剩余时间

        self.setWindowTitle("PPOCRLabel  --  " + f"Time Left: {time_left}")
        # 设置窗口标题显示剩余时间
    # 处理列表控件信号，向列表控件中添加项，并滚动到最后一项
    def handleListWidgetSingal(self, i):
        self.listWidget.addItem(i)
        titem = self.listWidget.item(self.listWidget.count() - 1)
        self.listWidget.scrollToItem(titem)

    # 处理结束信号，根据条件设置按钮的可用性
    def handleEndsignalSignal(self, i, str):
        if i == 0 and str == "readAll":
            self.buttonBox.button(BB.Ok).setEnabled(True)
            self.buttonBox.button(BB.Cancel).setEnabled(False)

    # 拒绝操作，停止线程并接受关闭事件
    def reject(self):
        print("reject")
        self.thread_1.handle = -1
        self.thread_1.quit()
        while not self.thread_1.isFinished():
            pass
        self.accept()

    # 验证操作，接受关闭事件
    def validate(self):
        self.accept()

    # 后处理操作，尝试修剪文本内容并打印
    def postProcess(self):
        try:
            self.edit.setText(self.edit.text().trimmed())
        except AttributeError:
            # 处理修剪文本时的异常情况
            self.edit.setText(self.edit.text())
            print(self.edit.text())

    # 弹出对话框，启动线程并执行对话框
    def popUp(self):
        self.thread_1.start()
        return 1 if self.exec_() else None

    # 处理关闭事件，拒绝操作
    def closeEvent(self, event):
        print("???")
        self.reject()
```