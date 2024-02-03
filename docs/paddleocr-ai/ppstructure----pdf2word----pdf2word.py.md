# `.\PaddleOCR\ppstructure\pdf2word\pdf2word.py`

```
# 版权声明
# 版权所有 (c) 2022 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言管理权限和限制。
#
# 导入所需的库
import sys
import tarfile
import os
import time
import datetime
import functools
import cv2
import platform
import numpy as np
import fitz
from PIL import Image
from pdf2docx.converter import Converter
from qtpy.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar, \
                           QGridLayout, QMessageBox, QLabel, QFileDialog, QCheckBox
from qtpy.QtCore import Signal, QThread, QObject
from qtpy.QtGui import QImage, QPixmap, QIcon

# 获取当前文件的路径
file = os.path.dirname(os.path.abspath(__file__))
# 获取根目录的路径
root = os.path.abspath(os.path.join(file, '../../'))
# 将当前文件路径添加到系统路径中
sys.path.append(file)
# 将根目录路径插入到系统路径的最前面
sys.path.insert(0, root)

# 导入自定义模块
from ppstructure.predict_system import StructureSystem, save_structure_res
from ppstructure.utility import parse_args, draw_structure_result
from ppocr.utils.network import download_with_progressbar
from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx
# from ScreenShotWidget import ScreenShotWidget

# 定义应用名称和版本号
__APPNAME__ = "pdf2word"
__VERSION__ = "0.2.2"

# 英文 URL 链接
URLs_EN = {
    # 下载超英文轻量级PP-OCRv3模型的检测模型并解压
    "en_PP-OCRv3_det_infer":
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
    # 下载英文轻量级PP-OCRv3模型的识别模型并解压
    "en_PP-OCRv3_rec_infer":
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar",
    # 下载超轻量级英文表格英文模型并解压
    "en_ppstructure_mobile_v2.0_SLANet_infer":
    # 英文版面分析模型的下载链接
    "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar",
    # picodet_lcnet_x1_0_fgd_layout_infer模型的下载链接
    "picodet_lcnet_x1_0_fgd_layout_infer":
    "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar",
# 英文字典，包含"rec_char_dict_path"和"layout_dict_path"两个键值对
DICT_EN = {
    "rec_char_dict_path": "en_dict.txt",
    "layout_dict_path": "layout_publaynet_dict.txt",
}

# 中文 URL 字典，包含下载链接和对应的描述信息
URLs_CN = {
    "cn_PP-OCRv3_det_infer": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar",
    "cn_PP-OCRv3_rec_infer": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar",
    "cn_ppstructure_mobile_v2.0_SLANet_infer": "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar",
    "picodet_lcnet_x1_0_fgd_layout_cdla_infer": "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar",
}

# 中文字典，包含"rec_char_dict_path"和"layout_dict_path"两个键值对
DICT_CN = {
    "rec_char_dict_path": "ppocr_keys_v1.txt",
    "layout_dict_path": "layout_cdla_dict.txt",
}

# 将 QImage 转换为 opencv MAT 格式的函数
def QImageToCvMat(incomingImage) -> np.array:
    '''  
    Converts a QImage into an opencv MAT format  
    '''

    # 将 QImage 转换为 RGBA8888 格式
    incomingImage = incomingImage.convertToFormat(QImage.Format.Format_RGBA8888)

    # 获取图片的宽度和高度
    width = incomingImage.width()
    height = incomingImage.height()

    # 获取图片数据的指针
    ptr = incomingImage.bits()
    ptr.setsize(height * width * 4)
    # 将指针数据转换为 numpy 数组
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    return arr

# 读取图片的函数
def readImage(image_file) -> list:
    # 检查文件名是否以 'pdf' 结尾，如果是则处理 PDF 文件
    if os.path.basename(image_file)[-3:] == 'pdf':
        # 初始化空列表用于存储图片
        imgs = []
        # 打开 PDF 文件
        with fitz.open(image_file) as pdf:
            # 遍历 PDF 文件的每一页
            for pg in range(0, pdf.pageCount):
                # 获取当前页对象
                page = pdf[pg]
                # 创建一个矩阵对象，用于控制图片大小
                mat = fitz.Matrix(2, 2)
                # 获取当前页的像素图像对象
                pm = page.getPixmap(matrix=mat, alpha=False)

                # 如果图片宽度或高度大于 2000 像素，则不放大图片
                if pm.width > 2000 or pm.height > 2000:
                    # 重新获取像素图像对象，不放大
                    pm = page.getPixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                # 将像素图像对象转换为 PIL 图像对象
                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                # 将 PIL 图像对象转换为 OpenCV 图像对象
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # 将处理后的图片添加到列表中
                imgs.append(img)
    else:
        # 如果不是 PDF 文件，则直接读取图片文件
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        # 如果成功读取到图片，则将其放入列表中
        if img is not None:
            imgs = [img]

    # 返回处理后的图片列表
    return imgs
# 定义一个名为 Worker 的类，继承自 QThread 类
class Worker(QThread):
    # 定义信号 progressBarValue，用于传递进度条的数值
    progressBarValue = Signal(int)
    # 定义信号 progressBarRange，用于传递进度条的范围
    progressBarRange = Signal(int)
    # 定义信号 endsignal，用于传递结束信号
    endsignal = Signal()
    # 定义信号 exceptedsignal，用于传递异常信息
    exceptedsignal = Signal(str)  #发送一个异常信号
    # 初始化 loopFlag 为 True
    loopFlag = True

    # 初始化函数，接受 predictors、save_pdf、vis_font_path、use_pdf2docx_api 四个参数
    def __init__(self, predictors, save_pdf, vis_font_path, use_pdf2docx_api):
        # 调用父类的初始化函数
        super(Worker, self).__init__()
        # 初始化实例变量 predictors、save_pdf、vis_font_path、use_pdf2docx_api
        self.predictors = predictors
        self.save_pdf = save_pdf
        self.vis_font_path = vis_font_path
        self.lang = 'EN'
        self.imagePaths = []
        self.use_pdf2docx_api = use_pdf2docx_api
        self.outputDir = None
        self.totalPageCnt = 0
        self.pageCnt = 0
        # 设置线程栈大小为 1024 * 1024
        self.setStackSize(1024 * 1024)

    # 设置图片路径的方法
    def setImagePath(self, imagePaths):
        self.imagePaths = imagePaths

    # 设置语言的方法
    def setLang(self, lang):
        self.lang = lang

    # 设置输出目录的方法
    def setOutputDir(self, outputDir):
        self.outputDir = outputDir

    # 设置是否使用 PDF 解析器的方法
    def setPDFParser(self, enabled):
        self.use_pdf2docx_api = enabled

    # 重置页面计数的方法
    def resetPageCnt(self):
        self.pageCnt = 0

    # 重置总页面计数的方法
    def resetTotalPageCnt(self):
        self.totalPageCnt = 0
    # 定义一个方法，用于对图片进行 OCR 预测
    def ppocrPrecitor(self, imgs, img_name):
        # 存储所有结果
        all_res = []
        # 更新进度条范围
        self.totalPageCnt += len(imgs)
        self.progressBarRange.emit(self.totalPageCnt)
        
        # 处理每一页图片
        for index, img in enumerate(imgs):
            # 使用对应语言的 OCR 预测器进行预测
            res, time_dict = self.predictors[self.lang](img)

            # 保存输出结果
            save_structure_res(res, self.outputDir, img_name)
            # 绘制结果图片
            # draw_img = draw_structure_result(img, res, self.vis_font_path)
            # img_save_path = os.path.join(self.outputDir, img_name, 'show_{}.jpg'.format(index))
            # if res != []:
            #     cv2.imwrite(img_save_path, draw_img)

            # 恢复结果布局
            h, w, _ = img.shape
            res = sorted_layout_boxes(res, w)
            all_res += res
            self.pageCnt += 1
            self.progressBarValue.emit(self.pageCnt)

        # 如果有结果，则尝试将结果转换为 docx 格式
        if all_res != []:
            try:
                convert_info_docx(imgs, all_res, self.outputDir, img_name)
            except Exception as ex:
                print("error in layout recovery image:{}, err msg: {}".format(
                    img_name, ex))
        # 打印预测时间
        print("Predict time : {:.3f}s".format(time_dict['all']))
        print('result save to {}'.format(self.outputDir))
    # 定义一个方法用于执行任务
    def run(self):
        # 重置页面计数
        self.resetPageCnt()
        # 重置总页面计数
        self.resetTotalPageCnt()
        try:
            # 创建输出目录，如果目录已存在则不抛出异常
            os.makedirs(self.outputDir, exist_ok=True)
            # 遍历图片路径列表
            for i, image_file in enumerate(self.imagePaths):
                # 如果不需要循环处理，则跳出循环
                if not self.loopFlag:
                    break
                # 使用 use_pdf2docx_api 进行 PDF 解析
                if self.use_pdf2docx_api \
                    and os.path.basename(image_file)[-3:] == 'pdf':
                    # 增加总页面计数
                    self.totalPageCnt += 1
                    # 发送总页面计数信号
                    self.progressBarRange.emit(self.totalPageCnt)
                    print(
                        '===============using use_pdf2docx_api===============')
                    # 获取图片文件名
                    img_name = os.path.basename(image_file).split('.')[0]
                    # 设置 docx 文件路径
                    docx_file = os.path.join(self.outputDir,
                                             '{}.docx'.format(img_name))
                    # 创建 Converter 对象
                    cv = Converter(image_file)
                    # 转换为 docx 文件
                    cv.convert(docx_file)
                    # 关闭 Converter 对象
                    cv.close()
                    print('docx save to {}'.format(docx_file))
                    # 增加页面计数
                    self.pageCnt += 1
                    # 发送页面计数信号
                    self.progressBarValue.emit(self.pageCnt)
                else:
                    # 使用 PPOCR 进行 PDF/Image 解析
                    imgs = readImage(image_file)
                    # 如果图片列表为空，则继续下一轮循环
                    if len(imgs) == 0:
                        continue
                    # 获取图片文件名
                    img_name = os.path.basename(image_file).split('.')[0]
                    # 创建图片输出目录
                    os.makedirs(
                        os.path.join(self.outputDir, img_name), exist_ok=True)
                    # 使用 ppocrPrecitor 处理图片
                    self.ppocrPrecitor(imgs, img_name)
                # 文件处理完成
            # 发送任务结束信号
            self.endsignal.emit()
            # 捕获异常并发送给 UI 进程
        except Exception as e:
            self.exceptedsignal.emit(str(e))  # 将异常发送给UI进程
class APP_Image2Doc(QWidget):
    # 定义一个名为APP_Image2Doc的类，继承自QWidget类
    def __init__(self):
        # 初始化函数，创建一个实例时会自动调用
        super().__init__()
        # 调用父类的初始化函数

        # settings
        self.imagePaths = []
        # 初始化一个空列表，用于存储图片路径
        self.screenShot = None
        # 初始化一个空的截图对象
        self.save_pdf = False
        # 初始化一个布尔值，表示是否保存为PDF
        self.output_dir = None
        # 初始化一个空的输出目录路径
        self.vis_font_path = os.path.join(root, "doc", "fonts", "simfang.ttf")
        # 设置字体路径
        self.use_pdf2docx_api = False
        # 初始化一个布尔值，表示是否使用pdf2docx API

        # ProgressBar
        self.pb = QProgressBar()
        # 创建一个进度条对象
        self.pb.setRange(0, 100)
        # 设置进度条范围为0到100
        self.pb.setValue(0)
        # 设置进度条初始值为0

        # 初始化界面
        self.setupUi()
        # 调用setupUi函数初始化界面

        # 下载模型
        self.downloadModels(URLs_EN)
        # 下载英文模型
        self.downloadModels(URLs_CN)
        # 下载中文模型

        # 初始化模型
        predictors = {
            'EN': self.initPredictor('EN'),
            'CN': self.initPredictor('CN'),
        }
        # 初始化预测器对象，包括英文和中文

        # 设置工作进程
        self._thread = Worker(predictors, self.save_pdf, self.vis_font_path,
                              self.use_pdf2docx_api)
        # 创建一个工作线程对象
        self._thread.progressBarValue.connect(
            self.handleProgressBarUpdateSingal)
        # 连接进度条数值信号和处理函数
        self._thread.endsignal.connect(self.handleEndsignalSignal)
        # 连接结束信号和处理函数
        self._thread.progressBarRange.connect(self.handleProgressBarRangeSingal)
        # 连接进度条范围信号和处理函数
        self._thread.exceptedsignal.connect(self.handleThreadException)
        # 连接异常信号和处理函数
        self.time_start = 0  # save start time
        # 初始化一个时间变量，用于记录开始时间
    # 设置用户界面
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.setWindowTitle(__APPNAME__ + " " + __VERSION__)

        # 创建一个网格布局
        layout = QGridLayout()

        # 创建一个打开文件按钮，并设置图标
        self.openFileButton = QPushButton("打开文件")
        self.openFileButton.setIcon(QIcon(QPixmap("./icons/folder-plus.png")))
        layout.addWidget(self.openFileButton, 0, 0, 1, 1)
        self.openFileButton.clicked.connect(self.handleOpenFileSignal)

        # 创建一个中文转换按钮，并设置图标
        self.startCNButton = QPushButton("中文转换")
        self.startCNButton.setIcon(QIcon(QPixmap("./icons/chinese.png")))
        layout.addWidget(self.startCNButton, 0, 1, 1, 1)
        self.startCNButton.clicked.connect(
            functools.partial(self.handleStartSignal, 'CN', False))

        # 创建一个英文转换按钮，并设置图标
        self.startENButton = QPushButton("英文转换")
        self.startENButton.setIcon(QIcon(QPixmap("./icons/english.png")))
        layout.addWidget(self.startENButton, 0, 2, 1, 1)
        self.startENButton.clicked.connect(
            functools.partial(self.handleStartSignal, 'EN', False))

        # 创建一个PDF解析按钮
        self.PDFParserButton = QPushButton('PDF解析', self)
        layout.addWidget(self.PDFParserButton, 0, 3, 1, 1)
        self.PDFParserButton.clicked.connect(
            functools.partial(self.handleStartSignal, 'CN', True))

        # 创建一个显示结果按钮，并设置图标
        self.showResultButton = QPushButton("显示结果")
        self.showResultButton.setIcon(QIcon(QPixmap("./icons/folder-open.png")))
        layout.addWidget(self.showResultButton, 0, 4, 1, 1)
        self.showResultButton.clicked.connect(self.handleShowResultSignal)

        # 将进度条添加到布局中
        layout.addWidget(self.pb, 2, 0, 1, 5)
        # 创建一个时间估计标签
        self.timeEstLabel = QLabel(("Time Left: --"))
        layout.addWidget(self.timeEstLabel, 3, 0, 1, 5)

        # 将布局应用到界面上
        self.setLayout(layout)
    # 下载模型文件并解压缩
    def downloadModels(self, URLs):
        # 定义需要解压缩的文件名列表
        tar_file_name_list = [
            'inference.pdiparams', 'inference.pdiparams.info',
            'inference.pdmodel', 'model.pdiparams', 'model.pdiparams.info',
            'model.pdmodel'
        ]
        # 设置模型文件存储路径
        model_path = os.path.join(root, 'inference')
        # 如果路径不存在则创建
        os.makedirs(model_path, exist_ok=True)

        # 遍历下载并解压缩模型
        for name in URLs.keys():
            url = URLs[name]
            print("Try downloading file: {}".format(url))
            tarname = url.split('/')[-1]
            tarpath = os.path.join(model_path, tarname)
            # 如果文件已存在则跳过下载
            if os.path.exists(tarpath):
                print("File have already exist. skip")
            else:
                try:
                    download_with_progressbar(url, tarpath)
                except Exception as e:
                    print(
                        "Error occurred when downloading file, error message:")
                    print(e)

            # 解压缩模型文件
            try:
                with tarfile.open(tarpath, 'r') as tarObj:
                    storage_dir = os.path.join(model_path, name)
                    os.makedirs(storage_dir, exist_ok=True)
                    for member in tarObj.getmembers():
                        filename = None
                        for tar_file_name in tar_file_name_list:
                            if tar_file_name in member.name:
                                filename = tar_file_name
                        if filename is None:
                            continue
                        file = tarObj.extractfile(member)
                        with open(os.path.join(storage_dir, filename),
                                  'wb') as f:
                            f.write(file.read())
            except Exception as e:
                print("Error occurred when unziping file, error message:")
                print(e)
    # 处理打开文件信号的方法
    def handleOpenFileSignal(self):
        '''
        可以多选图像文件
        '''
        # 弹出文件选择对话框，允许选择多个图片文件
        selectedFiles = QFileDialog.getOpenFileNames(
            self, "多文件选择", "/", "图片文件 (*.png *.jpeg *.jpg *.bmp *.pdf)")[0]
        # 如果选择了文件
        if len(selectedFiles) > 0:
            # 将选择的文件路径存储在imagePaths中
            self.imagePaths = selectedFiles
            # 丢弃截图的临时图像
            self.screenShot = None
            # 设置进度条的值为0
            self.pb.setValue(0)

    # def screenShotSlot(self):
    #     '''
    #     选定图像文件和截图的转换过程只能同时进行一个
    #     截图只能同时转换一个
    #     '''
    #     self.screenShotWg.start()
    #     if self.screenShotWg.captureImage:
    #         self.screenShot = self.screenShotWg.captureImage
    #         self.imagePaths.clear() # discard openfile temp list
    #         self.pb.setRange(0, 1)
    #         self.pb.setValue(0)
    # 处理开始信号，根据参数 lang 和 pdfParser 的值进行处理
    def handleStartSignal(self, lang='EN', pdfParser=False):
        # 如果存在截图
        if self.screenShot:  # for screenShot
            # 生成截图文件名
            img_name = 'screenshot_' + time.strftime("%Y%m%d%H%M%S",
                                                     time.localtime())
            # 将截图转换为图像对象
            image = QImageToCvMat(self.screenShot)
            # 预测并保存结果
            self.predictAndSave(image, img_name, lang)
            # 更新进度条
            self.pb.setValue(1)
            # 弹出信息框，提示文档提取完成
            QMessageBox.information(self, u'Information', "文档提取完成")
        # 如果存在图片路径
        elif len(self.imagePaths) > 0:  # for image file selection
            # 必须在开始之前设置图片路径列表和语言
            self.output_dir = os.path.join(
                os.path.dirname(self.imagePaths[0]),
                "output")  # output_dir shold be same as imagepath
            # 设置工作线程的输出目录
            self._thread.setOutputDir(self.output_dir)
            # 设置工作线程的图片路径
            self._thread.setImagePath(self.imagePaths)
            # 设置工作线程的语言
            self._thread.setLang(lang)
            # 设置工作线程的 PDF 解析器
            self._thread.setPDFParser(pdfParser)
            # 禁用按钮
            self.openFileButton.setEnabled(False)
            self.startCNButton.setEnabled(False)
            self.startENButton.setEnabled(False)
            self.PDFParserButton.setEnabled(False)
            # 启动工作线程
            self._thread.start()
            # 记录开始时间
            self.time_start = time.time()  # log start time
            # 弹出信息框，提示开始转换
            QMessageBox.information(self, u'Information', "开始转换")
        else:
            # 提示选择要识别的文件或截图
            QMessageBox.warning(self, u'Information', "请选择要识别的文件或截图")

    # 处理显示结果信号
    def handleShowResultSignal(self):
        # 如果输出目录为空，则返回
        if self.output_dir is None:
            return
        # 如果输出目录存在
        if os.path.exists(self.output_dir):
            # 如果是 Windows 系统，使用系统默认程序打开输出目录
            if platform.system() == 'Windows':
                os.startfile(self.output_dir)
            # 如果不是 Windows 系统，使用系统命令打开输出目录
            else:
                os.system('open ' + os.path.normpath(self.output_dir))
        else:
            # 提示输出文件不存在
            QMessageBox.information(self, u'Information', "输出文件不存在")
    # 处理进度条更新信号，根据传入的值设置进度条的当前值
    def handleProgressBarUpdateSingal(self, i):
        self.pb.setValue(i)
        # 计算识别剩余时间
        lenbar = self.pb.maximum()
        avg_time = (time.time() - self.time_start
                    ) / i  # 使用平均时间来防止时间波动
        time_left = str(datetime.timedelta(seconds=avg_time * (
            lenbar - i))).split(".")[0]  # 去除微秒部分
        self.timeEstLabel.setText(f"Time Left: {time_left}")  # 显示剩余时间

    # 处理进度条范围信号，根据传入的最大值设置进度条的范围
    def handleProgressBarRangeSingal(self, max):
        self.pb.setRange(0, max)

    # 处理结束信号，启用按钮，并显示转换结束的信息框
    def handleEndsignalSignal(self):
        # 启用按钮
        self.openFileButton.setEnabled(True)
        self.startCNButton.setEnabled(True)
        self.startENButton.setEnabled(True)
        self.PDFParserButton.setEnabled(True)
        # 显示转换结束的信息框
        QMessageBox.information(self, u'Information', "转换结束")

    # 处理复选框改变信号，根据复选框的状态设置 PDF 解析器
    def handleCBChangeSignal(self):
        self._thread.setPDFParser(self.checkBox.isChecked())

    # 处理线程异常，终止线程并显示错误信息框
    def handleThreadException(self, message):
        self._thread.quit()
        QMessageBox.information(self, 'Error', message)
# 主函数入口
def main():
    # 创建应用程序对象
    app = QApplication(sys.argv)

    # 创建图像转文档的窗口对象
    window = APP_Image2Doc()  # 创建对象
    # 显示窗口
    window.show()  # 全屏显示窗口

    # 处理应用程序事件
    QApplication.processEvents()
    # 执行应用程序，直到退出
    sys.exit(app.exec())


# 如果当前脚本作为主程序执行，则调用主函数
if __name__ == "__main__":
    main()
```