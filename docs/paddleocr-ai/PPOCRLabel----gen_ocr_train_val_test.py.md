# `.\PaddleOCR\PPOCRLabel\gen_ocr_train_val_test.py`

```
# 设置文件编码为 UTF-8
# 导入操作系统、文件操作、文件夹操作、随机数生成、命令行参数解析模块
import os
import shutil
import random
import argparse

# 创建或删除指定路径下的文件夹
def isCreateOrDeleteFolder(path, flag):
    # 拼接文件夹路径
    flagPath = os.path.join(path, flag)
    
    # 如果文件夹存在，则删除
    if os.path.exists(flagPath):
        shutil.rmtree(flagPath)
    
    # 创建新的空文件夹
    os.makedirs(flagPath)
    
    # 获取文件夹的绝对路径
    flagAbsPath = os.path.abspath(flagPath)
    
    return flagAbsPath

# 划分训练集、验证集、测试集
def splitTrainVal(root, absTrainRootPath, absValRootPath, absTestRootPath, trainTxt, valTxt, testTxt, flag):
    # 获取数据集的绝对路径
    dataAbsPath = os.path.abspath(root)
    
    # 根据不同的标志选择标签文件路径
    if flag == "det":
        labelFilePath = os.path.join(dataAbsPath, args.detLabelFileName)
    elif flag == "rec":
        labelFilePath = os.path.join(dataAbsPath, args.recLabelFileName)
    
    # 读取标签文件内容
    labelFileRead = open(labelFilePath, "r", encoding="UTF-8")
    labelFileContent = labelFileRead.readlines()
    
    # 随机打乱标签文件内容
    random.shuffle(labelFileContent)
    
    # 获取标签文件内容的长度
    labelRecordLen = len(labelFileContent)
    # 遍历标签文件内容，获取索引和每条记录信息
    for index, labelRecordInfo in enumerate(labelFileContent):
        # 根据制表符分割记录信息，获取图片相对路径和标签
        imageRelativePath = labelRecordInfo.split('\t')[0]
        imageLabel = labelRecordInfo.split('\t')[1]
        # 获取图片名称
        imageName = os.path.basename(imageRelativePath)

        # 根据标志位选择不同的路径拼接方式
        if flag == "det":
            imagePath = os.path.join(dataAbsPath, imageName)
        elif flag == "rec":
            imagePath = os.path.join(dataAbsPath, "{}\\{}".format(args.recImageDirName, imageName))

        # 按预设的比例划分训练集、验证集、测试集
        trainValTestRatio = args.trainValTestRatio.split(":")
        trainRatio = eval(trainValTestRatio[0]) / 10
        valRatio = trainRatio + eval(trainValTestRatio[1]) / 10
        curRatio = index / labelRecordLen

        # 根据比例将图片复制到对应的训练集、验证集、测试集目录，并写入对应的文本文件
        if curRatio < trainRatio:
            imageCopyPath = os.path.join(absTrainRootPath, imageName)
            shutil.copy(imagePath, imageCopyPath)
            trainTxt.write("{}\t{}".format(imageCopyPath, imageLabel))
        elif curRatio >= trainRatio and curRatio < valRatio:
            imageCopyPath = os.path.join(absValRootPath, imageName)
            shutil.copy(imagePath, imageCopyPath)
            valTxt.write("{}\t{}".format(imageCopyPath, imageLabel))
        else:
            imageCopyPath = os.path.join(absTestRootPath, imageName)
            shutil.copy(imagePath, imageCopyPath)
            testTxt.write("{}\t{}".format(imageCopyPath, imageLabel))
# 删除指定路径下的文件，如果文件存在的话
def removeFile(path):
    # 检查路径是否存在
    if os.path.exists(path):
        # 删除文件
        os.remove(path)

# 生成检测和识别的训练集和验证集
def genDetRecTrainVal(args):
    # 创建或删除指定文件夹，并返回绝对路径
    detAbsTrainRootPath = isCreateOrDeleteFolder(args.detRootPath, "train")
    detAbsValRootPath = isCreateOrDeleteFolder(args.detRootPath, "val")
    detAbsTestRootPath = isCreateOrDeleteFolder(args.detRootPath, "test")
    recAbsTrainRootPath = isCreateOrDeleteFolder(args.recRootPath, "train")
    recAbsValRootPath = isCreateOrDeleteFolder(args.recRootPath, "val")
    recAbsTestRootPath = isCreateOrDeleteFolder(args.recRootPath, "test")

    # 删除指定路径下的文件
    removeFile(os.path.join(args.detRootPath, "train.txt"))
    removeFile(os.path.join(args.detRootPath, "val.txt"))
    removeFile(os.path.join(args.detRootPath, "test.txt"))
    removeFile(os.path.join(args.recRootPath, "train.txt"))
    removeFile(os.path.join(args.recRootPath, "val.txt"))
    removeFile(os.path.join(args.recRootPath, "test.txt"))

    # 打开指定路径下的文件，以追加模式写入，指定编码为UTF-8
    detTrainTxt = open(os.path.join(args.detRootPath, "train.txt"), "a", encoding="UTF-8")
    detValTxt = open(os.path.join(args.detRootPath, "val.txt"), "a", encoding="UTF-8")
    detTestTxt = open(os.path.join(args.detRootPath, "test.txt"), "a", encoding="UTF-8")
    recTrainTxt = open(os.path.join(args.recRootPath, "train.txt"), "a", encoding="UTF-8")
    recValTxt = open(os.path.join(args.recRootPath, "val.txt"), "a", encoding="UTF-8")
    recTestTxt = open(os.path.join(args.recRootPath, "test.txt"), "a", encoding="UTF-8")

    # 分割训练集和验证集
    splitTrainVal(args.datasetRootPath, detAbsTrainRootPath, detAbsValRootPath, detAbsTestRootPath, detTrainTxt, detValTxt,
                  detTestTxt, "det")

    # 遍历数据集根目录下的文件和文件夹
    for root, dirs, files in os.walk(args.datasetRootPath):
        # 遍历文件夹
        for dir in dirs:
            # 如果文件夹名为'crop_img'，则进行分割训练集和验证集
            if dir == 'crop_img':
                splitTrainVal(root, recAbsTrainRootPath, recAbsValRootPath, recAbsTestRootPath, recTrainTxt, recValTxt,
                              recTestTxt, "rec")
            else:
                continue
        break

if __name__ == "__main__":
    # 定义命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加训练集、验证集、测试集的比例参数
    parser.add_argument(
        "--trainValTestRatio",
        type=str,
        default="6:2:2",
        help="ratio of trainset:valset:testset")
    # 添加数据集根路径参数
    parser.add_argument(
        "--datasetRootPath",
        type=str,
        default="../train_data/",
        help="path to the dataset marked by ppocrlabel, E.g, dataset folder named 1,2,3..."
    )
    # 添加划分检测数据集路径参数
    parser.add_argument(
        "--detRootPath",
        type=str,
        default="../train_data/det",
        help="the path where the divided detection dataset is placed")
    # 添加划分识别数据集路径参数
    parser.add_argument(
        "--recRootPath",
        type=str,
        default="../train_data/rec",
        help="the path where the divided recognition dataset is placed"
    )
    # 添加检测标注文件名参数
    parser.add_argument(
        "--detLabelFileName",
        type=str,
        default="Label.txt",
        help="the name of the detection annotation file")
    # 添加识别标注文件名参数
    parser.add_argument(
        "--recLabelFileName",
        type=str,
        default="rec_gt.txt",
        help="the name of the recognition annotation file"
    )
    # 添加识别图像文件夹名称参数
    parser.add_argument(
        "--recImageDirName",
        type=str,
        default="crop_img",
        help="the name of the folder where the cropped recognition dataset is located"
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数生成划分好的检测和识别训练集、验证集、测试集
    genDetRecTrainVal(args)
```