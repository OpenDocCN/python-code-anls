# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\utils\cal_recall\rrc_evaluation_funcs.py`

```py
#!/usr/bin/env python2
#encoding: UTF-8
# 导入所需的模块
import json
import sys
sys.path.append('./')
import zipfile
import re
import sys
import os
import codecs
import traceback
import numpy as np
from utils import order_points_clockwise

# 打印帮助信息
def print_help():
    sys.stdout.write(
        'Usage: python %s.py -g=<gtFile> -s=<submFile> [-o=<outputFolder> -p=<jsonParams>]'
        % sys.argv[0])
    sys.exit(2)

# 从 ZIP 文件中加载符合正则表达式的文件名作为键值的数组
def load_zip_file_keys(file, fileNameRegExp=''):
    """
    Returns an array with the entries of the ZIP file that match with the regular expression.
    The key's are the names or the file or the capturing group definied in the fileNameRegExp
    """
    try:
        # 加载 ZIP 文件
        archive = zipfile.ZipFile(file, mode='r', allowZip64=True)
    except:
        raise Exception('Error loading the ZIP archive.')

    pairs = []

    # 遍历 ZIP 文件中的文件名
    for name in archive.namelist():
        addFile = True
        keyName = name
        # 如果有指定文件名的正则表达式
        if fileNameRegExp != "":
            m = re.match(fileNameRegExp, name)
            if m == None:
                addFile = False
            else:
                if len(m.groups()) > 0:
                    keyName = m.group(1)

        if addFile:
            pairs.append(keyName)

    return pairs

# 从 ZIP 文件中加载符合正则表达式的文件内容作为值的数组
def load_zip_file(file, fileNameRegExp='', allEntries=False):
    """
    Returns an array with the contents (filtered by fileNameRegExp) of a ZIP file.
    The key's are the names or the file or the capturing group definied in the fileNameRegExp
    allEntries validates that all entries in the ZIP file pass the fileNameRegExp
    """
    try:
        # 加载 ZIP 文件
        archive = zipfile.ZipFile(file, mode='r', allowZip64=True)
    except:
        raise Exception('Error loading the ZIP archive')

    pairs = []
    # 遍历 ZIP 文件中的所有文件名
    for name in archive.namelist():
        # 默认将文件添加到结果中
        addFile = True
        # 将文件名作为键名
        keyName = name
        # 如果存在文件名正则表达式
        if fileNameRegExp != "":
            # 使用正则表达式匹配文件名
            m = re.match(fileNameRegExp, name)
            # 如果匹配结果为空
            if m == None:
                # 不添加该文件到结果中
                addFile = False
            else:
                # 如果匹配结果有分组
                if len(m.groups()) > 0:
                    # 使用第一个分组作为键名
                    keyName = m.group(1)

        # 如果需要添加文件到结果中
        if addFile:
            # 将文件名和文件数据添加到结果列表中
            pairs.append([keyName, archive.read(name)])
        else:
            # 如果需要返回所有条目
            if allEntries:
                # 抛出异常，表示 ZIP 条目无效
                raise Exception('ZIP entry not valid: %s' % name)

    # 将结果列表转换为字典并返回
    return dict(pairs)
# 加载文件夹中的文件内容，根据文件名正则表达式进行过滤
def load_folder_file(file, fileNameRegExp='', allEntries=False):
    # 存储文件名和内容的键值对列表
    pairs = []
    # 遍历文件夹中的文件
    for name in os.listdir(file):
        # 默认添加文件
        addFile = True
        keyName = name
        # 如果有文件名正则表达式
        if fileNameRegExp != "":
            # 使用正则表达式匹配文件名
            m = re.match(fileNameRegExp, name)
            # 如果匹配失败，则不添加文件
            if m == None:
                addFile = False
            else:
                # 如果有捕获组
                if len(m.groups()) > 0:
                    keyName = m.group(1)

        # 如果需要添加文件
        if addFile:
            # 将文件名和内容添加到键值对列表中
            pairs.append([keyName, open(os.path.join(file, name)).read()])
        else:
            # 如果需要验证所有文件
            if allEntries:
                raise Exception('ZIP entry not valid: %s' % name)

    # 将键值对列表转换为字典并返回
    return dict(pairs)


# 解码 UTF-8 编码的内容
def decode_utf8(raw):
    # 尝试解码为 UTF-8 编码的内容，失败则返回 None
    try:
        raw = codecs.decode(raw, 'utf-8', 'replace')
        # 提取存在的 BOM（字节顺序标记）
        raw = raw.encode('utf8')
        if raw.startswith(codecs.BOM_UTF8):
            raw = raw.replace(codecs.BOM_UTF8, '', 1)
        return raw.decode('utf-8')
    except:
        return None


# 验证文件中的每一行内容
def validate_lines_in_file(fileName,
                           file_contents,
                           CRLF=True,
                           LTRB=True,
                           withTranscription=False,
                           withConfidence=False,
                           imWidth=0,
                           imHeight=0):
    # 解码文件内容为 UTF-8 编码
    utf8File = decode_utf8(file_contents)
    # 如果解码失败，则抛出异常
    if (utf8File is None):
        raise Exception("The file %s is not UTF-8" % fileName)

    # 根据换行符拆分文件内容为行列表
    lines = utf8File.split("\r\n" if CRLF else "\n")
    # 遍历输入的行列表
    for line in lines:
        # 去除每行中的换行符
        line = line.replace("\r", "").replace("\n", "")
        # 如果行不为空
        if (line != ""):
            # 尝试验证该行是否有效，传入参数包括左上右下坐标、是否包含转录、是否包含置信度、图像宽度、图像高度
            try:
                validate_tl_line(line, LTRB, withTranscription, withConfidence,
                                 imWidth, imHeight)
            # 如果验证出现异常
            except Exception as e:
                # 抛出异常，包含文件名、行内容、异常信息
                raise Exception(
                    ("Line in sample not valid. Sample: %s Line: %s Error: %s" %
                     (fileName, line, str(e))).encode('utf-8', 'replace'))
# 验证文本行的格式是否正确，如果不正确则会引发异常
# 如果指定了最大宽度和最大高度，则所有点必须在图像边界内
# 可能的值有：
# LTRB=True: xmin,ymin,xmax,ymax[,confidence][,transcription] 
# LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription] 
def validate_tl_line(line,
                     LTRB=True,
                     withTranscription=True,
                     withConfidence=True,
                     imWidth=0,
                     imHeight=0):
    # 调用函数以获取文本行的值
    get_tl_line_values(line, LTRB, withTranscription, withConfidence, imWidth,
                       imHeight)


# 从文本行获取值
# 验证文本行的格式是否正确，如果不正确则会引发异常
# 如果指定了最大宽度和最大高度，则所有点必须在图像边界内
# 可能的值有：
# LTRB=True: xmin,ymin,xmax,ymax[,confidence][,transcription] 
# LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription] 
# 返回文本行的值。点，[置信度]，[转录]
def get_tl_line_values(line,
                       LTRB=True,
                       withTranscription=False,
                       withConfidence=False,
                       imWidth=0,
                       imHeight=0):
    # 置信度初始化为0.0
    confidence = 0.0
    # 转录初始化为空字符串
    transcription = ""
    # 点的列表初始化为空
    points = []

    # 点的数量为4
    numPoints = 4

    # 如果需要置信度
    if withConfidence:
        try:
            # 尝试将第numPoints + 1个匹配项转换为浮点数
            confidence = float(m.group(numPoints + 1))
        except ValueError:
            # 如果转换失败，则引发异常
            raise Exception("Confidence value must be a float")
    # 如果需要包含转录信息
    if withTranscription:
        # 计算转录信息在匹配对象中的位置
        posTranscription = numPoints + (2 if withConfidence else 1)
        # 获取转录信息
        transcription = m.group(posTranscription)
        # 匹配转录信息中的双引号
        m2 = re.match(r'^\s*\"(.*)\"\s*$', transcription)
        # 如果匹配成功
        if m2 != None:
            # 提取双引号内的值，并替换转义字符
            transcription = m2.group(1).replace("\\\\", "\\").replace("\\\"", "\"")
    
    # 返回点数、置信度和转录信息
    return points, confidence, transcription
# 验证点是否在给定边界内
def validate_point_inside_bounds(x, y, imWidth, imHeight):
    # 如果 x 值小于 0 或大于图像宽度，则抛出异常
    if (x < 0 or x > imWidth):
        raise Exception("X value (%s) not valid. Image dimensions: (%s,%s)" %
                        (xmin, imWidth, imHeight))
    # 如果 y 值小于 0 或大于图像高度，则抛出异常
    if (y < 0 or y > imHeight):
        raise Exception(
            "Y value (%s)  not valid. Image dimensions: (%s,%s) Sample: %s Line:%s"
            % (ymin, imWidth, imHeight))


# 验证四个点是否按顺时针顺序排列
def validate_clockwise_points(points):
    """
    验证四个点组成的多边形是否按顺时针顺序排列。
    """

    # 如果点的数量不等于 8，则抛出异常
    if len(points) != 8:
        raise Exception("Points list not valid." + str(len(points)))

    # 将点坐标转换为二维数组
    point = [[int(points[0]), int(points[1])],
             [int(points[2]), int(points[3])],
             [int(points[4]), int(points[5])],
             [int(points[6]), int(points[7])]]
    
    # 计算每条边的乘积
    edge = [(point[1][0] - point[0][0]) * (point[1][1] + point[0][1]),
            (point[2][0] - point[1][0]) * (point[2][1] + point[1][1]),
            (point[3][0] - point[2][0]) * (point[3][1] + point[2][1]),
            (point[0][0] - point[3][0]) * (point[0][1] + point[3][1])]

    # 计算边乘积的总和
    summatory = edge[0] + edge[1] + edge[2] + edge[3]
    
    # 如果总和大于 0，则抛出异常，说明点不是按顺时针顺序排列
    if summatory > 0:
        raise Exception(
            "Points are not clockwise. The coordinates of bounding quadrilaterals have to be given in clockwise order. Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is the standard one, with the image origin at the upper left, the X axis extending to the right and Y axis extending downwards."
        )
# 从文件内容中获取文本行的数值信息，包括坐标、置信度和文本内容等
def get_tl_line_values_from_file_contents(content,
                                          CRLF=True,
                                          LTRB=True,
                                          withTranscription=False,
                                          withConfidence=False,
                                          imWidth=0,
                                          imHeight=0,
                                          sort_by_confidences=True):
    """
    Returns all points, confindences and transcriptions of a file in lists. Valid line formats:
    xmin,ymin,xmax,ymax,[confidence],[transcription]
    x1,y1,x2,y2,x3,y3,x4,y4,[confidence],[transcription]
    """
    # 初始化存储坐标、文本内容和置信度的列表
    pointsList = []
    transcriptionsList = []
    confidencesList = []

    # 将内容按照指定的换行符分割成行
    lines = content.split("\r\n" if CRLF else "\n")
    for line in lines:
        # 去除行中的换行符
        line = line.replace("\r", "").replace("\n", "")
        if (line != ""):
            # 获取文本行的数值信息，包括坐标、置信度和文本内容
            points, confidence, transcription = get_tl_line_values(
                line, LTRB, withTranscription, withConfidence, imWidth,
                imHeight)
            pointsList.append(points)
            transcriptionsList.append(transcription)
            confidencesList.append(confidence)

    # 如果需要包含置信度信息，并且置信度列表不为空且需要按照置信度排序
    if withConfidence and len(confidencesList) > 0 and sort_by_confidences:
        import numpy as np
        # 根据置信度对列表进行排序
        sorted_ind = np.argsort(-np.array(confidencesList))
        confidencesList = [confidencesList[i] for i in sorted_ind]
        pointsList = [pointsList[i] for i in sorted_ind]
        transcriptionsList = [transcriptionsList[i] for i in sorted_ind]

    # 返回坐标、置信度和文本内容的列表
    return pointsList, confidencesList, transcriptionsList


# 主要评估函数，用于评估数据并展示结果
def main_evaluation(p,
                    default_evaluation_params_fn,
                    validate_data_fn,
                    evaluate_method_fn,
                    show_result=True,
                    per_sample=True):
    """
    """
    This process validates a method, evaluates it and if it succed generates a ZIP file with a JSON entry for each sample.
    Params:
    p: Dictionary of parmeters with the GT/submission locations. If None is passed, the parameters send by the system are used.
    default_evaluation_params_fn: points to a function that returns a dictionary with the default parameters used for the evaluation
    validate_data_fn: points to a method that validates the corrct format of the submission
    evaluate_method_fn: points to a function that evaluated the submission and return a Dictionary with the results
    """
    # 获取默认的评估参数
    evalParams = default_evaluation_params_fn()
    # 更新评估参数
    if 'p' in p.keys():
        evalParams.update(p['p'] if isinstance(p['p'], dict) else json.loads(p['p'][1:-1]))

    # 初始化结果字典
    resDict = {
        'calculated': True,
        'Message': '',
        'method': '{}',
        'per_sample': '{}'
    }
    try:
        # 调用评估方法
        evalData = evaluate_method_fn(p['g'], p['s'], evalParams)
        # 更新结果字典
        resDict.update(evalData)

    except Exception as e:
        # 打印异常信息
        traceback.print_exc()
        resDict['Message'] = str(e)
        resDict['calculated'] = False

    # 如果存在输出路径
    if 'o' in p:
        # 如果输出路径不存在，则创建
        if not os.path.exists(p['o']):
            os.makedirs(p['o'])

        # 设置结果输出文件名
        resultsOutputname = p['o'] + '/results.zip'
        # 创建 ZIP 文件对象
        outZip = zipfile.ZipFile(resultsOutputname, mode='w', allowZip64=True)

        # 删除不需要的键
        del resDict['per_sample']
        if 'output_items' in resDict.keys():
            del resDict['output_items']

        # 将结果字典以 JSON 格式写入 ZIP 文件
        outZip.writestr('method.json', json.dumps(resDict))

    # 如果计算失败
    if not resDict['calculated']:
        # 如果需要显示结果，则输出错误信息
        if show_result:
            sys.stderr.write('Error!\n' + resDict['Message'] + '\n\n')
        # 如果存在输出路径，则关闭 ZIP 文件
        if 'o' in p:
            outZip.close()
        # 返回结果字典
        return resDict
    # 如果参数中包含字母'o'
    if 'o' in p:
        # 如果 per_sample 参数为 True
        if per_sample == True:
            # 遍历 evalData 字典中 per_sample 键对应的值，将每个键值对以 JSON 格式写入输出 ZIP 文件
            for k, v in evalData['per_sample'].iteritems():
                outZip.writestr(k + '.json', json.dumps(v))

            # 如果 evalData 字典中包含键'output_items'
            if 'output_items' in evalData.keys():
                # 遍历 evalData 字典中 output_items 键对应的值，将每个键值对写入输出 ZIP 文件
                for k, v in evalData['output_items'].iteritems():
                    outZip.writestr(k, v)

        # 关闭输出 ZIP 文件
        outZip.close()

    # 如果 show_result 为 True
    if show_result:
        # 在标准输出中打印"Calculated!"
        sys.stdout.write("Calculated!")
        # 在标准输出中打印 resDict 字典中'method'键对应的值的 JSON 格式
        sys.stdout.write(json.dumps(resDict['method']))

    # 返回结果字典
    return resDict
# 主要的验证函数，用于验证一个方法
def main_validation(default_evaluation_params_fn, validate_data_fn):
    """
    This process validates a method
    Params:
    default_evaluation_params_fn: points to a function that returns a dictionary with the default parameters used for the evaluation
    validate_data_fn: points to a method that validates the corrct format of the submission
    """
    # 尝试执行以下代码块，捕获可能出现的异常
    try:
        # 从命令行参数中解析参数并创建字典
        p = dict([s[1:].split('=') for s in sys.argv[1:]])
        # 调用默认评估参数函数，获取默认参数字典
        evalParams = default_evaluation_params_fn()
        # 如果参数字典中包含 'p' 键
        if 'p' in p.keys():
            # 更新评估参数字典，根据 'p' 键对应的值来更新
            evalParams.update(p['p'] if isinstance(p['p'], dict) else
                              json.loads(p['p'][1:-1]))

        # 调用验证数据函数，验证提交的数据格式
        validate_data_fn(p['g'], p['s'], evalParams)
        # 打印成功信息
        print('SUCCESS')
        # 退出程序，返回状态码 0
        sys.exit(0)
    # 捕获异常并打印异常信息
    except Exception as e:
        print(str(e))
        # 退出程序，返回状态码 101
        sys.exit(101)
```