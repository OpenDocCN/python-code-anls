# `.\MinerU\tests\overall_indicator.py`

```
# 导入所需的库
import json  # 用于处理 JSON 数据
import pandas as pd  # 用于数据操作和分析
import numpy as np  # 用于科学计算
import re  # 用于正则表达式操作
from nltk.translate.bleu_score import sentence_bleu  # 导入 NLTK 的 BLEU 分数计算
import time  # 用于时间相关操作
import argparse  # 用于处理命令行参数
import os  # 用于与操作系统交互
from sklearn.metrics import classification_report, confusion_matrix  # 导入分类报告和混淆矩阵的评估函数
from collections import Counter  # 用于计数对象
from sklearn import metrics  # 导入 sklearn 的评估指标
from pandas import isnull  # 导入用于检查缺失值的函数


def indicator_cal(json_standard, json_test):
    # 将输入的 JSON 数据转换为 pandas DataFrame 格式
    json_standard = pd.DataFrame(json_standard)
    json_test = pd.DataFrame(json_test)

    '''数据集总体指标'''
    
    # 从测试数据集中选择 'id' 和 'mid_json' 列
    a = json_test[['id', 'mid_json']]
    # 从标准数据集中选择 'id', 'mid_json' 和 'pass_label' 列
    b = json_standard[['id', 'mid_json', 'pass_label']]
    # 对两个数据集根据 'id' 列进行外连接合并
    outer_merge = pd.merge(a, b, on='id', how='outer')
    # 重命名合并后的列
    outer_merge.columns = ['id', 'standard_mid_json', 'test_mid_json', 'pass_label']
    # 检查标准中间 JSON 是否存在
    standard_exist = outer_merge.standard_mid_json.apply(lambda x: not isnull(x))
    # 检查测试中间 JSON 是否存在
    test_exist = outer_merge.test_mid_json.apply(lambda x: not isnull(x))

    overall_report = {}  # 初始化总体报告字典
    # 计算准确率
    overall_report['accuracy'] = metrics.accuracy_score(standard_exist, test_exist)
    # 计算精确率
    overall_report['precision'] = metrics.precision_score(standard_exist, test_exist)
    # 计算召回率
    overall_report['recall'] = metrics.recall_score(standard_exist, test_exist)
    # 计算 F1 分数
    overall_report['f1_score'] = metrics.f1_score(standard_exist, test_exist)

    # 对两个数据集根据 'id' 列进行内连接合并
    inner_merge = pd.merge(a, b, on='id', how='inner')
    # 重命名合并后的列
    inner_merge.columns = ['id', 'standard_mid_json', 'test_mid_json', 'pass_label']
    # 从合并结果中提取标准中间 JSON 数据
    json_standard = inner_merge['standard_mid_json']  # check一下是否对齐
    # 从合并结果中提取测试中间 JSON 数据
    json_test = inner_merge['test_mid_json']

    '''批量读取中间生成的json文件'''
    # 初始化存储测试数据的列表
    test_inline_equations = []  # 存储行内方程
    test_interline_equations = []  # 存储行间方程
    test_inline_euqations_bboxs = []  # 存储行内方程的边界框
    test_interline_equations_bboxs = []  # 存储行间方程的边界框
    test_dropped_text_bboxes = []  # 存储被丢弃的文本边界框
    test_dropped_text_tag = []  # 存储被丢弃的文本标签
    test_dropped_image_bboxes = []  # 存储被丢弃的图像边界框
    test_dropped_table_bboxes = []  # 存储被丢弃的表格边界框
    test_preproc_num = []  # 存储阅读顺序
    test_para_num = []  # 存储段落数量
    test_para_text = []  # 存储段落文本
    # 遍历 json_test 中的每个元素
        for i in json_test:
            # 将每个元素转换为 DataFrame 格式
            mid_json=pd.DataFrame(i)
            # 去除 DataFrame 的最后一列
            mid_json=mid_json.iloc[:,:-1]
            # 遍历 'inline_equations' 行中的每个元素
            for j1 in mid_json.loc['inline_equations',:]:
                # 初始化文本和边界框列表
                page_in_text=[]
                page_in_bbox=[]
                # 遍历每个元素
                for k1 in j1:
                    # 添加 LaTeX 文本到文本列表
                    page_in_text.append(k1['latex_text'])
                    # 添加边界框到边界框列表
                    page_in_bbox.append(k1['bbox'])
                # 将文本和边界框分别添加到对应列表
                test_inline_equations.append(page_in_text)
                test_inline_euqations_bboxs.append(page_in_bbox)
            # 遍历 'interline_equations' 行中的每个元素
            for j2 in mid_json.loc['interline_equations',:]:
                # 初始化文本和边界框列表
                page_in_text=[]
                page_in_bbox=[]
                # 遍历每个元素
                for k2 in j2:
                    # 添加 LaTeX 文本到文本列表
                    page_in_text.append(k2['latex_text'])
                # 将文本和边界框分别添加到对应列表
                test_interline_equations.append(page_in_text)
                test_interline_equations_bboxs.append(page_in_bbox)
    
            # 遍历 'droped_text_block' 行中的每个元素
            for j3 in mid_json.loc['droped_text_block',:]:
                # 初始化边界框和标签列表
                page_in_bbox=[]
                page_in_tag=[]
                # 遍历每个元素
                for k3 in j3:
                    # 添加边界框到边界框列表
                    page_in_bbox.append(k3['bbox'])
                    # 如果 k3 中存在 'tag' 键
                    if 'tag' in k3.keys():
                        # 添加标签到标签列表
                        page_in_tag.append(k3['tag'])
                    else:
                        # 如果没有标签，添加 'None'
                        page_in_tag.append('None')
                # 将标签和边界框分别添加到对应列表
                test_dropped_text_tag.append(page_in_tag)
                test_dropped_text_bboxes.append(page_in_bbox)
            # 遍历 'droped_image_block' 行中的每个元素
            for j4 in mid_json.loc['droped_image_block',:]:
                    # 将图像边界框添加到对应列表
                    test_dropped_image_bboxes.append(j4)
            # 遍历 'droped_table_block' 行中的每个元素
            for j5 in mid_json.loc['droped_table_block',:]:
                    # 将表格边界框添加到对应列表
                    test_dropped_table_bboxes.append(j5)
            # 遍历 'preproc_blocks' 行中的每个元素
            for j6 in mid_json.loc['preproc_blocks',:]:
                # 初始化页面编号列表
                page_in=[]
                # 遍历每个元素
                for k6 in j6:
                    # 添加编号到列表
                    page_in.append(k6['number'])
                # 将编号添加到对应列表
                test_preproc_num.append(page_in)
    
            # 初始化 PDF 文本列表
            test_pdf_text=[]     
            # 遍历 'para_blocks' 行中的每个元素
            for j7 in mid_json.loc['para_blocks',:]:
                # 将段落数量添加到对应列表
                test_para_num.append(len(j7))  
                # 遍历每个元素
                for k7 in j7:
                    # 添加段落文本到文本列表
                    test_pdf_text.append(k7['text'])  
            # 将文本添加到段落文本列表
            test_para_text.append(test_pdf_text)
    
        # 初始化标准化各类数据的列表
        standard_inline_equations=[]
        standard_interline_equations=[]
        standard_inline_euqations_bboxs=[]
        standard_interline_equations_bboxs=[]
        standard_dropped_text_bboxes=[]
        standard_dropped_text_tag=[]
        standard_dropped_image_bboxes=[]
        standard_dropped_table_bboxes=[] 
        standard_preproc_num=[]#阅读顺序
        standard_para_num=[]
        standard_para_text=[]
    # 遍历 json_standard 中的每一个元素
    for i in json_standard:
        # 将当前元素转换为 DataFrame 格式
        mid_json=pd.DataFrame(i)
        # 去掉最后一列，保留其余列
        mid_json=mid_json.iloc[:,:-1]
        # 遍历 'inline_equations' 行中的每一个元素
        for j1 in mid_json.loc['inline_equations',:]:
            # 初始化用于存储文本和边界框的列表
            page_in_text=[]
            page_in_bbox=[]
            # 遍历当前元素的每一个部分
            for k1 in j1:
                # 将 'latex_text' 添加到文本列表
                page_in_text.append(k1['latex_text'])
                # 将 'bbox' 添加到边界框列表
                page_in_bbox.append(k1['bbox'])
            # 将收集到的文本添加到标准行内公式列表
            standard_inline_equations.append(page_in_text)
            # 将收集到的边界框添加到标准行内公式边界框列表
            standard_inline_euqations_bboxs.append(page_in_bbox)
        # 遍历 'interline_equations' 行中的每一个元素
        for j2 in mid_json.loc['interline_equations',:]:
            # 初始化用于存储文本和边界框的列表
            page_in_text=[]
            page_in_bbox=[]
            # 遍历当前元素的每一个部分
            for k2 in j2:
                # 将 'latex_text' 添加到文本列表
                page_in_text.append(k2['latex_text'])
                # 将 'bbox' 添加到边界框列表
                page_in_bbox.append(k2['bbox'])
            # 将收集到的文本添加到标准行间公式列表
            standard_interline_equations.append(page_in_text)
            # 将收集到的边界框添加到标准行间公式边界框列表
            standard_interline_equations_bboxs.append(page_in_bbox)
        # 遍历 'droped_text_block' 行中的每一个元素
        for j3 in mid_json.loc['droped_text_block',:]:
            # 初始化用于存储边界框和标签的列表
            page_in_bbox=[]
            page_in_tag=[]
            # 遍历当前元素的每一个部分
            for k3 in j3:
                # 将 'bbox' 添加到边界框列表
                page_in_bbox.append(k3['bbox'])
                # 如果存在 'tag'，则添加标签，否则添加 'None'
                if 'tag' in k3.keys():
                    page_in_tag.append(k3['tag'])
                else:
                    page_in_tag.append('None')
            # 将收集到的边界框添加到标准丢失文本边界框列表
            standard_dropped_text_bboxes.append(page_in_bbox)
            # 将收集到的标签添加到标准丢失文本标签列表
            standard_dropped_text_tag.append(page_in_tag)
        # 遍历 'droped_image_block' 行中的每一个元素
        for j4 in mid_json.loc['droped_image_block',:]:
                # 将每个元素添加到标准丢失图像边界框列表
                standard_dropped_image_bboxes.append(j4)
        # 遍历 'droped_table_block' 行中的每一个元素
        for j5 in mid_json.loc['droped_table_block',:]:
                # 将每个元素添加到标准丢失表格边界框列表
                standard_dropped_table_bboxes.append(j5)
        # 遍历 'preproc_blocks' 行中的每一个元素
        for j6 in mid_json.loc['preproc_blocks',:]:
            # 初始化用于存储页面编号的列表
            page_in=[]
            # 遍历当前元素的每一个部分
            for k6 in j6:
                # 将 'number' 添加到页面编号列表
                page_in.append(k6['number'])
            # 将收集到的页面编号添加到标准预处理编号列表
            standard_preproc_num.append(page_in)     

        # 初始化用于存储标准 PDF 文本的列表
        standard_pdf_text=[]
        # 遍历 'para_blocks' 行中的每一个元素
        for j7 in mid_json.loc['para_blocks',:]:
            # 将段落数量添加到标准段落数量列表
            standard_para_num.append(len(j7))  
            # 遍历当前元素的每一个部分
            for k7 in j7:
                # 将 'text' 添加到标准 PDF 文本列表
                standard_pdf_text.append(k7['text'])
        # 将收集到的 PDF 文本添加到标准段落文本列表
        standard_para_text.append(standard_pdf_text)


    # 在计算指标之前最好先确认基本统计信息是否一致


    # 计算 PDF 之间的总体编辑距离和 bleu
    # 这里只计算正例的 PDF
    test_para_text=np.asarray(test_para_text, dtype = object)[inner_merge['pass_label']=='yes']
    standard_para_text=np.asarray(standard_para_text, dtype = object)[inner_merge['pass_label']=='yes']

    # 初始化用于存储 PDF 编辑距离和 bleu 分数的列表
    pdf_dis=[]
    pdf_bleu=[]
    # 遍历测试段落文本和标准段落文本
    for a,b in zip(test_para_text,standard_para_text):
        # 将测试文本的每个部分合并成字符串
        a1=[ ''.join(i) for i in a]
        # 将标准文本的每个部分合并成字符串
        b1=[ ''.join(i) for i in b]
        # 计算编辑距离并添加到列表
        pdf_dis.append(Levenshtein_Distance(a1,b1))
        # 计算 bleu 分数并添加到列表
        pdf_bleu.append(sentence_bleu([a1],b1))
    # 计算并存储 PDF 之间的平均编辑距离
    overall_report['pdf间的平均编辑距离']=np.mean(pdf_dis)
    # 计算并存储 PDF 之间的平均 bleu 分数
    overall_report['pdf间的平均bleu']=np.mean(pdf_bleu)


    # 行内公式编辑距离和 bleu
    dis1=[]
    bleu1=[]

    # 将测试行内公式的每个部分合并成字符串
    test_inline_equations=[ ''.join(i) for i in test_inline_equations]
    # 将标准行内公式的每个部分合并成字符串
    standard_inline_equations=[ ''.join(i) for i in standard_inline_equations]
    # 遍历测试行内公式和标准行内公式
        for a,b in zip(test_inline_equations,standard_inline_equations):
            # 如果两个公式都为空，则继续下一次循环
            if len(a)==0 and len(b)==0:
                continue
            else:
                # 如果两个公式相同
                if a==b:
                    # 向编辑距离列表添加0
                    dis1.append(0)
                    # 向BLEU分数列表添加1
                    bleu1.append(1)
                else:
                    # 计算Levenshtein距离并添加到列表
                    dis1.append(Levenshtein_Distance(a,b))
                    # 计算BLEU分数并添加到列表
                    bleu1.append(sentence_bleu([a],b))
        # 计算行内公式的平均编辑距离
        inline_equations_edit=np.mean(dis1)
        # 计算行内公式的平均BLEU分数
        inline_equations_bleu=np.mean(bleu1)
    
        '''行内公式bbox匹配相关指标'''
        # 计算行内公式的bbox匹配指标
        inline_equations_bbox_report=bbox_match_indicator(test_inline_euqations_bboxs,standard_inline_euqations_bboxs)
    
        '''行间公式编辑距离和bleu'''
        # 初始化行间公式的编辑距离和BLEU分数列表
        dis2=[]
        bleu2=[]
    
        # 将测试行间公式列表中的每个元素合并为字符串
        test_interline_equations=[ ''.join(i) for i in test_interline_equations]
        # 将标准行间公式列表中的每个元素合并为字符串
        standard_interline_equations=[ ''.join(i) for i in standard_interline_equations]
    
        # 遍历测试行间公式和标准行间公式
        for a,b in zip(test_interline_equations,standard_interline_equations):
            # 如果两个公式都为空，则继续下一次循环
            if len(a)==0 and len(b)==0:
                continue
            else:
                # 如果两个公式相同
                if a==b:
                    # 向编辑距离列表添加0
                    dis2.append(0)
                    # 向BLEU分数列表添加1
                    bleu2.append(1)
                else:
                    # 计算Levenshtein距离并添加到列表
                    dis2.append(Levenshtein_Distance(a,b))
                    # 计算BLEU分数并添加到列表
                    bleu2.append(sentence_bleu([a],b))
        # 计算行间公式的平均编辑距离
        interline_equations_edit=np.mean(dis2)
        # 计算行间公式的平均BLEU分数
        interline_equations_bleu=np.mean(bleu2)
    
        '''行间公式bbox匹配相关指标'''
        # 计算行间公式的bbox匹配指标
        interline_equations_bbox_report=bbox_match_indicator(test_interline_equations_bboxs,standard_interline_equations_bboxs)
    
        '''可以先检查page和bbox数量是否一致'''
    
        '''dropped_text_block的bbox匹配相关指标'''
        # 初始化测试文本和标准文本的bbox列表
        test_text_bbox=[]
        standard_text_bbox=[]
        # 初始化测试标签和标准标签列表
        test_tag=[]
        standard_tag=[]
    
        # 初始化索引
        index=0
    # 遍历测试和标准的文本框坐标
    for a,b in zip(test_dropped_text_bboxes,standard_dropped_text_bboxes):
        # 初始化测试和标准页面标签及其边框列表
        test_page_tag=[]
        standard_page_tag=[]
        test_page_bbox=[]
        standard_page_bbox=[]
        # 如果两个边框均为空，则跳过
        if len(a)==0 and len(b)==0:
            pass
        else:
            # 遍历标准边框的索引
            for i in range(len(b)):
                judge=0  # 初始化判断标志
                # 添加标准标签到列表
                standard_page_tag.append(standard_dropped_text_tag[index][i])
                # 标记标准边框存在
                standard_page_bbox.append(1)
                # 遍历测试边框的索引
                for j in range(len(a)):
                    # 检查边框是否匹配
                    if bbox_offset(b[i],a[j]):
                        judge=1  # 设置判断标志为匹配
                        # 添加测试标签到列表
                        test_page_tag.append(test_dropped_text_tag[index][j])
                        # 标记测试边框存在
                        test_page_bbox.append(1)
                        break  # 找到匹配后退出内层循环
                # 如果没有匹配，添加'None'
                if judge==0:
                    test_page_tag.append('None')
                    test_page_bbox.append(0)  # 标记为缺失

            # 检查测试标签与标准标签的长度关系
            if len(test_dropped_text_tag[index])+test_page_tag.count('None')>len(standard_dropped_text_tag[index]):  # 有多删的情况出现
                test_page_tag1=test_page_tag.copy()  # 复制当前测试标签列表
                # 如果包含'None'，则移除
                if 'None' in test_page_tag:
                    test_page_tag1.remove('None')
                else:
                    test_page_tag1=test_page_tag

                # 计算标签的差异
                diff=list((Counter(test_dropped_text_tag[index]) - Counter(test_page_tag1)).elements())
              
                # 将差异添加到标签和边框列表
                test_page_tag.extend(diff)
                standard_page_tag.extend(['None']*len(diff))
                test_page_bbox.extend([1]*len(diff))
                standard_page_bbox.extend([0]*len(diff))

            # 将结果添加到最终标签和边框列表
            test_tag.extend(test_page_tag)
            standard_tag.extend(standard_page_tag)
            test_text_bbox.extend(test_page_bbox)
            standard_text_bbox.extend(standard_page_bbox)

        index+=1  # 更新索引

    # 创建报告字典并计算各项指标
    text_block_report = {}
    text_block_report['accuracy']=metrics.accuracy_score(standard_text_bbox,test_text_bbox)
    text_block_report['precision']=metrics.precision_score(standard_text_bbox,test_text_bbox)
    text_block_report['recall']=metrics.recall_score(standard_text_bbox,test_text_bbox)
    text_block_report['f1_score']=metrics.f1_score(standard_text_bbox,test_text_bbox)

    # 生成标签的分类报告，删除不必要的项
    text_block_tag_report = classification_report(y_true=standard_tag , y_pred=test_tag,output_dict=True)
    del text_block_tag_report['None']
    del text_block_tag_report["macro avg"]
    del text_block_tag_report["weighted avg"]

    # 计算图像块的边框匹配指标，可能存在数据格式不一致
    image_block_report=bbox_match_indicator(test_dropped_image_bboxes,standard_dropped_image_bboxes)
    
    # 计算表格块的边框匹配指标
    table_block_report=bbox_match_indicator(test_dropped_table_bboxes,standard_dropped_table_bboxes)
    
    # 计算阅读顺序编辑距离的均值
    preproc_num_dis=[]
    for a,b in zip(test_preproc_num,standard_preproc_num):
        preproc_num_dis.append(Levenshtein_Distance(a,b))  # 计算每对的编辑距离
    preproc_num_edit=np.mean(preproc_num_dis)  # 计算平均编辑距离

    # 分段准确率的处理
    # 将测试参数转换为 NumPy 数组
    test_para_num=np.array(test_para_num)
    # 将标准参数转换为 NumPy 数组
    standard_para_num=np.array(standard_para_num)
    # 计算测试参数与标准参数相等的平均值，得到准确率
    acc_para=np.mean(test_para_num==standard_para_num)

    
    # 创建一个空的 Pandas DataFrame 用于存储输出结果
    output=pd.DataFrame()
    # 将总体指标添加到 DataFrame 中
    output['总体指标']=[overall_report]
    # 将行内公式的平均编辑距离添加到 DataFrame 中
    output['行内公式平均编辑距离']=[inline_equations_edit]
    # 将行间公式的平均编辑距离添加到 DataFrame 中
    output['行间公式平均编辑距离']=[interline_equations_edit]
    # 将行内公式的平均 bleu 分数添加到 DataFrame 中
    output['行内公式平均bleu']=[inline_equations_bleu]
    # 将行间公式的平均 bleu 分数添加到 DataFrame 中
    output['行间公式平均bleu']=[interline_equations_bleu]
    # 将行内公式识别的相关指标添加到 DataFrame 中
    output['行内公式识别相关指标']=[inline_equations_bbox_report]
    # 将行间公式识别的相关指标添加到 DataFrame 中
    output['行间公式识别相关指标']=[interline_equations_bbox_report]
    # 将阅读顺序的平均编辑距离添加到 DataFrame 中
    output['阅读顺序平均编辑距离']=[preproc_num_edit]
    # 将分段准确率添加到 DataFrame 中
    output['分段准确率']=[acc_para]
    # 将删除的 text block 的相关指标添加到 DataFrame 中
    output['删除的text block的相关指标']=[text_block_report]
    # 将删除的 image block 的相关指标添加到 DataFrame 中
    output['删除的image block的相关指标']=[image_block_report]
    # 将删除的 table block 的相关指标添加到 DataFrame 中
    output['删除的table block的相关指标']=[table_block_report]
    # 将删除的 text block 的 tag 相关指标添加到 DataFrame 中
    output['删除的text block的tag相关指标']=[text_block_tag_report]
    

    # 返回包含所有指标的 DataFrame
    return output
# 计算编辑距离的函数
def Levenshtein_Distance(str1, str2):
    # 创建一个矩阵，行数为str1长度+1，列数为str2长度+1，初始化为行索引和列索引的和
    matrix = [[ i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    # 遍历str1的每个字符
    for i in range(1, len(str1)+1):
        # 遍历str2的每个字符
        for j in range(1, len(str2)+1):
            # 如果当前字符相等，设置距离d为0
            if(str1[i-1] == str2[j-1]):
                d = 0
            else:
                # 否则，设置距离d为1
                d = 1
            # 更新矩阵中的当前值为上、左、左上对角的最小值加d
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)
    # 返回编辑距离，即矩阵右下角的值
    return matrix[len(str1)][len(str2)]


# 计算bbox偏移量是否符合标准的函数
def bbox_offset(b_t,b_s):
    '''b_t是test_doc里的bbox,b_s是standard_doc里的bbox'''
    # 解包test和standard文档中的bbox坐标
    x1_t,y1_t,x2_t,y2_t=b_t
    x1_s,y1_s,x2_s,y2_s=b_s
    # 计算重叠区域的左上角x坐标
    x1=max(x1_t,x1_s)
    # 计算重叠区域的右下角x坐标
    x2=min(x2_t,x2_s)
    # 计算重叠区域的左上角y坐标
    y1=max(y1_t,y1_s)
    # 计算重叠区域的右下角y坐标
    y2=min(y2_t,y2_s)
    # 计算重叠面积
    area_overlap=(x2-x1)*(y2-y1)
    # 计算两个bbox的总面积
    area_t=(x2_t-x1_t)*(y2_t-y1_t)+(x2_s-x1_s)*(y2_s-y1_s)-area_overlap
    # 判断重叠情况是否符合标准
    if area_t-area_overlap==0 or area_overlap/(area_t-area_overlap)>0.95:
        return True
    else:
        return False
    

# bbox匹配和对齐函数，输出相关指标
# 输入的是以page为单位的bbox列表
def bbox_match_indicator(test_bbox_list,standard_bbox_list):
    
    # 初始化测试和标准bbox列表
    test_bbox=[]
    standard_bbox=[]
    # 同时遍历测试和标准的bbox列表
    for a,b in zip(test_bbox_list,standard_bbox_list):
        # 存储当前页面的bbox
        test_page_bbox=[]
        standard_page_bbox=[]
        # 如果两个页面的bbox都为空，跳过
        if len(a)==0 and len(b)==0:
            pass
        else:
            # 遍历标准bbox
            for i in b:
                # 检查当前bbox是否有效
                if len(i)!=4:
                    continue
                else:
                    judge=0
                    # 标记标准bbox的有效性
                    standard_page_bbox.append(1)
                    # 检查测试bbox与标准bbox的重叠情况
                    for j in a:
                        if bbox_offset(i,j):
                            judge=1
                            test_page_bbox.append(1)
                            break
                    # 如果没有匹配，标记为无效
                    if judge==0:
                        test_page_bbox.append(0)
                        
            # 计算多删的情况
            diff_num=len(a)+test_page_bbox.count(0)-len(b)
            if diff_num>0:#有多删的情况出现
                # 扩展测试和标准页面的有效性列表
                test_page_bbox.extend([1]*diff_num)
                standard_page_bbox.extend([0]*diff_num)

            # 合并到总列表中
            test_bbox.extend(test_page_bbox)
            standard_bbox.extend(standard_page_bbox)

    # 创建一个字典来存储评价指标
    block_report = {}
    # 计算准确率、精确率、召回率和F1分数
    block_report['accuracy']=metrics.accuracy_score(standard_bbox,test_bbox)
    block_report['precision']=metrics.precision_score(standard_bbox,test_bbox)
    block_report['recall']=metrics.recall_score(standard_bbox,test_bbox)
    block_report['f1_score']=metrics.f1_score(standard_bbox,test_bbox)

    # 返回指标报告
    return block_report


# 创建一个命令行参数解析器
parser = argparse.ArgumentParser()
# 添加test和standard参数
parser.add_argument('--test', type=str)
parser.add_argument('--standard', type=str)
# 解析命令行参数
args = parser.parse_args()
# 获取测试和标准文档的路径
pdf_json_test = args.test
pdf_json_standard = args.standard
# 如果该脚本是主程序运行
if __name__ == '__main__':
    
   # 从指定的 JSON 文件逐行读取数据，并将每行转换为 Python 字典，存储在列表中
   pdf_json_test = [json.loads(line) 
                        for line in open(pdf_json_test, 'r', encoding='utf-8')]
   # 从另一个指定的 JSON 文件逐行读取数据，并将每行转换为 Python 字典，存储在列表中
   pdf_json_standard = [json.loads(line) 
                    for line in open(pdf_json_standard, 'r', encoding='utf-8')]
   
   # 调用函数计算指标，传入标准和测试的 JSON 数据
   overall_indicator=indicator_cal(pdf_json_standard,pdf_json_test)

   '''计算的指标输出到 overall_indicator_output.json 文件中'''
   # 将计算出的指标以 JSON 格式保存到指定文件，设置记录方式、行分隔和 ASCII 处理
   overall_indicator.to_json('overall_indicator_output.json',orient='records',lines=True,force_ascii=False)
```