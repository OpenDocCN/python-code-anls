# `.\MinerU\magic_pdf\pre_proc\citationmarker_remove.py`

```
# 去掉正文的引文引用marker的文档链接
"""
去掉正文的引文引用marker
https://aicarrier.feishu.cn/wiki/YLOPwo1PGiwFRdkwmyhcZmr0n3d
"""
# 导入正则表达式库
import re
# 导入NLP模型的工具库（当前注释掉）
# from magic_pdf.libs.nlp_utils import NLPModels


# 初始化NLP模型（当前注释掉）
# __NLP_MODEL = NLPModels()

def check_1(spans, cur_span_i):
    """寻找前一个char,如果是句号，逗号，那么就是角标"""
    # 如果当前索引是0，则返回False，表示不是角标
    if cur_span_i==0:
        return False # 不是角标
    # 获取前一个span
    pre_span = spans[cur_span_i-1]
    # 获取前一个span的最后一个字符
    pre_char = pre_span['chars'][-1]['c']
    # 如果最后一个字符是句号或逗号，则返回True，表示是角标
    if pre_char in ['。', '，', '.', ',']:
        return True
    
    # 否则返回False
    return False


# def check_2(spans, cur_span_i):
#     """检查前面一个span的最后一个单词，如果长度大于5，全都是字母，并且不含大写，就是角标"""
#     # 定义一个正则表达式模式，匹配特定格式的名字缩写
#     pattern = r'\b[A-Z]\.\s[A-Z][a-z]*\b' # 形如A. Bcde, L. Bcde, 人名的缩写
#
#     # 如果当前索引是0且span数量大于1
#     if cur_span_i==0 and len(spans)>1:
#         # 获取下一个span
#         next_span = spans[cur_span_i+1]
#         # 合并下一个span的字符为文本
#         next_txt = "".join([c['c'] for c in next_span['chars']])
#         # 使用NLP模型检测实体类别
#         result = __NLP_MODEL.detect_entity_catgr_using_nlp(next_txt)
#         # 如果检测到的人物、地点或组织，返回True
#         if result in ["PERSON", "GPE", "ORG"]:
#             return True
#
#         # 如果匹配到正则模式，则返回True
#         if re.findall(pattern, next_txt):
#             return True
#
#         # 否则返回False
#         return False # 不是角标
#     # 如果当前索引是0且span数量为1，返回False
#     elif cur_span_i==0 and len(spans)==1: # 角标占用了整行？谨慎删除
#         return False
#
#     # 如果这个span是最后一个span
#     if cur_span_i==len(spans)-1:
#         # 获取前一个span
#         pre_span = spans[cur_span_i-1]
#         # 合并前一个span的字符为文本
#         pre_txt = "".join([c['c'] for c in pre_span['chars']])
#         # 获取前一个文本的最后一个单词
#         pre_word = pre_txt.split(' ')[-1]
#         # 使用NLP模型检测实体类别
#         result = __NLP_MODEL.detect_entity_catgr_using_nlp(pre_txt)
#         # 如果检测到的人物、地点或组织，返回True
#         if result in ["PERSON", "GPE", "ORG"]:
#             return True
#
#         # 如果匹配到正则模式，则返回True
#         if re.findall(pattern, pre_txt):
#             return True
#
#         # 返回前一个单词的长度大于5且全是字母且全为小写
#         return len(pre_word) > 5 and pre_word.isalpha() and pre_word.islower()
#     else: # 既不是第一个span，也不是最后一个span
#         # 获取前一个和后一个span
#         pre_span = spans[cur_span_i-1]
#         next_span = spans[cur_span_i+1]
#         cur_span = spans[cur_span_i]
#         # 找到前一个和后一个span里的距离最近的单词
#         pre_distance = 10000 # 一个很大的数
#         next_distance = 10000 # 一个很大的数
#         # 反向遍历前一个span的字符
#         for c in pre_span['chars'][::-1]:
#             # 如果字符是字母，计算距离
#             if c['c'].isalpha():
#                 pre_distance = cur_span['bbox'][0] - c['bbox'][2]
#                 break
#         # 正向遍历下一个span的字符
#         for c in next_span['chars']:
#             # 如果字符是字母，计算距离
#             if c['c'].isalpha():
#                 next_distance = c['bbox'][0] - cur_span['bbox'][2]
#                 break
#
#         # 比较前后距离，确定所属span
#         if pre_distance<next_distance:
#             belong_to_span = pre_span
#         else:
#             belong_to_span = next_span
#
#         # 合并所属span的字符为文本
#         txt = "".join([c['c'] for c in belong_to_span['chars']])
#         # 获取文本的最后一个单词
#         pre_word = txt.split(' ')[-1]
#         # 使用NLP模型检测实体类别
#         result = __NLP_MODEL.detect_entity_catgr_using_nlp(txt)
#         # 如果检测到的人物、地点或组织，返回True
#         if result in ["PERSON", "GPE", "ORG"]:
#             return True
#
#         # 如果匹配到正则模式，则返回True
#         if re.findall(pattern, txt):
#             return True
#
#         # 返回前一个单词的长度大于5且全是字母且全为小写
#         return len(pre_word) > 5 and pre_word.isalpha() and pre_word.islower()


def check_3(spans, cur_span_i):
    """上标里有[], 有*， 有-， 有逗号"""
    # 处理形如[2-3]和[22]的上标
    # 处理形如2,3,4的上标
    # 合并当前span的字符为文本并去除两端空白
    cur_span_txt = ''.join(c['c'] for c in spans[cur_span_i]['chars']).strip()
    # 定义一个包含不良字符的列表
    bad_char = ['[', ']', '*', ',']
    
    # 检查当前文本是否包含不良字符且包含数字
    if any([c in cur_span_txt for c in bad_char]) and any(character.isdigit() for character in cur_span_txt):
        # 如果条件满足，返回 True
        return True
    
    # 定义正则表达式模式，如2-3和a-b的格式
    patterns = [r'\d+-\d+', r'[a-zA-Z]-[a-zA-Z]', r'[a-zA-Z],[a-zA-Z]']
    # 遍历每个模式进行匹配
    for pattern in patterns:  
        # 使用正则表达式检查当前文本是否符合模式
        match = re.match(pattern, cur_span_txt)
        # 如果匹配成功，则返回 True
        if match is not None:
            return True
    
    # 如果没有匹配成功，返回 False
    return False
# 定义一个函数，用于移除引用标记
def remove_citation_marker(with_char_text_blcoks):
    # 遍历包含字符文本块的列表
    for blk in with_char_text_blcoks:
        # 遍历每个文本块中的行
        for line in blk['lines']:
            # 如果行中的span个数少于2，直接忽略
            if len(line['spans'])<=1:
                continue

            # 初始化高度最高的span为第一个span的边界框
            max_hi_span = line['spans'][0]['bbox']
            min_font_sz = 10000 # 初始化行内最小字体大小
            max_font_sz = 0   # 初始化行内最大字体大小
                
            # 遍历行内的每个span
            for s in line['spans']:
                # 更新高度最高的span为当前span（如果当前span更高）
                if max_hi_span[3]-max_hi_span[1]<s['bbox'][3]-s['bbox'][1]:
                    max_hi_span = s['bbox']
                # 更新行内最小字体大小
                if min_font_sz>s['size']:
                    min_font_sz = s['size']
                # 更新行内最大字体大小
                if max_font_sz<s['size']:
                    max_font_sz = s['size']
                        
            # 计算高度最高span的中间y坐标
            base_span_mid_y = (max_hi_span[3]+max_hi_span[1])/2
            
            # 用于存储要删除的span
            span_to_del = []
            # 遍历行中的每个span
            for i, span in enumerate(line['spans']):
                # 计算当前span的高度
                span_hi = span['bbox'][3]-span['bbox'][1]
                # 计算当前span的中间y坐标
                span_mid_y = (span['bbox'][3]+span['bbox'][1])/2
                # 获取当前span的字体大小
                span_font_sz = span['size']
                
                # 如果当前span的字体大小与最大字体大小差异小于1，则跳过
                if max_font_sz-span_font_sz<1:
                    continue

                # 过滤掉高度为0的情况
                if span_hi==0 or min_font_sz==0:
                    continue

                # 判断是否需要删除当前span
                if (base_span_mid_y-span_mid_y)/span_hi>0.2 or (base_span_mid_y-span_mid_y>0 and abs(span_font_sz-min_font_sz)/min_font_sz<0.1):
                    """
                    1. 检查前一个字符是否是句号或逗号，若是，则认为是角标
                    2. 检查前面是否是一个长度大于5的单词，而非短字母，认为是角标
                    3. 检查上标中是否包含数字和逗号组合，认为是角标
                    4. 根据距离判断角标是属于前文还是后文
                    """
                    if (check_1(line['spans'], i) or
                        # check_2(line['spans'], i) or
                        check_3(line['spans'], i)
                    ):
                        """将此角标标记为删除，更新行的文本"""
                        span_to_del.append(span)
            # 如果有要删除的span
            if len(span_to_del)>0:
                # 从行中删除标记的span
                for span in span_to_del:
                    line['spans'].remove(span)
                # 更新行的文本为剩余span的字符组合
                line['text'] = ''.join([c['c'] for s in line['spans'] for c in s['chars']])
    
    # 返回处理后的文本块
    return with_char_text_blcoks
```