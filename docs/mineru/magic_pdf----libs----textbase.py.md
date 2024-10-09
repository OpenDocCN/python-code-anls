# `.\MinerU\magic_pdf\libs\textbase.py`

```
# 导入数学库以便后续可能需要的数学函数
import math


# 增加字典中指定键的值，若键不存在则初始化
def __inc_dict_val(mp, key, val_inc:int):
    # 检查字典中是否存在指定键
    if mp.get(key):
        # 若存在，则增加其对应的值
        mp[key] = mp[key] + val_inc
    else:
        # 若不存在，则将该键初始化为增量值
        mp[key] = val_inc
        
    

# 获取文本块的基本信息，包括颜色、字号和字体
def get_text_block_base_info(block):
    """
    获取这个文本块里的字体的颜色、字号、字体
    按照正文字数最多的返回
    """
    
    # 创建一个计数器字典，用于统计不同样式文本的长度
    counter = {}
    
    # 遍历文本块中的每一行
    for line in block['lines']:
        # 遍历行中的每个文本片段
        for span in line['spans']:
            # 获取当前片段的颜色
            color = span['color']
            # 获取并四舍五入当前片段的字号
            size = round(span['size'], 2)
            # 获取当前片段的字体
            font = span['font']
            
            # 计算当前片段的文本长度
            txt_len = len(span['text'])
            # 更新计数器，记录当前样式的文本长度
            __inc_dict_val(counter, (color, size, font), txt_len)
            
    
    # 找出计数器中正文字数最多的样式
    c, s, ft = max(counter, key=counter.get)
    
    # 返回颜色、字号和字体
    return c, s, ft
```