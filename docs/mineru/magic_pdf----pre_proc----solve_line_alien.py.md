# `.\MinerU\magic_pdf\pre_proc\solve_line_alien.py`

```
# 解决行内文本间距过大问题，将处理后的字典返回
def solve_inline_too_large_interval(pdf_info_dict: dict) -> dict:  # text_block -> json中的preproc_block
    """解决行内文本间距过大问题"""
    # 遍历 pdf_info_dict 字典的每一页
    for i in range(len(pdf_info_dict)):
        # 获取当前页的预处理文本块
        text_blocks = pdf_info_dict[f'page_{i}']['preproc_blocks']

        # 遍历当前页的所有文本块
        for block in text_blocks:
            # 初始化前一个行框的坐标
            x_pre_1, y_pre_1, x_pre_2, y_pre_2 = 0, 0, 0, 0
            
            # 遍历当前文本块中的每一行
            for line in block['lines']:
                # 获取当前行的边界框坐标
                x_cur_1, y_cur_1, x_cur_2, y_cur_2 = line['bbox']
                # line_box = [x1, y1, x2, y2] 
                # 检查当前行的上边界和下边界是否与前一行相同
                if int(y_cur_1) == int(y_pre_1) and int(y_cur_2) == int(y_pre_2):
                    # 如果当前行只有一个文本跨度，前面加一个空格
                    # if len(line['spans']) == 1:
                    line['spans'][0]['text'] = ' ' + line['spans'][0]['text']
                
                # 更新前一个行框的坐标为当前行的坐标
                x_pre_1, y_pre_1, x_pre_2, y_pre_2 = line['bbox'] 

    # 返回处理后的 pdf_info_dict 字典
    return pdf_info_dict
```