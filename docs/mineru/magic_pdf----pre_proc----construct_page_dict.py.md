# `.\MinerU\magic_pdf\pre_proc\construct_page_dict.py`

```
# 定义构建页面组件的函数，接收多个参数用于页面内容构建
def construct_page_component(page_id, image_info, table_info, text_blocks_preproc, layout_bboxes, inline_eq_info,
                             interline_eq_info, raw_pymu_blocks,
                             removed_text_blocks, removed_image_blocks, images_backup, droped_table_block, table_backup,
                             layout_tree,
                             page_w, page_h, footnote_bboxes_tmp):
    # 初始化返回字典
    return_dict = {}

    # 初始化段落块字典
    return_dict['para_blocks'] = {}
    # 保存预处理文本块
    return_dict['preproc_blocks'] = text_blocks_preproc
    # 保存图像信息
    return_dict['images'] = image_info
    # 保存表格信息
    return_dict['tables'] = table_info
    # 保存行间方程信息
    return_dict['interline_equations'] = interline_eq_info
    # 保存行内方程信息
    return_dict['inline_equations'] = inline_eq_info
    # 保存布局边界框信息
    return_dict['layout_bboxes'] = layout_bboxes
    # 保存原始Pymu块信息
    return_dict['pymu_raw_blocks'] = raw_pymu_blocks
    # 初始化全局统计字典
    return_dict['global_statistic'] = {}

    # 保存被移除的文本块
    return_dict['droped_text_block'] = removed_text_blocks
    # 保存被移除的图像块
    return_dict['droped_image_block'] = removed_image_blocks
    # 初始化被移除的表格块为空列表
    return_dict['droped_table_block'] = []
    # 保存图像备份
    return_dict['image_backup'] = images_backup
    # 初始化被移除的表格备份为空列表
    return_dict['table_backup'] = []
    # 保存页面索引
    return_dict['page_idx'] = page_id
    # 保存页面尺寸
    return_dict['page_size'] = [page_w, page_h]
    # 保存布局树，用于辅助分析布局
    return_dict['_layout_tree'] = layout_tree  # 辅助分析layout作用
    # 保存脚注边界框临时信息
    return_dict['footnote_bboxes_tmp'] = footnote_bboxes_tmp

    # 返回构建的字典
    return return_dict


# 定义OCR构建页面组件的函数，接收多个参数用于页面内容构建
def ocr_construct_page_component(blocks, layout_bboxes, page_id, page_w, page_h, layout_tree,
                                 images, tables, interline_equations, inline_equations,
                                 dropped_text_block, dropped_image_block, dropped_table_block, dropped_equation_block,
                                 need_remove_spans_bboxes_dict):
    # 初始化返回字典
    return_dict = {
        # 保存预处理块
        'preproc_blocks': blocks,
        # 保存布局边界框信息
        'layout_bboxes': layout_bboxes,
        # 保存页面索引
        'page_idx': page_id,
        # 保存页面尺寸
        'page_size': [page_w, page_h],
        # 保存布局树
        '_layout_tree': layout_tree,
        # 保存图像信息
        'images': images,
        # 保存表格信息
        'tables': tables,
        # 保存行间方程信息
        'interline_equations': interline_equations,
        # 保存行内方程信息
        'inline_equations': inline_equations,
        # 保存被移除的文本块
        'droped_text_block': dropped_text_block,
        # 保存被移除的图像块
        'droped_image_block': dropped_image_block,
        # 保存被移除的表格块
        'droped_table_block': dropped_table_block,
        # 保存被移除的方程块
        'dropped_equation_block': dropped_equation_block,
        # 保存需要移除的边界框信息
        'droped_bboxes': need_remove_spans_bboxes_dict,
    }
    # 返回构建的字典
    return return_dict


# 定义OCR构建页面组件的第二版本，接收多个参数用于页面内容构建
def ocr_construct_page_component_v2(blocks, layout_bboxes, page_id, page_w, page_h, layout_tree,
                                    images, tables, interline_equations, discarded_blocks, need_drop, drop_reason):
    # 创建一个字典以返回多个数据块
        return_dict = {
            # 包含预处理后的文本块
            'preproc_blocks': blocks,
            # 包含页面布局的边界框
            'layout_bboxes': layout_bboxes,
            # 当前页面的索引
            'page_idx': page_id,
            # 页面尺寸的列表，包含宽和高
            'page_size': [page_w, page_h],
            # 布局树的结构
            '_layout_tree': layout_tree,
            # 图片的列表
            'images': images,
            # 表格的列表
            'tables': tables,
            # 行间方程的列表
            'interline_equations': interline_equations,
            # 被丢弃的文本块
            'discarded_blocks': discarded_blocks,
            # 是否需要丢弃标志
            'need_drop': need_drop,
            # 丢弃原因的描述
            'drop_reason': drop_reason,
        }
        # 返回构建的字典
        return return_dict
```