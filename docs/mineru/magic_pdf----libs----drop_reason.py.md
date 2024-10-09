# `.\MinerU\magic_pdf\libs\drop_reason.py`

```
# 定义一个表示不同丢弃原因的类
class DropReason:
    # 文字块有水平互相覆盖，导致无法准确定位文字顺序
    TEXT_BLCOK_HOR_OVERLAP = "text_block_horizontal_overlap" 
    # 需保留的block水平覆盖
    USEFUL_BLOCK_HOR_OVERLAP = "useful_block_horizontal_overlap" 
    # 复杂的布局，暂时不支持
    COMPLICATED_LAYOUT = "complicated_layout" 
    # 目前不支持分栏超过2列的
    TOO_MANY_LAYOUT_COLUMNS = "too_many_layout_columns" 
    # 含有带色块的PDF，色块会改变阅读顺序，目前不支持带底色文字块的PDF。
    COLOR_BACKGROUND_TEXT_BOX = "color_background_text_box" 
    # 含特殊图片，计算量太大，从而丢弃
    HIGH_COMPUTATIONAL_lOAD_BY_IMGS = "high_computational_load_by_imgs" 
    # 特殊的SVG图，计算量太大，从而丢弃
    HIGH_COMPUTATIONAL_lOAD_BY_SVGS = "high_computational_load_by_svgs" 
    # 计算量超过负荷，当前方法下计算量消耗过大
    HIGH_COMPUTATIONAL_lOAD_BY_TOTAL_PAGES = "high_computational_load_by_total_pages" 
    # 版面分析失败
    MISS_DOC_LAYOUT_RESULT = "missing doc_layout_result" 
    # 解析中发生异常
    Exception = "_exception" 
    # PDF是加密的
    ENCRYPTED = "encrypted" 
    # PDF页面总数为0
    EMPTY_PDF = "total_page=0" 
    # 不是文字版PDF，无法直接解析
    NOT_IS_TEXT_PDF = "not_is_text_pdf" 
    # 无法清晰的分段
    DENSE_SINGLE_LINE_BLOCK = "dense_single_line_block" 
    # 探测标题失败
    TITLE_DETECTION_FAILED = "title_detection_failed" 
    # 分析标题级别失败（例如一级、二级、三级标题）
    TITLE_LEVEL_FAILED = "title_level_failed" 
    # 识别段落失败
    PARA_SPLIT_FAILED = "para_split_failed" 
    # 段落合并失败
    PARA_MERGE_FAILED = "para_merge_failed" 
    # 不支持的语种
    NOT_ALLOW_LANGUAGE = "not_allow_language" 
    # 特殊PDF的标识
    SPECIAL_PDF = "special_pdf"
    # 无法精确判断文字分栏
    PSEUDO_SINGLE_COLUMN = "pseudo_single_column" 
    # 无法分析页面的版面
    CAN_NOT_DETECT_PAGE_LAYOUT = "can_not_detect_page_layout" 
    # 缩放导致 bbox 面积为负
    NEGATIVE_BBOX_AREA = "negative_bbox_area" 
    # 无法分离重叠的block
    OVERLAP_BLOCKS_CAN_NOT_SEPARATION = "overlap_blocks_can_t_separation" 
```