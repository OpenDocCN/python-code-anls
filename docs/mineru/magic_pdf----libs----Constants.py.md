# `.\MinerU\magic_pdf\libs\Constants.py`

```
# span维度自定义字段
"""
span维度自定义字段
"""
# 跨页合并的标志，用于指示当前 span 是否跨页
CROSS_PAGE = "cross_page"

# block维度自定义字段
"""
block维度自定义字段
"""
# 标志，指示 block 中的行是否被删除
LINES_DELETED = "lines_deleted"

# 结构等价表的标识符
# struct eqtable
STRUCT_EQTABLE = "struct_eqtable"

# 表识别的最大时间默认值，单位为毫秒
# table recognition max time default value
TABLE_MAX_TIME_VALUE = 400

# 表结果的最大长度限制
# pp_table_result_max_length
TABLE_MAX_LEN = 480

# 表结构算法的名称
# pp table structure algorithm
TABLE_MASTER = "TableMaster"

# 表结构主字典的文件名
# table master structure dict
TABLE_MASTER_DICT = "table_master_structure_dict.txt"

# 表结构主文件的目录
# table master dir
TABLE_MASTER_DIR = "table_structure_tablemaster_infer/"

# 目标检测模型的目录
# pp detect model dir
DETECT_MODEL_DIR = "ch_PP-OCRv3_det_infer"

# 识别模型的目录
# pp rec model dir
REC_MODEL_DIR = "ch_PP-OCRv3_rec_infer"

# 识别字符字典的路径
# pp rec char dict path
REC_CHAR_DICT = "ppocr_keys_v1.txt"
```