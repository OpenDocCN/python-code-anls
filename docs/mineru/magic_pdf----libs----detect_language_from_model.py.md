# `.\MinerU\magic_pdf\libs\detect_language_from_model.py`

```
# 从 collections 模块导入 Counter，用于统计语言出现的次数
from collections import Counter

# 从 magic_pdf.libs.language 模块导入 detect_lang 函数，用于检测文本语言
from magic_pdf.libs.language import detect_lang

# 定义函数，接受一个模型列表作为参数，返回检测出的语言
def get_language_from_model(model_list: list):
    # 初始化一个空列表，用于存储每个页面的语言
    language_lst = []
    # 遍历模型列表中的每个 OCR 页面信息
    for ocr_page_info in model_list:
        # 初始化页面文本为空字符串
        page_text = ""
        # 获取当前页面的布局检测结果
        layout_dets = ocr_page_info["layout_dets"]
        # 遍历当前页面的所有布局检测结果
        for layout_det in layout_dets:
            # 获取布局的类别 ID
            category_id = layout_det["category_id"]
            # 定义允许的类别 ID 列表
            allow_category_id_list = [15]
            # 如果当前类别 ID 在允许列表中
            if category_id in allow_category_id_list:
                # 将该布局的文本添加到页面文本中
                page_text += layout_det["text"]
        # 使用 detect_lang 函数检测页面文本的语言
        page_language = detect_lang(page_text)
        # 将检测出的语言添加到语言列表中
        language_lst.append(page_language)
    # 统计 language_lst 中每种语言的个数
    count_dict = Counter(language_lst)
    # 找出出现次数最多的语言
    language = max(count_dict, key=count_dict.get)
    # 返回出现次数最多的语言
    return language
```