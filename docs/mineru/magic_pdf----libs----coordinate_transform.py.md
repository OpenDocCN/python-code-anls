# `.\MinerU\magic_pdf\libs\coordinate_transform.py`

```
# 根据模型页面信息和页面对象获取缩放比例
def get_scale_ratio(model_page_info, page):
    # 生成页面的位图表示，分辨率为 72 dpi
    pix = page.get_pixmap(dpi=72)
    # 获取位图的宽度并转换为整数
    pymu_width = int(pix.w)
    # 获取位图的高度并转换为整数
    pymu_height = int(pix.h)
    # 从模型页面信息中提取页面宽度
    width_from_json = model_page_info['page_info']['width']
    # 从模型页面信息中提取页面高度
    height_from_json = model_page_info['page_info']['height']
    # 计算水平缩放比例
    horizontal_scale_ratio = width_from_json / pymu_width
    # 计算垂直缩放比例
    vertical_scale_ratio = height_from_json / pymu_height
    # 返回水平和垂直缩放比例
    return horizontal_scale_ratio, vertical_scale_ratio
```