# `.\marker\marker\models.py`

```py
# 从 marker.cleaners.equations 模块中导入 load_texify_model 函数
from marker.cleaners.equations import load_texify_model
# 从 marker.ordering 模块中导入 load_ordering_model 函数
from marker.ordering import load_ordering_model
# 从 marker.postprocessors.editor 模块中导入 load_editing_model 函数
from marker.postprocessors.editor import load_editing_model
# 从 marker.segmentation 模块中导入 load_layout_model 函数
from marker.segmentation import load_layout_model

# 定义一个函数用于加载所有模型
def load_all_models():
    # 调用 load_editing_model 函数，加载编辑模型
    edit = load_editing_model()
    # 调用 load_ordering_model 函数，加载排序模型
    order = load_ordering_model()
    # 调用 load_layout_model 函数，加载布局模型
    layout = load_layout_model()
    # 调用 load_texify_model 函数，加载 TeXify 模型
    texify = load_texify_model()
    # 将加载的模型按顺序存储在列表中
    model_lst = [texify, layout, order, edit]
    # 返回模型列表
    return model_lst
```