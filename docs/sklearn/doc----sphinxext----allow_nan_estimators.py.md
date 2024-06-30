# `D:\src\scipysrc\scikit-learn\doc\sphinxext\allow_nan_estimators.py`

```
from contextlib import suppress
# 导入上下文管理模块的 suppress 函数，用于忽略指定异常

from docutils import nodes
# 导入 docutils 库中的 nodes 模块，用于操作和生成文档树节点

from docutils.parsers.rst import Directive
# 导入 docutils 库中的 rst 解析器模块中的 Directive 类，用于定义自定义的 reStructuredText 指令

from sklearn.utils import all_estimators
# 导入 sklearn 库中的 all_estimators 函数，用于获取所有的估计器类

from sklearn.utils._testing import SkipTest
# 导入 sklearn 库中的 SkipTest 类，用于测试跳过异常的处理

from sklearn.utils.estimator_checks import _construct_instance
# 导入 sklearn 库中的 _construct_instance 函数，用于构造估计器类的实例

class AllowNanEstimators(Directive):
    @staticmethod
    def make_paragraph_for_estimator_type(estimator_type):
        # 创建一个新的列表项节点
        intro = nodes.list_item()
        # 添加强调文本节点，说明允许 NaN 值的估计器类型
        intro += nodes.strong(text="Estimators that allow NaN values for type ")
        # 添加文本字面值节点，显示估计器类型的名称
        intro += nodes.literal(text=f"{estimator_type}")
        # 添加强调文本节点，换行
        intro += nodes.strong(text=":\n")
        exists = False
        # 创建一个新的项目列表节点
        lst = nodes.bullet_list()
        # 遍历所有指定类型估计器的名称和类
        for name, est_class in all_estimators(type_filter=estimator_type):
            # 忽略 SkipTest 异常
            with suppress(SkipTest):
                # 构造当前估计器类的实例
                est = _construct_instance(est_class)

            # 如果当前估计器实例允许 NaN 值
            if est._get_tags().get("allow_nan"):
                # 获取模块名称，并生成链接到类文档的 URL
                module_name = ".".join(est_class.__module__.split(".")[:2])
                class_title = f"{est_class.__name__}"
                class_url = f"./generated/{module_name}.{class_title}.html"
                # 创建新的列表项节点
                item = nodes.list_item()
                # 创建新的段落节点
                para = nodes.paragraph()
                # 添加指向类文档的外部引用节点
                para += nodes.reference(
                    class_title, text=class_title, internal=False, refuri=class_url
                )
                exists = True
                # 添加段落到列表项中
                item += para
                # 添加列表项到项目列表中
                lst += item
        # 添加项目列表到介绍节点中
        intro += lst
        # 返回包含结果节点的列表，如果没有找到允许 NaN 值的估计器则返回 None
        return [intro] if exists else None

    def run(self):
        # 创建新的项目列表节点
        lst = nodes.bullet_list()
        # 遍历指定的估计器类型列表
        for i in ["cluster", "regressor", "classifier", "transformer"]:
            # 生成对应类型的段落节点列表或 None
            item = self.make_paragraph_for_estimator_type(i)
            # 如果生成的段落节点不为空
            if item is not None:
                # 将其添加到项目列表中
                lst += item
        # 返回包含结果节点的列表
        return [lst]


def setup(app):
    # 向应用添加允许 NaN 值估计器的自定义指令
    app.add_directive("allow_nan_estimators", AllowNanEstimators)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
```