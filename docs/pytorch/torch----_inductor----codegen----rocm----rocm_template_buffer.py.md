# `.\pytorch\torch\_inductor\codegen\rocm\rocm_template_buffer.py`

```
# 导入必要的模块，允许未类型化的定义
# mypy: allow-untyped-defs
from ...ir import TemplateBuffer

# 定义 ROCmTemplateBuffer 类，继承自 TemplateBuffer 类
class ROCmTemplateBuffer(TemplateBuffer):
    # 初始化方法，接收布局(layout)、输入(inputs)、生成内核渲染函数(make_kernel_render)、工作空间大小(workspace_size)和模板(template)
    def __init__(
        self,
        layout,
        inputs,
        make_kernel_render,
        workspace_size: int,
        template: "ROCmTemplate",  # 声明 template 参数为 ROCmTemplate 类型，忽略类型检查 # noqa: F821
    ):
        # 调用父类 TemplateBuffer 的初始化方法
        super().__init__(layout, inputs, make_kernel_render)
        # 设置实例变量 self.workspace_size 为传入的 workspace_size
        # 表示此模板所需的全局内存大小（以字节为单位）
        self.workspace_size = workspace_size
        # 设置实例变量 self.template 为传入的 template
        self.template = template

    # 定义方法 get_workspace_size，返回实例的工作空间大小
    def get_workspace_size(self):
        # 如果 self.workspace_size 不为 None，则返回 self.workspace_size，否则返回 0
        return self.workspace_size if self.workspace_size is not None else 0
```