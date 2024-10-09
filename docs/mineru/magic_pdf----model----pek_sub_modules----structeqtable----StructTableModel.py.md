# `.\MinerU\magic_pdf\model\pek_sub_modules\structeqtable\StructTableModel.py`

```
# 从 struct_eqtable.model 模块导入 StructTable 类
from struct_eqtable.model import StructTable
# 从 pypandoc 模块导入 convert_text 函数
from pypandoc import convert_text

# 定义一个名为 StructTableModel 的类
class StructTableModel:
    # 类的初始化方法，接受模型路径、最大新令牌、最大处理时间和设备类型
    def __init__(self, model_path, max_new_tokens=2048, max_time=400, device = 'cpu'):
        # 初始化
        self.model_path = model_path  # 保存模型路径
        self.max_new_tokens = max_new_tokens # 最大输出令牌长度
        self.max_time = max_time # 处理超时（秒）
        # 如果设备是 'cuda'，则在 GPU 上加载模型
        if device == 'cuda':
            self.model = StructTable(self.model_path, self.max_new_tokens, self.max_time).cuda()
        # 否则在 CPU 上加载模型
        else:
            self.model = StructTable(self.model_path, self.max_new_tokens, self.max_time)

    # 定义一个方法，将图像转换为 LaTeX 表格
    def image2latex(self, image) -> str:
        table_latex = self.model.forward(image)  # 使用模型处理图像，返回 LaTeX 表格
        return table_latex  # 返回 LaTeX 表格字符串

    # 定义一个方法，将图像转换为 HTML 表格
    def image2html(self, image) -> str:
        table_latex = self.image2latex(image)  # 调用 image2latex 方法获取 LaTeX 表格
        table_html = convert_text(table_latex, 'html', format='latex')  # 将 LaTeX 转换为 HTML
        return table_html  # 返回 HTML 表格字符串
```