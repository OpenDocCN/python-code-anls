# `stable-diffusion-webui\scripts\custom_code.py`

```py
# 导入自定义模块 scripts
import modules.scripts as scripts
# 导入 gradio 库并重命名为 gr
import gradio as gr
# 导入 ast 模块
import ast
# 导入 copy 模块
import copy

# 从 processing 模块中导入 Processed 类
from modules.processing import Processed
# 从 shared 模块中导入 cmd_opts 对象
from modules.shared import cmd_opts

# 定义函数，将表达式转换为表达式对象
def convertExpr2Expression(expr):
    # 设置表达式的行号和列偏移为 0
    expr.lineno = 0
    expr.col_offset = 0
    # 创建一个新的表达式对象，行号和列偏移也为 0
    result = ast.Expression(expr.value, lineno=0, col_offset=0)

    return result

# 定义函数，类似于 exec() 但可以返回值
def exec_with_return(code, module):
    """
    like exec() but can return values
    https://stackoverflow.com/a/52361938/5862977
    """
    # 解析代码字符串为 AST
    code_ast = ast.parse(code)

    # 复制 AST 对象
    init_ast = copy.deepcopy(code_ast)
    init_ast.body = code_ast.body[:-1]

    last_ast = copy.deepcopy(code_ast)
    last_ast.body = code_ast.body[-1:]

    # 执行初始化部分的 AST
    exec(compile(init_ast, "<ast>", "exec"), module.__dict__)
    # 如果最后一个 AST 是表达式，则返回其值
    if type(last_ast.body[0]) == ast.Expr:
        return eval(compile(convertExpr2Expression(last_ast.body[0]), "<ast>", "eval"), module.__dict__)
    else:
        # 否则执行最后一个 AST
        exec(compile(last_ast, "<ast>", "exec"), module.__dict__)

# 定义 Script 类，继承自 scripts.Script
class Script(scripts.Script):

    # 定义 title 方法，返回 "Custom code"
    def title(self):
        return "Custom code"

    # 定义 show 方法，根据 is_img2img 返回 cmd_opts.allow_code
    def show(self, is_img2img):
        return cmd_opts.allow_code

    # 定义 ui 方法，根据 is_img2img 返回 UI 元素列表
    def ui(self, is_img2img):
        # 设置示例代码
        example = """from modules.processing import process_images

p.width = 768
p.height = 768
p.batch_size = 2
p.steps = 10

return process_images(p)
"""

        # 创建 Code UI 元素
        code = gr.Code(value=example, language="python", label="Python code", elem_id=self.elem_id("code"))
        # 创建 Number UI 元素
        indent_level = gr.Number(label='Indent level', value=2, precision=0, elem_id=self.elem_id("indent_level"))

        # 返回 UI 元素列表
        return [code, indent_level]
    # 定义一个方法，用于执行给定的代码块
    def run(self, p, code, indent_level):
        # 断言 --allow-code 选项已启用
        assert cmd_opts.allow_code, '--allow-code option must be enabled'

        # 初始化用于显示结果的数据
        display_result_data = [[], -1, ""]

        # 定义一个显示结果的方法，更新显示结果数据
        def display(imgs, s=display_result_data[1], i=display_result_data[2]):
            display_result_data[0] = imgs
            display_result_data[1] = s
            display_result_data[2] = i

        # 导入 ModuleType 类
        from types import ModuleType
        # 创建一个名为 "testmodule" 的模块对象
        module = ModuleType("testmodule")
        # 将全局变量添加到模块对象的字典中
        module.__dict__.update(globals())
        # 将参数 p 和显示方法 display 添加到模块对象中
        module.p = p
        module.display = display

        # 根据缩进级别生成缩进字符串
        indent = " " * indent_level
        # 将代码块中的换行符替换为带有缩进的换行符
        indented = code.replace('\n', f"\n{indent}")
        # 构建一个包含代码块的函数体字符串
        body = f"""def __webuitemp__():
{indent}{indented}
__webuitemp__()"""
        # 调用 __webuitemp__ 函数
        result = exec_with_return(body, module)
        # 如果结果是 Processed 类型，则直接返回结果
        if isinstance(result, Processed):
            return result
        # 否则，返回一个包含 p 和 display_result_data 的 Processed 对象
        return Processed(p, *display_result_data)
```