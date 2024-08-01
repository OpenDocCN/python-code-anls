# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\pages\prompt-f0ec387782dbed90.js`

```py
# 定义一个函数，接收一个参数 text，用于生成一个包含该参数字符串的 HTML 段落标签的字符串
def paragraph(text):
    # 使用 f-string 构建包含 text 参数的 HTML 段落标签字符串，保存到变量 para 中
    para = f"<p>{text}</p>"
    # 返回构建好的 HTML 段落标签字符串
    return para

# 调用函数 paragraph，将字符串 "Hello, World!" 传入作为参数，将返回的结果保存到变量 html_para 中
html_para = paragraph("Hello, World!")
# 打印变量 html_para 中保存的 HTML 段落标签字符串到控制台
print(html_para)
```