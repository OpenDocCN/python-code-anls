# `.\PaddleOCR\tools\end2end\draw_html.py`

```
# 导入所需的库
import os
import argparse

# 将字符串转换为布尔值
def str2bool(v):
    return v.lower() in ("true", "t", "1")

# 初始化命令行参数解析器
def init_args():
    parser = argparse.ArgumentParser()
    # 添加命令行参数：图像文件夹路径，默认为空字符串
    parser.add_argument("--image_dir", type=str, default="")
    # 添加命令行参数：保存 HTML 文件路径，默认为 "./default.html"
    parser.add_argument("--save_html_path", type=str, default="./default.html")
    # 添加命令行参数：图像宽度，默认为 640
    parser.add_argument("--width", type=int, default=640)
    return parser

# 解析命令行参数
def parse_args():
    parser = init_args()
    return parser.parse_args()

# 绘制调试图像
def draw_debug_img(args):

    # 获取保存 HTML 文件路径
    html_path = args.save_html_path

    # 初始化错误计数器
    err_cnt = 0
    # 使用写入模式打开 HTML 文件
    with open(html_path, 'w') as html:
        # 写入 HTML 标签和表格标签
        html.write('<html>\n<body>\n')
        html.write('<table border="1">\n')
        # 写入 HTML meta 标签，指定字符编码为 utf-8
        html.write("<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />")
        # 初始化图片列表和图片路径
        image_list = []
        path = args.image_dir
        # 遍历指定目录下的文件
        for i, filename in enumerate(sorted(os.listdir(path))):
            # 如果文件名以 "txt" 结尾，则跳过
            if filename.endswith("txt"): continue
            # 构建图片路径
            base = "{}/{}".format(path, filename)
            # 写入 HTML 表格行标签
            html.write("<tr>\n")
            # 写入文件名和 GT 标签
            html.write(f'<td> {filename}\n GT')
            # 写入 GT 图片标签
            html.write(f'<td>GT\n<img src="{base}" width={args.width}></td>')
            # 关闭表格行标签
            html.write("</tr>\n")
        # 写入 CSS 样式
        html.write('<style>\n')
        html.write('span {\n')
        html.write('    color: red;\n')
        html.write('}\n')
        html.write('</style>\n')
        # 写入表格和 HTML 结尾标签
        html.write('</table>\n')
        html.write('</html>\n</body>\n')
    # 打印 HTML 文件保存路径
    print(f"The html file saved in {html_path}")
    # 返回函数
    return
# 如果当前脚本被直接执行而非被导入，则执行以下代码块
if __name__ == "__main__":

    # 解析命令行参数并将其存储在args变量中
    args = parse_args()

    # 调用draw_debug_img函数，传入args参数
    draw_debug_img(args)
```