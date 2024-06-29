# `D:\src\scipysrc\matplotlib\tools\visualize_tests.py`

```py
# 解析命令行参数模块
import argparse
# 操作系统相关的功能模块
import os
# 提供默认字典功能的模块
from collections import defaultdict

# 非 PNG 图像文件扩展名列表
NON_PNG_EXTENSIONS = ['pdf', 'svg', 'eps']

# HTML 模板，用于生成显示 Matplotlib 测试结果的网页
html_template = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Matplotlib test result visualization</title>
<style media="screen">
img{{
  width:100%;
  max-width:800px;
}}
</style>
</head><body>
{failed}
{body}
</body></html>
"""

# 子目录的 HTML 模板，用于显示每个子目录下的测试结果
subdir_template = """<h2>{subdir}</h2><table>
<thead><tr><th>name</th><th>actual</th><th>expected</th><th>diff</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
"""

# 仅显示失败的测试结果的 HTML 模板
failed_template = """<h2>Only Failed</h2><table>
<thead><tr><th>name</th><th>actual</th><th>expected</th><th>diff</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
"""

# 表格行的模板，显示每个测试结果条目的信息
row_template = ('<tr>'
                '<td>{0}{1}</td>'
                '<td>{2}</td>'
                '<td><a href="{3}"><img src="{3}"></a></td>'
                '<td>{4}</td>'
                '</tr>')

# 链接图像的模板，用于生成带链接的图像显示
linked_image_template = '<a href="{0}"><img src="{0}"></a>'


def run(show_browser=True):
    """
    构建一个用于视觉比较的网站页面
    """
    # 图像存放的目录
    image_dir = "result_images"
    # 获取所有子目录的生成器，每个子目录是一个目录名
    _subdirs = (name
                for name in os.listdir(image_dir)
                if os.path.isdir(os.path.join(image_dir, name)))

    # 用于存放失败测试结果行的列表
    failed_rows = []
    # 用于存放主体部分 HTML 的列表
    body_sections = []
    for subdir in sorted(_subdirs):
        # 对子目录进行排序遍历
        if subdir == "test_compare_images":
            # 如果子目录是"test_compare_images"，跳过这个特定的子目录
            # 这些图片用于测试图像比较函数。
            continue

        pictures = defaultdict(dict)
        # 创建一个默认字典来存储图片信息

        for file in os.listdir(os.path.join(image_dir, subdir)):
            # 遍历当前子目录下的所有文件
            if os.path.isdir(os.path.join(image_dir, subdir, file)):
                # 如果是目录则跳过
                continue
            fn, fext = os.path.splitext(file)
            # 获取文件名和扩展名

            if fext != ".png":
                # 如果不是PNG文件，则跳过
                continue

            if "-failed-diff" in fn:
                # 如果文件名中包含"-failed-diff"
                file_type = 'diff'
                test_name = fn[:-len('-failed-diff')]
            elif "-expected" in fn:
                # 如果文件名中包含"-expected"
                for ext in NON_PNG_EXTENSIONS:
                    if fn.endswith(f'_{ext}'):
                        display_extension = f'_{ext}'
                        extension = ext
                        fn = fn[:-len(display_extension)]
                        break
                else:
                    display_extension = ''
                    extension = 'png'
                file_type = 'expected'
                test_name = fn[:-len('-expected')] + display_extension
            else:
                # 否则认为是实际生成的图片
                file_type = 'actual'
                test_name = fn

            # 将图片路径添加到字典中，使用"/"作为URL分隔符
            pictures[test_name][file_type] = '/'.join((subdir, file))

        subdir_rows = []
        # 存储当前子目录的行信息

        for name, test in sorted(pictures.items()):
            # 对每个测试的图片进行排序
            expected_image = test.get('expected', '')
            actual_image = test.get('actual', '')

            if 'diff' in test:
                # 如果存在差异图片
                status = " (failed)"
                failed = f'<a href="{test["diff"]}">diff</a>'
                current = linked_image_template.format(actual_image)
                failed_rows.append(row_template.format(name, "", current,
                                                       expected_image, failed))
            elif 'actual' not in test:
                # 如果实际图片不存在，测试失败
                status = " (failed)"
                failed = '--'
                current = '(Failure in test, no image produced)'
                failed_rows.append(row_template.format(name, "", current,
                                                       expected_image, failed))
            else:
                # 否则测试通过
                status = " (passed)"
                failed = '--'
                current = linked_image_template.format(actual_image)

            # 将当前测试的行信息添加到当前子目录的行列表中
            subdir_rows.append(row_template.format(name, status, current,
                                                   expected_image, failed))

        # 将当前子目录的表格形式添加到整体内容中
        body_sections.append(
            subdir_template.format(subdir=subdir, rows='\n'.join(subdir_rows)))

    if failed_rows:
        # 如果有失败的测试行，则生成失败的表格
        failed = failed_template.format(rows='\n'.join(failed_rows))
    else:
        failed = ''
    # 拼接所有子目录的表格内容
    body = ''.join(body_sections)
    # 将所有内容放入HTML模板中
    html = html_template.format(failed=failed, body=body)
    # 构建 index.html 文件的完整路径
    index = os.path.join(image_dir, "index.html")
    
    # 打开 index.html 文件以写入模式，并写入生成的 HTML 内容
    with open(index, "w") as f:
        f.write(html)
    
    # 根据 show_browser 变量确定是否显示浏览器窗口
    show_message = not show_browser
    
    # 如果需要显示浏览器窗口
    if show_browser:
        try:
            # 尝试导入并打开系统默认浏览器显示 index.html
            import webbrowser
            webbrowser.open(index)
        except Exception:
            # 若出现异常（无法打开浏览器），则设置显示消息标志为 True
            show_message = True
    
    # 若需要显示消息（无法打开浏览器或未设置显示浏览器）
    if show_message:
        # 打印提示信息，指示用户在浏览器中打开 index.html 进行可视化比较
        print(f"Open {index} in a browser for a visual comparison.")
# 如果该脚本作为主程序运行（而非作为模块被导入），则执行以下代码块
if __name__ == '__main__':
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个命令行参数 --no-browser，如果设置则将 action 设为 'store_true'
    # 这意味着当使用 --no-browser 参数时，args.no_browser 的值为 True
    parser.add_argument('--no-browser', action='store_true',
                        help="Don't show browser after creating index page.")
    # 解析命令行参数，并将结果存储在 args 对象中
    args = parser.parse_args()
    # 调用 run 函数，并根据 --no-browser 参数决定是否显示浏览器
    run(show_browser=not args.no_browser)
```