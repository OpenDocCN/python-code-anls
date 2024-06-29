# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\web_application_server_sgskip.py`

```
# 导入需要的库
import base64              # 导入 base64 编码解码库
from io import BytesIO     # 导入字节流处理库中的 BytesIO 类

from flask import Flask    # 导入 Flask 框架

from matplotlib.figure import Figure   # 导入 Matplotlib 的 Figure 类


app = Flask(__name__)  # 创建一个 Flask 应用实例


@app.route("/")
def hello():
    # 生成图形，不使用 pyplot
    fig = Figure()       # 创建一个空的 Figure 对象
    ax = fig.subplots()  # 在 Figure 对象上创建一个子图
    ax.plot([1, 2])      # 在子图上绘制简单的曲线

    # 将图形保存到临时缓冲区
    buf = BytesIO()              # 创建一个 BytesIO 对象
    fig.savefig(buf, format="png")  # 将图形保存为 PNG 格式到缓冲区

    # 将图形数据转换为 base64 编码的字符串
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    # 将图像嵌入到 HTML 输出中
    return f"<img src='data:image/png;base64,{data}'/>"

# %%
#
# 由于上述代码是一个 Flask 应用程序，应使用
# `flask 命令行工具 <https://flask.palletsprojects.com/en/latest/cli/>`_
# 假设工作目录包含此脚本：
#
# Unix-like 系统
#
# .. code-block:: console
#
#  FLASK_APP=web_application_server_sgskip flask run
#
# Windows
#
# .. code-block:: console
#
#  set FLASK_APP=web_application_server_sgskip
#  flask run
#
#
# 用于 HTML 的可点击图像
# -------------------------
#
# Andrew Dalke of `Dalke Scientific <http://www.dalkescientific.com>`_
# 写了一篇很棒的 `文章
# <http://www.dalkescientific.com/writings/diary/archive/2005/04/24/interactive_html.html>`_
# 关于如何使用 Matplotlib 的 agg PNG 创建 HTML 点击映射。我们也希望将此功能添加到 SVG 中。
# 如果您有兴趣为这些工作做出贡献，那将是很棒的。
```