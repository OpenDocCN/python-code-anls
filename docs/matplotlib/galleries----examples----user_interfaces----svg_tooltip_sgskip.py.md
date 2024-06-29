# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\svg_tooltip_sgskip.py`

```py
# 导入所需的模块和库
from io import BytesIO  # 导入字节流处理模块
import xml.etree.ElementTree as ET  # 导入 XML 解析模块
import matplotlib.pyplot as plt  # 导入 matplotlib 绘图库

# 注册 SVG 命名空间
ET.register_namespace("", "http://www.w3.org/2000/svg")

# 创建 matplotlib 图形和坐标轴
fig, ax = plt.subplots()

# 创建要添加提示信息的图形对象
rect1 = plt.Rectangle((10, -20), 10, 5, fc='blue')  # 创建蓝色矩形
rect2 = plt.Rectangle((-20, 15), 10, 5, fc='green')  # 创建绿色矩形

shapes = [rect1, rect2]  # 所有的图形对象
labels = ['This is a blue rectangle.', 'This is a green rectangle']  # 对应每个图形的标签文本

# 遍历图形对象和标签
for i, (item, label) in enumerate(zip(shapes, labels)):
    patch = ax.add_patch(item)  # 添加图形对象到坐标轴
    # 创建标注对象并设置其样式和位置
    annotate = ax.annotate(labels[i], xy=item.get_xy(), xytext=(0, 0),
                           textcoords='offset points', color='w', ha='center',
                           fontsize=8, bbox=dict(boxstyle='round, pad=.5',
                                                 fc=(.1, .1, .1, .92),
                                                 ec=(1., 1., 1.), lw=1,
                                                 zorder=1))
    ax.add_patch(patch)  # 添加图形对象到坐标轴
    patch.set_gid(f'mypatch_{i:03d}')  # 设置图形对象的全局 ID
    annotate.set_gid(f'mytooltip_{i:03d}')  # 设置标注对象的全局 ID

# 设置图形的坐标轴范围和比例
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.set_aspect('equal')

f = BytesIO()  # 创建一个字节流对象用于存储图形
plt.savefig(f, format="svg")  # 将图形保存为 SVG 格式到字节流对象中

# --- 添加交互性 ---

# 从 SVG 文件创建 XML 树
tree, xmlid = ET.XMLID(f.getvalue())
tree.set('onload', 'init(event)')  # 设置 SVG 树的 onload 属性

# 遍历所有图形对象
for i in shapes:
    index = shapes.index(i)  # 获取图形对象在列表中的索引
    tooltip = xmlid[f'mytooltip_{index:03d}']  # 获取对应索引的提示信息对象
    tooltip.set('visibility', 'hidden')  # 设置提示信息对象的可见性为隐藏
    mypatch = xmlid[f'mypatch_{index:03d}']  # 获取对应索引的图形对象
    # 设置图形对象的鼠标悬停和移出事件回调函数
    mypatch.set('onmouseover', "ShowTooltip(this)")
    mypatch.set('onmouseout', "HideTooltip(this)")

# 定义 ShowTooltip 和 HideTooltip 函数的 JavaScript 脚本
script = """
    <script type="text/ecmascript">
    <![CDATA[

    function init(event) {
        if ( window.svgDocument == null ) {
            svgDocument = event.target.ownerDocument;
            }
        }

    function ShowTooltip(obj) {
        var cur = obj.id.split("_")[1];  // 提取当前对象的编号
        var tip = svgDocument.getElementById('mytooltip_' + cur);  // 获取对应编号的提示信息对象
        tip.setAttribute('visibility', "visible")  // 设置提示信息对象可见
        }
    # 定义一个 JavaScript 函数，用于隐藏工具提示。
    function HideTooltip(obj) {
        # 从对象的 id 属性中提取出编号部分，使用下划线 "_" 分割后取第二部分作为 cur
        var cur = obj.id.split("_")[1];
        # 根据 cur 构建工具提示元素的 id，使用 svgDocument.getElementById 方法获取该元素
        var tip = svgDocument.getElementById('mytooltip_' + cur);
        # 设置获取到的工具提示元素的 visibility 属性为 "hidden"，以隐藏工具提示
        tip.setAttribute('visibility', "hidden")
    }
# 在 XML 树的顶部插入脚本，并保存到文件。
# tree 是一个 XML 元素列表，insert 方法用于在指定位置插入一个元素。
tree.insert(0, ET.XML(script))

# 使用 ElementTree 库创建一个 XML 元素树对象，并将其写入到名为 'svg_tooltip.svg' 的 SVG 文件中。
# ET.ElementTree(tree) 创建了一个 ElementTree 对象，write 方法将该对象写入到文件。
ET.ElementTree(tree).write('svg_tooltip.svg')
```