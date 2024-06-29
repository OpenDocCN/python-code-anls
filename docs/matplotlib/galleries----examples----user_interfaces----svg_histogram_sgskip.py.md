# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\svg_histogram_sgskip.py`

```
# 导入需要的模块
from io import BytesIO  # 导入BytesIO类，用于创建内存中的二进制数据流
import json  # 导入json模块，用于处理JSON格式数据
import xml.etree.ElementTree as ET  # 导入xml.etree.ElementTree模块，用于处理XML数据

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图
import numpy as np  # 导入numpy模块，用于数值计算

plt.rcParams['svg.fonttype'] = 'none'  # 设置matplotlib绘图时SVG字体的类型为none，确保字体不嵌入SVG中

# 注册命名空间，避免在XML中使用默认命名空间
ET.register_namespace("", "http://www.w3.org/2000/svg")

# 设置随机种子以保证图像可复现性
np.random.seed(19680801)

# --- 创建直方图、图例和标题 ---
plt.figure()  # 创建一个新的图形对象
r = np.random.randn(100)  # 生成100个符合标准正态分布的随机数
r1 = r + 1  # 将r中的每个元素加1，形成新的随机数数组
labels = ['Rabbits', 'Frogs']  # 定义直方图的标签
H = plt.hist([r, r1], label=labels)  # 绘制直方图，并将返回值存储在H中
containers = H[-1]  # 获取直方图中的容器（patches）
leg = plt.legend(frameon=False)  # 创建图例，不显示框架
plt.title("From a web browser, click on the legend\n"
          "marker to toggle the corresponding histogram.")  # 设置图形的标题

# --- 为将要修改的SVG对象添加ids ---

hist_patches = {}  # 创建一个空字典，用于存储直方图的patches
for ic, c in enumerate(containers):  # 遍历每个容器（patches）
    hist_patches[f'hist_{ic}'] = []  # 为每个容器创建一个空列表
    for il, element in enumerate(c):  # 遍历每个容器中的元素（patches）
        element.set_gid(f'hist_{ic}_patch_{il}')  # 为每个元素设置全局唯一的id属性
        hist_patches[f'hist_{ic}'].append(f'hist_{ic}_patch_{il}')  # 将id添加到对应的hist_patches中

# 为图例的patches设置ids
for i, t in enumerate(leg.get_patches()):  # 遍历图例中的patches
    t.set_gid(f'leg_patch_{i}')  # 为每个patch设置全局唯一的id属性

# 为图例的texts设置ids
for i, t in enumerate(leg.get_texts()):  # 遍历图例中的texts
    t.set_gid(f'leg_text_{i}')  # 为每个text设置全局唯一的id属性

# 将SVG保存在一个虚拟的文件对象中
f = BytesIO()
plt.savefig(f, format="svg")  # 将当前图形保存为SVG格式，并写入到BytesIO对象f中

# 从SVG文件内容创建XML树
tree, xmlid = ET.XMLID(f.getvalue())  # 使用ET从BytesIO对象f中的SVG内容创建XML树，同时获取id属性

# --- 添加交互性 ---

# 为patch对象添加属性
for i, t in enumerate(leg.get_patches()):  # 再次遍历图例中的patches
    el = xmlid[f'leg_patch_{i}']  # 获取对应id的XML元素
    el.set('cursor', 'pointer')  # 设置鼠标悬停样式为手型
    el.set('onclick', "toggle_hist(this)")  # 设置点击事件，调用JavaScript函数toggle_hist(this)

# 为text对象添加属性
for i, t in enumerate(leg.get_texts()):  # 再次遍历图例中的texts
    el = xmlid[f'leg_text_{i}']  # 获取对应id的XML元素
    el.set('cursor', 'pointer')  # 设置鼠标悬停样式为手型
    el.set('onclick', "toggle_hist(this)")


# 设置元素 el 的 onclick 事件处理函数为 toggle_hist(this)
# 创建脚本定义函数 `toggle_hist`。
# 创建一个全局变量 `container`，用于存储每个直方图的补丁 ID。
# 然后定义函数 "toggle_element"，用于设置每个直方图的所有补丁和标记本身的可见性属性以及标记本身的不透明度。

script = """
<script type="text/ecmascript">
<![CDATA[
var container = %s

function toggle(oid, attribute, values) {
    /* Toggle the style attribute of an object between two values.

    Parameters
    ----------
    oid : str
      Object identifier.
    attribute : str
      Name of style attribute.
    values : [on state, off state]
      The two values that are switched between.
    */
    var obj = document.getElementById(oid);
    var a = obj.style[attribute];

    // 根据当前状态切换样式属性的值
    a = (a == values[0] || a == "") ? values[1] : values[0];
    obj.style[attribute] = a;
    }

function toggle_hist(obj) {

    var num = obj.id.slice(-1);

    // 切换补丁和文本的不透明度
    toggle('leg_patch_' + num, 'opacity', [1, 0.3]);
    toggle('leg_text_' + num, 'opacity', [1, 0.5]);

    var names = container['hist_'+num]

    // 遍历当前直方图的所有补丁，切换其不透明度
    for (var i=0; i < names.length; i++) {
        toggle(names[i], 'opacity', [1, 0])
    };
    }
]]>
</script>
""" % json.dumps(hist_patches)

# 添加过渡效果到 CSS 中
css = tree.find('.//{http://www.w3.org/2000/svg}style')
css.text = css.text + "g {-webkit-transition:opacity 0.4s ease-out;" + \
    "-moz-transition:opacity 0.4s ease-out;}"

# 在文档树中插入脚本并保存到文件中
tree.insert(0, ET.XML(script))

# 将修改后的文档树写入 SVG 文件
ET.ElementTree(tree).write("svg_histogram.svg")
```