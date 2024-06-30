# `D:\src\scipysrc\scikit-learn\doc\sphinxext\dropdown_anchors.py`

```
# 导入正则表达式模块
import re
# 导入docutils库中的nodes模块
from docutils import nodes
# 导入sphinx.transforms.post_transforms模块中的SphinxPostTransform类
from sphinx.transforms.post_transforms import SphinxPostTransform
# 导入sphinx_design.dropdown中的dropdown_main和dropdown_title函数
from sphinx_design.dropdown import dropdown_main, dropdown_title

# 定义一个继承自SphinxPostTransform的类DropdownAnchorAdder
class DropdownAnchorAdder(SphinxPostTransform):
    """Insert anchor links to the sphinx-design dropdowns.

    Some of the dropdowns were originally headers that had automatic anchors, so we
    need to make sure that the old anchors still work. See the original implementation
    (in JS): https://github.com/scikit-learn/scikit-learn/pull/27409

    The structure of each sphinx-design dropdown node is expected to be:

    <dropdown_main ...>
        <dropdown_title ...>
            ...icon      <-- This exists if the "icon" option of the sphinx-design
                             dropdown is set; we do not use it in our documentation

            ...title     <-- This may contain multiple nodes, e.g. literal nodes if
                             there are inline codes; we use the concatenated text of
                             all these nodes to generate the anchor ID

            Here we insert the anchor link!

            <container ...>  <-- The "dropdown closed" marker
            <container ...>  <-- The "dropdown open" marker
        </dropdown_title>
        <container...>
            ...main contents
        </container>
    </dropdown_main>
    """

    # 设置默认处理优先级，优先级为9999，确保在所有其他转换后应用
    default_priority = 9999
    # 指定支持的文档格式为html
    formats = ["html"]
    def run(self):
        """
        Run the post transformation.
        """
        # Counter to store the duplicated summary text to add it as a suffix in the
        # anchor ID
        anchor_id_counters = {}

        # Iterate through all dropdown elements in the document
        for sd_dropdown in self.document.findall(dropdown_main):
            # Grab the dropdown title
            sd_dropdown_title = sd_dropdown.next_node(dropdown_title)

            # Concatenate the text of relevant nodes as the title text
            # Since we do not have the prefix icon, the relevant nodes are the very
            # first child node until the third last node (last two are markers)
            title_text = "".join(
                node.astext() for node in sd_dropdown_title.children[:-2]
            )

            # Generate anchor ID from the first line of title text
            anchor_id = re.sub(r"\s+", "-", title_text.strip().split("\n")[0]).lower()
            
            # Check if anchor ID already exists in counters; increment counter if so
            if anchor_id in anchor_id_counters:
                anchor_id_counters[anchor_id] += 1
                anchor_id = f"{anchor_id}-{anchor_id_counters[anchor_id]}"
            else:
                anchor_id_counters[anchor_id] = 1
            
            # Append the generated anchor ID to the 'ids' attribute of sd_dropdown
            sd_dropdown["ids"].append(anchor_id)

            # Create the anchor element and insert it after the title text
            # This is done directly with raw HTML
            anchor_html = (
                f'<a class="headerlink" href="#{anchor_id}" '
                'title="Link to this dropdown">#</a>'
            )
            anchor_node = nodes.raw("", anchor_html, format="html")
            sd_dropdown_title.insert(-2, anchor_node)  # before the two markers
# 定义一个名为 setup 的函数，用于设置应用程序的配置
def setup(app):
    # 调用 app 对象的 add_post_transform 方法，将 DropdownAnchorAdder 类添加为后置转换处理器
    app.add_post_transform(DropdownAnchorAdder)
```