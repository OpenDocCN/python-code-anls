
# `matplotlib\galleries\examples\subplots_axes_and_figures\gridspec_customization.py` 详细设计文档

这是一个matplotlib示例代码，演示了使用GridSpec创建子图布局，通过width_ratios和height_ratios控制子图的相对大小，以及通过left、right、top、bottom、wspace、hspace参数控制子图之间的间距。

## 整体流程

```mermaid
graph TD
    A[开始] --> B[导入matplotlib.pyplot和GridSpec]
B --> C[定义annotate_axes函数]
C --> D[创建第一个Figure对象]
D --> E[创建GridSpec: 2x2布局, width_ratios=[1,2], height_ratios=[4,1]]
E --> F[使用gs索引创建4个子图: ax1, ax2, ax3, ax4]
F --> G[调用annotate_axes为每个子图添加标注]
G --> H[创建第二个Figure对象]
H --> I[创建第一个GridSpec: 3x3, left=0.05, right=0.48, wspace=0.05]
I --> J[使用gs1创建3个子图: ax1, ax2, ax3]
J --> K[创建第二个GridSpec: 3x3, left=0.55, right=0.98, hspace=0.05]
K --> L[使用gs2创建3个子图: ax4, ax5, ax6]
L --> M[调用annotate_axes为每个子图添加标注]
M --> N[调用plt.show()显示图形]
N --> O[结束]
```

## 类结构

```
matplotlib.pyplot (模块)
├── Figure
│   ├── axes (子图列表)
│   └── suptitle()
├── GridSpec (matplotlib.gridspec)
│   └── 索引访问 (用于获取子图位置)
└── Axes (通过add_subplot创建)
    ├── text()
    └── tick_params()
```

## 全局变量及字段


### `fig`
    
当前图形实例，用于添加子图和设置整体属性

类型：`Figure对象`
    


### `gs`
    
第一个布局规格，定义2x2网格，宽高比为[1,2]和[4,1]

类型：`GridSpec对象`
    


### `gs1`
    
第二个布局规格-左半部分，3x3网格，控制左右间距和垂直间距

类型：`GridSpec对象`
    


### `gs2`
    
第二个布局规格-右半部分，3x3网格，控制水平间距和上下间距

类型：`GridSpec对象`
    


### `ax1`
    
第一个子图axes实例，位于gs布局的左上角

类型：`Axes对象`
    


### `ax2`
    
第二个子图axes实例，位于gs布局的右上角

类型：`Axes对象`
    


### `ax3`
    
第三个子图axes实例，位于gs布局的左下角

类型：`Axes对象`
    


### `ax4`
    
第四个子图axes实例，位于gs布局的右下角

类型：`Axes对象`
    


### `ax5`
    
第五个子图axes实例，位于gs1布局的左下角

类型：`Axes对象`
    


### `ax6`
    
第六个子图axes实例，位于gs2布局的右下角

类型：`Axes对象`
    


    

## 全局函数及方法




### `annotate_axes(fig)`

该函数用于为传入的 matplotlib Figure 对象中的所有子图添加中心文本标签（如 "ax1", "ax2" 等），并隐藏每个子图的 x 轴和 y 轴刻度标签，以便在演示布局时更清晰地标识子图。

参数：

- `fig`：`matplotlib.figure.Figure`，要标注的 Figure 对象，包含一个或多个子图（Axes）。

返回值：`None`，该函数没有返回值。

#### 流程图

```mermaid
graph TD
    A([开始]) --> B[遍历 fig.axes 中的每个子图 ax]
    B --> C{是否还有未处理的子图}
    C -->|是| D[在当前子图 ax 的中心位置添加文本标签 "ax" + (索引+1)]
    D --> E[隐藏当前子图 ax 的 x 轴和 y 轴刻度标签]
    E --> B
    C -->|否| F([结束])
```

#### 带注释源码

```python
def annotate_axes(fig):
    """
    为 Figure 对象中的所有子图添加中心文本标签，并隐藏刻度标签。
    
    参数:
        fig: matplotlib.figure.Figure 对象，包含多个子图。
    """
    # 遍历 Figure 对象中的所有子图（Axes），并获取索引 i 和子图对象 ax
    for i, ax in enumerate(fig.axes):
        # 在每个子图的中心位置（相对坐标 0.5, 0.5）添加文本标签
        # 标签内容为 "ax" 加上当前子图的编号（从 1 开始）
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        # 隐藏子图的 x 轴和 y 轴刻度标签，使布局更简洁
        ax.tick_params(labelbottom=False, labelleft=False)
```



## 关键组件





### GridSpec 网格布局系统

用于定义二维子图布局的规格类，支持通过行列数量、宽度比例、高度比例以及间距参数来精确控制子图的相对大小和位置关系。

### width_ratios 和 height_ratios 相对大小控制

通过列表参数指定各列或各行的相对宽度和高度比例，例如 [1, 2] 表示第一列宽度是第二列的一半。

### wspace 和 hspace 子图间距控制

wspace 控制子图之间的水平间距，hspace 控制垂直间距，以figure的宽高为基准进行归一化设置。

### left/right/top/bottom 边界位置控制

用于设置整个子图区域在figure中的位置边界，left为左边距，right为右边距，bottom为下边距，top为上边距。

### fig.add_subplot 子图创建函数

根据GridSpec的索引规格将子图Axes对象添加到figure中，支持切片语法如 gs[:-1, :] 来创建跨越多个网格区域的子图。

### annotate_axes 辅助标注函数

用于在每个子图中心添加"axN"标签的辅助函数，便于演示和调试时识别各个子图位置。



## 问题及建议





### 已知问题

- **缺少类型注解**：函数参数和返回值没有类型提示，降低了代码可读性和可维护性。
- **硬编码配置值**：子图比例（width_ratios、height_ratios）和间距参数（left、right、wspace、hspace）直接硬编码，难以复用和调整。
- **重复代码模式**：创建子图并添加标注的逻辑在两个figure中重复，可以进一步抽象。
- **图形资源未显式管理**：使用`plt.figure()`创建图形但未显式关闭，可能导致资源泄漏（虽然在交互式环境下影响较小）。
- **魔法注释# %%**：代码中包含Jupyter Notebook的cell分隔符，但实际作为独立脚本运行，注释含义不明确。
- **变量命名可优化**：gs1、gs2、ax1-ax6等命名过于简单，缺乏描述性。

### 优化建议

- **添加类型注解**：
  ```python
  def annotate_axes(fig: plt.Figure) -> None: ...
  def create_subplot_layout(fig: plt.Figure, gs: GridSpec, layout: str) -> List[plt.Axes]: ...
  ```
  
- **配置外部化**：将GridSpec参数提取为配置字典或 dataclass，提高复用性：
  ```python
  GRID_CONFIGS = {
      'size_demo': {'nrows': 2, 'ncols': 2, 'width_ratios': [1, 2], 'height_ratios': [4, 1]},
      'spacing_demo_left': {'nrows': 3, 'ncols': 3, 'left': 0.05, 'right': 0.48, 'wspace': 0.05},
      'spacing_demo_right': {'nrows': 3, 'ncols': 3, 'left': 0.55, 'right': 0.98, 'hspace': 0.05},
  }
  ```

- **封装子图创建逻辑**：将子图布局模式抽象为独立函数，减少重复代码。

- **移除或明确注释意图**：删除`# %%`或添加说明其用途的注释。

- **使用明确变量名**：将gs1/gs2改为gridspec_sizes/gridspec_spacing，ax1-ax6使用更描述性的名称（如ax_main、ax_side等）。

- **图形资源管理**：在生产环境中考虑使用上下文管理器或显式调用fig.clf()和plt.close()。



## 其它




### 设计目标与约束

本代码示例旨在演示matplotlib中GridSpec的高级用法，包括：(1) 使用width_ratios和height_ratios控制子图的相对尺寸；(2) 使用left、right、bottom、top、wspace、hspace参数控制子图间距和布局。约束条件包括：需要matplotlib 3.0+版本支持GridSpec的切片操作，所有子图必须通过fig.add_subplot()添加，GridSpec对象在创建后不可修改。

### 错误处理与异常设计

代码中未包含显式的错误处理逻辑。潜在异常场景包括：(1) width_ratios或height_ratios长度与行列数不匹配时抛出ValueError；(2) 当left>=right或bottom>=top时导致无效的子图布局；(3) add_subplot()调用时GridSpec索引越界会抛出IndexError。实际应用中应添加参数验证逻辑。

### 数据流与状态机

数据流：用户定义GridSpec参数(width_ratios, height_ratios, left, right, top, bottom, wspace, hspace) → GridSpec对象创建 → fig.add_subplot()使用GridSpec索引创建Axes对象 → annotate_axes()遍历所有Axes添加文本标签 → plt.show()渲染显示。状态机转换：初始化状态(创建figure) → GridSpec配置状态 → 子图创建状态 → 渲染完成状态。

### 外部依赖与接口契约

主要依赖：matplotlib.pyplot模块（绘图框架）、matplotlib.gridspec模块（GridSpec网格布局类）。接口契约：annotate_axes(fig)函数接收Figure对象，遍历其axes属性获取所有子图；fig.add_subplot(gs[…])接受GridSpec索引切片；GridSpec构造参数包括nrows、nrows及所有布局控制参数。

### 性能考虑

当前实现对于少量子图（≤9个）性能可接受。潜在性能优化点：(1) 批量创建子图时可缓存Axes对象避免重复查询fig.axes；(2) annotate_axes函数中每次调用ax.text()会触发重绘，可考虑使用set_text()批量更新；(3) 对于大规模网格(>100个子图)，建议使用GridSpecFromSubplotSpec或预计算布局。

### 安全性考虑

代码为示例脚本，无用户输入处理，无安全风险。生产环境中需注意：(1) 避免将未经验证的外部参数传入GridSpec构造；(2) 防止通过ax.text()注入恶意格式化字符串（当前使用固定格式"ax%d"相对安全）。

### 测试策略建议

单元测试应覆盖：(1) 各种width_ratios/height_ratios组合的布局正确性；(2) 边界值测试(left=0, right=1, left<right等)；(3) GridSpec索引切片边界验证；(4) annotate_axes对不同Axes数量的兼容性。集成测试应验证最终渲染输出的视觉正确性。

### 使用示例和用例

典型用例包括：(1) 创建主从布局的主图（宽高比4:1）和侧边图（宽高比1:2）；(2) 创建非对称网格布局如L型、U型布局；(3) 创建密集型小图网格用于数据探索。扩展应用可结合SubplotSpec的切片功能实现复杂布局，如嵌套GridSpec实现混合布局。

    