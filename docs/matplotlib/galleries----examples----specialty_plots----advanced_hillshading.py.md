
# `matplotlib\galleries\examples\specialty_plots\advanced_hillshading.py` 详细设计文档

A Matplotlib demonstration script illustrating three advanced hillshading techniques: correctly attaching colorbars to shaded plots, handling data outliers using vmin/vmax, and applying shading and coloring to different datasets independently.

## 整体流程

```mermaid
graph TD
    Start[Script Execution] --> FuncCall{Call Functions}
    FuncCall --> D1[display_colorbar]
    FuncCall --> D2[avoid_outliers]
    FuncCall --> D3[shade_other_data]
    FuncCall --> Show[plt.show]
    subgraph display_colorbar
    D1 --> D1_Grid[Generate Grid (x, y)]
    D1_Grid --> D1_Z[Compute z = 10*cos(x²+y²)]
    D1_Z --> D1_LS[LightSource.shade]
    D1_LS --> D1_Plot[plt.subplots & ax.imshow]
    D1_Plot --> D1_CB[fig.colorbar]
    end
    subgraph avoid_outliers
    D2 --> D2_Grid[Generate Grid]
    D2_Grid --> D2_Z[Compute z + Add Outliers]
    D2_Z --> D2_Plot1[Shade Full Range -> Plot]
    D2_Z --> D2_Plot2[Shade (vmin=-10, vmax=10) -> Plot]
    end
    subgraph shade_other_data
    D3 --> D3_Grid[Generate Grid]
    D3_Grid --> D3_Zs[Compute z1 (sin) & z2 (cos)]
    D3_Zs --> D3_Norm[Normalize z2]
    D3_Norm --> D3_Shade[LightSource.shade_rgb]
    D3_Shade --> D3_Plot[Plot Result]
    end
    D1_CB --> Show
    D2_Plot2 --> Show
    D3_Plot --> Show
    Show --> End[End]
```

## 类结构

```
Module (Top Level)
└── No user-defined classes. The script is procedural and relies on Matplotlib's Object-Oriented API (Figure, Axes, LightSource, Normalize).
```

## 全局变量及字段


### `plt`
    
matplotlib.pyplot模块，用于创建图形、图表和可视化

类型：`module`
    


### `np`
    
numpy模块，用于数值计算和数组操作

类型：`module`
    


### `LightSource`
    
matplotlib.colors中的类，用于创建光源效果进行山体阴影渲染

类型：`class`
    


### `Normalize`
    
matplotlib.colors中的类，用于数据归一化处理

类型：`class`
    


### `global function.display_colorbar`
    
显示带有正确数值颜色条的山体阴影图

类型：`function`
    


### `global function.avoid_outliers`
    
使用自定义归一化控制着色图的显示z范围以处理异常值

类型：`function`
    


### `global function.shade_other_data`
    
演示通过阴影和颜色显示不同变量的山体阴影技术

类型：`function`
    
    

## 全局函数及方法



### `display_colorbar`

该函数用于在 matplotlib 中为山体阴影图（hillshade plot）显示正确的数值颜色条（colorbar）。它通过创建一个代理图像（proxy artist）来解决阴影图像无法直接显示颜色条的问题，展示了在同时使用 LightSource 生成的光照效果和颜色条时的标准做法。

参数： 无

返回值： `None`，该函数直接展示图形，不返回任何值

#### 流程图

```mermaid
flowchart TD
    A[开始] --> B[创建网格: np.mgrid[-4:2:200j, -4:2:200j]]
    B --> C[计算z值: z = 10 * np.cos(x**2 + y**2)]
    C --> D[获取colormap: plt.colormaps['copper']]
    D --> E[创建LightSource: 方位角315°, 高度角45°]
    E --> F[生成阴影RGB图像: ls.shade]
    F --> G[创建图形和轴: plt.subplots]
    G --> H[显示阴影RGB图像: ax.imshow]
    H --> I[创建代理图像: ax.imshow用于colorbar]
    I --> J[移除代理图像: im.remove]
    J --> K[添加颜色条: fig.colorbar]
    K --> L[设置标题]
    L --> M[结束]
```

#### 带注释源码

```python
def display_colorbar():
    """Display a correct numeric colorbar for a shaded plot."""
    # 创建一个200x200的网格，范围从-4到2
    # y, x 分别代表网格的垂直和水平坐标
    y, x = np.mgrid[-4:2:200j, -4:2:200j]
    
    # 计算z值：基于x和y的余弦函数，生成波浪形地形数据
    # 乘以10是为了增加高度变化的幅度
    z = 10 * np.cos(x**2 + y**2)

    # 获取'copper'颜色映射，用于可视化高度数据
    cmap = plt.colormaps["copper"]
    
    # 创建LightSource对象
    # 参数315表示光源方位角（从西北方向照射）
    # 参数45表示光源高度角（45度角照射）
    ls = LightSource(315, 45)
    
    # 使用shade函数将z值转换为带有阴影效果的RGB图像
    # 这个rgb图像已经是计算好的颜色值，不再包含原始的z值信息
    rgb = ls.shade(z, cmap)

    # 创建图形窗口和子图轴
    fig, ax = plt.subplots()
    
    # 在轴上显示阴影后的RGB图像
    # interpolation='bilinear'使图像显示更平滑
    ax.imshow(rgb, interpolation='bilinear')

    # 使用代理艺术家（proxy artist）技术来显示颜色条
    # 关键点：虽然我们显示的是rgb图像，但颜色条需要原始的z值和cmap
    # 因此创建一个临时的图像对象用于生成颜色条
    im = ax.imshow(z, cmap=cmap)
    
    # 立即移除这个临时图像，避免它在最终图中显示
    # 我们只需要它的颜色映射信息来创建颜色条
    im.remove()
    
    # 创建颜色条
    # colorbar需要原始的映射器（这里是im），而不需要实际的图像显示
    fig.colorbar(im, ax=ax)

    # 设置图表标题
    ax.set_title('Using a colorbar with a shaded plot', size='x-large')
```



### `avoid_outliers`

该函数用于演示如何在阴影图中通过手动设置z值的显示范围（vmin和vmax）来避免异常值（outliers）对可视化结果的影响，使图像细节更清晰。

参数： 无

返回值：`None`，该函数不返回任何值，仅用于展示matplotlib绘制的图形

#### 流程图

```mermaid
flowchart TD
    A[函数开始] --> B[创建网格坐标: y, x = np.mgrid[-4:2:200j, -4:2:200j]]
    B --> C[计算z值: z = 10 * cos(x² + y²)]
    C --> D[添加异常值: z[100,105] = 2000, z[120,110] = -9000]
    D --> E[创建LightSource对象: ls = LightSource(315, 45)]
    E --> F[创建包含2个子图的图表: fig, (ax1, ax2) = plt.subplots]
    F --> G[生成全范围阴影图: rgb = ls.shade(z, cmap)]
    G --> H[在ax1上显示全范围图像]
    H --> I[生成限定范围阴影图: rgb = ls.shade(z, cmap, vmin=-10, vmax=10)]
    I --> J[在ax2上显示限定范围图像]
    J --> K[设置总标题: fig.suptitle]
    K --> L[函数结束]
```

#### 带注释源码

```python
def avoid_outliers():
    """Use a custom norm to control the displayed z-range of a shaded plot."""
    # 创建网格坐标范围，生成-4到2的200个采样点的网格
    y, x = np.mgrid[-4:2:200j, -4:2:200j]
    # 使用cos函数生成基础数据z值，范围大致在-10到10之间
    z = 10 * np.cos(x**2 + y**2)

    # 人为添加异常值（outliers）以演示如何处理
    z[100, 105] = 2000   # 极高的异常值
    z[120, 110] = -9000  # 极低的异常值

    # 创建光源对象，315度方位角，45度高度角
    ls = LightSource(315, 45)
    # 创建包含2列子图的图表布局
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4.5))

    # 使用copper色彩映射生成全数据范围的阴影图
    rgb = ls.shade(z, plt.colormaps["copper"])
    # 在ax1子图上显示全范围数据（包含异常值）
    ax1.imshow(rgb, interpolation='bilinear')
    ax1.set_title('Full range of data')

    # 手动设置vmin和vmax限制显示范围，排除异常值干扰
    rgb = ls.shade(z, plt.colormaps["copper"], vmin=-10, vmax=10)
    # 在ax2子图上显示限定范围的图像（异常值被截断）
    ax2.imshow(rgb, interpolation='bilinear')
    ax2.set_title('Manually set range')

    # 设置整个图表的总标题
    fig.suptitle('Avoiding Outliers in Shaded Plots', size='x-large')
```




### `shade_other_data`

该函数演示了如何通过山体阴影（hillshading）显示一个变量（z1），同时使用颜色映射显示另一个变量（z2），从而在同一图像中呈现多维数据。

参数：

- 该函数没有参数

返回值：`None`，无返回值（该函数主要创建并显示matplotlib图形）

#### 流程图

```mermaid
graph TD
    A[开始 shade_other_data] --> B[使用 np.mgrid 创建网格坐标 y, x]
    B --> C[计算 z1 = sin(x²) 用于山体阴影]
    C --> D[计算 z2 = cos(x² + y²) 用于颜色映射]
    D --> E[创建 Normalize 对象 norm, 范围为 z2 的 min 到 max]
    E --> F[获取 RdBu 颜色映射]
    F --> G[创建 LightSource 对象 ls, 角度 315°/45°]
    G --> H[调用 ls.shade_rgb 生成山体阴影图像]
    H --> I[创建 matplotlib 子图 fig 和 ax]
    I --> J[使用 ax.imshow 显示生成的 RGB 图像]
    J --> K[设置图表标题]
    K --> L[结束函数]
```

#### 带注释源码

```python
def shade_other_data():
    """Demonstrates displaying different variables through shade and color."""
    # 使用 np.mgrid 创建二维网格坐标
    # y 范围: -4 到 2, 200个采样点
    # x 范围: -4 到 2, 200个采样点
    y, x = np.mgrid[-4:2:200j, -4:2:200j]
    
    # 计算 z1: 作为山体阴影的高度数据
    # 使用 sin(x²) 创建起伏的地形效果
    z1 = np.sin(x**2)  # Data to hillshade
    
    # 计算 z2: 作为颜色映射的数据
    # 使用 cos(x² + y²) 创建复杂的颜色分布
    z2 = np.cos(x**2 + y**2)  # Data to color

    # 创建归一化对象, 将 z2 的值映射到 [0, 1] 范围
    # 用于颜色映射的输入
    norm = Normalize(z2.min(), z2.max())
    
    # 获取 'RdBu' (红蓝) 颜色映射
    # 红蓝映射常用于显示正负对立的数据
    cmap = plt.colormaps["RdBu"]

    # 创建 LightSource 光源对象
    # 参数 315: 光源方位角 (从西北方向照射)
    # 参数 45: 光源高度角 (45度角照射)
    ls = LightSource(315, 45)
    
    # 调用 shade_rgb 生成山体阴影图像
    # cmap(norm(z2)): 将 z2 归一化后映射到颜色
    # z1: 作为山体阴影的高度图
    # 返回 RGB 图像数组
    rgb = ls.shade_rgb(cmap(norm(z2)), z1)

    # 创建 matplotlib 子图
    # fig: 图形对象
    # ax: 坐标轴对象
    fig, ax = plt.subplots()
    
    # 使用 imshow 显示生成的 RGB 图像
    # interpolation='bilinear': 使用双线性插值使图像更平滑
    ax.imshow(rgb, interpolation='bilinear')
    
    # 设置图表标题
    ax.set_title('Shade by one variable, color by another', size='x-large')
```


## 关键组件




### LightSource 类

matplotlib.colors 模块中的光照源类，用于生成山体阴影效果。构造函数接受方位角(azdeg)和高度角(altdeg)参数来定义光源位置。

### shade() 方法

将二维高度数据转换为RGB图像，应用山体阴影效果。参数包括：z(高度数据数组)、cmap(颜色映射)、mode(渲染模式)等，返回渲染后的RGB数组。

### shade_rgb() 方法

使用一个变量的数据进行阴影，另一个变量的数据进行着色显示。参数：rgb(颜色映射后的RGB值)、elevation(高度数据)，返回组合后的RGB图像。

### Normalize 类

matplotlib.colors 中的数据标准化类，用于将数据值映射到[0,1]范围，以便进行颜色映射。

### mgrid 网格生成

使用 numpy.mgrid 创建二维网格坐标，用于生成测试用的高度数据(x, y)和z值。

### vmin/vmax 参数

用于控制颜色映射的显示范围，可以排除极端异常值，使图像显示更合理。

### 颜色映射 (cmap)

使用 "copper"、"RdBu" 等 colormap 将数值映射到颜色，实现数据的可视化表达。

### 代理艺术家 (Proxy Artist)

通过 im.remove() 技巧，在不显示原始数据的情况下创建颜色条，实现阴影图像的颜色条显示。


## 问题及建议



### 已知问题

-   **代码重复**：三个函数中存在大量重复代码，包括网格生成（`np.mgrid[-4:2:200j, -4:2:200j]`）重复3次、`LightSource(315, 45)` 实例化重复3次、colormap获取重复3次
-   **魔法数字**：光照角度参数`315`和`45`在多处硬编码，未提取为常量，导致维护困难
-   **资源管理不当**：未显式关闭matplotlib的figure对象，可能导致内存泄漏
-   **缺乏灵活性**：网格大小（200j）、坐标范围（-4:2）、colormap名称等参数硬编码，难以适应不同场景
-   **可测试性差**：函数直接绘图而非返回数据或对象，难以进行单元测试
-   **缺少错误处理**：未对可能的异常进行处理，如colormap不存在、数据维度不匹配等情况
-   **入口点设计**：模块级直接调用函数和`plt.show()`，不适合作为可导入模块使用
-   **类型提示缺失**：无函数参数和返回值的类型标注，降低了代码可读性和IDE支持

### 优化建议

-   将重复代码提取为公共函数，如创建网格、初始化LightSource等
-   定义模块级常量（如`DEFAULT_LIGHT_ANGLE`、`DEFAULT_GRID_SIZE`等）替代魔法数字
-   使用`plt.close(fig)`或上下文管理器管理figure生命周期
-   添加命令行参数解析（如`argparse`）以提高脚本灵活性
-   重构函数使其返回数据或可复用对象，而非直接绘图，提高可测试性
-   添加类型提示（type hints）和更完善的文档字符串
-   添加异常处理机制，如try-except捕获并处理可能的错误

## 其它





### 设计目标与约束

本代码旨在演示matplotlib中LightSource类的山体阴影（hillshading）可视化技术的常见用法，包括颜色条显示、数据范围控制、多变量渲染等场景。设计约束包括：依赖matplotlib和numpy库，使用特定版本兼容的API（如plt.colormaps在较新版本中的使用），以及需要图形界面支持以显示结果。

### 错误处理与异常设计

代码主要依赖matplotlib的内部错误处理机制。未对输入数据进行显式验证，但在avoid_outliers()函数中通过vmin/vmax参数处理异常值，避免极端值影响可视化效果。若z数组为空或包含无效值，LightSource.shade()方法将抛出相应异常。

### 数据流与状态机

数据流遵循以下路径：生成网格坐标(x, y) → 计算高度数据z → 创建LightSource对象 → 调用shade()方法生成RGB图像 → 通过imshow()渲染到Axes对象。状态机主要体现在plt.show()调用时的交互式图形显示状态。

### 外部依赖与接口契约

主要依赖：matplotlib.pyplot（图形渲染）、numpy（数值计算）、matplotlib.colors（LightSource和Normalize类）。接口契约：所有函数均无参数，无返回值（直接显示图形），通过修改全局matplotlib状态实现可视化效果。

### 性能考虑

代码使用200x200的网格分辨率（200j），在avoid_outliers()中创建两个子图会增加内存占用。shade_rgb()方法在大数据集上可能较慢，可考虑降采样处理。plt.colormaps返回Colormap对象，存在版本兼容性差异。

### 安全性考虑

代码为纯可视化示例，无用户输入处理，无安全风险。但注意plt.show()会阻塞主线程，在某些GUI框架中需异步处理。

### 测试策略

建议测试场景包括：不同尺寸网格的性能测试、验证颜色映射与阴影的正确叠加、测试vmin/vmax边界值处理、验证跨matplotlib版本的兼容性。

### 部署与使用注意事项

运行时需要图形后端支持（如TkAgg、Agg等）。在无头服务器环境需设置matplotlib后端为Agg。推荐通过python -c调用或导入模块方式使用，函数可直接调用展示不同 hillshading 效果。

### 代码可扩展性分析

当前实现可直接扩展支持：自定义光源方向和高度角参数化、增加更多色彩映射方案支持、添加动画效果、集成到Web应用（需保存为图片而非plt.show()）。


    