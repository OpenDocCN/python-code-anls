
# `matplotlib\galleries\examples\images_contours_and_fields\tricontour_smooth_delaunay.py` 详细设计文档

A matplotlib demonstration script that performs high-resolution tricontouring of randomly generated points using Delaunay triangulation, TriAnalyzer for filtering flat triangles at the border, and UniformTriRefiner for mesh refinement to create smooth contour plots.

## 整体流程

```mermaid
graph TD
    A[开始] --> B[生成随机测试点 x_test, y_test]
    B --> C[计算解析函数值 z_test = experiment_res(x_test, y_test)]
    C --> D[创建 Delaunay 三角剖分 tri = Triangulation(x_test, y_test)]
    D --> E{init_mask_frac > 0?}
    E -- 是 --> F[随机掩码部分三角形 mask_init]
    E -- 否 --> G[使用 TriAnalyzer 分析三角网质量]
    F --> G
    G --> H[获取扁平三角形掩码 mask = TriAnalyzer.get_flat_tri_mask()]
    H --> I[应用掩码到三角剖分 tri.set_mask(mask)]
    I --> J[创建 UniformTriRefiner 细化器]
    J --> K[细化数据 tri_refi, z_test_refi = refine_field()]
    K --> L[计算预期解析结果 z_expected]
    L --> M[配置 matplotlib 绘图参数]
    M --> N[绘制 tricontour 等高线]
    N --> O{plot_tri?}
    O -- 是 --> P[绘制基础三角网]
    O -- 否 --> Q{plot_masked_tri?}
    P --> Q
    Q -- 是 --> R[绘制被掩码的三角形]
    Q -- 否 --> S[plt.show 显示图形]
```

## 类结构

```
独立脚本 (无类定义)
└── 主要函数: experiment_res(x, y)
```

## 全局变量及字段


### `n_test`
    
测试数据点数量，用于生成随机测试点

类型：`int`
    


### `subdiv`
    
初始网格递归细分次数，控制细化密度

类型：`int`
    


### `init_mask_frac`
    
要掩码的初始三角形比例，模拟无效数据

类型：`float`
    


### `min_circle_ratio`
    
最小圆比率阈值，用于过滤扁平三角形

类型：`float`
    


### `random_gen`
    
随机数生成器，用于生成测试数据和掩码

类型：`np.random.RandomState`
    


### `x_test`
    
测试点 x 坐标，范围在 [-1, 1] 内

类型：`np.ndarray`
    


### `y_test`
    
测试点 y 坐标，范围在 [-1, 1] 内

类型：`np.ndarray`
    


### `z_test`
    
测试点的解析函数值，由 experiment_res 计算得出

类型：`np.ndarray`
    


### `tri`
    
Delaunay 三角剖分对象，表示测试点的三角网格

类型：`Triangulation`
    


### `ntri`
    
三角形数量

类型：`int`
    


### `mask_init`
    
初始掩码数组，标记需要掩码的三角形

类型：`np.ndarray`
    


### `masked_tri`
    
被随机掩码的三角形索引

类型：`np.ndarray`
    


### `mask`
    
最终应用的扁平三角形掩码

类型：`np.ndarray`
    


### `refiner`
    
三角网细化器，用于细化三角网格

类型：`UniformTriRefiner`
    


### `tri_refi`
    
细化后的三角剖分

类型：`Triangulation`
    


### `z_test_refi`
    
细化后的插值数据

类型：`np.ndarray`
    


### `z_expected`
    
解析函数的预期值，用于对比分析

类型：`np.ndarray`
    


### `flat_tri`
    
扁平三角形三角剖分，用于可视化被排除的三角形

类型：`Triangulation`
    


### `plot_tri`
    
是否绘制基础三角网

类型：`bool`
    


### `plot_masked_tri`
    
是否绘制被掩码的三角形

类型：`bool`
    


### `plot_refi_tri`
    
是否绘制细化三角网

类型：`bool`
    


### `plot_expected`
    
是否绘制预期等高线

类型：`bool`
    


### `levels`
    
等高线级别数组

类型：`np.ndarray`
    


### `fig`
    
图形对象

类型：`matplotlib.figure.Figure`
    


### `ax`
    
坐标轴对象

类型：`matplotlib.axes.Axes`
    


    

## 全局函数及方法



### `experiment_res`

该函数是一个分析测试函数，用于模拟实验结果。它接收二维坐标 (x, y)，通过计算两个极坐标系统下的指数和余弦组合，再加上二次项，生成复杂的数值模式，最后将结果归一化到 [0, 1] 区间。

参数：

- `x`：`float`，表示输入的 x 坐标值
- `y`：`float`，表示输入的 y 坐标值

返回值：`np.ndarray`，返回归一化后的实验结果值，范围在 [0, 1] 之间

#### 流程图

```mermaid
flowchart TD
    A[开始: 输入坐标 x, y] --> B[将 x 乘以 2]
    B --> C[计算 r1: 点 (x,y) 到 (0.5, 0.5) 的距离]
    C --> D[计算 theta1: 点 (x,y) 到 (0.5, 0.5) 的角度]
    D --> E[计算 r2: 点 (x,y) 到 (-0.2, -0.2) 的距离]
    E --> F[计算 theta2: 点 (x,y) 到 (-0.2, -0.2) 的角度]
    F --> G[计算第一项: 4 * (exp((r1/10)²) - 1) * 30 * cos(3 * theta1)]
    G --> H[计算第二项: (exp((r2/10)²) - 1) * 30 * cos(5 * theta2)]
    H --> I[计算第三项: 2 * (x² + y²)]
    I --> J[组合: z = 第一项 + 第二项 + 第三项]
    J --> K[计算归一化分子: np.max(z) - z]
    K --> L[计算归一化分母: np.max(z) - np.min(z)]
    L --> M[返回归一化结果: z_normalized = 分子 / 分母]
    M --> N[结束: 输出 np.ndarray]
```

#### 带注释源码

```python
def experiment_res(x, y):
    """An analytic function representing experiment results.
    
    该函数创建一个复杂的分析测试模式，用于模拟实验数据。
    它结合了两个极坐标系统（中心分别在 (0.5, 0.5) 和 (-0.2, -0.2)）
    的指数衰减余弦波形，以及一个径向的二次增长项。
    
    参数:
        x: float - 输入的 x 坐标值
        y: float - 输入的 y 坐标值
    
    返回:
        np.ndarray - 归一化到 [0, 1] 范围的实验结果值
    """
    # 将 x 坐标缩放 2 倍，改变函数的水平方向特性
    x = 2 * x
    
    # 第一个极坐标系统：中心在 (0.5, 0.5)
    # 计算当前点到中心的距离 r1
    r1 = np.sqrt((0.5 - x)**2 + (0.5 - y)**2)
    # 计算当前点到中心的角度 theta1（使用 arctan2 获取完整象限信息）
    theta1 = np.arctan2(0.5 - x, 0.5 - y)
    
    # 第二个极坐标系统：中心在 (-0.2, -0.2)
    # 计算当前点到中心的距离 r2
    r2 = np.sqrt((-x - 0.2)**2 + (-y - 0.2)**2)
    # 计算当前点到中心的角度 theta2
    theta2 = np.arctan2(-x - 0.2, -y - 0.2)
    
    # 计算综合实验结果 z
    # 第一项：以 (0.5, 0.5) 为中心的指数衰减余弦波（3倍频）
    # 第二项：以 (-0.2, -0.2) 为中心的指数衰减余弦波（5倍频）
    # 第三项：径向二次项，使函数在原点附近有抛物面特性
    z = (4 * (np.exp((r1/10)**2) - 1) * 30 * np.cos(3 * theta1) +
         (np.exp((r2/10)**2) - 1) * 30 * np.cos(5 * theta2) +
         2 * (x**2 + y**2))
    
    # 归一化处理：将 z 值映射到 [0, 1] 区间
    # 分子：当前值到最大值的距离
    # 分母：最大值到最小值的跨度
    return (np.max(z) - z) / (np.max(z) - np.min(z))
```

## 关键组件




### Triangulation（三角剖分）

使用matplotlib.tri.Triangulation对随机生成的测试点进行Delaunay三角剖分，生成基础的三角网格结构。

### TriAnalyzer（三角网格分析器）

使用matplotlib.tri.TriAnalyzer分析三角网格，检测并掩码化位于边界上的扁平三角形，以提高等高线绘制质量。

### UniformTriRefiner（均匀三角网格细化器）

使用matplotlib.tri.UniformTriRefiner对初始三角网格进行递归细分，并对数据进行插值，生成高分辨率的细化网格用于平滑等高线绘制。

### experiment_res（分析测试函数）

定义一个解析函数，基于极坐标变换生成具有多个极值点的实验数据，用于测试等高线绘制效果。

### 随机数据生成与掩码处理

使用numpy.random.RandomState生成指定数量的随机测试点，并根据init_mask_frac参数随机掩码部分三角形，模拟无效数据场景。

### tricontour（三角等高线绘制）

使用Axes.tricontour方法在细化后的三角网格上绘制等高线，支持多级别设置和颜色映射。

### 扁平三角形可视化

将TriAnalyzer识别出的扁平三角形单独存储并使用红色线条绘制，用于演示网格质量过滤效果。


## 问题及建议




### 已知问题

- **硬编码的魔数（Magic Numbers）**：代码中多处使用硬编码数值，如种子值`19680801`、颜色`'Blues'`、`'0.7'`、`'0.97'`、`'red'`、线条宽度`[2.0, 0.5, 1.0, 0.5]`等，缺乏配置化管理。
- **全局作用域代码过多**：所有逻辑（包括数据生成、三角剖分、绘图）均在模块级别执行，未封装成函数或类，难以测试和复用。
- **缺少类型注解（Type Hints）**：`experiment_res`函数及所有全局变量均无类型注解，降低了代码的可读性和静态分析能力。
- **未使用的计算资源**：当`plot_expected=False`时，`z_expected`仍会被计算（`z_expected = experiment_res(tri_refi.x, tri_refi.y)`），造成不必要的计算开销。
- **参数验证缺失**：未对关键参数进行有效性检查，如`n_test`应至少为3、`subdiv`应为正整数、`init_mask_frac`应在[0,1]范围内。
- **重复的绘图逻辑**：`ax.tricontour`和`ax.triplot`的多次调用存在重复的模式代码，可提取为辅助函数。
- **变量命名不一致**：`random_gen`使用下划线命名法，但`x_test`、`y_test`、`z_test`未遵循PEP 8的私有变量约定（单下划线前缀）。
- **注释格式问题**：文件开头的文档字符串使用了Sphinx reStructuredText格式，但该脚本并非可导入的模块，这种用法不够规范。
- **TriAnalyzer掩码逻辑不透明**：`TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio)`的掩码逻辑对调用者不够透明，缺少对掩码覆盖率的可视化或日志输出。

### 优化建议

- **引入配置类或字典**：使用`dataclass`或`NamedTuple`封装配置参数（`n_test`、`subdiv`、`init_mask_frac`等），便于统一管理和传递。
- **封装为函数或类**：将数据生成、三角剖分、网格优化、绘图等步骤分别封装为独立函数，或创建`TricontourDemo`类提高模块化程度。
- **添加类型注解**：为`experiment_res`函数添加`-> np.ndarray`返回类型，为参数添加类型提示。
- **条件计算优化**：使用惰性计算（lazy evaluation）或将`z_expected`的计算移入`if plot_expected:`分支内。
- **输入验证**：在函数入口处添加参数校验，抛出`ValueError`或`AssertionError`异常处理非法输入。
- **提取绘图辅助函数**：将重复的`tricontour`和`triplot`调用模式提取为`plot_contours`和`plot_mesh`辅助函数。
- **常量分离**：将魔数提取为模块级常量（如`DEFAULT_SEED`、`DEFAULT_CMAP`、`COLORS`字典），并添加文档字符串说明。
- **添加日志或进度提示**：在耗时的网格细分（`refine_field`）操作前后添加`print`或`logging`输出，提升用户体验。
- **扩展错误处理**：捕获`Triangulation`、`TriAnalyzer`可能抛出的异常（如点共线、点数不足等），提供友好的错误信息。
- **考虑性能优化**：对于大规模数据（`n_test` > 1000），可考虑使用`joblib`并行计算或预先分配数组内存。


## 其它




### 设计目标与约束

本代码旨在演示如何使用matplotlib的tri模块进行高质量的三角等高线绘制。具体目标包括：1）展示如何处理Delaunay三角剖分中的无效（扁平）三角形；2）演示如何使用UniformTriRefiner进行网格细化以获得平滑的等高线；3）提供一个可复用的分析函数作为测试数据。约束条件包括：subdiv参数不宜设置过高（>3可能导致三角形数量爆炸），min_circle_ratio建议值为0.01，init_mask_frac用于模拟无效数据场景。

### 错误处理与异常设计

代码主要依赖matplotlib和numpy库，异常处理主要体现在：1）TriAnalyzer.get_flat_tri_mask()可能返回空掩码；2）UniformTriRefiner.refine_field()在输入数据不合法时可能抛出异常；3）np.random.RandomState的种子设置保证可复现性。当前代码未显式捕获异常，属于演示性质代码，生产环境需增加参数校验和异常捕获逻辑。

### 数据流与状态机

数据流如下：随机点生成(x_test, y_test) → 实验函数计算z值(z_test) → Delaunay三角剖分(Triangulation) → 设置初始掩码(mask_init) → TriAnalyzer分析获取扁平三角形掩码(mask) → UniformTriRefiner细化(refine_field) → 绘制等高线(tricontour)。状态转换：初始态(随机点) → 三角化态 → 掩码态 → 细化态 → 可视化态。

### 外部依赖与接口契约

核心依赖包括：matplotlib.pyplot(绘图)、numpy(数值计算)、matplotlib.tri模块(TriAnalyzer, Triangulation, UniformTriRefiner)。接口契约：experiment_res(x, y)接收numpy数组返回归一化后的z值；Triangulation(x, y)接收坐标数组返回三角网格对象；TriAnalyzer(tri).get_flat_tri_mask(ratio)返回布尔掩码数组；UniformTriRefiner(tri).refine_field(z, subdiv=n)返回细化后的网格和z值。

### 性能考虑

关键性能参数：n_test=200个测试点，subdiv=3次递归细分。理论输出三角形数量为4^3 * ntri，当前配置下可正常运行。如需更高分辨率需权衡内存占用和渲染时间。TriAnalyzer的get_flat_tri_mask方法复杂度为O(ntri)，UniformTriRefiner复杂度为O(ntri * 4^subdiv)。

### 配置参数说明

n_test: 测试数据点数，建议范围3-5000；subdiv: 递归细分次数，建议≤3；init_mask_frac: 初始掩码比例，用于模拟无效数据，0表示不掩码；min_circle_ratio: 最小圆比率阈值，建议0.01，-1表示保留所有三角形；levels: 等高线级别数组，从0到1步长0.025；plot_tri/plot_masked_tri/plot_refi_tri/plot_expected: 布尔开关控制各类图形输出。

### 兼容性考虑

代码兼容matplotlib 3.x版本和numpy 1.x版本。使用triplot和tricontour等API在matplotlib 2.0+保持稳定。RandomState种子19680801确保跨平台结果可复现。代码末尾的docstring包含References节，说明了所使用函数和类的官方文档链接。

### 可扩展性设计

代码结构支持以下扩展：1）替换experiment_res函数以使用不同的分析函数；2）通过修改levels数组调整等高线密度；3）添加新的绘图选项（如tricontourf填充等高线）；4）可以封装为类或函数以提高复用性。当前为演示脚本形式，直接运行产生图形输出。

    