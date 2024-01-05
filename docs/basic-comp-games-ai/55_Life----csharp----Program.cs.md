# `55_Life\csharp\Program.cs`

```
// 设置最大宽度和最大高度的常量
const int maxWidth = 70;
const int maxHeight = 24;

// 打印提示信息，读取用户输入的图案，并转换为 Pattern 对象
Console.WriteLine("ENTER YOUR PATTERN:");
var pattern = new Pattern(ReadPattern(limitHeight: maxHeight).ToArray());

// 根据图案的高度和宽度计算最小和最大的 x、y 坐标
var minX = 10 - pattern.Height / 2;
var minY = 34 - pattern.Width / 2;
var maxX = maxHeight - 1;
var maxY = maxWidth - 1;

// 创建一个指定高度和宽度的矩阵
var matrix = new Matrix(height: maxHeight, width: maxWidth);
// 初始化模拟器，并将图案放置在矩阵中
var simulation = InitializeSimulation(pattern, matrix);

// 打印表头信息
PrintHeader();
// 处理模拟器的运行
ProcessSimulation();

// 读取用户输入的图案，并返回一个字符串集合
IEnumerable<string> ReadPattern(int limitHeight)
{
    for (var i = 0; i < limitHeight; i++)  # 使用循环遍历限定高度范围内的内容
    {
        var input = Console.ReadLine();  # 从控制台读取用户输入
        if (input.ToUpper() == "DONE")  # 如果用户输入转换为大写后等于"DONE"
        {
            break;  # 跳出循环
        }

        // In the original version, BASIC would trim the spaces in the beginning of an input, so the original
        // game allowed you to input an '.' before the spaces to circumvent this limitation. This behavior was
        // kept for compatibility.
        if (input.StartsWith('.'))  # 如果输入以'.'开头
            yield return input.Substring(1, input.Length - 1);  # 返回去除'.'后的输入内容

        yield return input;  # 返回输入内容
    }
}

void PrintHeader()  # 打印标题
{
    # 定义一个名为PrintCentered的函数，接受一个字符串参数text
    void PrintCentered(string text)
    {
        # 定义一个常量pageWidth，表示页面宽度为64
        const int pageWidth = 64;

        # 计算空格数，使得文本居中显示
        var spaceCount = (pageWidth - text.Length) / 2;
        # 输出空格
        Console.Write(new string(' ', spaceCount));
        # 输出文本
        Console.WriteLine(text);
    }

    # 调用PrintCentered函数，输出居中显示的"LIFE"
    PrintCentered("LIFE");
    # 调用PrintCentered函数，输出居中显示的"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    PrintCentered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    # 输出空行
    Console.WriteLine();
    # 输出空行
    Console.WriteLine();
    # 输出空行
    Console.WriteLine();
}

# 定义一个名为InitializeSimulation的函数，接受Pattern和Matrix两个参数
Simulation InitializeSimulation(Pattern pattern, Matrix matrixToInitialize) {
    # 创建一个新的Simulation对象
    var newSimulation = new Simulation();
    // 将模式转录到模拟的中间，并计算初始人口
    for (var x = 0; x < pattern.Height; x++)
    {
        for (var y = 0; y < pattern.Width; y++)
        {
            // 如果模式内容为空格，则跳过
            if (pattern.Content[x][y] == ' ')
                continue;

            // 在初始化矩阵中将对应位置标记为稳定状态
            matrixToInitialize[minX + x, minY + y] = CellState.Stable;
            // 增加模拟的人口数量
            newSimulation.IncreasePopulation();
        }
    }

    // 返回新的模拟对象
    return newSimulation;
}

// 获取迭代之间的暂停时间
TimeSpan GetPauseBetweenIterations()
{
    // 如果参数长度不等于2，则返回零时间间隔
    if (args.Length != 2) return TimeSpan.Zero;
    # 将参数转换为小写
    var parameter = args[0].ToLower();
    # 检查参数中是否包含"wait"关键字
    if (parameter.Contains("wait"))
    {
        # 获取等待时间的数值
        var value = args[1];
        # 尝试将数值转换为整数，如果成功则返回对应的时间间隔
        if (int.TryParse(value, out var sleepMilliseconds))
            return TimeSpan.FromMilliseconds(sleepMilliseconds);
    }
    # 如果参数中不包含"wait"关键字或者转换失败，则返回零时间间隔
    return TimeSpan.Zero;
}

void ProcessSimulation()
{
    # 获取迭代之间的暂停时间
    var pauseBetweenIterations = GetPauseBetweenIterations();
    # 初始化是否无效的标志
    var isInvalid = false;

    # 进入无限循环
    while (true)
    {
        # 根据是否无效输出相应的文本
        var invalidText = isInvalid ? "INVALID!" : "";
        Console.WriteLine($"GENERATION: {simulation.Generation}\tPOPULATION: {simulation.Population} {invalidText}");
        // 开始新一代的模拟
        simulation.StartNewGeneration();

        // 初始化下一个搜索区域的最小和最大坐标
        var nextMinX = maxHeight - 1;
        var nextMinY = maxWidth - 1;
        var nextMaxX = 0;
        var nextMaxY = 0;

        // 创建一个 StringBuilder 用于存储矩阵输出
        var matrixOutput = new StringBuilder();

        // 在搜索区域之前打印空行
        for (var x = 0; x < minX; x++)
        {
            matrixOutput.AppendLine();
        }

        // 刷新矩阵并更新搜索区域
        for (var x = minX; x <= maxX; x++)
        {
            // 创建一个由空格字符组成的列表，用于打印行
            var printedLine = Enumerable.Repeat(' ', maxWidth).ToList();
            for (var y = minY; y <= maxY; y++)
            {
                # 遍历 y 坐标范围内的每一个点
                if (matrix[x, y] == CellState.Dying)
                {
                    # 如果当前点状态为 Dying，则将其状态设置为 Empty，并继续下一个循环
                    matrix[x, y] = CellState.Empty;
                    continue;
                }
                if (matrix[x, y] == CellState.New)
                {
                    # 如果当前点状态为 New，则将其状态设置为 Stable
                    matrix[x, y] = CellState.Stable;
                }
                else if (matrix[x, y] != CellState.Stable)
                {
                    # 如果当前点状态不为 Stable，则继续下一个循环
                    continue;
                }

                # 将当前点的 y 坐标位置标记为 '*'
                printedLine[y] = '*';

                # 更新下一个 x 坐标的最小值和最大值
                nextMinX = Math.Min(x, nextMinX);
                nextMaxX = Math.Max(x, nextMaxX);
                nextMinY = Math.Min(y, nextMinY);  # 更新下一个搜索区域的最小 Y 值
                nextMaxY = Math.Max(y, nextMaxY);  # 更新下一个搜索区域的最大 Y 值
            }

            matrixOutput.AppendLine(string.Join(separator: null, values: printedLine));  # 将打印行的值连接成字符串，并添加到输出矩阵中

        }

        // prints empty lines after search area  # 在搜索区域之后打印空行
        for (var x = maxX + 1; x < maxHeight; x++)  # 遍历最大 X 值之后的高度
        {
            matrixOutput.AppendLine();  # 添加空行到输出矩阵中
        }
        Console.Write(matrixOutput);  # 将输出矩阵打印到控制台

        void UpdateSearchArea()  # 更新搜索区域的函数
        {
            minX = nextMinX;  # 更新最小 X 值
            maxX = nextMaxX;  # 更新最大 X 值
            minY = nextMinY;  # 更新最小 Y 值
            maxY = nextMaxY;  # 更新最大 Y 值
# 设置X轴的最大值
const int limitX = 21;
# 设置Y轴的最大值
const int limitY = 67;

# 如果X轴的最小值小于2，则将最小值设为2，并将isInvalid标记为True
if (minX < 2)
{
    minX = 2;
    isInvalid = true;
}

# 如果X轴的最大值大于limitX，则将最大值设为limitX，并将isInvalid标记为True
if (maxX > limitX)
{
    maxX = limitX;
    isInvalid = true;
}

# 如果Y轴的最小值小于2，则将最小值设为2，并将isInvalid标记为True
if (minY < 2)
{
    minY = 2;
    isInvalid = true;
}
            }

            if (maxY > limitY)
            {
                maxY = limitY;  # 如果最大Y坐标超出了限制Y坐标，将最大Y坐标设为限制Y坐标
                isInvalid = true;  # 将isInvalid标记为true，表示坐标无效
            }
        }
        UpdateSearchArea();  # 更新搜索区域

        for (var x = minX - 1; x <= maxX + 1; x++)  # 从最小X坐标-1开始，遍历到最大X坐标+1
        {
            for (var y = minY - 1; y <= maxY + 1; y++)  # 从最小Y坐标-1开始，遍历到最大Y坐标+1
            {
                int CountNeighbors()  # 定义一个函数CountNeighbors，用于计算邻居数量
                {
                    var neighbors = 0;  # 初始化邻居数量为0
                    for (var i = x - 1; i <= x + 1; i++)  # 从当前X坐标-1开始，遍历到当前X坐标+1
                    {
                        for (var j = y - 1; j <= y + 1; j++)  # 从当前Y坐标-1开始，遍历到当前Y坐标+1
                {
                    # 遍历矩阵中指定位置的邻居细胞状态，统计稳定和濒死状态的邻居数量
                    if (matrix[i, j] == CellState.Stable || matrix[i, j] == CellState.Dying)
                        neighbors++;
                }
            }

            return neighbors;
        }

        # 调用CountNeighbors函数，获取指定位置细胞的邻居数量
        var neighbors = CountNeighbors();
        # 如果指定位置细胞为空
        if (matrix[x, y] == CellState.Empty)
        {
            # 如果邻居数量为3，则将该位置细胞状态设置为新生，并增加模拟中的人口数量
            if (neighbors == 3)
            {
                matrix[x, y] = CellState.New;
                simulation.IncreasePopulation();
            }
        }
        # 如果指定位置细胞不为空且邻居数量小于3或大于4
        else if (neighbors is < 3 or > 4)
        {
                matrix[x, y] = CellState.Dying;  // 将矩阵中坐标为(x, y)的细胞状态设置为正在死亡
            }
            else
            {
                simulation.IncreasePopulation();  // 否则，增加模拟中的人口数量
            }
        }
    }

    // 扩展搜索区域以适应新的细胞
    minX--;  // 最小X坐标减一
    minY--;  // 最小Y坐标减一
    maxX++;  // 最大X坐标加一
    maxY++;  // 最大Y坐标加一

    if (pauseBetweenIterations > TimeSpan.Zero)  // 如果迭代之间有暂停时间
        Thread.Sleep(pauseBetweenIterations);  // 线程休眠暂停时间
}
# 定义一个名为 Pattern 的类
public class Pattern
{
    # 声明一个名为 Content 的字符串数组属性
    public string[] Content { get; }
    # 声明一个名为 Height 的整数属性
    public int Height { get; }
    # 声明一个名为 Width 的整数属性
    public int Width { get; }

    # 定义一个构造函数，接受一个只读字符串集合作为参数
    public Pattern(IReadOnlyCollection<string> patternLines)
    {
        # 将 Height 属性设置为 patternLines 的元素数量
        Height = patternLines.Count;
        # 将 Width 属性设置为 patternLines 中最长字符串的长度
        Width = patternLines.Max(x => x.Length);
        # 调用 NormalizeWidth 方法，将结果赋给 Content 属性
        Content = NormalizeWidth(patternLines);
    }

    # 定义一个名为 NormalizeWidth 的私有方法，接受一个只读字符串集合作为参数
    private string[] NormalizeWidth(IReadOnlyCollection<string> patternLines)
    {
        # 对 patternLines 中的每个字符串进行处理，使其长度与 Width 相同，然后转换为数组
        return patternLines
            .Select(x => x.PadRight(Width, ' '))
            .ToArray();
    }
}
/// <summary>
/// 表示模拟中给定单元格的状态。
/// </summary>
internal enum CellState
{
    Empty = 0, // 空状态
    Stable = 1, // 稳定状态
    Dying = 2, // 死亡状态
    New = 3 // 新状态
}

public class Simulation
{
    public int Generation { get; private set; } // 代数属性，只能在类内部设置

    public int Population { get; private set; } // 人口数量属性，只能在类内部设置

    public void StartNewGeneration()
    {
        Generation++;  // 增加 Generation 变量的值
        Population = 0;  // 将 Population 变量的值设为 0
    }

    public void IncreasePopulation()  // 定义一个方法，用于增加 Population 变量的值
    {
        Population++;  // 增加 Population 变量的值
    }
}

/// <summary>
/// This class was created to aid debugging, through the implementation of the ToString() method.
/// </summary>
class Matrix  // 定义一个名为 Matrix 的类
{
    private readonly CellState[,] _matrix;  // 声明一个名为 _matrix 的只读二维数组变量

    public Matrix(int height, int width)  // 定义一个构造函数，接受 height 和 width 两个参数
    {
        _matrix = new CellState[height, width];  // 初始化 _matrix 变量为指定大小的 CellState 类型的二维数组
    }

    // 定义一个索引器，用于获取和设置矩阵中指定位置的元素状态
    public CellState this[int x, int y]
    {
        get => _matrix[x, y]; // 获取矩阵中指定位置的元素状态
        set => _matrix[x, y] = value; // 设置矩阵中指定位置的元素状态
    }

    // 重写 ToString 方法，返回矩阵的字符串表示
    public override string ToString()
    {
        var stringBuilder = new StringBuilder(); // 创建一个 StringBuilder 对象，用于构建字符串
        for (var x = 0; x < _matrix.GetLength(0); x++) // 遍历矩阵的行
        {
            for (var y = 0; y < _matrix.GetLength(1); y++) // 遍历矩阵的列
            {
                var character = _matrix[x, y] == 0 ? " ": ((int)_matrix[x, y]).ToString(); // 根据矩阵中元素的状态确定要追加到字符串的字符
                stringBuilder.Append(character); // 将字符追加到字符串中
            }

            stringBuilder.AppendLine(); // 在每行末尾添加换行符
        }
        return stringBuilder.ToString();  # 返回拼接后的字符串
    }
}
```