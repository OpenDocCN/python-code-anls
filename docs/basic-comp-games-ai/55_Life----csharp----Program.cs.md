# `basic-computer-games\55_Life\csharp\Program.cs`

```
// 设置最大宽度和最大高度的常量
const int maxWidth = 70;
const int maxHeight = 24;

// 输出提示信息，要求输入模式
Console.WriteLine("ENTER YOUR PATTERN:");
// 从控制台读取输入的模式，并转换为数组
var pattern = new Pattern(ReadPattern(limitHeight: maxHeight).ToArray());

// 计算模式在矩阵中的位置
var minX = 10 - pattern.Height / 2;
var minY = 34 - pattern.Width / 2;
var maxX = maxHeight - 1;
var maxY = maxWidth - 1;

// 创建指定高度和宽度的矩阵
var matrix = new Matrix(height: maxHeight, width: maxWidth);
// 初始化模拟器
var simulation = InitializeSimulation(pattern, matrix);

// 打印标题
PrintHeader();
// 处理模拟器
ProcessSimulation();

// 读取模式的方法，返回一个字符串的集合
IEnumerable<string> ReadPattern(int limitHeight)
{
    for (var i = 0; i < limitHeight; i++)
    {
        var input = Console.ReadLine();
        if (input.ToUpper() == "DONE")
        {
            break;
        }

        // 在原始版本中，BASIC 会修剪输入开头的空格，所以原始游戏允许在空格前输入 '.' 来规避这个限制。为了兼容性，保留了这种行为。
        if (input.StartsWith('.'))
            yield return input.Substring(1, input.Length - 1);

        yield return input;
    }
}

// 打印标题的方法
void PrintHeader()
{
    // 打印居中的文本
    void PrintCentered(string text)
    {
        const int pageWidth = 64;

        var spaceCount = (pageWidth - text.Length) / 2;
        Console.Write(new string(' ', spaceCount));
        Console.WriteLine(text);
    }

    PrintCentered("LIFE");
    PrintCentered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    Console.WriteLine();
    Console.WriteLine();
    Console.WriteLine();
}

// 初始化模拟器的方法
Simulation InitializeSimulation(Pattern pattern, Matrix matrixToInitialize) {
    var newSimulation = new Simulation();

    // 将模式转录到模拟的中间位置，并计算初始人口
    for (var x = 0; x < pattern.Height; x++)
    {
        # 遍历 pattern 的宽度范围
        for (var y = 0; y < pattern.Width; y++)
        {
            # 如果 pattern 中当前位置的内容为' '，则跳过当前循环，继续下一次循环
            if (pattern.Content[x][y] == ' ')
                continue;

            # 在初始化的矩阵中，将对应位置标记为稳定状态
            matrixToInitialize[minX + x, minY + y] = CellState.Stable;
            # 增加新模拟的人口数量
            newSimulation.IncreasePopulation();
        }
    }

    # 返回新的模拟对象
    return newSimulation;
}

TimeSpan GetPauseBetweenIterations()
{
    // 检查参数个数，如果不是2个则返回0
    if (args.Length != 2) return TimeSpan.Zero;

    // 将参数转换为小写
    var parameter = args[0].ToLower();
    // 如果参数包含"wait"关键字
    if (parameter.Contains("wait"))
    {
        // 获取第二个参数的值
        var value = args[1];
        // 尝试将值转换为整数，如果成功则返回对应的时间间隔
        if (int.TryParse(value, out var sleepMilliseconds))
            return TimeSpan.FromMilliseconds(sleepMilliseconds);
    }

    // 返回0时间间隔
    return TimeSpan.Zero;
}

void ProcessSimulation()
{
    // 获取迭代之间的暂停时间间隔
    var pauseBetweenIterations = GetPauseBetweenIterations();
    // 设置是否无效的标志为false
    var isInvalid = false;

    // 无限循环
    while (true)
    }
}

public class Pattern
{
    public string[] Content { get; }
    public int Height { get; }
    public int Width { get; }

    // 构造函数，根据给定的模式行集合初始化模式对象
    public Pattern(IReadOnlyCollection<string> patternLines)
    {
        // 设置模式的高度为行数
        Height = patternLines.Count;
        // 设置模式的宽度为行中最长的长度
        Width = patternLines.Max(x => x.Length);
        // 根据最大宽度对模式行进行规范化处理
        Content = NormalizeWidth(patternLines);
    }

    // 根据最大宽度对模式行进行规范化处理
    private string[] NormalizeWidth(IReadOnlyCollection<string> patternLines)
    {
        return patternLines
            .Select(x => x.PadRight(Width, ' '))
            .ToArray();
    }
}

/// <summary>
/// 表示模拟中给定细胞的状态。
/// </summary>
internal enum CellState
{
    Empty = 0,
    Stable = 1,
    Dying = 2,
    New = 3
}

public class Simulation
{
    public int Generation { get; private set; }

    public int Population { get; private set; }

    // 开始新一代模拟
    public void StartNewGeneration()
    {
        Generation++;
        Population = 0;
    }

    // 增加人口数量
    public void IncreasePopulation()
    {
        Population++;
    }
}

/// <summary>
/// 该类用于辅助调试，通过实现ToString()方法。
/// </summary>
class Matrix
{
    private readonly CellState[,] _matrix;

    // 构造函数，根据给定的高度和宽度初始化矩阵
    public Matrix(int height, int width)
    {
        _matrix = new CellState[height, width];
    }

    // 获取或设置指定位置的细胞状态
    public CellState this[int x, int y]
    {
        get => _matrix[x, y];
        set => _matrix[x, y] = value;
    }

    // 重写ToString()方法
    public override string ToString()
    {
        // 创建一个 StringBuilder 对象，用于构建字符串
        var stringBuilder = new StringBuilder();
        // 遍历矩阵的行
        for (var x = 0; x < _matrix.GetLength(0); x++)
        {
            // 遍历矩阵的列
            for (var y = 0; y < _matrix.GetLength(1); y++)
            {
                // 如果矩阵中的值为 0，则用空格代替，否则转换为字符串并添加到 StringBuilder 中
                var character = _matrix[x, y] == 0 ? " ": ((int)_matrix[x, y]).ToString();
                stringBuilder.Append(character);
            }
            // 在每行末尾添加换行符
            stringBuilder.AppendLine();
        }
        // 将 StringBuilder 转换为字符串并返回
        return stringBuilder.ToString();
    }
# 闭合前面的函数定义
```