# `basic-computer-games\55_Life\csharp\Program.cs`

```

// 设置最大宽度和最大高度
const int maxWidth = 70;
const int maxHeight = 24;

// 打印提示信息，读取用户输入的图案
Console.WriteLine("ENTER YOUR PATTERN:");
var pattern = new Pattern(ReadPattern(limitHeight: maxHeight).ToArray());

// 计算图案在矩阵中的位置
var minX = 10 - pattern.Height / 2;
var minY = 34 - pattern.Width / 2;
var maxX = maxHeight - 1;
var maxY = maxWidth - 1;

// 创建矩阵和模拟对象
var matrix = new Matrix(height: maxHeight, width: maxWidth);
var simulation = InitializeSimulation(pattern, matrix);

// 打印游戏标题
PrintHeader();
// 处理模拟
ProcessSimulation();

// 读取用户输入的图案
IEnumerable<string> ReadPattern(int limitHeight)
{
    for (var i = 0; i < limitHeight; i++)
    {
        var input = Console.ReadLine();
        if (input.ToUpper() == "DONE")
        {
            break;
        }

        // 如果输入以'.'开头，则去掉'.'后返回
        if (input.StartsWith('.'))
            yield return input.Substring(1, input.Length - 1);

        yield return input;
    }
}

// 打印游戏标题
void PrintHeader()
{
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

// 初始化模拟
Simulation InitializeSimulation(Pattern pattern, Matrix matrixToInitialize) {
    var newSimulation = new Simulation();

    // 将图案转录到模拟的中间位置，并计算初始人口
    for (var x = 0; x < pattern.Height; x++)
    {
        for (var y = 0; y < pattern.Width; y++)
        {
            if (pattern.Content[x][y] == ' ')
                continue;

            matrixToInitialize[minX + x, minY + y] = CellState.Stable;
            newSimulation.IncreasePopulation();
        }
    }

    return newSimulation;
}

// 获取迭代之间的暂停时间
TimeSpan GetPauseBetweenIterations()
{
    if (args.Length != 2) return TimeSpan.Zero;

    var parameter = args[0].ToLower();
    if (parameter.Contains("wait"))
    {
        var value = args[1];
        if (int.TryParse(value, out var sleepMilliseconds))
            return TimeSpan.FromMilliseconds(sleepMilliseconds);
    }

    return TimeSpan.Zero;
}

// 模拟处理
void ProcessSimulation()
{
    // 实现模拟处理的逻辑
}

// 图案类
public class Pattern
{
    public string[] Content { get; }
    public int Height { get; }
    public int Width { get; }

    public Pattern(IReadOnlyCollection<string> patternLines)
    {
        Height = patternLines.Count;
        Width = patternLines.Max(x => x.Length);
        Content = NormalizeWidth(patternLines);
    }

    private string[] NormalizeWidth(IReadOnlyCollection<string> patternLines)
    {
        return patternLines
            .Select(x => x.PadRight(Width, ' '))
            .ToArray();
    }
}

// 表示模拟中给定单元格的状态
internal enum CellState
{
    Empty = 0,
    Stable = 1,
    Dying = 2,
    New = 3
}

// 模拟类
public class Simulation
{
    public int Generation { get; private set; }

    public int Population { get; private set; }

    public void StartNewGeneration()
    {
        Generation++;
        Population = 0;
    }

    public void IncreasePopulation()
    {
        Population++;
    }
}

// 用于调试的矩阵类
class Matrix
{
    private readonly CellState[,] _matrix;

    public Matrix(int height, int width)
    {
        _matrix = new CellState[height, width];
    }

    public CellState this[int x, int y]
    {
        get => _matrix[x, y];
        set => _matrix[x, y] = value;
    }

    public override string ToString()
    {
        var stringBuilder = new StringBuilder();
        for (var x = 0; x < _matrix.GetLength(0); x++)
        {
            for (var y = 0; y < _matrix.GetLength(1); y++)
            {
                var character = _matrix[x, y] == 0 ? " ": ((int)_matrix[x, y]).ToString();
                stringBuilder.Append(character);
            }

            stringBuilder.AppendLine();
        }
        return stringBuilder.ToString();
    }
}

```