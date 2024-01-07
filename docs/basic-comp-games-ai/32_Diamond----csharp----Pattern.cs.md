# `basic-computer-games\32_Diamond\csharp\Pattern.cs`

```

// 使用 System.Text 命名空间
// 使用 Diamond.Resources.Resource 中的静态资源
namespace Diamond;

// Pattern 类
internal class Pattern
{
    // 只读字段 _io，类型为 IReadWrite 接口
    private readonly IReadWrite _io;

    // Pattern 类的构造函数，接受一个 IReadWrite 类型的参数 io
    public Pattern(IReadWrite io)
    {
        // 将参数 io 赋值给字段 _io
        _io = io;
        // 调用 io 的 Write 方法，传入 Streams.Introduction 参数
        io.Write(Streams.Introduction);
    }

    // Draw 方法
    public void Draw()
    {
        // 从 io 中读取一个数字，提示为 Prompts.TypeNumber
        var diamondSize = _io.ReadNumber(Prompts.TypeNumber);
        // 写入一个空行
        _io.WriteLine();

        // 计算 diamondCount，即 60 除以 diamondSize 的整数部分
        var diamondCount = (int)(60 / diamondSize);

        // 创建一个只读的字符串列表 diamondLines，内容为 GetDiamondLines(diamondSize) 的结果
        var diamondLines = new List<string>(GetDiamondLines(diamondSize)).AsReadOnly();

        // 循环 diamondCount 次
        for (int patternRow = 0; patternRow < diamondCount; patternRow++)
        {
            // 循环 diamondLines 的长度次
            for (int diamondRow = 0; diamondRow < diamondLines.Count; diamondRow++)
            {
                // 创建一个 StringBuilder 对象 line
                var line = new StringBuilder();
                // 循环 diamondCount 次
                for (int patternColumn = 0; patternColumn < diamondCount; patternColumn++)
                {
                    // 在 line 后面添加 (patternColumn * diamondSize) 个空格，再添加 diamondLines[diamondRow]
                    line.PadToLength((int)(patternColumn * diamondSize)).Append(diamondLines[diamondRow]);
                }
                // 在 io 中写入 line
                _io.WriteLine(line);
            }
        }
    }

    // 静态方法 GetDiamondLines，返回一个字符串列表
    public static IEnumerable<string> GetDiamondLines(float size)
    {
        // 循环，i 从 1 开始，每次增加 2，直到 i 大于等于 size
        for (var i = 1; i <= size; i += 2)
        {
            // 返回 GetLine(i) 的结果
            yield return GetLine(i);
        }

        // 循环，i 从 size - 2 开始，每次减少 2，直到 i 大于等于 1
        for (var i = size - 2; i >= 1; i -= 2)
        {
            // 返回 GetLine(i) 的结果
            yield return GetLine(i);
        }

        // 定义 GetLine 方法，接受一个 float 类型的参数 i
        string GetLine(float i) =>
            // 返回一个字符串，内容为 (size - i) / 2 个空格，Math.Min(i, 2) 个 'C'，Math.Max(0, i - 2) 个 '!'
            string.Concat(
                new string(' ', (int)(size - i) / 2),
                new string('C', Math.Min((int)i, 2)),
                new string('!', Math.Max(0, (int)i - 2)));
    }
}

```