# `basic-computer-games\32_Diamond\csharp\Pattern.cs`

```
using System.Text;
using static Diamond.Resources.Resource;

namespace Diamond;
// 创建名为 Diamond 的命名空间
internal class Pattern
{
    private readonly IReadWrite _io;
    // 声明私有只读字段 _io，类型为 IReadWrite 接口

    public Pattern(IReadWrite io)
    {
        _io = io;
        // 构造函数，接受一个 IReadWrite 类型的参数，并将其赋值给 _io 字段
        io.Write(Streams.Introduction);
        // 调用 io 对象的 Write 方法，将 Streams.Introduction 写入
    }

    public void Draw()
    {
        var diamondSize = _io.ReadNumber(Prompts.TypeNumber);
        // 从 _io 对象中读取一个数字，赋值给 diamondSize
        _io.WriteLine();
        // 调用 _io 对象的 WriteLine 方法，输出一个空行

        var diamondCount = (int)(60 / diamondSize);
        // 计算 diamondCount，将 60 除以 diamondSize，并转换为整数

        var diamondLines = new List<string>(GetDiamondLines(diamondSize)).AsReadOnly();
        // 创建一个只读的字符串列表 diamondLines，内容为 GetDiamondLines(diamondSize) 的结果

        for (int patternRow = 0; patternRow < diamondCount; patternRow++)
        {
            // 循环 diamondCount 次，每次循环执行以下操作
            for (int diamondRow = 0; diamondRow < diamondLines.Count; diamondRow++)
            {
                // 循环 diamondLines 列表的长度次，每次循环执行以下操作
                var line = new StringBuilder();
                // 创建一个 StringBuilder 对象 line
                for (int patternColumn = 0; patternColumn < diamondCount; patternColumn++)
                {
                    // 循环 diamondCount 次，每次循环执行以下操作
                    line.PadToLength((int)(patternColumn * diamondSize)).Append(diamondLines[diamondRow]);
                    // 在 line 后面添加 diamondLines[diamondRow]，并根据 patternColumn * diamondSize 进行填充
                }
                _io.WriteLine(line);
                // 调用 _io 对象的 WriteLine 方法，输出 line
            }
        }
    }

    public static IEnumerable<string> GetDiamondLines(float size)
    {
        // 创建一个公共的静态方法 GetDiamondLines，接受一个 float 类型的参数 size
        for (var i = 1; i <= size; i += 2)
        {
            // 循环，每次循环 i 增加 2
            yield return GetLine(i);
            // 返回 GetLine(i) 的结果
        }

        for (var i = size - 2; i >= 1; i -= 2)
        {
            // 循环，每次循环 i 减少 2
            yield return GetLine(i);
            // 返回 GetLine(i) 的结果
        }

        string GetLine(float i) =>
            // 创建一个名为 GetLine 的方法，接受一个 float 类型的参数 i
            string.Concat(
                new string(' ', (int)(size - i) / 2),
                // 连接一个由空格组成的字符串，长度为 (size - i) / 2
                new string('C', Math.Min((int)i, 2)),
                // 连接一个由字符 'C' 组成的字符串，长度为 i 和 2 中较小的那个
                new string('!', Math.Max(0, (int)i - 2)));
                // 连接一个由字符 '!' 组成的字符串，长度为 (i - 2) 和 0 中较大的那个
    }
}
```