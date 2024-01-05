# `32_Diamond\csharp\Pattern.cs`

```
using System.Text;  # 导入 System.Text 模块，用于处理文本数据
using static Diamond.Resources.Resource;  # 导入 Diamond.Resources.Resource 模块的所有内容

namespace Diamond;  # 定义 Diamond 命名空间

internal class Pattern  # 定义内部类 Pattern
{
    private readonly IReadWrite _io;  # 声明私有成员变量 _io，类型为 IReadWrite 接口

    public Pattern(IReadWrite io)  # 定义构造函数，参数为 io
    {
        _io = io;  # 将参数 io 赋值给成员变量 _io
        io.Write(Streams.Introduction);  # 调用 io 对象的 Write 方法，将 Streams.Introduction 写入
    }

    public void Draw()  # 定义 Draw 方法
    {
        var diamondSize = _io.ReadNumber(Prompts.TypeNumber);  # 调用 _io 对象的 ReadNumber 方法，将结果赋值给 diamondSize
        _io.WriteLine();  # 调用 _io 对象的 WriteLine 方法
```
```python
        var diamondCount = (int)(60 / diamondSize);  # 计算钻石的数量，根据给定的钻石大小

        var diamondLines = new List<string>(GetDiamondLines(diamondSize)).AsReadOnly();  # 获取钻石的线条，并将其转换为只读列表

        for (int patternRow = 0; patternRow < diamondCount; patternRow++)  # 遍历钻石的行数
        {
            for (int diamondRow = 0; diamondRow < diamondLines.Count; diamondRow++)  # 遍历每一行的钻石线条
            {
                var line = new StringBuilder();  # 创建一个新的字符串构建器
                for (int patternColumn = 0; patternColumn < diamondCount; patternColumn++)  # 遍历钻石的列数
                {
                    line.PadToLength((int)(patternColumn * diamondSize)).Append(diamondLines[diamondRow]);  # 在行末尾添加钻石线条
                }
                _io.WriteLine(line);  # 输出每一行的钻石图案
            }
        }
    }

    public static IEnumerable<string> GetDiamondLines(float size)  # 定义一个方法，用于获取钻石的线条
# 从1开始，每隔2个取一个数，调用GetLine函数并返回结果
for (var i = 1; i <= size; i += 2)
{
    yield return GetLine(i);
}

# 从size-2开始，每隔2个取一个数，调用GetLine函数并返回结果
for (var i = size - 2; i >= 1; i -= 2)
{
    yield return GetLine(i);
}

# 定义GetLine函数，接受一个浮点数i作为参数，返回一个字符串
string GetLine(float i) =>
    string.Concat(
        new string(' ', (int)(size - i) / 2),  # 创建由空格组成的字符串，长度为(size - i) / 2
        new string('C', Math.Min((int)i, 2)),  # 创建由字符'C'组成的字符串，长度为i和2的最小值
        new string('!', Math.Max(0, (int)i - 2)));  # 创建由字符'!'组成的字符串，长度为i-2和0的最大值
```