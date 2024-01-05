# `d:/src/tocomm/basic-computer-games\58_Love\csharp\LovePattern.cs`

```
using System.IO;  // 导入 System.IO 命名空间，用于文件操作
using System.Text;  // 导入 System.Text 命名空间，用于处理字符串

namespace Love  // 命名空间 Love
{
    internal class LovePattern  // 定义内部类 LovePattern
    {
        private const int _lineLength = 60;  // 定义常量 _lineLength，值为 60
        private readonly int[] _segmentLengths = new[] {  // 定义只读整型数组 _segmentLengths
            // 数组元素省略
        };
    }
}
# 定义一个整型数组，用于存储一系列数字
private readonly int[] _segmentLengths = {
    12, 5, 1, 11, 8, 13, 27, 1, 11, 8, 13, 27, 1, 60
};
# 定义一个 StringBuilder 对象，用于存储生成的模式
private readonly StringBuilder _pattern = new();

# 定义一个构造函数，接受一个字符串参数
public LovePattern(string message)
{
    # 调用 Fill 方法，传入一个 SourceCharacters 对象
    Fill(new SourceCharacters(_lineLength, message));
}

# 定义一个填充模式的方法，接受一个 SourceCharacters 对象作为参数
private void Fill(SourceCharacters source)
{
    # 初始化一个变量 lineLength 用于存储当前行的长度
    var lineLength = 0;

    # 遍历 _segmentLengths 数组中的每个数字
    foreach (var segmentLength in _segmentLengths)
    {
        # 从 source 中获取指定长度的字符，并添加到 _pattern 中
        foreach (var character in source.GetCharacters(segmentLength))
        {
            _pattern.Append(character);
        }
        # 更新当前行的长度
        lineLength += segmentLength;
    }
}
            if (lineLength >= _lineLength)
            {
                _pattern.AppendLine();  // 如果行长度大于等于指定的行长度，就在模式中添加一个换行符
                lineLength = 0;  // 重置行长度为0
            }
        }
    }

    public override string ToString() =>
        new StringBuilder()
            .AppendLines(10)  // 在字符串构建器中添加10个空行
            .Append(_pattern)  // 在字符串构建器中添加模式
            .AppendLines(10)  // 在字符串构建器中再次添加10个空行
            .ToString();  // 将字符串构建器转换为最终的字符串并返回
}
```