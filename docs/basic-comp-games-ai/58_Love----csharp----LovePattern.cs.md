# `basic-computer-games\58_Love\csharp\LovePattern.cs`

```py
// 引入命名空间 System.IO 和 System.Text
using System.IO;
using System.Text;

// 声明命名空间 Love
namespace Love
{
    // 声明内部类 LovePattern
    internal class LovePattern
    {
        // 声明私有常量 _lineLength，赋值为 60
        private const int _lineLength = 60;
        // 声明只读整型数组 _segmentLengths，包含一系列整数
        private readonly int[] _segmentLengths = new[] {
            // 数组包含大量整数，表示每个段落的长度
        };
        // 声明只读字符串构建器 _pattern
        private readonly StringBuilder _pattern = new();

        // 声明公共构造函数 LovePattern，接受字符串参数 message
        public LovePattern(string message)
        {
            // 调用 Fill 方法，传入 SourceCharacters 对象
            Fill(new SourceCharacters(_lineLength, message));
        }

        // 声明私有方法 Fill，接受 SourceCharacters 类型的参数 source
        private void Fill(SourceCharacters source)
        {
            // 声明整型变量 lineLength，赋值为 0
            var lineLength = 0;

            // 遍历 _segmentLengths 数组中的每个元素
            foreach (var segmentLength in _segmentLengths)
            {
                // 遍历 source.GetCharacters(segmentLength) 返回的字符集合
                foreach (var character in source.GetCharacters(segmentLength))
                {
                    // 将字符追加到 _pattern 中
                    _pattern.Append(character);
                }
                // 更新 lineLength
                lineLength += segmentLength;
                // 如果 lineLength 大于等于 _lineLength
                if (lineLength >= _lineLength)
                {
                    // 在 _pattern 中添加换行符
                    _pattern.AppendLine();
                    // 重置 lineLength 为 0
                    lineLength = 0;
                }
            }
        }

        // 重写 ToString 方法
        public override string ToString() =>
            // 创建新的字符串构建器，追加空行、_pattern、再追加空行，返回字符串
            new StringBuilder()
                .AppendLines(10)
                .Append(_pattern)
                .AppendLines(10)
                .ToString();
    }
}
```