# `basic-computer-games\00_Common\dotnet\Games.Common\IO\Token.cs`

```
using System.Text;
using System.Text.RegularExpressions;

namespace Games.Common.IO;
// 命名空间声明

internal class Token
{
    private static readonly Regex _numberPattern = new(@"^[+\-]?\d*(\.\d*)?([eE][+\-]?\d*)?");
    // 声明一个只读的正则表达式对象，用于匹配数字模式

    internal Token(string value)
    {
        String = value;
        // 初始化 Token 对象的 String 属性为传入的值

        var match = _numberPattern.Match(String);
        // 使用正则表达式匹配传入的值

        IsNumber = float.TryParse(match.Value, out var number);
        // 判断匹配结果是否为数字，并尝试解析为浮点数

        Number = (IsNumber, number) switch
        {
            (false, _) => float.NaN,
            (true, float.PositiveInfinity) => float.MaxValue,
            (true, float.NegativeInfinity) => float.MinValue,
            (true, _) => number
        };
        // 根据匹配结果设置 Number 属性的值
    }

    public string String { get; }
    // 声明 String 属性，用于存储 Token 对象的字符串值
    public bool IsNumber { get; }
    // 声明 IsNumber 属性，用于存储 Token 对象是否为数字
    public float Number { get; }
    // 声明 Number 属性，用于存储 Token 对象的数字值

    public override string ToString() => String;
    // 重写 ToString 方法，返回 Token 对象的字符串值

    internal class Builder
    {
        private readonly StringBuilder _builder = new();
        // 声明一个 StringBuilder 对象，用于构建 Token 对象的字符串值
        private bool _isQuoted;
        // 声明一个布尔变量，用于标记 Token 对象是否被引用
        private int _trailingWhiteSpaceCount;
        // 声明一个整型变量，用于记录 Token 对象末尾的空白字符数量

        public Builder Append(char character)
        {
            _builder.Append(character);
            // 将传入的字符追加到 StringBuilder 对象中

            _trailingWhiteSpaceCount = char.IsWhiteSpace(character) ? _trailingWhiteSpaceCount + 1 : 0;
            // 判断追加的字符是否为空白字符，更新末尾空白字符数量

            return this;
            // 返回 Builder 对象本身
        }

        public Builder SetIsQuoted()
        {
            _isQuoted = true;
            // 设置 Token 对象为引用状态
            return this;
            // 返回 Builder 对象本身
        }

        public Token Build()
        {
            if (!_isQuoted) { _builder.Length -= _trailingWhiteSpaceCount; }
            // 如果 Token 对象不是引用状态，则移除末尾的空白字符

            return new Token(_builder.ToString());
            // 返回一个新的 Token 对象，其字符串值为 StringBuilder 对象的内容
        }
    }
}
```