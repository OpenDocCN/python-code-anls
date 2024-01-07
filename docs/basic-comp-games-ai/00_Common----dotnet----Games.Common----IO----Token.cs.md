# `basic-computer-games\00_Common\dotnet\Games.Common\IO\Token.cs`

```

// 使用 System.Text 和 System.Text.RegularExpressions 命名空间
using System.Text;
using System.Text.RegularExpressions;

// 声明 Games.Common.IO 命名空间
namespace Games.Common.IO
{
    // 声明 Token 类，为内部类
    internal class Token
    {
        // 声明静态只读的 _numberPattern 正则表达式对象
        private static readonly Regex _numberPattern = new(@"^[+\-]?\d*(\.\d*)?([eE][+\-]?\d*)?");

        // Token 类的构造函数，接受一个字符串参数
        internal Token(string value)
        {
            // 将参数值赋给 String 属性
            String = value;

            // 使用正则表达式匹配字符串
            var match = _numberPattern.Match(String);

            // 判断匹配结果是否为数字，如果是则转换为 float 类型
            IsNumber = float.TryParse(match.Value, out var number);
            Number = (IsNumber, number) switch
            {
                // 根据匹配结果设置 Number 属性的值
                (false, _) => float.NaN,
                (true, float.PositiveInfinity) => float.MaxValue,
                (true, float.NegativeInfinity) => float.MinValue,
                (true, _) => number
            };
        }

        // 声明 String 属性，存储 Token 的字符串值
        public string String { get; }
        // 声明 IsNumber 属性，表示 Token 是否为数字
        public bool IsNumber { get; }
        // 声明 Number 属性，存储 Token 的数字值
        public float Number { get; }

        // 重写 ToString 方法，返回 Token 的字符串值
        public override string ToString() => String;

        // 声明 Builder 内部类
        internal class Builder
        {
            // 声明 StringBuilder 对象 _builder
            private readonly StringBuilder _builder = new();
            // 声明 _isQuoted 变量，表示 Token 是否被引号引用
            private bool _isQuoted;
            // 声明 _trailingWhiteSpaceCount 变量，表示末尾空白字符的数量
            private int _trailingWhiteSpaceCount;

            // 声明 Append 方法，向 Token 中添加字符
            public Builder Append(char character)
            {
                _builder.Append(character);

                // 判断添加的字符是否为空白字符，更新 _trailingWhiteSpaceCount
                _trailingWhiteSpaceCount = char.IsWhiteSpace(character) ? _trailingWhiteSpaceCount + 1 : 0;

                return this;
            }

            // 声明 SetIsQuoted 方法，设置 Token 为被引号引用
            public Builder SetIsQuoted()
            {
                _isQuoted = true;
                return this;
            }

            // 声明 Build 方法，构建 Token 对象
            public Token Build()
            {
                // 如果 Token 没有被引号引用，则移除末尾空白字符
                if (!_isQuoted) { _builder.Length -= _trailingWhiteSpaceCount; }
                return new Token(_builder.ToString());
            }
        }
    }
}

```