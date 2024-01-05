# `00_Common\dotnet\Games.Common\IO\Token.cs`

```
using System.Text;  // 导入 System.Text 命名空间，用于操作字符串和文本
using System.Text.RegularExpressions;  // 导入 System.Text.RegularExpressions 命名空间，用于正则表达式匹配

namespace Games.Common.IO;  // 声明 Games.Common.IO 命名空间

internal class Token  // 声明 Token 类
{
    private static readonly Regex _numberPattern = new(@"^[+\-]?\d*(\.\d*)?([eE][+\-]?\d*)?");  // 声明一个静态只读的正则表达式对象，用于匹配数字模式

    internal Token(string value)  // 声明 Token 类的构造函数，接受一个字符串参数
    {
        String = value;  // 将传入的字符串赋值给 Token 对象的 String 属性

        var match = _numberPattern.Match(String);  // 使用正则表达式匹配传入的字符串

        IsNumber = float.TryParse(match.Value, out var number);  // 尝试将匹配到的字符串转换为浮点数，判断是否为数字并赋值给 IsNumber 属性
        Number = (IsNumber, number) switch  // 使用元组模式匹配判断 IsNumber 和 number 的值
        {
            (false, _) => float.NaN,  // 如果不是数字，则将 Number 属性赋值为 float.NaN
            (true, float.PositiveInfinity) => float.MaxValue,  // 如果是正无穷大，则将 Number 属性赋值为 float.MaxValue
            (true, float.NegativeInfinity) => float.MinValue,  // 如果条件为真且值为负无穷大，则返回float.MinValue
            (true, _) => number  // 如果条件为真，则返回number
        };
    }

    public string String { get; }  // 获取字符串属性
    public bool IsNumber { get; }  // 获取是否为数字属性
    public float Number { get; }  // 获取数字属性

    public override string ToString() => String;  // 重写ToString方法，返回字符串属性的值

    internal class Builder  // 内部类Builder
    {
        private readonly StringBuilder _builder = new();  // 创建StringBuilder对象
        private bool _isQuoted;  // 是否引用的标志
        private int _trailingWhiteSpaceCount;  // 尾部空白字符计数

        public Builder Append(char character)  // 在字符串构建器中追加字符
        {
            _builder.Append(character);
            _trailingWhiteSpaceCount = char.IsWhiteSpace(character) ? _trailingWhiteSpaceCount + 1 : 0;  // 如果当前字符是空白字符，则将_trailingWhiteSpaceCount加1，否则设为0

            return this;  // 返回当前的Builder对象
        }

        public Builder SetIsQuoted()  // 设置是否为引用
        {
            _isQuoted = true;  // 将_isQuoted标记设为true
            return this;  // 返回当前的Builder对象
        }

        public Token Build()  // 构建Token对象
        {
            if (!_isQuoted) { _builder.Length -= _trailingWhiteSpaceCount; }  // 如果不是引用，则从_builder中减去_trailingWhiteSpaceCount个字符
            return new Token(_builder.ToString());  // 返回一个新的Token对象，其内容为_builder的字符串表示
        }
    }
}
```