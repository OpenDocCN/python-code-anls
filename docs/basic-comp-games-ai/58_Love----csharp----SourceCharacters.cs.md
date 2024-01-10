# `basic-computer-games\58_Love\csharp\SourceCharacters.cs`

```
// 命名空间声明，定义了 Love 命名空间
namespace Love
{
    // 定义了 SourceCharacters 类，用于处理源字符
    internal class SourceCharacters
    {
        // 声明私有只读字段 _lineLength，用于存储行长度
        private readonly int _lineLength;
        // 声明私有只读字段 _chars，用于存储字符数组
        private readonly char[][] _chars;
        // 声明私有字段 _currentRow，用于存储当前行数
        private int _currentRow;
        // 声明私有字段 _currentIndex，用于存储当前索引
        private int _currentIndex;

        // 构造函数，初始化 SourceCharacters 类的实例
        public SourceCharacters(int lineLength, string message)
        {
            // 将参数 lineLength 赋值给 _lineLength
            _lineLength = lineLength;
            // 创建一个包含两个长度为 lineLength 的字符数组的二维数组，并赋值给 _chars
            _chars = new[] { new char[lineLength], new char[lineLength] };

            // 循环遍历 lineLength 次
            for (int i = 0; i < lineLength; i++)
            {
                // 将 message 中的字符赋值给 _chars[0]，循环使用 message 中的字符
                _chars[0][i] = message[i % message.Length];
                // 将空格字符赋值给 _chars[1]
                _chars[1][i] = ' ';
            }
        }

        // 方法，返回只读字符跨度，用于获取指定数量的字符
        public ReadOnlySpan<char> GetCharacters(int count)
        {
            // 创建只读字符跨度 span，从 _chars[_currentRow] 的 _currentIndex 开始，长度为 count
            var span = new ReadOnlySpan<char>(_chars[_currentRow], _currentIndex, count);

            // 切换当前行
            _currentRow = 1 - _currentRow;
            // 更新当前索引
            _currentIndex += count;
            // 如果当前索引超过行长度
            if (_currentIndex >= _lineLength)
            {
                // 重置当前索引和当前行
                _currentIndex = _currentRow = 0;
            }

            // 返回字符跨度
            return span;
        }
    }
}
```