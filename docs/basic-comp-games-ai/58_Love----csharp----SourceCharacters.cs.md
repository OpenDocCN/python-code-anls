# `basic-computer-games\58_Love\csharp\SourceCharacters.cs`

```

// 命名空间声明
using System;

// 声明内部类 SourceCharacters
namespace Love
{
    internal class SourceCharacters
    {
        // 声明私有只读字段 _lineLength 和 _chars
        private readonly int _lineLength;
        private readonly char[][] _chars;
        // 声明私有字段 _currentRow 和 _currentIndex
        private int _currentRow;
        private int _currentIndex;

        // 构造函数，初始化 _lineLength 和 _chars
        public SourceCharacters(int lineLength, string message)
        {
            _lineLength = lineLength;
            _chars = new[] { new char[lineLength], new char[lineLength] };

            // 使用 message 初始化 _chars[0]
            for (int i = 0; i < lineLength; i++)
            {
                _chars[0][i] = message[i % message.Length];
                _chars[1][i] = ' ';
            }
        }

        // 方法，返回指定数量的字符
        public ReadOnlySpan<char> GetCharacters(int count)
        {
            // 创建只读字符范围 span
            var span = new ReadOnlySpan<char>(_chars[_currentRow], _currentIndex, count);

            // 更新 _currentRow 和 _currentIndex
            _currentRow = 1 - _currentRow;
            _currentIndex += count;
            // 如果 _currentIndex 大于等于 _lineLength，则重置 _currentIndex 和 _currentRow
            if (_currentIndex >= _lineLength)
            {
                _currentIndex = _currentRow = 0;
            }

            // 返回 span
            return span;
        }
    }
}

```