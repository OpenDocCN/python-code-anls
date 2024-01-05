# `d:/src/tocomm/basic-computer-games\58_Love\csharp\SourceCharacters.cs`

```
using System;  # 导入 System 模块

namespace Love;  # 命名空间为 Love

internal class SourceCharacters  # 定义名为 SourceCharacters 的类
{
    private readonly int _lineLength;  # 定义只读整型变量 _lineLength
    private readonly char[][] _chars;  # 定义只读二维字符数组 _chars
    private int _currentRow;  # 定义整型变量 _currentRow
    private int _currentIndex;  # 定义整型变量 _currentIndex

    public SourceCharacters(int lineLength, string message)  # 定义名为 SourceCharacters 的构造函数，接受 lineLength 和 message 两个参数
    {
        _lineLength = lineLength;  # 将参数 lineLength 赋值给 _lineLength
        _chars = new[] { new char[lineLength], new char[lineLength] };  # 创建一个包含两个长度为 lineLength 的字符数组的二维数组

        for (int i = 0; i < lineLength; i++)  # 循环 lineLength 次
        {
            _chars[0][i] = message[i % message.Length];  # 将 message 中的字符赋值给 _chars[0] 中的对应位置
            _chars[1][i] = ' ';  # 将空格赋值给 _chars[1] 中的对应位置
    public ReadOnlySpan<char> GetCharacters(int count)
    {
        // 创建一个只读的字符 Span，从 _chars 数组中的 _currentRow 行和 _currentIndex 列开始，长度为 count
        var span = new ReadOnlySpan<char>(_chars[_currentRow], _currentIndex, count);

        // 更新 _currentRow 和 _currentIndex 的值
        _currentRow = 1 - _currentRow;
        _currentIndex += count;

        // 如果 _currentIndex 大于等于 _lineLength，则将 _currentIndex 和 _currentRow 重置为 0
        if (_currentIndex >= _lineLength)
        {
            _currentIndex = _currentRow = 0;
        }

        // 返回创建的 Span
        return span;
    }
}
```