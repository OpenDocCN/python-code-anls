# `basic-computer-games\16_Bug\csharp\Parts\PartCollection.cs`

```py
using System.Text;
using BugGame.Resources;

namespace BugGame.Parts;
// 声明 BugGame.Parts 命名空间

internal class PartCollection
{
    private readonly int _maxCount;
    // 声明只读整型变量 _maxCount
    private readonly Message _addedMessage;
    // 声明只读 Message 类型变量 _addedMessage
    private readonly Message _fullMessage;
    // 声明只读 Message 类型变量 _fullMessage
    private int _count;
    // 声明整型变量 _count

    public PartCollection(int maxCount, Message addedMessage, Message fullMessage)
    {
        _maxCount = maxCount;
        // 将参数 maxCount 赋值给 _maxCount
        _addedMessage = addedMessage;
        // 将参数 addedMessage 赋值给 _addedMessage
        _fullMessage = fullMessage;
        // 将参数 fullMessage 赋值给 _fullMessage
    }

    public bool IsComplete => _count == _maxCount;
    // 判断 _count 是否等于 _maxCount，返回布尔值

    public bool TryAddOne(out Message message)
    {
        if (_count < _maxCount)
        {
            _count++;
            // 如果 _count 小于 _maxCount，_count 加一
            message = _addedMessage.ForValue(_count);
            // 将 _count 传入 _addedMessage.ForValue 方法，将返回值赋给 message
            return true;
            // 返回 true
        }

        message = _fullMessage;
        // 将 _fullMessage 赋给 message
        return false;
        // 返回 false
    }

    protected void AppendTo(StringBuilder builder, int offset, int length, char character)
    {
        if (_count == 0) { return; }
        // 如果 _count 等于 0，直接返回

        for (var i = 0; i < length; i++)
        {
            builder.Append(' ', offset);
            // 在 builder 中添加 offset 个空格

            for (var j = 0; j < _count; j++)
            {
                builder.Append(character).Append(' ');
                // 在 builder 中添加 character 和一个空格
            }
            builder.AppendLine();
            // 在 builder 中添加换行符
        }
    }
}
```