# `16_Bug\csharp\Parts\PartCollection.cs`

```
# 使用 System.Text 和 BugGame.Resources 命名空间
using System.Text;
using BugGame.Resources;

# 声明 BugGame.Parts 命名空间下的 PartCollection 类
namespace BugGame.Parts;

# 声明 PartCollection 类为 internal 类
internal class PartCollection
{
    # 声明私有只读属性 _maxCount，_addedMessage，_fullMessage 和 _count
    private readonly int _maxCount;
    private readonly Message _addedMessage;
    private readonly Message _fullMessage;
    private int _count;

    # 声明 PartCollection 类的构造函数，接受 maxCount，addedMessage 和 fullMessage 作为参数
    public PartCollection(int maxCount, Message addedMessage, Message fullMessage)
    {
        # 将传入的参数赋值给对应的私有只读属性
        _maxCount = maxCount;
        _addedMessage = addedMessage;
        _fullMessage = fullMessage;
    }

    # 声明 PartCollection 类的 IsComplete 属性，返回 _count 是否等于 _maxCount 的布尔值
    public bool IsComplete => _count == _maxCount;
}
    # 尝试向计数器中添加一个，如果成功则返回 true，并返回添加的消息
    public bool TryAddOne(out Message message)
    {
        # 如果计数小于最大值
        if (_count < _maxCount)
        {
            # 增加计数
            _count++;
            # 为当前计数创建消息
            message = _addedMessage.ForValue(_count);
            # 返回 true
            return true;
        }

        # 如果计数已满，返回指定的消息
        message = _fullMessage;
        # 返回 false
        return false;
    }

    # 向字符串构建器中追加指定长度的指定字符
    protected void AppendTo(StringBuilder builder, int offset, int length, char character)
    {
        # 如果计数为 0，则直接返回
        if (_count == 0) { return; }

        # 循环追加指定长度的指定字符到字符串构建器中
        for (var i = 0; i < length; i++)
        {
            builder.Append(' ', offset);  # 在字符串构建器中添加指定数量的空格，偏移量为offset

            for (var j = 0; j < _count; j++)  # 循环_count次
            {
                builder.Append(character).Append(' ');  # 在字符串构建器中添加指定字符和空格
            }
            builder.AppendLine();  # 在字符串构建器中添加换行符
        }
    }
}
```