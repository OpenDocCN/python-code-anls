# `basic-computer-games\16_Bug\csharp\Parts\Part.cs`

```
// 使用 BugGame.Resources 命名空间
using BugGame.Resources;

// BugGame.Parts 命名空间下的 Part 类，实现了 IPart 接口
internal class Part : IPart
{
    // 用于存储添加消息的私有字段
    private readonly Message _addedMessage;
    // 用于存储重复消息的私有字段
    private readonly Message _duplicateMessage;

    // Part 类的构造函数，接受添加消息和重复消息作为参数
    public Part(Message addedMessage, Message duplicateMessage)
    {
        // 初始化 _addedMessage 字段
        _addedMessage = addedMessage;
        // 初始化 _duplicateMessage 字段
        _duplicateMessage = duplicateMessage;
    }

    // 虚属性，表示零件是否完整
    public virtual bool IsComplete => IsPresent;

    // 表示零件是否存在的受保护属性
    protected bool IsPresent { get; private set; }

    // 获取零件名称
    public string Name => GetType().Name;

    // 尝试添加零件，返回是否添加成功，并输出消息
    public bool TryAdd(out Message message)
    {
        // 如果零件已存在，则输出重复消息并返回 false
        if (IsPresent)
        {
            message = _duplicateMessage;
            return false;
        }

        // 输出添加消息，将零件状态设置为存在，并返回 true
        message = _addedMessage;
        IsPresent = true;
        return true;
    }
}
```