# `basic-computer-games\16_Bug\csharp\Parts\Part.cs`

```

// 使用 BugGame.Resources 命名空间
using BugGame.Resources;

// 声明 BugGame.Parts 命名空间下的 Part 类，实现 IPart 接口
internal class Part : IPart
{
    // 声明私有的 Message 类型的_addedMessage和_duplicateMessage字段
    private readonly Message _addedMessage;
    private readonly Message _duplicateMessage;

    // Part 类的构造函数，接受两个 Message 类型的参数
    public Part(Message addedMessage, Message duplicateMessage)
    {
        // 初始化_addedMessage和_duplicateMessage字段
        _addedMessage = addedMessage;
        _duplicateMessage = duplicateMessage;
    }

    // 声明虚拟属性 IsComplete，返回 IsPresent 属性的值
    public virtual bool IsComplete => IsPresent;

    // 声明受保护的 IsPresent 属性，可读可写
    protected bool IsPresent { get; private set; }

    // 声明 Name 属性，返回当前对象的类型名
    public string Name => GetType().Name;

    // 声明 TryAdd 方法，尝试添加部件，返回是否添加成功，并通过 out 参数返回相应的消息
    public bool TryAdd(out Message message)
    {
        // 如果部件已存在，则返回重复消息并返回 false
        if (IsPresent)
        {
            message = _duplicateMessage;
            return false;
        }

        // 否则返回添加成功消息，将 IsPresent 设置为 true，并返回 true
        message = _addedMessage;
        IsPresent = true;
        return true;
    }
}

```