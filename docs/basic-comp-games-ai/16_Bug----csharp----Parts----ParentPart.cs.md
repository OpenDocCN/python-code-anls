# `basic-computer-games\16_Bug\csharp\Parts\ParentPart.cs`

```

// 使用 BugGame.Resources 命名空间中的资源
using BugGame.Resources;

// 声明 BugGame.Parts 命名空间中的抽象类 ParentPart，继承自 Part 类
internal abstract class ParentPart : Part
{
    // 构造函数，接受两个 Message 类型的参数，调用基类的构造函数
    public ParentPart(Message addedMessage, Message duplicateMessage)
        : base(addedMessage, duplicateMessage)
    {
    }

    // 尝试添加一个部件，返回是否添加成功以及消息
    public bool TryAdd(IPart part, out Message message)
        => (part.GetType() == GetType(), IsPresent) switch
        {
            // 如果部件类型与当前类型相同且已存在，则尝试添加并返回结果
            (true, _) => TryAdd(out message),
            // 如果部件类型与当前类型不同且不存在，则报告无法添加并返回结果
            (false, false) => ReportDoNotHave(out message),
            // 其他情况下调用 TryAddCore 方法进行添加并返回结果
            _ => TryAddCore(part, out message)
        };

    // 抽象方法，尝试添加部件的核心逻辑
    protected abstract bool TryAddCore(IPart part, out Message message);

    // 报告无法添加部件并返回结果
    private bool ReportDoNotHave(out Message message)
    {
        message = Message.DoNotHaveA(this);
        return false;
    }
}

```