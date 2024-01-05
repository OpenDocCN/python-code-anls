# `d:/src/tocomm/basic-computer-games\16_Bug\csharp\Parts\ParentPart.cs`

```
using BugGame.Resources;  # 导入 BugGame 资源模块

namespace BugGame.Parts;  # 声明 BugGame.Parts 命名空间

internal abstract class ParentPart : Part  # 声明一个抽象类 ParentPart，继承自 Part 类
{
    public ParentPart(Message addedMessage, Message duplicateMessage)  # 声明一个构造函数，接受两个 Message 类型的参数
        : base(addedMessage, duplicateMessage)  # 调用父类 Part 的构造函数，传入两个 Message 类型的参数
    {
    }

    public bool TryAdd(IPart part, out Message message)  # 声明一个公共方法 TryAdd，接受一个 IPart 类型的参数和一个 Message 类型的输出参数
        => (part.GetType() == GetType(), IsPresent) switch  # 使用元组和 switch 表达式进行条件判断
        {
            (true, _) => TryAdd(out message),  # 如果 part 的类型和当前对象的类型相同，则调用 TryAdd 方法，将结果赋给 message
            (false, false) => ReportDoNotHave(out message),  # 如果 part 的类型和当前对象的类型不同，并且 IsPresent 为 false，则调用 ReportDoNotHave 方法，将结果赋给 message
            _ => TryAddCore(part, out message)  # 其他情况下调用 TryAddCore 方法，将结果赋给 message
        };

    protected abstract bool TryAddCore(IPart part, out Message message);  # 声明一个抽象方法 TryAddCore，接受一个 IPart 类型的参数和一个 Message 类型的输出参数
    # 定义一个私有方法，用于报告某种情况下的消息，并返回 false
    def ReportDoNotHave(out message):
        message = Message.DoNotHaveA(this)
        return False
```