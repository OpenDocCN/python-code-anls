# `basic-computer-games\16_Bug\csharp\Parts\Neck.cs`

```py
using System.Text;  // 导入 System.Text 命名空间，用于使用 StringBuilder 类
using BugGame.Resources;  // 导入 BugGame.Resources 命名空间，用于使用 Message 类

namespace BugGame.Parts;  // 声明 BugGame.Parts 命名空间
internal class Neck : ParentPart  // 声明 Neck 类，继承自 ParentPart 类
{
    private Head _head = new();  // 声明私有字段 _head，初始化为新的 Head 对象

    public Neck()  // 声明构造函数 Neck
        : base(Message.NeckAdded, Message.NeckNotNeeded)  // 调用父类的构造函数，传入 Message.NeckAdded 和 Message.NeckNotNeeded
    {
    }

    public override bool IsComplete => _head.IsComplete;  // 声明 IsComplete 属性，返回 _head.IsComplete 的值

    protected override bool TryAddCore(IPart part, out Message message)  // 声明受保护的 TryAddCore 方法，接受一个 IPart 类型的参数 part 和一个 Message 类型的输出参数 message
        => part switch  // 使用 switch 表达式
        {
            Head => _head.TryAdd(out message),  // 如果 part 是 Head 类型，则调用 _head.TryAdd 方法，并将结果赋值给 message
            Feeler => _head.TryAdd(part, out message),  // 如果 part 是 Feeler 类型，则调用 _head.TryAdd 方法，并将结果赋值给 message
            _ => throw new NotSupportedException($"Can't add a {part.Name} to a {Name}.")  // 如果 part 不是上述类型，则抛出 NotSupportedException 异常
        };

    public void AppendTo(StringBuilder builder, char feelerCharacter)  // 声明 AppendTo 方法，接受一个 StringBuilder 类型的参数 builder 和一个 char 类型的参数 feelerCharacter
    {
        if (IsPresent)  // 如果 IsPresent 属性为真
        {
            _head.AppendTo(builder, feelerCharacter);  // 调用 _head.AppendTo 方法，传入 builder 和 feelerCharacter 参数
            builder.AppendLine("          N N").AppendLine("          N N");  // 在 builder 中追加指定的字符串
        }
    }
}
```