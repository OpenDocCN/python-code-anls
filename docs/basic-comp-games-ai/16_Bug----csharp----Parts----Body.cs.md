# `basic-computer-games\16_Bug\csharp\Parts\Body.cs`

```py
using System.Text;  // 导入 System.Text 命名空间，用于使用 StringBuilder 类
using BugGame.Resources;  // 导入 BugGame.Resources 命名空间，用于使用 Message 类

namespace BugGame.Parts;  // 声明 BugGame.Parts 命名空间
internal class Body : ParentPart  // 声明 Body 类，继承自 ParentPart 类
{
    private readonly Neck _neck = new();  // 声明并初始化私有字段 _neck，类型为 Neck
    private readonly Tail _tail = new();  // 声明并初始化私有字段 _tail，类型为 Tail
    private readonly Legs _legs = new();  // 声明并初始化私有字段 _legs，类型为 Legs

    public Body()  // 声明 Body 类的构造函数
        : base(Message.BodyAdded, Message.BodyNotNeeded)  // 调用父类的构造函数，传入参数 Message.BodyAdded 和 Message.BodyNotNeeded
    {
    }

    public override bool IsComplete => _neck.IsComplete && _tail.IsComplete && _legs.IsComplete;  // 声明 IsComplete 属性，返回 _neck.IsComplete 和 _tail.IsComplete 和 _legs.IsComplete 的逻辑与结果

    protected override bool TryAddCore(IPart part, out Message message)  // 声明 TryAddCore 方法，重写父类的方法
        => part switch  // 使用 switch 表达式
        {
            Neck => _neck.TryAdd(out message),  // 如果 part 类型为 Neck，则调用 _neck.TryAdd 方法，并将结果赋值给 message
            Head or Feeler => _neck.TryAdd(part, out message),  // 如果 part 类型为 Head 或 Feeler，则调用 _neck.TryAdd 方法，并将结果赋值给 message
            Tail => _tail.TryAdd(out message),  // 如果 part 类型为 Tail，则调用 _tail.TryAdd 方法，并将结果赋值给 message
            Leg => _legs.TryAddOne(out message),  // 如果 part 类型为 Leg，则调用 _legs.TryAddOne 方法，并将结果赋值给 message
            _ => throw new NotSupportedException($"Can't add a {part.Name} to a {Name}.")  // 其他情况抛出 NotSupportedException 异常
        };

    public void AppendTo(StringBuilder builder, char feelerCharacter)  // 声明 AppendTo 方法，接受 StringBuilder 和 char 类型的参数
    {
        if (IsPresent)  // 如果 IsPresent 属性为真
        {
            _neck.AppendTo(builder, feelerCharacter);  // 调用 _neck 的 AppendTo 方法，传入 StringBuilder 和 feelerCharacter 参数
            builder  // 使用 StringBuilder 对象 builder 进行操作
                .AppendLine("     BBBBBBBBBBBB")  // 在 builder 中追加一行文本
                .AppendLine("     B          B")  // 在 builder 中追加一行文本
                .AppendLine("     B          B");  // 在 builder 中追加一行文本
            _tail.AppendTo(builder);  // 调用 _tail 的 AppendTo 方法，传入 StringBuilder 参数
            builder  // 继续在 builder 中进行操作
                .AppendLine("     BBBBBBBBBBBB");  // 在 builder 中追加一行文本
            _legs.AppendTo(builder);  // 调用 _legs 的 AppendTo 方法，传入 StringBuilder 参数
        }
    }
}
```