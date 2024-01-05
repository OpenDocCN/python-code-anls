# `d:/src/tocomm/basic-computer-games\16_Bug\csharp\Parts\Body.cs`

```
using System.Text;  // 导入 System.Text 命名空间，用于处理字符串和文本
using BugGame.Resources;  // 导入 BugGame.Resources 命名空间，用于引用游戏资源

namespace BugGame.Parts;  // 声明 BugGame.Parts 命名空间

internal class Body : ParentPart  // 声明 Body 类，继承自 ParentPart 类
{
    private readonly Neck _neck = new();  // 声明私有字段 _neck，并初始化为新的 Neck 实例
    private readonly Tail _tail = new();  // 声明私有字段 _tail，并初始化为新的 Tail 实例
    private readonly Legs _legs = new();  // 声明私有字段 _legs，并初始化为新的 Legs 实例

    public Body()  // 声明 Body 类的构造函数
        : base(Message.BodyAdded, Message.BodyNotNeeded)  // 调用父类的构造函数，并传入参数 Message.BodyAdded 和 Message.BodyNotNeeded
    {
    }

    public override bool IsComplete => _neck.IsComplete && _tail.IsComplete && _legs.IsComplete;  // 重写父类的 IsComplete 属性，判断是否完整

    protected override bool TryAddCore(IPart part, out Message message)  // 重写父类的 TryAddCore 方法
        => part switch  // 使用 switch 语句对 part 进行匹配
        {
            Neck => _neck.TryAdd(out message), // 如果 part 是 Neck，则尝试将其添加到 _neck 中，并返回消息
            Head or Feeler => _neck.TryAdd(part, out message), // 如果 part 是 Head 或 Feeler，则尝试将其添加到 _neck 中，并返回消息
            Tail => _tail.TryAdd(out message), // 如果 part 是 Tail，则尝试将其添加到 _tail 中，并返回消息
            Leg => _legs.TryAddOne(out message), // 如果 part 是 Leg，则尝试将其添加到 _legs 中，并返回消息
            _ => throw new NotSupportedException($"Can't add a {part.Name} to a {Name}.") // 如果 part 不是上述任何一种情况，则抛出异常
        };

    public void AppendTo(StringBuilder builder, char feelerCharacter)
    {
        if (IsPresent) // 如果 part 存在
        {
            _neck.AppendTo(builder, feelerCharacter); // 将 _neck 的内容添加到 StringBuilder 中，并使用 feelerCharacter
            builder
                .AppendLine("     BBBBBBBBBBBB") // 在 StringBuilder 中添加一行内容
                .AppendLine("     B          B") // 在 StringBuilder 中添加一行内容
                .AppendLine("     B          B"); // 在 StringBuilder 中添加一行内容
            _tail.AppendTo(builder); // 将 _tail 的内容添加到 StringBuilder 中
            builder
                .AppendLine("     BBBBBBBBBBBB"); // 在 StringBuilder 中添加一行内容
# 将_legs对象添加到builder中
_legs.AppendTo(builder);
# 结束当前的代码块
}
# 结束当前的类定义
}
# 结束当前的命名空间定义
}
```