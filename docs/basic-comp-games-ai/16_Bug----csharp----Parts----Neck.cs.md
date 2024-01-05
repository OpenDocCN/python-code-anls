# `16_Bug\csharp\Parts\Neck.cs`

```
using System.Text;  # 导入 System.Text 模块
using BugGame.Resources;  # 导入 BugGame.Resources 模块

namespace BugGame.Parts;  # 定义 BugGame.Parts 命名空间

internal class Neck : ParentPart  # 定义 Neck 类，继承自 ParentPart 类
{
    private Head _head = new();  # 创建一个私有的 Head 类型的 _head 变量

    public Neck()  # 定义 Neck 类的构造函数
        : base(Message.NeckAdded, Message.NeckNotNeeded)  # 调用父类的构造函数，传入 Message.NeckAdded 和 Message.NeckNotNeeded 参数
    {
    }

    public override bool IsComplete => _head.IsComplete;  # 重写父类的 IsComplete 属性，判断 _head 是否完成

    protected override bool TryAddCore(IPart part, out Message message)  # 重写父类的 TryAddCore 方法
        => part switch  # 使用 switch 语句判断 part 的类型
        {
            Head => _head.TryAdd(out message),  # 如果 part 是 Head 类型，则调用 _head 的 TryAdd 方法，并将结果赋值给 message
            Feeler => _head.TryAdd(part, out message),
            _ => throw new NotSupportedException($"Can't add a {part.Name} to a {Name}.")
        };
```
这部分代码是一个 switch 语句，根据 Feeler 的值进行不同的操作。如果 Feeler 为 true，则调用 _head.TryAdd 方法，将 part 添加到对象中，并将结果存储在 message 中。如果 Feeler 为 false，则抛出一个 NotSupportedException 异常，异常消息为 "Can't add a {part.Name} to a {Name}."。

```
    public void AppendTo(StringBuilder builder, char feelerCharacter)
    {
        if (IsPresent)
        {
            _head.AppendTo(builder, feelerCharacter);
            builder.AppendLine("          N N").AppendLine("          N N");
        }
    }
```
这部分代码是一个公共方法 AppendTo，它接受一个 StringBuilder 对象和一个字符 feelerCharacter 作为参数。如果 IsPresent 为 true，则调用 _head.AppendTo 方法，将内容添加到 StringBuilder 对象中，并在后面添加两行特定的字符串。
```