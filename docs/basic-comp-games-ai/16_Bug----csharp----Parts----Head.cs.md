# `basic-computer-games\16_Bug\csharp\Parts\Head.cs`

```
using System.Text;  // 导入 System.Text 命名空间，用于使用 StringBuilder 类
using BugGame.Resources;  // 导入 BugGame.Resources 命名空间，用于使用 Message 类

namespace BugGame.Parts;  // 声明 BugGame.Parts 命名空间
internal class Head : ParentPart  // 声明 Head 类，继承自 ParentPart 类
{
    private Feelers _feelers = new();  // 声明私有字段 _feelers，初始化为新的 Feelers 对象

    public Head()  // 声明 Head 类的构造函数
        : base(Message.HeadAdded, Message.HeadNotNeeded)  // 调用父类的构造函数，传入两个 Message 类型的参数
    {
    }

    public override bool IsComplete => _feelers.IsComplete;  // 声明 IsComplete 属性，返回 _feelers.IsComplete 属性的值

    protected override bool TryAddCore(IPart part, out Message message)  // 声明 TryAddCore 方法，重写父类的方法
        => part switch  // 使用 switch 表达式
        {
            Feeler => _feelers.TryAddOne(out message),  // 如果 part 是 Feeler 类型，则调用 _feelers.TryAddOne 方法，并返回其结果
            _ => throw new NotSupportedException($"Can't add a {part.Name} to a {Name}.")  // 如果 part 不是 Feeler 类型，则抛出 NotSupportedException 异常
        };

    public void AppendTo(StringBuilder builder, char feelerCharacter)  // 声明 AppendTo 方法，接受一个 StringBuilder 对象和一个字符参数
    {
        if (IsPresent)  // 如果 IsPresent 属性为真
        {
            _feelers.AppendTo(builder, feelerCharacter);  // 调用 _feelers 的 AppendTo 方法，向 StringBuilder 对象中添加内容
            builder  // 使用 StringBuilder 对象
                .AppendLine("        HHHHHHH")  // 向 StringBuilder 对象中添加一行内容
                .AppendLine("        H     H")  // 向 StringBuilder 对象中添加一行内容
                .AppendLine("        H O O H")  // 向 StringBuilder 对象中添加一行内容
                .AppendLine("        H     H")  // 向 StringBuilder 对象中添加一行内容
                .AppendLine("        H  V  H")  // 向 StringBuilder 对象中添加一行内容
                .AppendLine("        HHHHHHH");  // 向 StringBuilder 对象中添加一行内容
        }
    }
}
```