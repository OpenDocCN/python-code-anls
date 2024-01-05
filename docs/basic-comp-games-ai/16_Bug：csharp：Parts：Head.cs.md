# `d:/src/tocomm/basic-computer-games\16_Bug\csharp\Parts\Head.cs`

```
using System.Text;  // 导入 System.Text 命名空间，用于使用其中的类和方法
using BugGame.Resources;  // 导入 BugGame.Resources 命名空间，用于使用其中的资源

namespace BugGame.Parts;  // 声明 BugGame.Parts 命名空间

internal class Head : ParentPart  // 声明 Head 类，继承自 ParentPart 类
{
    private Feelers _feelers = new();  // 声明私有字段 _feelers，并初始化为一个新的 Feelers 对象

    public Head()  // 声明 Head 类的构造函数
        : base(Message.HeadAdded, Message.HeadNotNeeded)  // 调用父类的构造函数，并传入参数 Message.HeadAdded 和 Message.HeadNotNeeded
    {
    }

    public override bool IsComplete => _feelers.IsComplete;  // 声明 IsComplete 属性，返回 _feelers.IsComplete 的值

    protected override bool TryAddCore(IPart part, out Message message)  // 声明 TryAddCore 方法，接受一个 IPart 类型的参数 part，并返回一个布尔值和一个 Message 类型的输出参数 message
        => part switch  // 使用 switch 语句对 part 进行匹配
        {
            Feeler => _feelers.TryAddOne(out message),  // 如果 part 是 Feeler 类型，则调用 _feelers.TryAddOne 方法，并将结果赋值给 message
_ => throw new NotSupportedException($"Can't add a {part.Name} to a {Name}.")
```
- 这是一个 lambda 表达式，用于抛出一个 NotSupportedException 异常，异常消息包含 part.Name 和 Name 的值。

```
public void AppendTo(StringBuilder builder, char feelerCharacter)
```
- 这是一个公共方法，用于将感觉器的内容附加到 StringBuilder 对象中，同时使用指定的字符作为分隔符。

```
if (IsPresent)
```
- 检查当前对象是否存在。

```
_feelers.AppendTo(builder, feelerCharacter);
```
- 调用 _feelers 对象的 AppendTo 方法，将感觉器的内容附加到 StringBuilder 对象中。

```
builder
    .AppendLine("        HHHHHHH")
    .AppendLine("        H     H")
    .AppendLine("        H O O H")
    .AppendLine("        H     H")
    .AppendLine("        H  V  H")
    .AppendLine("        HHHHHHH");
```
- 将一些文本内容附加到 StringBuilder 对象中，用于表示某种图形或形状。

```
}
```
- 方法的结束标记。
```