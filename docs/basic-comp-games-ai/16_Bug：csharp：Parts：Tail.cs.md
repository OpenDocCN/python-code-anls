# `d:/src/tocomm/basic-computer-games\16_Bug\csharp\Parts\Tail.cs`

```
using System.Text;  // 导入 System.Text 命名空间，用于使用 StringBuilder 类
using BugGame.Resources;  // 导入 BugGame.Resources 命名空间，用于使用 Message 类

namespace BugGame.Parts;  // 声明 BugGame.Parts 命名空间
internal class Tail : Part  // 声明 Tail 类，继承自 Part 类
{
    public Tail()  // 声明 Tail 类的构造函数
        : base(Message.TailAdded, Message.TailNotNeeded)  // 调用基类 Part 的构造函数，传入 Message.TailAdded 和 Message.TailNotNeeded
    {
    }

    public void AppendTo(StringBuilder builder)  // 声明 AppendTo 方法，接受一个 StringBuilder 对象作为参数
    {
        if (IsPresent)  // 如果 IsPresent 属性为真
        {
            builder.AppendLine("TTTTTB          B");  // 在 StringBuilder 对象中追加一行字符串
        }
    }
}
```