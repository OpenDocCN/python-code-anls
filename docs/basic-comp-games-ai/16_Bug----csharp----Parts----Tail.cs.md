# `basic-computer-games\16_Bug\csharp\Parts\Tail.cs`

```py
# 导入 System.Text 命名空间
using System.Text;
# 导入 BugGame.Resources 命名空间
using BugGame.Resources;

# 声明 BugGame.Parts 命名空间下的 Tail 类，继承自 Part 类
internal class Tail : Part
{
    # 声明 Tail 类的构造函数，调用基类的构造函数，传入 Message.TailAdded 和 Message.TailNotNeeded
    public Tail()
        : base(Message.TailAdded, Message.TailNotNeeded)
    {
    }

    # 声明 AppendTo 方法，接受一个 StringBuilder 对象作为参数
    public void AppendTo(StringBuilder builder)
    {
        # 如果 Tail 对象存在
        if (IsPresent)
        {
            # 在 StringBuilder 对象中追加一行字符串 "TTTTTB          B"
            builder.AppendLine("TTTTTB          B");
        }
    }
}
```