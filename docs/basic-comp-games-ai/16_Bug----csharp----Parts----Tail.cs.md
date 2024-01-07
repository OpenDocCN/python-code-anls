# `basic-computer-games\16_Bug\csharp\Parts\Tail.cs`

```

// 使用 System.Text 命名空间中的 StringBuilder 类
using System.Text;
// 使用 BugGame.Resources 命名空间中的资源

// 声明 Tail 类，继承自 Part 类
namespace BugGame.Parts
{
    internal class Tail : Part
    {
        // 构造函数，调用基类的构造函数，传入尾部添加和尾部不需要的消息
        public Tail()
            : base(Message.TailAdded, Message.TailNotNeeded)
        {
        }

        // 将尾部内容追加到 StringBuilder 对象中
        public void AppendTo(StringBuilder builder)
        {
            // 如果尾部存在，则在 StringBuilder 中追加指定的字符串
            if (IsPresent)
            {
                builder.AppendLine("TTTTTB          B");
            }
        }
    }
}

```