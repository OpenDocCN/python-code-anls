# `basic-computer-games\16_Bug\csharp\Resources\Message.cs`

```py
// BugGame.Parts 命名空间的引用
using BugGame.Parts;

// BugGame.Resources 命名空间的定义
namespace BugGame.Resources
{
    // Message 类的定义，限定为内部访问
    internal class Message
    {
        // 静态字段 Rolled 的定义，初始化为指定格式的消息
        public static Message Rolled = new("rolled a {0}");

        // 静态字段 BodyAdded 的定义，初始化为指定消息
        public static Message BodyAdded = new("now have a body.");
        public static Message BodyNotNeeded = new("do not need a body.");

        // 静态字段 NeckAdded 的定义，初始化为指定消息
        public static Message NeckAdded = new("now have a neck.");
        public static Message NeckNotNeeded = new("do not need a neck.");

        // 静态字段 HeadAdded 的定义，初始化为指定消息
        public static Message HeadAdded = new("needed a head.");
        public static Message HeadNotNeeded = new("I do not need a head.", "You have a head.");

        // 静态字段 TailAdded 的定义，初始化为指定消息
        public static Message TailAdded = new("I now have a tail.", "I now give you a tail.");
        public static Message TailNotNeeded = new("I do not need a tail.", "You already have a tail.");

        // 静态字段 FeelerAdded 的定义，初始化为指定消息
        public static Message FeelerAdded = new("I get a feeler.", "I now give you a feeler");
        public static Message FeelersFull = new("I have 2 feelers already.", "You have two feelers already");

        // 静态字段 LegAdded 的定义，初始化为指定格式的消息
        public static Message LegAdded = new("now have {0} legs");
        public static Message LegsFull = new("I have 6 feet.", "You have 6 feet already");

        // 静态字段 Complete 的定义，初始化为指定消息
        public static Message Complete = new("bug is finished.");

        // 私有构造函数，接受一个公共消息作为参数
        private Message(string common)
            : this("I " + common, "You " + common)
        {
        }

        // 私有构造函数，接受两个不同的消息作为参数
        private Message(string i, string you)
        {
            I = i;
            You = you;
        }

        // 只读属性，返回 I 消息
        public string I { get; }
        // 只读属性，返回 You 消息
        public string You { get; }

        // 静态方法，接受一个 Part 参数，返回一个新的 Message 对象
        public static Message DoNotHaveA(Part part) => new($"do not have a {part.Name}");

        // 实例方法，接受一个整数参数，返回一个新的 Message 对象
        public Message ForValue(int quantity) => new(string.Format(I, quantity), string.Format(You, quantity));
    }
}
```