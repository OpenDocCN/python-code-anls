# `basic-computer-games\16_Bug\csharp\Resources\Message.cs`

```

// 使用 BugGame.Parts 命名空间
using BugGame.Parts;

// BugGame.Resources 命名空间下的内部类 Message
internal class Message
{
    // 静态字段 Rolled，表示滚动的消息
    public static Message Rolled = new("rolled a {0}");

    // 静态字段 BodyAdded，表示添加身体的消息
    public static Message BodyAdded = new("now have a body.");
    // 静态字段 BodyNotNeeded，表示不需要身体的消息
    public static Message BodyNotNeeded = new("do not need a body.");

    // 静态字段 NeckAdded，表示添加脖子的消息
    public static Message NeckAdded = new("now have a neck.");
    // 静态字段 NeckNotNeeded，表示不需要脖子的消息
    public static Message NeckNotNeeded = new("do not need a neck.");

    // 静态字段 HeadAdded，表示添加头部的消息
    public static Message HeadAdded = new("needed a head.");
    // 静态字段 HeadNotNeeded，表示不需要头部的消息
    public static Message HeadNotNeeded = new("I do not need a head.", "You have a head.");

    // 静态字段 TailAdded，表示添加尾巴的消息
    public static Message TailAdded = new("I now have a tail.", "I now give you a tail.");
    // 静态字段 TailNotNeeded，表示不需要尾巴的消息
    public static Message TailNotNeeded = new("I do not need a tail.", "You already have a tail.");

    // 静态字段 FeelerAdded，表示添加触角的消息
    public static Message FeelerAdded = new("I get a feeler.", "I now give you a feeler");
    // 静态字段 FeelersFull，表示触角已满的消息
    public static Message FeelersFull = new("I have 2 feelers already.", "You have two feelers already");

    // 静态字段 LegAdded，表示添加腿的消息
    public static Message LegAdded = new("now have {0} legs");
    // 静态字段 LegsFull，表示腿已满的消息
    public static Message LegsFull = new("I have 6 feet.", "You have 6 feet already");

    // 静态字段 Complete，表示昆虫已完成的消息
    public static Message Complete = new("bug is finished.");

    // 私有构造函数，接受一个 common 参数
    private Message(string common)
        : this("I " + common, "You " + common)
    {
    }

    // 私有构造函数，接受两个参数 i 和 you
    private Message(string i, string you)
    {
        I = i;
        You = you;
    }

    // 只读属性 I，表示消息的第一人称形式
    public string I { get; }
    // 只读属性 You，表示消息的第二人称形式
    public string You { get; }

    // 静态方法 DoNotHaveA，接受一个 Part 参数，表示没有某个部位的消息
    public static Message DoNotHaveA(Part part) => new($"do not have a {part.Name}");

    // 实例方法 ForValue，接受一个 quantity 参数，表示根据数量返回消息
    public Message ForValue(int quantity) => new(string.Format(I, quantity), string.Format(You, quantity));
}

```