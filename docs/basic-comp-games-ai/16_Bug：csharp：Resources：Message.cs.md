# `16_Bug\csharp\Resources\Message.cs`

```
using BugGame.Parts;  // 导入 BugGame.Parts 命名空间

namespace BugGame.Resources;  // 声明 BugGame.Resources 命名空间

internal class Message  // 声明一个内部类 Message
{
    public static Message Rolled = new("rolled a {0}");  // 声明并初始化一个静态的 Message 对象 Rolled

    public static Message BodyAdded = new("now have a body.");  // 声明并初始化一个静态的 Message 对象 BodyAdded
    public static Message BodyNotNeeded = new("do not need a body.");  // 声明并初始化一个静态的 Message 对象 BodyNotNeeded

    public static Message NeckAdded = new("now have a neck.");  // 声明并初始化一个静态的 Message 对象 NeckAdded
    public static Message NeckNotNeeded = new("do not need a neck.");  // 声明并初始化一个静态的 Message 对象 NeckNotNeeded

    public static Message HeadAdded = new("needed a head.");  // 声明并初始化一个静态的 Message 对象 HeadAdded
    public static Message HeadNotNeeded = new("I do not need a head.", "You have a head.");  // 声明并初始化一个静态的 Message 对象 HeadNotNeeded

    public static Message TailAdded = new("I now have a tail.", "I now give you a tail.");  // 声明并初始化一个静态的 Message 对象 TailAdded
    public static Message TailNotNeeded = new("I do not need a tail.", "You already have a tail.");  // 声明并初始化一个静态的 Message 对象 TailNotNeeded
}
    # 创建一个静态的 Message 对象 FeelerAdded，包含两个消息字符串
    public static Message FeelerAdded = new("I get a feeler.", "I now give you a feeler");
    # 创建一个静态的 Message 对象 FeelersFull，包含两个消息字符串
    public static Message FeelersFull = new("I have 2 feelers already.", "You have two feelers already");

    # 创建一个静态的 Message 对象 LegAdded，包含一个消息字符串
    public static Message LegAdded = new("now have {0} legs");
    # 创建一个静态的 Message 对象 LegsFull，包含两个消息字符串
    public static Message LegsFull = new("I have 6 feet.", "You have 6 feet already");

    # 创建一个静态的 Message 对象 Complete，包含一个消息字符串
    public static Message Complete = new("bug is finished.");

    # 定义一个私有的构造函数，接受一个 common 字符串参数
    private Message(string common)
        # 调用另一个构造函数，传入拼接后的消息字符串
        : this("I " + common, "You " + common)
    {
    }

    # 定义一个私有的构造函数，接受两个字符串参数
    private Message(string i, string you)
    {
        # 将参数赋值给对象的属性
        I = i;
        You = you;
    }

    # 定义一个只读属性 I，用于获取消息的第一个字符串
    public string I { get; }
    public string You { get; }  # 定义一个公共属性 You，用于获取值

    public static Message DoNotHaveA(Part part) => new($"do not have a {part.Name}");  # 定义一个静态方法 DoNotHaveA，用于返回一个 Message 对象，内容为 "do not have a {part.Name}"

    public Message ForValue(int quantity) => new(string.Format(I, quantity), string.Format(You, quantity));  # 定义一个方法 ForValue，用于返回一个 Message 对象，内容为 string.Format(I, quantity) 和 string.Format(You, quantity)
```