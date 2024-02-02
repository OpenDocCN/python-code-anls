# `basic-computer-games\16_Bug\csharp\Parts\ParentPart.cs`

```py
// 使用 BugGame.Resources 命名空间
namespace BugGame.Parts
{
    // 抽象类 ParentPart 继承自 Part 类
    internal abstract class ParentPart : Part
    {
        // 构造函数，接受两个 Message 类型的参数
        public ParentPart(Message addedMessage, Message duplicateMessage)
            : base(addedMessage, duplicateMessage)
        {
        }

        // 尝试添加子部件的方法，返回布尔值和消息
        public bool TryAdd(IPart part, out Message message)
            => (part.GetType() == GetType(), IsPresent) switch
            {
                // 如果 part 的类型和当前实例的类型相同，调用 TryAdd 方法
                (true, _) => TryAdd(out message),
                // 如果 part 的类型和当前实例的类型不同，并且当前实例不包含子部件，调用 ReportDoNotHave 方法
                (false, false) => ReportDoNotHave(out message),
                // 其他情况调用 TryAddCore 方法
                _ => TryAddCore(part, out message)
            };

        // 抽象方法，尝试添加子部件的核心逻辑
        protected abstract bool TryAddCore(IPart part, out Message message);

        // 报告当前实例不包含子部件的方法
        private bool ReportDoNotHave(out Message message)
        {
            // 设置 message 为当前实例不包含子部件的消息
            message = Message.DoNotHaveA(this);
            return false;
        }
    }
}
```