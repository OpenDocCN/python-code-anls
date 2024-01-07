# `basic-computer-games\16_Bug\csharp\Parts\Neck.cs`

```

// 使用 System.Text 命名空间中的 StringBuilder 类
// 使用 BugGame.Resources 命名空间中的资源
namespace BugGame.Parts
{
    // Neck 类继承自 ParentPart 类
    internal class Neck : ParentPart
    {
        // 创建一个 Head 对象
        private Head _head = new();

        // 构造函数，调用父类的构造函数，传入消息 NeckAdded 和 NeckNotNeeded
        public Neck()
            : base(Message.NeckAdded, Message.NeckNotNeeded)
        {
        }

        // 重写父类的 IsComplete 属性
        public override bool IsComplete => _head.IsComplete;

        // 重写父类的 TryAddCore 方法
        protected override bool TryAddCore(IPart part, out Message message)
            => part switch
            {
                // 如果 part 是 Head 类型，则调用 _head 的 TryAdd 方法
                Head => _head.TryAdd(out message),
                // 如果 part 是 Feeler 类型，则调用 _head 的 TryAdd 方法
                Feeler => _head.TryAdd(part, out message),
                // 如果 part 不是 Head 或 Feeler 类型，则抛出 NotSupportedException 异常
                _ => throw new NotSupportedException($"Can't add a {part.Name} to a {Name}.")
            };

        // 定义一个 AppendTo 方法，将 Neck 的内容追加到 StringBuilder 中
        public void AppendTo(StringBuilder builder, char feelerCharacter)
        {
            // 如果 Neck 存在
            if (IsPresent)
            {
                // 调用 _head 的 AppendTo 方法，将内容追加到 StringBuilder 中
                _head.AppendTo(builder, feelerCharacter);
                // 在 StringBuilder 中追加字符串
                builder.AppendLine("          N N").AppendLine("          N N");
            }
        }
    }
}

```