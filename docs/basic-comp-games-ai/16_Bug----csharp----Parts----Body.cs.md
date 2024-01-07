# `basic-computer-games\16_Bug\csharp\Parts\Body.cs`

```

// 使用 System.Text 命名空间中的 StringBuilder 类
// 使用 BugGame.Resources 命名空间中的资源
namespace BugGame.Parts
{
    // Body 类继承自 ParentPart 类
    internal class Body : ParentPart
    {
        // 创建 Neck、Tail、Legs 对象
        private readonly Neck _neck = new();
        private readonly Tail _tail = new();
        private readonly Legs _legs = new();

        // 构造函数，调用基类的构造函数
        public Body()
            : base(Message.BodyAdded, Message.BodyNotNeeded)
        {
        }

        // 判断是否完整
        public override bool IsComplete => _neck.IsComplete && _tail.IsComplete && _legs.IsComplete;

        // 尝试添加部件的核心方法
        protected override bool TryAddCore(IPart part, out Message message)
            => part switch
            {
                Neck => _neck.TryAdd(out message),
                Head or Feeler => _neck.TryAdd(part, out message),
                Tail => _tail.TryAdd(out message),
                Leg => _legs.TryAddOne(out message),
                _ => throw new NotSupportedException($"Can't add a {part.Name} to a {Name}.")
            };

        // 将部件附加到 StringBuilder 对象
        public void AppendTo(StringBuilder builder, char feelerCharacter)
        {
            if (IsPresent)
            {
                _neck.AppendTo(builder, feelerCharacter);
                builder
                    .AppendLine("     BBBBBBBBBBBB")
                    .AppendLine("     B          B")
                    .AppendLine("     B          B");
                _tail.AppendTo(builder);
                builder
                    .AppendLine("     BBBBBBBBBBBB");
                _legs.AppendTo(builder);
            }
        }
    }
}

```