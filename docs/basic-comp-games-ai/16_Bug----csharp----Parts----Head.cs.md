# `basic-computer-games\16_Bug\csharp\Parts\Head.cs`

```

// 使用 System.Text 命名空间中的 StringBuilder 类
using System.Text;
// 使用 BugGame.Resources 命名空间中的资源

// 声明 Head 类，继承自 ParentPart 类
namespace BugGame.Parts
{
    internal class Head : ParentPart
    {
        // 创建一个 Feelers 对象
        private Feelers _feelers = new();

        // 构造函数，初始化 Head 类的实例
        public Head()
            : base(Message.HeadAdded, Message.HeadNotNeeded)
        {
        }

        // 重写 IsComplete 属性
        public override bool IsComplete => _feelers.IsComplete;

        // 重写 TryAddCore 方法，尝试添加部件到 Head 类
        protected override bool TryAddCore(IPart part, out Message message)
            => part switch
            {
                Feeler => _feelers.TryAddOne(out message), // 如果 part 是 Feeler 类型，则尝试添加到 _feelers 对象
                _ => throw new NotSupportedException($"Can't add a {part.Name} to a {Name}.") // 否则抛出异常
            };

        // 将头部信息追加到 StringBuilder 对象中
        public void AppendTo(StringBuilder builder, char feelerCharacter)
        {
            if (IsPresent) // 如果头部存在
            {
                _feelers.AppendTo(builder, feelerCharacter); // 将 _feelers 对象的信息追加到 StringBuilder 对象中
                builder
                    .AppendLine("        HHHHHHH") // 追加一行字符串
                    .AppendLine("        H     H") // 追加一行字符串
                    .AppendLine("        H O O H") // 追加一行字符串
                    .AppendLine("        H     H") // 追加一行字符串
                    .AppendLine("        H  V  H") // 追加一行字符串
                    .AppendLine("        HHHHHHH"); // 追加一行字符串
            }
        }
    }
}

```