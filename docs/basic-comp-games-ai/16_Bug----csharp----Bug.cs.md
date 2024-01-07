# `basic-computer-games\16_Bug\csharp\Bug.cs`

```

// 使用 System.Text 命名空间中的 StringBuilder 类
using System.Text;
// 引用 BugGame.Parts 和 BugGame.Resources 命名空间
using BugGame.Parts;
using BugGame.Resources;

// 声明 Bug 类，属于 BugGame 命名空间
namespace BugGame
{
    // Bug 类是 internal 类型，只能在当前程序集中访问
    internal class Bug
    {
        // 声明一个只读的 Body 对象 _body，并初始化为一个新的 Body 实例
        private readonly Body _body = new();

        // 声明 IsComplete 属性，返回 _body 的 IsComplete 属性值
        public bool IsComplete => _body.IsComplete;

        // 声明 TryAdd 方法，尝试向 _body 中添加一个部件，返回是否添加成功，并将消息存储在 out 参数 message 中
        public bool TryAdd(IPart part, out Message message) => _body.TryAdd(part, out message);

        // 声明 ToString 方法，返回一个包含 pronoun 和 feelerCharacter 的 Bug 字符串
        public string ToString(string pronoun, char feelerCharacter)
        {
            // 创建一个 StringBuilder 对象 builder，初始化为包含 pronoun 的 Bug 字符串
            var builder = new StringBuilder($"*****{pronoun} Bug*****").AppendLine().AppendLine().AppendLine();
            // 将 _body 的内容添加到 builder 中，使用 feelerCharacter 作为参数
            _body.AppendTo(builder, feelerCharacter);
            // 返回 builder 转换为字符串的结果
            return builder.ToString();
        }
    }
}

```