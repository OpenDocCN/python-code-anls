# `basic-computer-games\16_Bug\csharp\Bug.cs`

```
// 使用 System.Text 命名空间中的 StringBuilder 类
using System.Text;
// 引用 BugGame.Parts 和 BugGame.Resources 命名空间中的内容
using BugGame.Parts;
using BugGame.Resources;

// Bug 类定义在 BugGame 命名空间中
namespace BugGame
{
    // Bug 类被定义为 internal 类型，只能在当前程序集中访问
    internal class Bug
    {
        // Bug 类的私有字段 _body，类型为 Body，使用默认构造函数初始化
        private readonly Body _body = new();

        // Bug 类的公共属性 IsComplete，返回 _body 的 IsComplete 属性值
        public bool IsComplete => _body.IsComplete;

        // Bug 类的公共方法 TryAdd，尝试向 _body 中添加部件，返回是否添加成功以及消息
        public bool TryAdd(IPart part, out Message message) => _body.TryAdd(part, out message);

        // Bug 类的公共方法 ToString，返回 Bug 的字符串表示，包括代词和 feelerCharacter
        public string ToString(string pronoun, char feelerCharacter)
        {
            // 创建 StringBuilder 对象，初始化 Bug 的字符串表示
            var builder = new StringBuilder($"*****{pronoun} Bug*****").AppendLine().AppendLine().AppendLine();
            // 将 _body 的内容添加到 StringBuilder 中，使用指定的 feelerCharacter
            _body.AppendTo(builder, feelerCharacter);
            // 返回 StringBuilder 的字符串表示
            return builder.ToString();
        }
    }
}
```