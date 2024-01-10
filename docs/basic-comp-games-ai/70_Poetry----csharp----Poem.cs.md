# `basic-computer-games\70_Poetry\csharp\Poem.cs`

```
// 使用 Poetry.Resources.Resource 命名空间下的 Resource 类
using static Poetry.Resources.Resource;

// 声明 Poetry 命名空间
namespace Poetry
{
    // 声明 Poem 类，访问权限为 internal
    internal class Poem
    {
        // 声明 Compose 方法，参数为 IReadWrite 类型的 io 和 IRandom 类型的 random
        internal static void Compose(IReadWrite io, IRandom random)
        {
            // 调用 io 对象的 Write 方法，输出 Streams.Title 的内容
            io.Write(Streams.Title);

            // 创建 Context 对象，传入 io 和 random 对象
            var context = new Context(io, random);

            // 进入无限循环
            while (true)
            {
                // 调用 Context 对象的 WritePhrase 方法
                context.WritePhrase();
                // 调用 Context 对象的 MaybeWriteComma 方法
                context.MaybeWriteComma();
                // 调用 Context 对象的 WriteSpaceOrNewLine 方法

                // 进入内层无限循环
                while (true)
                {
                    // 调用 Context 对象的 Update 方法，传入 random 对象
                    context.Update(random);
                    // 调用 Context 对象的 MaybeIndent 方法
                    context.MaybeIndent();

                    // 如果 context.GroupNumberIsValid 为真，则跳出内层循环
                    if (context.GroupNumberIsValid) { break; }

                    // 重置 context 的组
                    context.ResetGroup();

                    // 如果 context.MaybeCompleteStanza() 为真，则跳出内层循环
                    if (context.MaybeCompleteStanza()) { break; }
                }
            }
        }
    }
}
```