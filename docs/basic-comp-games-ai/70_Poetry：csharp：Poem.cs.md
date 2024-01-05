# `d:/src/tocomm/basic-computer-games\70_Poetry\csharp\Poem.cs`

```
using static Poetry.Resources.Resource;  # 导入 Poetry 资源中的 Resource 类

namespace Poetry;  # 声明 Poetry 命名空间

internal class Poem  # 内部类 Poem
{
    internal static void Compose(IReadWrite io, IRandom random)  # 定义静态方法 Compose，接受 IReadWrite 和 IRandom 接口类型的参数
    {
        io.Write(Streams.Title);  # 使用 io 接口的 Write 方法输出 Streams.Title 的内容

        var context = new Context(io, random);  # 创建 Context 对象，传入 io 和 random 参数

        while (true)  # 进入无限循环
        {
            context.WritePhrase();  # 调用 Context 对象的 WritePhrase 方法
            context.MaybeWriteComma();  # 调用 Context 对象的 MaybeWriteComma 方法
            context.WriteSpaceOrNewLine();  # 调用 Context 对象的 WriteSpaceOrNewLine 方法

            while (true)  # 进入内部无限循环
            {
                context.Update(random);  // 更新上下文信息，传入随机数
                context.MaybeIndent();  // 可能进行缩进处理

                if (context.GroupNumberIsValid) { break; }  // 如果上下文中的组号有效，则跳出循环

                context.ResetGroup();  // 重置上下文中的组信息

                if (context.MaybeCompleteStanza()) { break; }  // 如果可能完成一个段落的处理，则跳出循环
            }
        }
    }
}
```