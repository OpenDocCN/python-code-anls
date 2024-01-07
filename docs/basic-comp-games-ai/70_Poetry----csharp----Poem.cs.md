# `basic-computer-games\70_Poetry\csharp\Poem.cs`

```

# 使用 Poetry 资源中的 Resource 类
using static Poetry.Resources.Resource;

# 命名空间为 Poetry
namespace Poetry;

# 内部类 Poem
internal class Poem
{
    # 静态方法 Compose，接受 IReadWrite 和 IRandom 接口
    internal static void Compose(IReadWrite io, IRandom random)
    {
        # 使用 io 接口写入标题
        io.Write(Streams.Title);

        # 创建 Context 对象，传入 io 和 random
        var context = new Context(io, random);

        # 无限循环
        while (true)
        {
            # 写入诗句
            context.WritePhrase();
            # 可能写入逗号
            context.MaybeWriteComma();
            # 写入空格或换行

            # 内部循环
            while (true)
            {
                # 更新上下文
                context.Update(random);
                # 可能缩进
                context.MaybeIndent();

                # 如果组号有效，则跳出内部循环
                if (context.GroupNumberIsValid) { break; }

                # 重置组
                context.ResetGroup();

                # 可能完成一节诗
                if (context.MaybeCompleteStanza()) { break; }
            }
        }
    }
}

```