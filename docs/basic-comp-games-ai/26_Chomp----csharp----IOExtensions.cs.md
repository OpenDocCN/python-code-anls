# `basic-computer-games\26_Chomp\csharp\IOExtensions.cs`

```

# 命名空间 Chomp 下的内部静态类 IOExtensions
namespace Chomp;

internal static class IOExtensions
{
    # 读取参数的扩展方法，返回一个元组包含三个值：浮点数、整数、整数
    public static (float, int, int) ReadParameters(this IReadWrite io)
        => (
            # 调用 ReadNumber 方法读取玩家数量
            (int)io.ReadNumber(Resource.Prompts.HowManyPlayers),
            # 调用 ReadNumberWithMax 方法读取行数，限制最大值为9
            io.ReadNumberWithMax(Resource.Prompts.HowManyRows, 9, Resource.Strings.TooManyRows),
            # 调用 ReadNumberWithMax 方法读取列数，限制最大值为9
            io.ReadNumberWithMax(Resource.Prompts.HowManyColumns, 9, Resource.Strings.TooManyColumns)
        );

    # 读取带有最大值限制的数字的私有方法
    private static int ReadNumberWithMax(this IReadWrite io, string initialPrompt, int max, string reprompt)
    {
        # 初始化提示信息
        var prompt = initialPrompt;

        # 循环直到输入的数字小于等于9
        while (true)
        {
            # 调用 ReadNumber 方法读取用户输入的数字
            var response = io.ReadNumber(prompt);
            # 如果输入的数字小于等于9，则返回该数字
            if (response <= 9) { return (int)response; }

            # 更新提示信息，要求用户重新输入
            prompt = $"{reprompt} {initialPrompt.ToLowerInvariant()}";
        }
    }
}

```