# `d:/src/tocomm/basic-computer-games\26_Chomp\csharp\IOExtensions.cs`

```
namespace Chomp;  # 命名空间声明

internal static class IOExtensions  # 声明一个内部静态类 IOExtensions
{
    public static (float, int, int) ReadParameters(this IReadWrite io)  # 声明一个静态方法 ReadParameters，接收一个 IReadWrite 类型的参数，并返回一个元组
        => (
            (int)io.ReadNumber(Resource.Prompts.HowManyPlayers),  # 调用 io 对象的 ReadNumber 方法，传入 HowManyPlayers 资源作为参数，并将结果转换为整数
            io.ReadNumberWithMax(Resource.Prompts.HowManyRows, 9, Resource.Strings.TooManyRows),  # 调用 io 对象的 ReadNumberWithMax 方法，传入 HowManyRows 资源、最大值 9 和 TooManyRows 资源作为参数
            io.ReadNumberWithMax(Resource.Prompts.HowManyColumns, 9, Resource.Strings.TooManyColumns)  # 调用 io 对象的 ReadNumberWithMax 方法，传入 HowManyColumns 资源、最大值 9 和 TooManyColumns 资源作为参数
        );

    private static int ReadNumberWithMax(this IReadWrite io, string initialPrompt, int max, string reprompt)  # 声明一个私有静态方法 ReadNumberWithMax，接收一个 IReadWrite 类型的参数、一个字符串 initialPrompt、一个整数 max 和一个字符串 reprompt，并返回一个整数
    {
        var prompt = initialPrompt;  # 初始化一个字符串变量 prompt，赋值为 initialPrompt

        while (true)  # 进入无限循环
        {
            var response = io.ReadNumber(prompt);  # 调用 io 对象的 ReadNumber 方法，传入 prompt 变量作为参数，并将结果赋值给 response
            if (response <= 9) { return (int)response; }  # 如果 response 小于等于 9，则将其转换为整数并返回
# 创建一个新的字符串 prompt，其值为 reprompt 和 initialPrompt 转换为小写后的字符串的拼接
prompt = $"{reprompt} {initialPrompt.ToLowerInvariant()}";
```