# `basic-computer-games\26_Chomp\csharp\IOExtensions.cs`

```py
// 命名空间 Chomp 下的内部静态类 IOExtensions
namespace Chomp
{
    internal static class IOExtensions
    {
        // 为 IReadWrite 接口添加 ReadParameters 方法，返回一个元组包含三个值
        public static (float, int, int) ReadParameters(this IReadWrite io)
            => (
                // 调用 io 的 ReadNumber 方法，传入提示字符串 Resource.Prompts.HowManyPlayers，将结果转换为 int 类型
                (int)io.ReadNumber(Resource.Prompts.HowManyPlayers),
                // 调用 io 的 ReadNumberWithMax 方法，传入提示字符串 Resource.Prompts.HowManyRows、最大值 9、超出最大值时的提示字符串 Resource.Strings.TooManyRows
                io.ReadNumberWithMax(Resource.Prompts.HowManyRows, 9, Resource.Strings.TooManyRows),
                // 调用 io 的 ReadNumberWithMax 方法，传入提示字符串 Resource.Prompts.HowManyColumns、最大值 9、超出最大值时的提示字符串 Resource.Strings.TooManyColumns
                io.ReadNumberWithMax(Resource.Prompts.HowManyColumns, 9, Resource.Strings.TooManyColumns)
            );

        // 私有静态方法，为 IReadWrite 接口添加 ReadNumberWithMax 方法，返回一个整数值，接受三个参数
        private static int ReadNumberWithMax(this IReadWrite io, string initialPrompt, int max, string reprompt)
        {
            // 初始化提示字符串
            var prompt = initialPrompt;

            // 循环，直到返回结果
            while (true)
            {
                // 调用 io 的 ReadNumber 方法，传入提示字符串 prompt
                var response = io.ReadNumber(prompt);
                // 如果 response 小于等于 9，则返回 response 的整数值
                if (response <= 9) { return (int)response; }

                // 更新提示字符串为 reprompt 和 initialPrompt 的组合
                prompt = $"{reprompt} {initialPrompt.ToLowerInvariant()}";
            }
        }
    }
}
```