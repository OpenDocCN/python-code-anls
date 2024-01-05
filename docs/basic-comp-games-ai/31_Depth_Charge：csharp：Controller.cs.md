# `31_Depth_Charge\csharp\Controller.cs`

```
// 引入 System 命名空间
using System;

// 定义名为 DepthCharge 的命名空间
namespace DepthCharge
{
    /// <summary>
    /// 包含从用户读取输入的函数。
    /// </summary>
    static class Controller
    {
        /// <summary>
        /// 从用户获取游戏区域的维度。
        /// </summary>
        /// <remarks>
        /// 注意，原始的 BASIC 版本允许维度值为 0 或更小。我们在这里进行一些额外的验证，以避免奇怪的行为。
        /// </remarks>
        public static int InputDimension()
        {
            // 调用 View 类的 PromptDimension 方法，提示用户输入游戏区域的维度
            View.PromptDimension();
            while (true)  # 创建一个无限循环
            {
                if (!Int32.TryParse(Console.ReadLine(), out var dimension))  # 从控制台读取用户输入的字符串，尝试将其转换为整数，如果成功则将结果赋值给dimension变量
                    View.ShowInvalidNumber();  # 如果用户输入的不是整数，则显示错误消息
                else
                if (dimension < 1)  # 如果用户输入的整数小于1
                    View.ShowInvalidDimension();  # 显示无效维度的错误消息
                else
                    return dimension;  # 如果用户输入的是有效的整数，则返回该整数
            }
        }

        /// <summary>
        /// Retrieves a set of coordinates from the user.
        /// </summary>
        /// <param name="trailNumber">
        /// The current trail number.
        /// </param>
        public static (int x, int y, int depth) InputCoordinates(int trailNumber)  # 定义一个公共静态方法InputCoordinates，该方法接受一个整数参数trailNumber，并返回一个包含三个整数的元组
# 提示用户猜测轨迹号
View.PromptGuess(trailNumber);

# 无限循环，直到用户输入正确格式的坐标
while (true)
{
    # 读取用户输入的坐标并以逗号分隔
    var coordinates = Console.ReadLine().Split(',');

    # 如果输入的坐标少于3个，显示提示信息
    if (coordinates.Length < 3)
        View.ShowTooFewCoordinates();
    else
    # 如果输入的坐标多于3个，显示提示信息
    if (coordinates.Length > 3)
        View.ShowTooManyCoordinates();
    else
    # 如果无法将输入的坐标转换为整数，显示提示信息
    if (!Int32.TryParse(coordinates[0], out var x) ||
        !Int32.TryParse(coordinates[1], out var y) ||
        !Int32.TryParse(coordinates[2], out var depth))
        View.ShowInvalidNumber();
    else
        # 如果输入的坐标格式正确，返回坐标的元组
        return (x, y, depth);
}
        }

        /// <summary>
        /// Retrieves the user's intention to play again (or not).
        /// </summary>
        public static bool InputPlayAgain()
        {
            // 调用 View 类的 PromptPlayAgain 方法，提示用户输入是否要再玩一次
            View.PromptPlayAgain();

            // 循环，直到用户输入有效的选项
            while (true)
            {
                // 读取用户输入
                switch (Console.ReadLine())
                {
                    // 如果用户输入 Y，则返回 true
                    case "Y":
                        return true;
                    // 如果用户输入 N，则返回 false
                    case "N":
                        return false;
                    // 如果用户输入其他内容，则提示用户输入有效的选项
                    default:
                        View.ShowInvalidYesOrNo();
                        break;
抱歉，给定的代码片段不完整，无法为每个语句添加注释。
```