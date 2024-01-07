# `basic-computer-games\31_Depth_Charge\csharp\Controller.cs`

```

// 命名空间 DepthCharge 包含了用于从用户那里读取输入的函数
namespace DepthCharge
{
    /// <summary>
    /// 包含了从用户那里读取输入的函数
    /// </summary>
    static class Controller
    {
        /// <summary>
        /// 从用户那里获取游戏区域的维度
        /// </summary>
        /// <remarks>
        /// 注意，原始的BASIC版本允许维度值为0或更小。我们在这里进行了一些额外的验证，以避免奇怪的行为。
        /// </remarks>
        public static int InputDimension()
        {
            View.PromptDimension();

            while (true)
            {
                if (!Int32.TryParse(Console.ReadLine(), out var dimension))
                    View.ShowInvalidNumber();
                else
                if (dimension < 1)
                    View.ShowInvalidDimension();
                else
                    return dimension;
            }
        }

        /// <summary>
        /// 从用户那里获取一组坐标
        /// </summary>
        /// <param name="trailNumber">
        /// 当前轨迹编号
        /// </param>
        public static (int x, int y, int depth) InputCoordinates(int trailNumber)
        {
            View.PromptGuess(trailNumber);

            while (true)
            {
                var coordinates = Console.ReadLine().Split(',');

                if (coordinates.Length < 3)
                    View.ShowTooFewCoordinates();
                else
                if (coordinates.Length > 3)
                    View.ShowTooManyCoordinates();
                else
                if (!Int32.TryParse(coordinates[0], out var x) ||
                    !Int32.TryParse(coordinates[1], out var y) ||
                    !Int32.TryParse(coordinates[2], out var depth))
                    View.ShowInvalidNumber();
                else
                    return (x, y, depth);
            }
        }

        /// <summary>
        /// 获取用户是否想再玩一次的意图
        /// </summary>
        public static bool InputPlayAgain()
        {
            View.PromptPlayAgain();

            while (true)
            {
                switch (Console.ReadLine())
                {
                    case "Y":
                        return true;
                    case "N":
                        return false;
                    default:
                        View.ShowInvalidYesOrNo();
                        break;
                }
            }
        }
    }
}

```