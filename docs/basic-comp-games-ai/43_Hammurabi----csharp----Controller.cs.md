# `basic-computer-games\43_Hammurabi\csharp\Controller.cs`

```

// 命名空间 Hammurabi
namespace Hammurabi
{
    /// <summary>
    /// 提供从用户输入读取的方法。
    /// </summary>
    public static class Controller
    {
        /// <summary>
        /// 持续提示用户输入数字，直到输入有效数字并更新游戏状态。
        /// </summary>
        /// <param name="state">
        /// 当前游戏状态。
        /// </param>
        /// <param name="prompt">
        /// 显示提示给用户的操作。
        /// </param>
        /// <param name="rule">
        /// 获取输入后要调用的规则。
        /// </param>
        /// <returns>
        /// 更新后的游戏状态。
        /// </returns>
        public static GameState UpdateGameState(
            GameState state,
            Action prompt,
            Func<GameState, int, (GameState newState, ActionResult result)> rule)
        {
            while (true)
            {
                prompt(); // 调用显示提示给用户的操作

                if (!Int32.TryParse(Console.ReadLine(), out var amount)) // 尝试将用户输入转换为整数
                {
                    View.ShowInvalidNumber(); // 显示无效数字的消息
                    continue;
                }

                var (newState, result) = rule(state, amount); // 调用规则处理输入

                switch (result) // 根据规则处理结果进行相应操作
                {
                    case ActionResult.InsufficientLand:
                        View.ShowInsufficientLand(state); // 显示土地不足的消息
                        break;
                    case ActionResult.InsufficientPopulation:
                        View.ShowInsufficientPopulation(state); // 显示人口不足的消息
                        break;
                    case ActionResult.InsufficientStores:
                        View.ShowInsufficientStores(state); // 显示储备不足的消息
                        break;
                    case ActionResult.Offense:
                        // 不确定为什么这里要结束游戏...
                        // 或许在70年代有这样的设定。
                        throw new GreatOffence(); // 抛出异常
                    default:
                        return newState; // 返回更新后的游戏状态
                }
            }
        }
    }
}

```