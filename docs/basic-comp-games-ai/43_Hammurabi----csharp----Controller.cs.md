# `basic-computer-games\43_Hammurabi\csharp\Controller.cs`

```
# 命名空间声明，定义了一个命名空间 Hammurabi
namespace Hammurabi
{
    /// <summary>
    # 提供了一个静态类 Controller，用于读取用户输入的方法
    /// </summary>
    public static class Controller
        /// <summary>
        /// Continuously prompts the user to enter a number until he or she
        /// enters a valid number and updates the game state.
        /// </summary>
        /// <param name="state">
        /// The current game state.
        /// </param>
        /// <param name="prompt">
        /// Action that will display the prompt to the user.
        /// </param>
        /// <param name="rule">
        /// The rule to invoke once input is retrieved.
        /// </param>
        /// <returns>
        /// The updated game state.
        /// </returns>
        public static GameState UpdateGameState(
            GameState state,
            Action prompt,
            Func<GameState, int, (GameState newState, ActionResult result)> rule)
        {
            // 无限循环，持续提示用户输入数字，直到输入有效数字并更新游戏状态
            while (true)
            {
                prompt(); // 调用显示提示给用户的动作

                // 尝试将用户输入转换为整数，如果失败则显示无效数字并继续循环
                if (!Int32.TryParse(Console.ReadLine(), out var amount))
                {
                    View.ShowInvalidNumber(); // 显示无效数字的消息
                    continue;
                }

                // 调用规则函数处理输入，获取新的状态和结果
                var (newState, result) = rule(state, amount);

                // 根据结果显示不同的消息
                switch (result)
                {
                    case ActionResult.InsufficientLand:
                        View.ShowInsufficientLand(state); // 显示土地不足的消息
                        break;
                    case ActionResult.InsufficientPopulation:
                        View.ShowInsufficientPopulation(state); // 显示人口不足的消息
                        break;
                    case ActionResult.InsufficientStores:
                        View.ShowInsufficientStores(state); // 显示商店不足的消息
                        break;
                    case ActionResult.Offense:
                        // 不确定为什么这里要结束游戏...
                        // 或许在70年代有这样的设计理念
                        throw new GreatOffence(); // 抛出异常
                    default:
                        return newState; // 返回新的状态
                }
            }
        }
# 闭合前面的函数定义
```