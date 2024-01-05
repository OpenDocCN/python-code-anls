# `d:/src/tocomm/basic-computer-games\43_Hammurabi\csharp\Controller.cs`

```
        /// <param name="validator">
        /// Function that will validate the user input.
        /// </param>
        public static void PromptForNumber(GameState state, Action prompt, Func<string, bool> validator)
        {
            bool validInput = false;
            int userInput = 0;

            while (!validInput)
            {
                prompt(); // Display the prompt to the user
                string input = Console.ReadLine(); // Read user input from the console

                if (validator(input)) // Validate user input using the provided validator function
                {
                    userInput = int.Parse(input); // Convert the validated input to an integer
                    validInput = true; // Set validInput to true to exit the loop
                }
                else
                {
                    Console.WriteLine("Invalid input. Please try again."); // Display error message for invalid input
                }
            }

            state.Update(userInput); // Update the game state with the validated user input
        }
    }
}
        /// The rule to invoke once input is retrieved.
        /// </param>
        /// <returns>
        /// The updated game state.
        /// </returns>
        public static GameState UpdateGameState(
            GameState state,  // 接受当前游戏状态作为参数
            Action prompt,  // 接受一个动作作为参数，用于提示用户输入
            Func<GameState, int, (GameState newState, ActionResult result)> rule)  // 接受一个规则函数作为参数，该函数接受当前游戏状态和输入的整数，返回新的游戏状态和操作结果
        {
            while (true)  // 进入无限循环，直到游戏状态更新完成
            {
                prompt();  // 调用提示用户输入的动作

                if (!Int32.TryParse(Console.ReadLine(), out var amount))  // 从控制台读取用户输入的整数，如果输入不是整数，则显示无效数字并继续循环
                {
                    View.ShowInvalidNumber();  // 显示无效数字的提示
                    continue;  // 继续下一次循环
                }
                # 调用 rule 函数，传入当前状态和数量参数，获取新的状态和结果
                var (newState, result) = rule(state, amount);

                # 根据结果进行不同的处理
                switch (result)
                {
                    # 如果结果为土地不足，显示土地不足的提示
                    case ActionResult.InsufficientLand:
                        View.ShowInsufficientLand(state);
                        break;
                    # 如果结果为人口不足，显示人口不足的提示
                    case ActionResult.InsufficientPopulation:
                        View.ShowInsufficientPopulation(state);
                        break;
                    # 如果结果为储备不足，显示储备不足的提示
                    case ActionResult.InsufficientStores:
                        View.ShowInsufficientStores(state);
                        break;
                    # 如果结果为进攻行为，抛出 GreatOffence 异常
                    case ActionResult.Offense:
                        # 不确定为什么要在这里结束游戏...
                        # 或许在70年代这样做有意义。
                        throw new GreatOffence();
                    # 其他情况下返回新的状态
                    default:
                        return newState;
                }
抱歉，给定的代码片段不完整，无法为每个语句添加注释。
```