# `d:/src/tocomm/basic-computer-games\28_Combat\csharp\Controller.cs`

```
            // BUG: This loop allows the player to assign negative values to
            // the armed forces, which should not be allowed
            // 循环允许玩家分配负值给武装部队，这是不应该被允许的

            // Prompt the player to distribute their armed forces
            Console.WriteLine("Distribute your armed forces:");

            // Get the number of infantry from the player
            Console.Write("Infantry: ");
            playerForces.Infantry = Convert.ToInt32(Console.ReadLine());

            // Get the number of cavalry from the player
            Console.Write("Cavalry: ");
            playerForces.Cavalry = Convert.ToInt32(Console.ReadLine());

            // Get the number of artillery from the player
            Console.Write("Artillery: ");
            playerForces.Artillery = Convert.ToInt32(Console.ReadLine());

            // Return the player's armed forces distribution
            return playerForces;
        }
    }
}
// 显示分配兵力的界面
View.ShowDistributeForces();

// 提示用户输入陆军规模，并将输入的值存储到变量 army 中
View.PromptArmySize(computerForces.Army);
var army = InputInteger();

// 提示用户输入海军规模，并将输入的值存储到变量 navy 中
View.PromptNavySize(computerForces.Navy);
var navy = InputInteger();

// 提示用户输入空军规模，并将输入的值存储到变量 airForce 中
View.PromptAirForceSize(computerForces.AirForce);
var airForce = InputInteger();

// 创建一个新的武装部队对象，其中包括用户输入的陆军、海军和空军规模
playerForces = new ArmedForces
{
    Army     = army,
    Navy     = navy,
    AirForce = airForce
};
            }
            while (playerForces.TotalTroops > computerForces.TotalTroops);
            // 循环直到玩家部队总数小于等于电脑部队总数

            return playerForces;
        }

        /// <summary>
        /// 获取用户下一次攻击的军事分支。
        /// </summary>
        public static MilitaryBranch GetAttackBranch(WarState state, bool isFirstTurn)
        {
            if (isFirstTurn)
                View.PromptFirstAttackBranch();
            else
                View.PromptNextAttackBranch(state.ComputerForces, state.PlayerForces);
            // 如果是第一轮攻击，则提示用户选择攻击军事分支，否则提示用户选择下一次攻击的军事分支

            // 如果用户在原始游戏中输入了无效的分支号码，代码会继续执行到军队情况。我们将保留这种行为。
            return Console.ReadLine() switch
            {
                "2" => MilitaryBranch.Navy,  // 如果输入为2，则返回Navy
                "3" => MilitaryBranch.AirForce,  // 如果输入为3，则返回AirForce
                _   => MilitaryBranch.Army  // 如果输入不是2或3，则返回Army
            };
        }

        /// <summary>
        /// Gets a valid attack size from the player for the given branch
        /// of the armed forces.
        /// </summary>
        /// <param name="troopsAvailable">
        /// The number of troops available.
        /// </param>
        public static int GetAttackSize(int troopsAvailable)
        {
            var attackSize = 0;  // 初始化攻击规模为0

            do
            {
                View.PromptAttackSize();  // 调用视图层的方法，提示用户输入攻击规模
                attackSize = InputInteger();  // 调用 InputInteger 方法获取用户输入的攻击规模
            }
            while (attackSize < 0 || attackSize > troopsAvailable);  // 当攻击规模小于0或大于可用部队数量时，继续循环

            return attackSize;  // 返回用户输入的攻击规模
        }

        /// <summary>
        /// Gets an integer value from the user.
        /// </summary>
        public static int InputInteger()  // 定义一个公共静态方法，用于获取用户输入的整数值
        {
            var value = default(int);  // 初始化一个整数变量 value

            while (!Int32.TryParse(Console.ReadLine(), out value))  // 当用户输入无法转换为整数时，继续循环
                View.PromptValidInteger();  // 调用视图层的方法，提示用户输入有效的整数

            return value;  // 返回用户输入的整数值
        }
    }
```

这部分代码是一个缩进错误，应该删除这两行代码。
```