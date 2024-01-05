# `42_Gunner\csharp\Program.cs`

```
            // 打印游戏介绍
            PrintIntro();

            // 初始化变量 keepPlaying 为 "Y"
            string keepPlaying = "Y";

            // 当 keepPlaying 为 "Y" 时循环执行游戏
            while (keepPlaying == "Y") {
                // 执行游戏
                PlayGame();
                // 提示用户是否再次玩游戏，并获取用户输入
                Console.WriteLine("TRY AGAIN (Y OR N)");
                keepPlaying = Console.ReadLine();
            }
        }

        static void PlayGame()
        {
            // 初始化变量 totalAttempts 为 0
            int totalAttempts = 0;
            int amountOfGames = 0;  // 初始化游戏次数为0

            while (amountOfGames < 4) {  // 当游戏次数小于4时执行循环

                int maximumRange = new Random().Next(0, 40000) + 20000;  // 生成一个随机的最大射程范围
                Console.WriteLine($"MAXIMUM RANGE OF YOUR GUN IS {maximumRange} YARDS." + Environment.NewLine + Environment.NewLine + Environment.NewLine);  // 打印最大射程范围

                int distanceToTarget = (int) (maximumRange * (0.1 + 0.8 * new Random().NextDouble()));  // 生成一个随机的目标距离
                Console.WriteLine($"DISTANCE TO THE TARGET IS {distanceToTarget} YARDS.");  // 打印目标距离

                (bool gameWon, int attempts) = HitTheTarget(maximumRange, distanceToTarget);  // 调用HitTheTarget方法，返回游戏是否胜利和尝试次数

                if(!gameWon) {  // 如果游戏未胜利
                    Console.WriteLine(Environment.NewLine + "BOOM !!!!   YOU HAVE JUST BEEN DESTROYED" + Environment.NewLine +
                        "BY THE ENEMY." + Environment.NewLine + Environment.NewLine + Environment.NewLine
                    );  // 打印游戏失败信息
                    PrintReturnToBase();  // 调用PrintReturnToBase方法
                    break;  // 退出循环
                } else {
                    amountOfGames += 1;  // 游戏次数加1
                    totalAttempts += attempts;  # 将当前尝试次数累加到总尝试次数中

                    Console.WriteLine($"TOTAL ROUNDS EXPENDED WERE:{totalAttempts}");  # 打印总尝试次数

                    if (amountOfGames < 4) {  # 如果游戏次数小于4
                        Console.WriteLine("THE FORWARD OBSERVER HAS SIGHTED MORE ENEMY ACTIVITY...");  # 打印观察员发现更多敌人活动
                    } else {  # 否则
                        if (totalAttempts > 18) {  # 如果总尝试次数大于18
                            PrintReturnToBase();  # 调用返回基地的函数
                        } else {  # 否则
                            Console.WriteLine($"NICE SHOOTING !!");  # 打印“射击得很好！”
                        }
                    }
                }
            }
        }

        static (bool, int) HitTheTarget(int maximumRange, int distanceToTarget)  # 定义一个函数，参数为最大射程和目标距离
        {
            int attempts = 0;  # 初始化尝试次数为0
            while (attempts < 6)  # 当尝试次数小于6时执行循环
            {
                int elevation = GetElevation();  # 获取高度信息

                int differenceBetweenTargetAndImpact = CalculateDifferenceBetweenTargetAndImpact(maximumRange, distanceToTarget, elevation);  # 计算目标和实际命中位置之间的距离差

                if (Math.Abs(differenceBetweenTargetAndImpact) < 100)  # 如果距离差的绝对值小于100
                {
                    Console.WriteLine($"*** TARGET DESTROYED *** {attempts} ROUNDS OF AMMUNITION EXPENDED.");  # 输出目标被摧毁的信息和尝试次数
                    return (true, attempts);  # 返回目标被摧毁的标志和尝试次数
                }
                else if (differenceBetweenTargetAndImpact > 100)  # 如果距离差大于100
                {
                    Console.WriteLine($"OVER TARGET BY {Math.Abs(differenceBetweenTargetAndImpact)} YARDS.");  # 输出超过目标的信息
                }
                else  # 如果距离差小于-100
                {
                    Console.WriteLine($"SHORT OF TARGET BY {Math.Abs(differenceBetweenTargetAndImpact)} YARDS.");  # 输出未达到目标的信息
                }
                attempts += 1;  # 增加尝试次数
            }
            return (false, attempts);  # 返回失败和尝试次数
        }

        static int CalculateDifferenceBetweenTargetAndImpact(int maximumRange, int distanceToTarget, int elevation)
        {
            double weirdNumber = 2 * elevation / 57.3;  # 计算奇怪的数字
            double distanceShot = maximumRange * Math.Sin(weirdNumber);  # 计算射击距离
            return (int)distanceShot - distanceToTarget;  # 返回目标和实际射击距离的差值
        }

        static void PrintReturnToBase()
        {
            Console.WriteLine("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!");  # 打印返回基地的消息
        }

        static int GetElevation()
        {
            Console.WriteLine("ELEVATION");  // 输出提示信息，要求用户输入仰角
            int elevation = int.Parse(Console.ReadLine());  // 从用户输入中获取仰角值并转换为整数类型
            if (elevation > 89) {  // 如果仰角大于89度
                Console.WriteLine("MAXIMUM ELEVATION IS 89 DEGREES");  // 输出最大仰角为89度的提示信息
                return GetElevation();  // 调用 GetElevation() 方法重新获取仰角
            }
            if (elevation < 1) {  // 如果仰角小于1度
                Console.WriteLine("MINIMUM ELEVATION IS 1 DEGREE");  // 输出最小仰角为1度的提示信息
                return GetElevation();  // 调用 GetElevation() 方法重新获取仰角
            }
            return elevation;  // 返回获取到的合法仰角值
        }

        static void PrintIntro()
        {
            Console.WriteLine(new String(' ', 30) + "GUNNER");  // 输出标题信息
            Console.WriteLine(new String(' ', 15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY" + Environment.NewLine + Environment.NewLine + Environment.NewLine);  // 输出初始介绍信息
            Console.WriteLine("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN");  // 输出游戏角色信息
            Console.WriteLine("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE");  // 输出指示信息
            Console.WriteLine("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN 100 YARDS");  // 输出游戏目标信息
# 打印一行文本到控制台，文本内容为"OF THE TARGET WILL DESTROY IT."，并换行
Console.WriteLine("OF THE TARGET WILL DESTROY IT." + Environment.NewLine)
```