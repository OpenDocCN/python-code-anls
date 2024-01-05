# `d:/src/tocomm/basic-computer-games\51_Hurkle\csharp\ConsoleHurkleView.cs`

```
            // 打印猜测的次数
            Console.WriteLine($"You missed! The hurkle is {failedGuessViewModel.Direction} of your guess.");
        }

        public void ShowWin()
        {
            // 打印玩家获胜的消息
            Console.WriteLine("Congratulations! You found the hurkle!");
        }

        public void ShowLose()
        {
            // 打印玩家失败的消息
            Console.WriteLine("Sorry, you didn't find the hurkle. Better luck next time!");
        }
    }
}
# 输出"GO "到控制台
Console.Write("GO ");
# 根据 failedGuessViewModel 的 Direction 属性进行不同的操作
switch(failedGuessViewModel.Direction)
{
    # 如果 Direction 为 East，则输出"EAST"到控制台
    case CardinalDirection.East:
        Console.WriteLine("EAST");
        break;
    # 如果 Direction 为 North，则输出"NORTH"到控制台
    case CardinalDirection.North:
        Console.WriteLine("NORTH");
        break;
    # 如果 Direction 为 South，则输出"SOUTH"到控制台
    case CardinalDirection.South:
        Console.WriteLine("SOUTH");
        break;
    # 如果 Direction 为 West，则输出"WEST"到控制台
    case CardinalDirection.West:
        Console.WriteLine("WEST");
        break;
    # 如果 Direction 为 NorthEast，则输出"NORTHEAST"到控制台
    case CardinalDirection.NorthEast:
        Console.WriteLine("NORTHEAST");
        break;
    # 如果 Direction 为 NorthWest，则输出"NORTHWEST"到控制台
    case CardinalDirection.NorthWest:
                    Console.WriteLine("NORTHWEST");  // 打印"NORTHWEST"，表示方向为西北
                    break;  // 跳出 switch 语句
                case CardinalDirection.SouthEast:  // 当方向为东南时
                    Console.WriteLine("SOUTHEAST");  // 打印"SOUTHEAST"，表示方向为东南
                    break;  // 跳出 switch 语句
                case CardinalDirection.SouthWest:  // 当方向为西南时
                    Console.WriteLine("SOUTHWEST");  // 打印"SOUTHWEST"，表示方向为西南
                    break;  // 跳出 switch 语句
            }

            Console.WriteLine();  // 打印空行

        }

        public void ShowLoss(LossViewModel lossViewModel)
        {
            Console.WriteLine();  // 打印空行
            Console.WriteLine($"SORRY, THAT'S {lossViewModel.MaxGuesses} GUESSES");  // 打印"SORRY, THAT'S {lossViewModel.MaxGuesses} GUESSES"，表示猜测次数已用完
            Console.WriteLine($"THE HURKLE IS AT {lossViewModel.HurkleLocation.X},{lossViewModel.HurkleLocation.Y}");  // 打印"THE HURKLE IS AT {lossViewModel.HurkleLocation.X},{lossViewModel.HurkleLocation.Y}"，表示 HURKLE 的位置坐标
        }
# 定义一个公共方法，用于展示游戏胜利的信息，接受一个 VictoryViewModel 对象作为参数
public void ShowVictory(VictoryViewModel victoryViewModel)
{
    # 在控制台输出空行
    Console.WriteLine();
    # 使用字符串插值输出玩家猜测的次数
    Console.WriteLine($"YOU FOUND HIM IN {victoryViewModel.CurrentGuessNumber} GUESSES!");
}
# 结束方法定义
}
# 结束类定义
}
```