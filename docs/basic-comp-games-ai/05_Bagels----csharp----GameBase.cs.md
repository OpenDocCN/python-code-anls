# `05_Bagels\csharp\GameBase.cs`

```
# 使用 System 命名空间
# 创建 GameBase 类
class GameBase:
    # 创建受保护的 Rnd 属性，用于生成随机数
    def __init__(self):
        self.Rnd = Random()

    # 提示玩家再次尝试，并等待他们按下 Y 或 N
    # 如果玩家想再次尝试，则返回 True，如果他们已经完成游戏，则返回 False
    def TryAgain(self):
        # 设置控制台前景色为白色
        Console.ForegroundColor = ConsoleColor.White
        # 打印提示信息
        Console.WriteLine("Would you like to try again? (Press 'Y' for yes or 'N' for no)")
        # 设置控制台前景色为黄色
        Console.ForegroundColor = ConsoleColor.Yellow
        # 打印提示符
        Console.Write("> ")
			char pressedKey; // 定义一个字符变量用于存储用户输入的按键
			// Keep looping until we get a recognised input
			do
			{
				// Read a key, don't display it on screen
				ConsoleKeyInfo key = Console.ReadKey(true); // 从控制台读取用户输入的按键信息，不在屏幕上显示
				// Convert to upper-case so we don't need to care about capitalisation
				pressedKey = Char.ToUpper(key.KeyChar); // 将用户输入的按键转换为大写，避免大小写敏感
				// Is this a key we recognise? If not, keep looping
			} while (pressedKey != 'Y' && pressedKey != 'N'); // 如果用户输入的按键不是'Y'或'N'，则继续循环
			// Display the result on the screen
			Console.WriteLine(pressedKey); // 在屏幕上显示用户输入的按键

			// Return true if the player pressed 'Y', false for anything else.
			return (pressedKey == 'Y'); // 如果用户输入的按键是'Y'，则返回true，否则返回false
		}

	}
}
```