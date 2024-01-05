# `d:/src/tocomm/basic-computer-games\51_Hurkle\csharp\Program.cs`

```
            // 设置变量 N 的值为 5
            int N = 5;
            // 设置变量 G 的值为 10
            int G = 10;
            var N=5; // 定义变量N为5，表示玩家有5次尝试的机会
            var G=10; // 定义变量G为10，表示游戏的网格大小为10*10
            /*
            210 PRINT
            220 PRINT "A HURKLE IS HIDING ON A";G;"BY";G;"GRID. HOMEBASE"
            230 PRINT "ON THE GRID IS POINT 0,0 AND ANY GRIDPOINT IS A"
            240 PRINT "PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. TRY TO"
            250 PRINT "GUESS THE HURKLE'S GRIDPOINT. YOU GET";N;"TRIES."
            260 PRINT "AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE"
            270 PRINT "DIRECTION TO GO TO LOOK FOR THE HURKLE."
            280 PRINT
            */
            // 使用字符串格式化通过'$'字符串
            Console.WriteLine(); // 输出空行
            Console.WriteLine($"A HURKLE IS HIDING ON A {G} BY {G} GRID. HOMEBASE"); // 输出游戏提示信息，包括网格大小
            Console.WriteLine(@"ON THE GRID IS POINT 0,0 AND ANY GRIDPOINT IS A"); // 输出游戏提示信息
            Console.WriteLine(@"PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. TRY TO"); // 输出游戏提示信息
            Console.WriteLine($"GUESS THE HURKLE'S GRIDPOINT. YOU GET {N} TRIES."); // 输出游戏提示信息，包括玩家的尝试次数
            Console.WriteLine(@"AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE"); // 输出游戏提示信息
            Console.WriteLine(@"DIRECTION TO GO TO LOOK FOR THE HURKLE."); // 输出游戏提示信息
            Console.WriteLine();  # 输出空行

            var view = new ConsoleHurkleView();  # 创建一个控制台视图对象
            var hurkle = new HurkleGame(N,G, view);  # 创建一个 Hurkle 游戏对象，传入参数 N、G 和视图对象
            while(true):  # 进入无限循环
                hurkle.PlayGame();  # 调用 Hurkle 游戏对象的 PlayGame 方法

                Console.WriteLine("PLAY AGAIN? (Y)ES/(N)O");  # 输出提示信息
                var playAgainResponse = Console.ReadLine();  # 读取用户输入的响应
                if(playAgainResponse.Trim().StartsWith("y", StringComparison.InvariantCultureIgnoreCase)):  # 如果用户输入以 "y" 开头（不区分大小写）
                    Console.WriteLine();  # 输出空行
                    Console.WriteLine("LET'S PLAY AGAIN. HURKLE IS HIDING");  # 输出提示信息
                    Console.WriteLine();  # 输出空行
                else:  # 如果用户输入不以 "y" 开头
                    Console.WriteLine("THANKS FOR PLAYING!");  # 输出感谢信息
                    break  # 退出循环
抱歉，给定的代码片段不完整，无法为每个语句添加注释。如果您有完整的代码片段需要解释，请提供完整的代码片段。谢谢！
```