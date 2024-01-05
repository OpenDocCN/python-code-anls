# `d:/src/tocomm/basic-computer-games\17_Bullfight\csharp\View.cs`

```
# 创建一个名为 View 的静态类，包含用于向用户显示信息的函数
# 创建一个名为 QualityString 的只读字符串数组，包含了不同质量的字符串
# 创建一个名为 ShowBanner 的静态函数，用于显示游戏的横幅信息
# 在控制台打印游戏的横幅信息
        // 显示游戏指令
        public static void ShowInstructions()
        {
            // 打印欢迎词
            Console.WriteLine("HELLO, ALL YOU BLOODLOVERS AND AFICIONADOS.");
            // 打印提示信息
            Console.WriteLine("HERE IS YOUR BIG CHANCE TO KILL A BULL.");
            Console.WriteLine();
            // 打印提示信息
            Console.WriteLine("ON EACH PASS OF THE BULL, YOU MAY TRY");
            // 打印提示信息
            Console.WriteLine("0 - VERONICA (DANGEROUS INSIDE MOVE OF THE CAPE)");
            // 打印提示信息
            Console.WriteLine("1 - LESS DANGEROUS OUTSIDE MOVE OF THE CAPE");
            // 打印提示信息
            Console.WriteLine("2 - ORDINARY SWIRL OF THE CAPE.");
            Console.WriteLine();
            // 打印提示信息
            Console.WriteLine("INSTEAD OF THE ABOVE, YOU MAY TRY TO KILL THE BULL");
            // 打印提示信息
            Console.WriteLine("ON ANY TURN: 4 (OVER THE HORNS), 5 (IN THE CHEST).");
            // 打印提示信息
            Console.WriteLine("BUT IF I WERE YOU,");
            // 打印提示信息
            Console.WriteLine("I WOULDN'T TRY IT BEFORE THE SEVENTH PASS.");
            Console.WriteLine();
            // 打印提示信息
            Console.WriteLine("THE CROWD WILL DETERMINE WHAT AWARD YOU DESERVE");
            // 打印提示信息
            Console.WriteLine("(POSTHUMOUSLY IF NECESSARY).");
            // 打印提示信息
            Console.WriteLine("THE BRAVER YOU ARE, THE BETTER THE AWARD YOU RECEIVE.");
            Console.WriteLine();
            // 打印提示信息
            Console.WriteLine("THE BETTER THE JOB THE PICADORES AND TOREADORES DO,");
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
                {
                    # 打印“你很幸运”
                    Console.WriteLine("YOU'RE LUCKY");
                }
                else
                if (matchStarted.BullQuality < Quality.Good)
                {
                    # 如果比赛开始时的牛的质量小于“好”，则打印“祝你好运。你会需要它。”
                    Console.WriteLine("GOOD LUCK.  YOU'LL NEED IT.");
                    Console.WriteLine();
                }

                # 打印空行
                Console.WriteLine();
            }

            static void ShowHelpQuality(string helperName, Quality helpQuality, int helpersKilled, int horsesKilled)
            {
                # 打印帮手名字和帮手的质量
                Console.WriteLine($"THE {helperName} DID A {QualityString[(int)helpQuality - 1]} JOB.");

                # 注意：下面的代码对于伤亡数字的生成做出了一些*强烈*的假设。
                # 它是这样编写的，以保留原始BASIC的行为
                // 根据 helpQuality 的不同情况进行不同的处理
                switch (helpQuality)
                {
                    case Quality.Poor:  // 如果 helpQuality 为 Poor
                        if (horsesKilled > 0)  // 如果有马匹被杀死
                            Console.WriteLine($"ONE OF THE HORSES OF THE {helperName} WAS KILLED.");  // 输出相应信息

                        if (helpersKilled > 0)  // 如果有帮手被杀死
                            Console.WriteLine($"ONE OF THE {helperName} WAS KILLED.");  // 输出相应信息
                        else
                            Console.WriteLine($"NO {helperName} WERE KILLED.");  // 输出相应信息
                        break;

                    case Quality.Awful:  // 如果 helpQuality 为 Awful
                        if (horsesKilled > 0)  // 如果有马匹被杀死
                            Console.WriteLine($" {horsesKilled} OF THE HORSES OF THE {helperName} KILLED.");  // 输出相应信息

                        Console.WriteLine($" {helpersKilled} OF THE {helperName} KILLED.");  // 输出相应信息
        // 结束当前循环
        break;
    }
}

// 显示当前通行证的开始
public static void ShowStartOfPass(int passNumber)
{
    // 打印空行
    Console.WriteLine();
    Console.WriteLine();
    // 打印当前通行证的编号
    Console.WriteLine($"PASS NUMBER {passNumber}");
}

// 显示玩家是否被公牛刺伤
public static void ShowPlayerGored(bool playerPanicked, bool firstGoring)
{
    // 根据玩家是否惊慌和是否是第一次刺伤，打印不同的信息
    Console.WriteLine((playerPanicked, firstGoring) switch
    {
        (true,  true) => "YOU PANICKED.  THE BULL GORED YOU.",
        (false, true) => "THE BULL HAS GORED YOU!",
        (_, false)    => "YOU ARE GORED AGAIN!"
    });
        }

        public static void ShowPlayerSurvives()
        {
            Console.WriteLine("YOU ARE STILL ALIVE.");  // 打印玩家仍然存活的消息
            Console.WriteLine();  // 打印空行
        }

        public static void ShowPlayerFoolhardy()
        {
            Console.WriteLine("YOU ARE BRAVE.  STUPID, BUT BRAVE.");  // 打印玩家勇敢但愚蠢的消息
        }

        public static void ShowFinalResult(ActionResult result, bool extremeBravery, Reward reward)
        {
            Console.WriteLine();  // 打印空行
            Console.WriteLine();  // 打印空行
            Console.WriteLine();  // 打印空行

            switch (result)  // 根据结果进行不同的处理
            {
                # 如果玩家逃跑
                case ActionResult.PlayerFlees:
                    # 输出“懦夫”
                    Console.WriteLine("COWARD");
                    # 跳出 switch 语句
                    break;
                # 如果公牛杀死玩家
                case ActionResult.BullKillsPlayer:
                    # 输出“你已经死了。”
                    Console.WriteLine("YOU ARE DEAD.");
                    # 跳出 switch 语句
                    break;
                # 如果玩家杀死公牛
                case ActionResult.PlayerKillsBull:
                    # 输出“你杀死了公牛！”
                    Console.WriteLine("YOU KILLED THE BULL!");
                    # 跳出 switch 语句
                    break;
            }

            # 如果结果是玩家逃跑
            if (result == ActionResult.PlayerFlees)
            {
                # 输出“观众嘘声持续十分钟。如果你再敢出现在擂台上，他们发誓会杀了你——
                Console.WriteLine("THE CROWD BOOS FOR TEN MINUTES.  IF YOU EVER DARE TO SHOW");
                Console.WriteLine("YOUR FACE IN A RING AGAIN, THEY SWEAR THEY WILL KILL YOU--");
                Console.WriteLine("UNLESS THE BULL DOES FIRST.");
            }
            # 如果结果不是玩家逃跑
            else
            {
                # 如果极度勇敢，则打印“人群疯狂欢呼！”
                if (extremeBravery)
                    Console.WriteLine("THE CROWD CHEERS WILDLY!");
                # 如果结果是玩家杀死了公牛，则打印“人群欢呼！”并换行
                else if (result == ActionResult.PlayerKillsBull)
                {
                    Console.WriteLine("THE CROWD CHEERS!");
                    Console.WriteLine();
                }

                # 打印“人群奖励你”
                Console.WriteLine("THE CROWD AWARDS YOU");
                # 根据奖励类型进行不同的打印
                switch (reward)
                {
                    case Reward.Nothing:
                        Console.WriteLine("NOTHING AT ALL.");
                        break;
                    case Reward.OneEar:
                        Console.WriteLine("ONE EAR OF THE BULL.");
                        break;
                    case Reward.TwoEars:
                        Console.WriteLine("BOTH EARS OF THE BULL!");
                        Console.WriteLine("OLE!");  # 打印"OLE!"
                        break;  # 跳出当前循环或者 switch 语句
                    default:  # 默认情况
                        Console.WriteLine("OLE!  YOU ARE 'MUY HOMBRE'!! OLE!  OLE!");  # 打印"OLE!  YOU ARE 'MUY HOMBRE'!! OLE!  OLE!"
                        break;  # 跳出当前循环或者 switch 语句
                }
            }

            Console.WriteLine();  # 打印空行
            Console.WriteLine("ADIOS");  # 打印"ADIOS"
            Console.WriteLine();  # 打印空行
            Console.WriteLine();  # 打印空行
            Console.WriteLine();  # 打印空行
        }

        public static void PromptShowInstructions()  # 定义一个名为 PromptShowInstructions 的静态方法
        {
            Console.Write("DO YOU WANT INSTRUCTIONS? ");  # 打印"DO YOU WANT INSTRUCTIONS? "
        }
# 提示玩家有一头公牛向他们冲来，询问是否想要杀死公牛
public static void PromptKillBull()
{
    Console.WriteLine("THE BULL IS CHARGING AT YOU!  YOU ARE THE MATADOR--");
    Console.Write("DO YOU WANT TO KILL THE BULL? ");
}

# 简短提示玩家有一头公牛向他们冲来，询问是否想要尝试杀死公牛
public static void PromptKillBullBrief()
{
    Console.Write("HERE COMES THE BULL.  TRY FOR A KILL? ");
}

# 提示玩家现在是关键时刻，询问他们打算如何尝试杀死公牛
public static void PromptKillMethod()
{
    Console.WriteLine();
    Console.WriteLine("IT IS THE MOMENT OF TRUTH.");
    Console.WriteLine();

    Console.Write("HOW DO YOU TRY TO KILL THE BULL? ");
}
# 提示用户输入披风的移动
public static void PromptCapeMove()
{
    Console.Write("WHAT MOVE DO YOU MAKE WITH THE CAPE? ");
}

# 简短提示用户输入披风的移动
public static void PromptCapeMoveBrief()
{
    Console.Write("CAPE MOVE? ");
}

# 提示用户不要慌，输入正确的数字
public static void PromptDontPanic()
{
    Console.WriteLine("DON'T PANIC, YOU IDIOT!  PUT DOWN A CORRECT NUMBER");
    Console.Write("? ");
}

# 提示用户是否从戒指中逃跑
public static void PromptRunFromRing()
{
    Console.Write("DO YOU RUN FROM THE RING? ");
}
    }
```

这部分代码是一个缩进错误，应该删除这两行代码。
```