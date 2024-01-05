# `28_Combat\csharp\View.cs`

```
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
            // 打印一条消息
            Console.WriteLine("I AM AT WAR WITH YOU.");
            // 打印一条消息
            Console.WriteLine("WE HAVE 72000 SOLDIERS APIECE.");
        }

        // 打印分配兵力的提示
        public static void ShowDistributeForces()
        {
            // 打印空行
            Console.WriteLine();
            // 打印分配兵力的提示
            Console.WriteLine("DISTRIBUTE YOUR FORCES.");
            // 打印表头
            Console.WriteLine("\tME\t  YOU");
        }

        // 打印指定的消息
        public static void ShowMessage(string message)
        {
            // 打印指定的消息
            Console.WriteLine(message);
        }

        // 打印最终战争结果
        public static void ShowResult(WarState finalState)
        {
            // 如果不是绝对胜利，则执行下面的代码
            if (!finalState.IsAbsoluteVictory)
            {
                Console.WriteLine();
                Console.WriteLine("FROM THE RESULTS OF BOTH OF YOUR ATTACKS,");
            }
```
这段代码是在控制台输出空行和一条消息。

```
            switch (finalState.FinalOutcome)
            {
            case WarResult.ComputerVictory:
                Console.WriteLine("YOU LOST-I CONQUERED YOUR COUNTRY.  IT SERVES YOU");
                Console.WriteLine("RIGHT FOR PLAYING THIS STUPID GAME!!!");
                break;
            case WarResult.PlayerVictory:
                Console.WriteLine("YOU WON, OH! SHUCKS!!!!");
                break;
            case WarResult.PeaceTreaty:
                Console.WriteLine("THE TREATY OF PARIS CONCLUDED THAT WE TAKE OUR");
                Console.WriteLine("RESPECTIVE COUNTRIES AND LIVE IN PEACE.");
                break;
            }
        }
```
这段代码是一个 switch 语句，根据 finalState.FinalOutcome 的值来执行不同的操作。根据不同的情况，输出不同的消息。如果 finalState.FinalOutcome 是 WarResult.ComputerVictory，则输出一条失败的消息；如果是 WarResult.PlayerVictory，则输出一条胜利的消息；如果是 WarResult.PeaceTreaty，则输出一条和平条约的消息。
# 提示计算机军队规模
public static void PromptArmySize(int computerArmySize)
{
    Console.Write($"ARMY\t{computerArmySize}\t? ");
}

# 提示计算机海军规模
public static void PromptNavySize(int computerNavySize)
{
    Console.Write($"NAVY\t{computerNavySize}\t? ");
}

# 提示计算机空军规模
public static void PromptAirForceSize(int computerAirForceSize)
{
    Console.Write($"A. F.\t{computerAirForceSize}\t? ");
}

# 提示首次攻击的兵种选择
public static void PromptFirstAttackBranch()
{
    Console.WriteLine("YOU ATTACK FIRST. TYPE (1) FOR ARMY; (2) FOR NAVY;");
    Console.WriteLine("AND (3) FOR AIR FORCE.");
    Console.Write("? ");
}
        }

        public static void PromptNextAttackBranch(ArmedForces computerForces, ArmedForces playerForces)
        {
            // BUG: More of a nit-pick really, but the order of columns in the
            //  table is reversed from what we showed when distributing troops.
            //  The tables should be consistent.
            // 打印提示信息，提醒玩家和电脑的军队、海军和空军数量
            Console.WriteLine();
            Console.WriteLine("\tYOU\tME");
            Console.WriteLine($"ARMY\t{playerForces.Army}\t{computerForces.Army}");
            Console.WriteLine($"NAVY\t{playerForces.Navy}\t{computerForces.Navy}");
            Console.WriteLine($"A. F.\t{playerForces.AirForce}\t{computerForces.AirForce}");

            // 打印提示信息，询问玩家下一步的行动
            Console.WriteLine("WHAT IS YOUR NEXT MOVE?");
            Console.WriteLine("ARMY=1  NAVY=2  AIR FORCE=3");
            Console.Write("? ");
        }

        public static void PromptAttackSize()
        {
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
```