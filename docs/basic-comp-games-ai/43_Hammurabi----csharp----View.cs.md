# `43_Hammurabi\csharp\View.cs`

```
// 命名空间 Hammurabi
namespace Hammurabi
{
    /// <summary>
    /// 提供各种方法来向用户呈现信息。
    /// </summary>
    public static class View
    {
        /// <summary>
        /// 显示给玩家的介绍横幅。
        /// </summary>
        public static void ShowBanner()
        {
            // 在控制台打印 HAMURABI
            Console.WriteLine("                                HAMURABI");
            // 在控制台打印 CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
            Console.WriteLine("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            // 在控制台打印空行
            Console.WriteLine();
            // 在控制台打印空行
            Console.WriteLine();
            // 在控制台打印空行
            Console.WriteLine();
            // 在控制台打印 TRY YOUR HAND AT GOVERNING ANCIENT SUMERIA
            Console.WriteLine("TRY YOUR HAND AT GOVERNING ANCIENT SUMERIA");
            Console.WriteLine("FOR A TEN-YEAR TERM OF OFFICE.");
```
这行代码是在控制台打印输出一段文字，显示市长的任期为十年。

        /// <summary>
        /// Shows a summary of the current state of the city.
        /// </summary>
这段代码是一个注释块，用于说明下面的函数是用来显示城市当前状态的摘要信息。

        public static void ShowCitySummary(GameState state)
这行代码定义了一个名为ShowCitySummary的公共静态函数，接受一个名为state的GameState类型参数。

            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
这几行代码是在控制台打印输出空行，用于格式化输出。

            Console.WriteLine("HAMURABI:  I BEG TO REPORT TO YOU,");
这行代码是在控制台打印输出一段文字，显示市长向玩家汇报城市的情况。

            Console.WriteLine($"IN YEAR {state.Year}, {state.Starvation} PEOPLE STARVED, {state.PopulationIncrease} CAME TO THE CITY,");
这行代码是在控制台打印输出一段文字，显示城市的年份、饥荒人口和人口增长情况。

            if (state.IsPlagueYear)
            {
                Console.WriteLine("A HORRIBLE PLAGUE STRUCK!  HALF THE PEOPLE DIED.");
            }
这段代码是一个条件语句，如果state.IsPlagueYear为真，则在控制台打印输出一段文字，显示城市遭受了可怕的瘟疫，一半的人口死亡。

            Console.WriteLine($"POPULATION IS NOW {state.Population}");
这行代码是在控制台打印输出一段文字，显示当前城市的人口数量。
            // 输出当前城市拥有的土地面积
            Console.WriteLine($"THE CITY NOW OWNS {state.Acres} ACRES.");
            // 输出每英亩收获的产量
            Console.WriteLine($"YOU HARVESTED {state.Productivity} BUSHELS PER ACRE.");
            // 输出老鼠吃掉的粮食数量
            Console.WriteLine($"THE RATS ATE {state.Spoilage} BUSHELS.");
            // 输出当前存储的粮食数量
            Console.WriteLine($"YOU NOW HAVE {state.Stores} BUSHELS IN STORE.");
            // 输出空行
            Console.WriteLine();
        }

        /// <summary>
        /// 显示当前土地的价格
        /// </summary>
        /// <param name="state"></param>
        public static void ShowLandPrice(GameState state)
        {
            // 输出土地的交易价格
            Console.WriteLine ($"LAND IS TRADING AT {state.LandPrice} BUSHELS PER ACRE.");
        }

        /// <summary>
        /// 显示一个部分分隔符
        /// </summary>
        public static void ShowSeparator()
        {
            Console.WriteLine();
        }
```
这行代码是一个空的代码块，没有实际作用。

```
        /// <summary>
        /// Inform the player that he or she has entered an invalid number.
        /// </summary>
        public static void ShowInvalidNumber()
        {
            Console.WriteLine("PLEASE ENTER A VALID NUMBER");
        }
```
这段代码定义了一个名为ShowInvalidNumber的公共静态方法，用于向玩家显示输入了无效数字的消息。

```
        /// <summary>
        /// Inform the player that he or she has insufficient acreage.
        /// </summary>
        public static void ShowInsufficientLand(GameState state)
        {
            Console.WriteLine($"HAMURABI:  THINK AGAIN.  YOU OWN ONLY {state.Acres} ACRES.  NOW THEN,");
        }
```
这段代码定义了一个名为ShowInsufficientLand的公共静态方法，用于向玩家显示拥有的土地不足的消息。消息中包含了玩家拥有的土地数量。
        /// <summary>
        /// Inform the player that he or she has insufficient population.
        /// </summary>
        public static void ShowInsufficientPopulation(GameState state)
        {
            // 在控制台输出当前人口数量不足的提示信息
            Console.WriteLine($"BUT YOU HAVE ONLY {state.Population} PEOPLE TO TEND THE FIELDS!  NOW THEN,");
        }

        /// <summary>
        /// Inform the player that he or she has insufficient grain stores.
        /// </summary>
        public static void ShowInsufficientStores(GameState state)
        {
            // 在控制台输出当前粮食储备不足的提示信息
            Console.WriteLine("HAMURABI:  THINK AGAIN.  YOU HAVE ONLY");
            Console.WriteLine($"{state.Stores} BUSHELS OF GRAIN.  NOW THEN,");
        }

        /// <summary>
        /// Show the player that he or she has caused great offence.
        /// </summary>
        public static void ShowGreatOffence()
        {
            // 在控制台输出空行
            Console.WriteLine();
            // 在控制台输出指定的字符串
            Console.WriteLine("HAMURABI:  I CANNOT DO WHAT YOU WISH.");
            // 在控制台输出指定的字符串
            Console.WriteLine("GET YOURSELF ANOTHER STEWARD!!!!!");
        }

        /// <summary>
        /// Shows the game's final result to the user.
        /// </summary>
        public static void ShowGameResult(GameResult result)
        {
            // 如果玩家没有被弹劾
            if (!result.WasPlayerImpeached)
            {
                // 在控制台输出指定的字符串，并使用结果对象的属性进行格式化
                Console.WriteLine($"IN YOUR 10-YEAR TERM OF OFFICE, {result.AverageStarvationRate} PERCENT OF THE");
                // 在控制台输出指定的字符串
                Console.WriteLine("POPULATION STARVED PER YEAR ON THE AVERAGE, I.E. A TOTAL OF");
                // 在控制台输出指定的字符串，并使用结果对象的属性进行格式化
                Console.WriteLine($"{result.TotalStarvation} PEOPLE DIED!!");

                // 在控制台输出指定的字符串
                Console.WriteLine("YOU STARTED WITH 10 ACRES PER PERSON AND ENDED WITH");
                // 在控制台输出指定的字符串，并使用结果对象的属性进行格式化
                Console.WriteLine($"{result.AcresPerPerson} ACRES PER PERSON.");
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，并封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
                    Console.WriteLine($"REALLY WASN'T TOO BAD AT ALL. {result.Assassins} PEOPLE");
                    // 打印出一个关于表现评级为“不错”的结果的消息
                    Console.WriteLine("WOULD DEARLY LIKE TO SEE YOU ASSASSINATED BUT WE ALL HAVE OUR");
                    // 打印出一个关于表现评级为“不错”的结果的消息
                    Console.WriteLine("TRIVIAL PROBLEMS.");
                    // 打印出一个关于表现评级为“不错”的结果的消息
                    break;
                case PerformanceRating.Terrific:
                    // 打印出一个关于表现评级为“出色”的结果的消息
                    Console.WriteLine("A FANTASTIC PERFORMANCE!!!  CHARLEMANGE, DISRAELI, AND");
                    // 打印出一个关于表现评级为“出色”的结果的消息
                    Console.WriteLine("JEFFERSON COMBINED COULD NOT HAVE DONE BETTER!");
                    // 打印出一个关于表现评级为“出色”的结果的消息
                    break;
            }
        }

        /// <summary>
        /// Shows a farewell message to the user.
        /// </summary>
        public static void ShowFarewell()
        {
            // 打印出一个告别消息
            Console.WriteLine("SO LONG FOR NOW.");
            // 打印一个空行
            Console.WriteLine();
        }
        /// <summary>
        /// 提示用户购买土地。
        /// </summary>
        public static void PromptBuyLand()
        {
            Console.Write("HOW MANY ACRES DO YOU WISH TO BUY? ");
        }

        /// <summary>
        /// 提示用户出售土地。
        /// </summary>
        public static void PromptSellLand()
        {
            Console.Write("HOW MANY ACRES DO YOU WISH TO SELL? ");
        }

        /// <summary>
        /// 提示用户喂养人口。
        /// </summary>
        public static void PromptFeedPeople()
        {
            Console.Write("HOW MANY BUSHELS DO YOU WISH TO FEED YOUR PEOPLE? ");
        }
```
这段代码是在控制台上打印出一个提示信息，询问用户想要给人们喂多少蒲式耳的粮食。

```
        /// <summary>
        /// Prompts the user to plant crops.
        /// </summary>
        public static void PromptPlantCrops()
        {
            Console.Write("HOW MANY ACRES DO YOU WISH TO PLANT WITH SEED? ");
        }
    }
}
```
这段代码是一个公共的方法，用于提示用户种植作物。在控制台上打印出一个提示信息，询问用户想要种植多少英亩的种子。
```