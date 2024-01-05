# `d:/src/tocomm/basic-computer-games\91_Train\csharp\Train\TrainGame.cs`

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
            do
            {
                PlayGame();  # 调用PlayGame()函数，开始游戏
            } while (TryAgain());  # 当TryAgain()函数返回true时，继续循环

        }

        private void PlayGame()
        {
            int carSpeed = (int)GenerateRandomNumber(40, 25);  # 生成40到25之间的随机整数，表示汽车速度
            int timeDifference = (int)GenerateRandomNumber(5, 15);  # 生成5到15之间的随机整数，表示时间差
            int trainSpeed = (int)GenerateRandomNumber(20, 19);  # 生成20到19之间的随机整数，表示火车速度

            Console.WriteLine($"A CAR TRAVELING {carSpeed} MPH CAN MAKE A CERTAIN TRIP IN");  # 输出汽车以特定速度行驶的信息
            Console.WriteLine($"{timeDifference} HOURS LESS THAN A TRAIN TRAVELING AT {trainSpeed} MPH");  # 输出汽车比火车快多少小时的信息
            Console.WriteLine("HOW LONG DOES THE TRIP TAKE BY CAR?");  # 提示用户输入汽车行程时间

            double userInputCarJourneyDuration = double.Parse(Console.ReadLine());  # 获取用户输入的汽车行程时间
            double actualCarJourneyDuration = CalculateCarJourneyDuration(carSpeed, timeDifference, trainSpeed);  # 计算实际汽车行程时间
            int percentageDifference = CalculatePercentageDifference(userInputCarJourneyDuration, actualCarJourneyDuration);  # 计算用户输入时间与实际时间的百分比差异
            if (IsWithinAllowedDifference(percentageDifference, ALLOWED_PERCENTAGE_DIFFERENCE))
            {
                // 如果用户输入的百分比差值在允许范围内，则输出GOOD! ANSWER WITHIN {percentageDifference} PERCENT.
                Console.WriteLine($"GOOD! ANSWER WITHIN {percentageDifference} PERCENT.");
            }
            else
            {
                // 如果用户输入的百分比差值超出允许范围，则输出SORRY.  YOU WERE OFF BY {percentageDifference} PERCENT.
                Console.WriteLine($"SORRY.  YOU WERE OFF BY {percentageDifference} PERCENT.");
            }
            // 输出正确答案的车程时间
            Console.WriteLine($"CORRECT ANSWER IS {actualCarJourneyDuration} HOURS.");
        }

        // 判断百分比差值是否在允许范围内
        public static bool IsWithinAllowedDifference(int percentageDifference, int allowedDifference)
        {
            return percentageDifference <= allowedDifference;
        }

        // 计算用户输入的车程时间与实际车程时间的百分比差值
        private static int CalculatePercentageDifference(double userInputCarJourneyDuration, double carJourneyDuration)
        {
            return (int)(Math.Abs((carJourneyDuration - userInputCarJourneyDuration) * 100 / userInputCarJourneyDuration) + .5);
        }
        // 计算汽车行程所需时间
        public static double CalculateCarJourneyDuration(double carSpeed, double timeDifference, double trainSpeed)
        {
            return timeDifference * trainSpeed / (carSpeed - trainSpeed);
        }

        // 生成随机数
        public double GenerateRandomNumber(int baseSpeed, int multiplier)
        {
            return multiplier * Rnd.NextDouble() + baseSpeed;
        }

        // 再试一次
        private bool TryAgain()
        {
            Console.WriteLine("ANOTHER PROBLEM (YES OR NO)? ");
            return IsInputYes(Console.ReadLine());
        }

        // 判断输入是否为“是”
        public static bool IsInputYes(string consoleInput)
        {
            var options = new string[] { "Y", "YES" };
            return options.Any(o => o.Equals(consoleInput, StringComparison.CurrentCultureIgnoreCase));
```
这行代码是一个返回语句，它使用 LINQ 查询语法来检查 options 列表中是否存在与 consoleInput 相等的元素，并且忽略大小写。

```
        private void DisplayIntroText()
        {
            Console.WriteLine("TRAIN");
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine("TIME - SPEED DISTANCE EXERCISE");
            Console.WriteLine();
        }
```
这是一个名为 DisplayIntroText 的方法，它用于在控制台上显示一些介绍性的文本。在这个方法中，使用 Console.WriteLine 来输出一些固定的文本内容。
```