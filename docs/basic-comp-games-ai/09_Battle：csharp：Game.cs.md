# `09_Battle\csharp\Game.cs`

```
                # 清空游戏场地，重新初始化为 7x7 的二维数组
                field = new int[7, 7];

                # 遍历每种船只类型
                foreach (var shipType in new []{ 1, 2, 3})
                {
                    foreach (var ship in new int[] { 1, 2 })  # 遍历包含船只类型的整数数组
                    {
                        while (!SetShip(shipType, ship)) { }  # 当船只未设置成功时，持续尝试设置船只
                    }
                }

                UserInteraction();  # 调用用户交互函数
            }
        }

        private bool SetShip(int shipType, int shipNum)  # 设置船只的函数，接受船只类型和船只编号作为参数
        {
            var shipSize = 4 - shipType;  # 根据船只类型计算船只大小
            int direction;  # 定义方向变量
            int[] A = new int[5];  # 创建长度为5的整数数组A
            int[] B = new int[5];  # 创建长度为5的整数数组B
            int row, col;  # 定义行和列变量

            do
            {
                # 生成随机的行数和列数
                row = Rnd(6) + 1;
                col = Rnd(6) + 1;
                # 生成随机的方向
                direction = Rnd(4) + 1;
            } while (field[row, col] > 0);

            # 初始化变量 M
            var M = 0;

            # 根据方向进行不同的操作
            switch (direction)
            {
                case 1:
                    # 设置船的位置
                    B[1] = col;
                    B[2] = 7;
                    B[3] = 7;

                    # 遍历船的大小
                    for (var K = 1; K <= shipSize; K++)
                    {
                        # 判断船是否超出边界或者与其他船重叠
                        if (!(M > 1 || B[K] == 6 || field[row, B[K] + 1] > 0))
                        {
                            # 更新船的位置
                            B[K + 1] = B[K] + 1;
                        continue;  // 跳过当前循环的剩余代码，继续下一次循环

                        M = 2;  // 初始化变量 M 为 2
                        var Z = 1;  // 初始化变量 Z 为 1

                        if (B[1] < B[2] && B[1] < B[3]) Z = B[1];  // 如果 B[1] 小于 B[2] 和 B[3]，则将 Z 赋值为 B[1]
                        if (B[2] < B[1] && B[2] < B[3]) Z = B[2];  // 如果 B[2] 小于 B[1] 和 B[3]，则将 Z 赋值为 B[2]
                        if (B[3] < B[1] && B[3] < B[2]) Z = B[3];  // 如果 B[3] 小于 B[1] 和 B[2]，则将 Z 赋值为 B[3]

                        if (Z == 1 || field[row, Z - 1] > 0) return false;  // 如果 Z 等于 1 或者 field[row, Z - 1] 大于 0，则返回 false

                        B[K + 1] = Z - 1;  // 将 B[K + 1] 赋值为 Z - 1
                    }

                    field[row, col] = 9 - 2 * shipType - shipNum;  // 将 field[row, col] 赋值为 9 - 2 * shipType - shipNum

                    for (var K = 1; K <= shipSize; K++)  // 循环，K 从 1 到 shipSize
                    {
                        field[row, B[K + 1]] = field[row, col];  // 将 field[row, B[K + 1]] 赋值为 field[row, col]
                    }
                    break;

                case 2:
                    A[1] = row;  // 将变量row的值赋给数组A的第一个元素
                    B[1] = col;  // 将变量col的值赋给数组B的第一个元素
                    A[2] = 0;    // 将0赋给数组A的第二个元素
                    A[3] = 0;    // 将0赋给数组A的第三个元素
                    B[2] = 0;    // 将0赋给数组B的第二个元素
                    B[3] = 0;    // 将0赋给数组B的第三个元素

                    for (var K = 1; K <= shipSize; K++)  // 循环，K从1到shipSize
                    {
                        if (!(M > 1  // 如果M大于1
                            || A[K] == 1 || B[K] == 1  // 或者A[K]等于1，或者B[K]等于1
                            || field[A[K] - 1, B[K] - 1] > 0  // 或者field[A[K] - 1, B[K] - 1]大于0
                            || (field[A[K] - 1, B[K]] > 0 && field[A[K] - 1, B[K]] == field[A[K], B[K] - 1])))  // 或者(field[A[K] - 1, B[K]]大于0并且field[A[K] - 1, B[K]]等于field[A[K], B[K] - 1]
                        {
                            A[K + 1] = A[K] - 1;  // 将A[K]减1的值赋给数组A的下一个元素
                            B[K + 1] = B[K] - 1;  // 将B[K]减1的值赋给数组B的下一个元素
                        continue;  // 跳过当前循环的剩余代码，继续下一次循环

                        M = 2;  // 初始化变量 M 为 2
                        var Z1 = 1;  // 初始化变量 Z1 为 1
                        var Z2 = 1;  // 初始化变量 Z2 为 1

                        if (A[1] > A[2] && A[1] > A[3]) Z1 = A[1];  // 如果 A[1] 大于 A[2] 和 A[3]，则将 Z1 赋值为 A[1]
                        if (A[2] > A[1] && A[2] > A[3]) Z1 = A[2];  // 如果 A[2] 大于 A[1] 和 A[3]，则将 Z1 赋值为 A[2]
                        if (A[3] > A[1] && A[3] > A[2]) Z1 = A[3];  // 如果 A[3] 大于 A[1] 和 A[2]，则将 Z1 赋值为 A[3]
                        if (B[1] > B[2] && B[1] > B[3]) Z2 = B[1];  // 如果 B[1] 大于 B[2] 和 B[3]，则将 Z2 赋值为 B[1]
                        if (B[2] > B[1] && B[2] > B[3]) Z2 = B[2];  // 如果 B[2] 大于 B[1] 和 B[3]，则将 Z2 赋值为 B[2]
                        if (B[3] > B[1] && B[3] > B[2]) Z2 = B[3];  // 如果 B[3] 大于 B[1] 和 B[2]，则将 Z2 赋值为 B[3]

                        if (Z1 == 6 || Z2 == 6  // 如果 Z1 等于 6 或者 Z2 等于 6
                            || field[Z1 + 1, Z2 + 1] > 0  // 或者 field[Z1 + 1, Z2 + 1] 大于 0
                            || (field[Z1, Z2 + 1] > 0 && field[Z1, Z2 + 1] == field[Z1 + 1, Z2])) return false;  // 或者 (field[Z1, Z2 + 1] 大于 0 并且 field[Z1, Z2 + 1] 等于 field[Z1 + 1, Z2])，则返回 false

                        A[K + 1] = Z1 + 1;  // 将 A[K + 1] 赋值为 Z1 + 1
                        B[K + 1] = Z2 + 1;  // 将 B[K + 1] 赋值为 Z2 + 1
                    }

                    # 根据船只类型和数量计算船只的值，并赋给对应位置的二维数组
                    field[row, col] = 9 - 2 * shipType - shipNum;

                    # 根据船只的大小，将对应位置的值设为船只的值
                    for (var K = 1; K <= shipSize; K++)
                    {
                        field[A[K + 1], B[K + 1]] = field[row, col];
                    }
                    break;

                case 3:
                    # 设置船只的位置
                    A[1] = row;
                    A[2] = 7;
                    A[3] = 7;

                    # 根据船只的大小，检查是否可以放置船只
                    for (var K = 1; K <= shipSize; K++)
                    {
                        if (!(M > 1 || A[K] == 6
                            || field[A[K] + 1, col] > 0))
                        {
                            A[K + 1] = A[K] + 1;  // 将数组 A 中索引为 K+1 的元素赋值为索引为 K 的元素加 1
                            continue;  // 跳过当前循环的剩余代码，继续下一次循环

                        }

                        M = 2;  // 将变量 M 赋值为 2
                        var Z = 1;  // 声明并初始化变量 Z 为 1

                        if (A[1] < A[2] && A[1] < A[3]) Z = A[1];  // 如果 A[1] 小于 A[2] 且小于 A[3]，则将 Z 赋值为 A[1]
                        if (A[2] < A[1] && A[2] < A[3]) Z = A[2];  // 如果 A[2] 小于 A[1] 且小于 A[3]，则将 Z 赋值为 A[2]
                        if (A[3] < A[1] && A[3] < A[2]) Z = A[3];  // 如果 A[3] 小于 A[1] 且小于 A[2]，则将 Z 赋值为 A[3]

                        if (Z == 1 || field[Z - 1, col] > 0) return false;  // 如果 Z 等于 1 或者 field[Z - 1, col] 大于 0，则返回 false

                        A[K + 1] = Z - 1;  // 将数组 A 中索引为 K+1 的元素赋值为 Z-1
                    }

                    field[row, col] = 9 - 2 * shipType - shipNum;  // 将 field[row, col] 赋值为 9 减去 2 乘以 shipType 再减去 shipNum

                    for (var K = 1; K <= shipSize; K++)  // 循环，K 从 1 到 shipSize
                    field[A[K + 1], col] = field[row, col];
```
将二维数组field中的某个位置的值赋给另一个位置。

```
                    }
                    break;
```
结束当前的case或循环。

```
                case 4:
                default:
                    A[1] = row;
                    B[1] = col;
                    A[2] = 7;
                    A[3] = 7;
                    B[2] = 0;
                    B[3] = 0;
```
设置变量A和B的初始值。

```
                    for (var K = 1; K <= shipSize; K++)
                    {
                        if (!(M > 1 || A[K] == 6 || B[K] == 1
                            || field[A[K] + 1, B[K] - 1] > 0
                            || (field[A[K] + 1, B[K]] > 0 && field[A[K] + 1, B[K]] == field[A[K], B[K] - 1])))
                        {
                            A[K + 1] = A[K] + 1;
```
循环遍历shipSize次，根据条件判断设置A[K + 1]的值。
# 设置数组 B 中的第 K + 1 个元素的值为数组 B 中的第 K 个元素的值减去 1
B[K + 1] = B[K] - 1;
# 继续执行下一次循环
continue;
# 设置变量 M 的值为 2
M = 2;
# 设置变量 Z1 的值为 1
var Z1 = 1;
# 设置变量 Z2 的值为 1
var Z2 = 1;

# 如果 A 数组中的第一个元素小于第二个元素并且小于第三个元素，则将 Z1 的值设置为 A 数组中的第一个元素
if (A[1] < A[2] && A[1] < A[3]) Z1 = A[1];
# 如果 A 数组中的第二个元素小于第一个元素并且小于第三个元素，则将 Z1 的值设置为 A 数组中的第二个元素
if (A[2] < A[1] && A[2] < A[3]) Z1 = A[2];
# 如果 A 数组中的第三个元素小于第一个元素并且小于第二个元素，则将 Z1 的值设置为 A 数组中的第三个元素
if (A[3] < A[1] && A[3] < A[2]) Z1 = A[3];
# 如果 B 数组中的第一个元素大于第二个元素并且大于第三个元素，则将 Z2 的值设置为 B 数组中的第一个元素
if (B[1] > B[2] && B[1] > B[3]) Z2 = B[1];
# 如果 B 数组中的第二个元素大于第一个元素并且大于第三个元素，则将 Z2 的值设置为 B 数组中的第二个元素
if (B[2] > B[1] && B[2] > B[3]) Z2 = B[2];
# 如果 B 数组中的第三个元素大于第一个元素并且大于第二个元素，则将 Z2 的值设置为 B 数组中的第三个元素
if (B[3] > B[1] && B[3] > B[2]) Z2 = B[3];

# 如果 Z1 的值等于 1 或者 Z2 的值等于 6 或者 field[Z1 - 1, Z2 + 1] 大于 0 或者 (field[Z1, Z2 + 1] 大于 0 并且 field[Z1, Z2 + 1] 等于 field[Z1 - 1, Z2])，则返回 false
if (Z1 == 1 || Z2 == 6
    || field[Z1 - 1, Z2 + 1] > 0
    || (field[Z1, Z2 + 1] > 0 && field[Z1, Z2 + 1] == field[Z1 - 1, Z2])) return false;

# 设置数组 A 中的第 K + 1 个元素的值为 Z1 - 1
A[K + 1] = Z1 - 1;
                    B[K + 1] = Z2 + 1;  // 将数组 B 的第 K+1 个元素赋值为 Z2+1

                    }

                    field[row, col] = 9 - 2 * shipType - shipNum;  // 将二维数组 field 的第 row 行 col 列的元素赋值为 9 - 2 * shipType - shipNum

                    for (var K = 1; K <= shipSize; K++)  // 循环遍历 K 从 1 到 shipSize
                    {
                        field[A[K + 1], B[K + 1]] = field[row, col];  // 将二维数组 field 的第 A[K+1] 行 B[K+1] 列的元素赋值为 field[row, col]
                    }

                    break;  // 跳出循环
            }

            return true;  // 返回 true
        }

        public void DisplayIntro()  // 定义一个公共方法 DisplayIntro
        {
            Console.ForegroundColor = ConsoleColor.Green;  // 设置控制台前景色为绿色
            Print(Tab(33) + "BATTLE");  // 调用 Print 方法，在控制台输出 "BATTLE"，并且通过 Tab 方法在前面添加 33 个空格
            # 打印出"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，使用Tab(15)进行缩进
            Print(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            # 战斗程序由Ray Westergard于1970年10月编写
            //-- BATTLE WRITTEN BY RAY WESTERGARD  10 / 70
            # 版权所有：1971年加利福尼亚大学理事会
            // COPYRIGHT 1971 BY THE REGENTS OF THE UNIV.OF CALIF.
            # 在加利福尼亚大学伯克利分校劳伦斯科学馆制作
            // PRODUCED AT THE LAWRENCE HALL OF SCIENCE, BERKELEY
        }

        # 用户交互函数
        public void UserInteraction()
        {
            # 打印空行
            Print();
            # 打印以下代码的坏家伙舰队部署情况已被捕获但尚未解码
            Print("THE FOLLOWING CODE OF THE BAD GUYS' FLEET DISPOSITION");
            Print("HAS BEEN CAPTURED BUT NOT DECODED:");
            Print();

            # 遍历6x6的二维数组
            for (var row = 1; row <= 6; row++)
            {
                for (var col = 1; col <= 6; col++)
                {
                    # 将field[col, row]的值转换为字符串并打印出来
                    Write(field[col, row].ToString());
                }
            // 调用Print函数
            Print();
        }

        // 调用Print函数
        Print();
        // 调用Print函数并输出指定的字符串
        Print("DE-CODE IT AND USE IT IF YOU CAN");
        // 调用Print函数并输出指定的字符串
        Print("BUT KEEP THE DE-CODING METHOD A SECRET.");
        // 调用Print函数
        Print();

        // 创建一个7x7的二维数组hit
        var hit = new int[7, 7];
        // 创建一个长度为4的一维数组lost
        var lost = new int[4];
        // 创建一个包含指定元素的一维数组shipHits
        var shipHits = new[] { 0, 2, 2, 1, 1, 0, 0 };
        // 初始化变量splashes为0
        var splashes = 0;
        // 初始化变量hits为0
        var hits = 0;

        // 调用Print函数并输出指定的字符串
        Print("START GAME");

        // 循环读取用户输入，并将输入的字符串转换为整数数组
        do
        {
            var input = Console.ReadLine().Split(',').Select(x => int.TryParse(x, out var num) ? num : 0).ToArray();
                # 如果输入无效，则打印错误信息并继续循环
                if (!IsValid(input))
                {
                    Print("INVALID INPUT.  TRY AGAIN.");
                    continue;
                }

                # 获取输入的列号和行号
                var col = input[0];
                var row = 7 - input[1];
                # 获取该位置上的船的编号
                var shipNum = field[row, col];

                # 如果该位置上没有船，则增加splashes计数，打印提示信息并继续循环
                if (shipNum == 0)
                {
                    splashes = splashes + 1;
                    Print("SPLASH!  TRY AGAIN.");
                    continue;
                }

                # 如果该位置上的船已经被击中超过3次，则打印提示信息
                if (shipHits[shipNum] > 3)
                {
                    Print("THERE USED TO BE A SHIP AT THAT POINT, BUT YOU SUNK IT.");
                    # 打印提示信息，要求玩家重新尝试
                    Print("SPLASH!  TRY AGAIN.");
                    # 增加尝试次数
                    splashes = splashes + 1;
                    # 继续下一次循环
                    continue;
                }

                # 如果该位置已经被击中
                if (hit[row, col] > 0)
                {
                    # 打印提示信息，要求玩家重新尝试
                    Print($"YOU ALREADY PUT A HOLE IN SHIP NUMBER {shipNum} AT THAT POINT.");
                    Print("SPLASH!  TRY AGAIN.");
                    # 增加尝试次数
                    splashes = splashes + 1;
                    # 继续下一次循环
                    continue;
                }

                # 增加击中次数
                hits = hits + 1;
                # 在击中数组中记录击中的船只编号
                hit[row, col] = shipNum;

                # 打印提示信息，显示直接击中了哪艘船只
                Print($"A DIRECT HIT ON SHIP NUMBER {shipNum}");
                # 增加被击中船只的击中次数
                shipHits[shipNum] = shipHits[shipNum] + 1;

                # 如果被击中船只的击中次数小于4
                {
                    Print("TRY AGAIN.");  # 打印提示信息
                    continue;  # 继续循环
                }

                var shipType = (shipNum - 1) / 2 + 1;  # 计算船只类型
                lost[shipType] = lost[shipType] + 1;  # 增加相应类型船只的损失数量

                Print("AND YOU SUNK IT.  HURRAH FOR THE GOOD GUYS.");  # 打印击沉提示
                Print("SO FAR, THE BAD GUYS HAVE LOST");  # 打印提示信息
                Write($"{lost[1]} DESTROYER(S), {lost[2]} CRUISER(S), AND ");  # 打印摧毁的驱逐舰和巡洋舰数量
                Print($"{lost[3]} AIRCRAFT CARRIER(S).");  # 打印摧毁的航空母舰数量
                Print($"YOUR CURRENT SPLASH/HIT RATIO IS {splashes / hits}");  # 打印当前的击中比率

                if ((lost[1] + lost[2] + lost[3]) < 6) continue;  # 如果损失的船只数量小于6，继续循环

                Print();  # 打印空行
                Print("YOU HAVE TOTALLY WIPED OUT THE BAD GUYS' FLEET");  # 打印击败敌方舰队的提示
                Print($"WITH A FINAL SPLASH/HIT RATIO OF {splashes / hits}");  # 打印最终的击中比率
                if ((splashes / hits) == 0)  # 如果击中次数等于0
                {
                    Print("CONGRATULATIONS -- A DIRECT HIT EVERY TIME.");  # 打印“恭喜 -- 每次都直接命中”
                }

                Print();  # 打印空行
                Print("****************************");  # 打印一行星号
                Print();  # 打印空行

                return;  # 返回

            } while (true);  # 无限循环

        }

        public bool IsValid(int[] input) => input.Length == 2 && input.All(Valid);  # 检查输入是否有效

        public bool Valid(int value) => value > 0 && value < 7;  # 检查值是否有效

        public void Print(string str = "") => Console.WriteLine(str);  # 打印函数
# 定义一个公共方法，用于向控制台输出指定的字符串值
public void Write(string value) => Console.Write(value);

# 定义一个公共方法，用于生成指定数量空格的字符串
public string Tab(int pos) => new String(' ', pos);

# 定义一个公共方法，用于生成一个小于指定种子值的随机数
public int Rnd(int seed) => random.Next(seed);
```