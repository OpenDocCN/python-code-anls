# `d:/src/tocomm/basic-computer-games\68_Orbit\csharp\program.cs`

```
using System.Text;  // 导入 System.Text 命名空间

namespace Orbit  // 命名空间 Orbit
{
    class Orbit  // 类 Orbit
    {
        private void DisplayIntro()  // 私有方法 DisplayIntro
        {
            Console.WriteLine();  // 输出空行
            Console.WriteLine("ORBIT".PadLeft(23));  // 输出 "ORBIT" 并左对齐到第 23 个位置
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 输出 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
            Console.WriteLine();  // 输出空行
            Console.WriteLine();  // 输出空行
            Console.WriteLine();  // 输出空行
            Console.WriteLine("");  // 输出空字符串
            Console.WriteLine("SOMEWHERE ABOVE YOUR PLANET IS A ROMULAN SHIP.");  // 输出 "SOMEWHERE ABOVE YOUR PLANET IS A ROMULAN SHIP."
            Console.WriteLine();  // 输出空行
            Console.WriteLine("THE SHIP IS IN A CONSTANT POLAR ORBIT.  ITS");  // 输出 "THE SHIP IS IN A CONSTANT POLAR ORBIT.  ITS"
            Console.WriteLine("DISTANCE FROM THE CENTER OF YOUR PLANET IS FROM");  // 输出 "DISTANCE FROM THE CENTER OF YOUR PLANET IS FROM"
            Console.WriteLine("10,000 TO 30,000 MILES AND AT ITS PRESENT VELOCITY CAN");  // 输出 "10,000 TO 30,000 MILES AND AT ITS PRESENT VELOCITY CAN"
# 输出提示信息
Console.WriteLine("CIRCLE YOUR PLANET ONCE EVERY 12 TO 36 HOURS.")
Console.WriteLine()
Console.WriteLine("UNFORTUNATELY, THEY ARE USING A CLOAKING DEVICE SO")
Console.WriteLine("YOU ARE UNABLE TO SEE THEM, BUT WITH A SPECIAL")
Console.WriteLine("INSTRUMENT YOU CAN TELL HOW NEAR THEIR SHIP YOUR")
Console.WriteLine("PHOTON BOMB EXPLODED.  YOU HAVE SEVEN HOURS UNTIL THEY")
Console.WriteLine("HAVE BUILT UP SUFFICIENT POWER IN ORDER TO ESCAPE")
Console.WriteLine("YOUR PLANET'S GRAVITY.")
Console.WriteLine()
Console.WriteLine("YOUR PLANET HAS ENOUGH POWER TO FIRE ONE BOMB AN HOUR.")
Console.WriteLine()
Console.WriteLine("AT THE BEGINNING OF EACH HOUR YOU WILL BE ASKED TO GIVE AN")
Console.WriteLine("ANGLE (BETWEEN 0 AND 360) AND A DISTANCE IN UNITS OF")
Console.WriteLine("100 MILES (BETWEEN 100 AND 300), AFTER WHICH YOUR BOMB'S")
Console.WriteLine("DISTANCE FROM THE ENEMY SHIP WILL BE GIVEN.")
Console.WriteLine()
Console.WriteLine("AN EXPLOSION WITHIN 5,000 MILES OF THE ROMULAN SHIP")
Console.WriteLine("WILL DESTROY IT.")
Console.WriteLine()
Console.WriteLine("BELOW IS A DIAGRAM TO HELP YOU VISUALIZE YOUR PLIGHT.")
# 输出空行
Console.WriteLine();
# 输出空行
Console.WriteLine();
# 输出一行包含90的字符串
Console.WriteLine("                          90");
# 输出一行包含一串0的字符串
Console.WriteLine("                    0000000000000");
# 输出一行包含一串0的字符串
Console.WriteLine("                 0000000000000000000");
# 输出一行包含一串0的字符串
Console.WriteLine("               000000           000000");
# 输出一行包含一串0的字符串
Console.WriteLine("             00000                 00000");
# 输出一行包含一串X的字符串
Console.WriteLine("            00000    XXXXXXXXXXX    00000");
# 输出一行包含一串X的字符串
Console.WriteLine("           00000    XXXXXXXXXXXXX    00000");
# 输出一行包含一串X的字符串
Console.WriteLine("          0000     XXXXXXXXXXXXXXX     0000");
# 输出一行包含一串X的字符串
Console.WriteLine("         0000     XXXXXXXXXXXXXXXXX     0000");
# 输出一行包含一串X的字符串
Console.WriteLine("        0000     XXXXXXXXXXXXXXXXXXX     0000");
# 输出一行包含180<==和==>0的字符串
Console.WriteLine("180<== 00000     XXXXXXXXXXXXXXXXXXX     00000 ==>0");
# 输出一行包含一串X的字符串
Console.WriteLine("        0000     XXXXXXXXXXXXXXXXXXX     0000");
# 输出一行包含一串X的字符串
Console.WriteLine("         0000     XXXXXXXXXXXXXXXXX     0000");
# 输出一行包含一串X的字符串
Console.WriteLine("          0000     XXXXXXXXXXXXXXX     0000");
# 输出一行包含一串X的字符串
Console.WriteLine("           00000    XXXXXXXXXXXXX    00000");
# 输出一行包含一串X的字符串
Console.WriteLine("            00000    XXXXXXXXXXX    00000");
# 输出一行包含一串0的字符串
Console.WriteLine("             00000                 00000");
# 输出一行包含一串0的字符串
Console.WriteLine("               000000           000000");
            Console.WriteLine("                 0000000000000000000");  // 打印一行数字
            Console.WriteLine("                    0000000000000");  // 打印一行数字
            Console.WriteLine("                         270");  // 打印一行数字
            Console.WriteLine();  // 打印空行
            Console.WriteLine("X - YOUR PLANET");  // 打印提示信息
            Console.WriteLine("O - THE ORBIT OF THE ROMULAN SHIP");  // 打印提示信息
            Console.WriteLine();  // 打印空行
            Console.WriteLine("ON THE ABOVE DIAGRAM, THE ROMULAN SHIP IS CIRCLING");  // 打印提示信息
            Console.WriteLine("COUNTERCLOCKWISE AROUND YOUR PLANET.  DON'T FORGET THAT");  // 打印提示信息
            Console.WriteLine("WITHOUT SUFFICIENT POWER THE ROMULAN SHIP'S ALTITUDE");  // 打印提示信息
            Console.WriteLine("AND ORBITAL RATE WILL REMAIN CONSTANT.");  // 打印提示信息
            Console.WriteLine();  // 打印空行
            Console.WriteLine("GOOD LUCK.  THE FEDERATION IS COUNTING ON YOU.");  // 打印祝福信息
       }

        private bool PromptYesNo(string Prompt)  // 定义一个名为PromptYesNo的私有方法，参数为Prompt
        {
            bool Success = false;  // 初始化一个布尔类型变量Success，并赋值为false

            while (!Success)  // 当Success为false时执行循环
            {
                // 输出提示信息
                Console.Write(Prompt);
                // 读取用户输入并去除首尾空格，转换为小写
                string LineInput = Console.ReadLine().Trim().ToLower();

                // 如果用户输入为 "yes"，返回 true
                if (LineInput.Equals("yes"))
                    return true;
                // 如果用户输入为 "no"，返回 false
                else if (LineInput.Equals("no"))
                    return false;
                // 如果用户输入既不是 "yes" 也不是 "no"，输出提示信息
                else
                    Console.WriteLine("Yes or No");
            }

            // 默认返回 false
            return false;
        }

        // 提示用户输入数字
        private int PromptForNumber(string Prompt)
        {
            // 初始化输入成功标志和返回结果
            bool InputSuccess = false;
            int ReturnResult = 0;
            while (!InputSuccess)
            {
                // 在控制台上显示提示信息，并获取用户输入的字符串，去除两端的空格
                Console.Write(Prompt);
                string Input = Console.ReadLine().Trim();
                // 尝试将用户输入的字符串转换为整数，如果成功则将转换后的结果赋值给ReturnResult，并将InputSuccess设为true
                InputSuccess = int.TryParse(Input, out ReturnResult);
                // 如果转换失败，则在控制台上显示错误信息
                if (!InputSuccess)
                    Console.WriteLine("*** Please enter a valid number ***");
            }   

            // 返回转换后的整数结果
            return ReturnResult;
        }

        private void PlayOneRound()
        {
            // 创建一个随机数生成器对象
            Random rand = new Random();
            // 初始化一个空字符串用于存储提示信息

            int A_AngleToShip = 0;
            int D_DistanceFromBombToShip = 0;
            int R_DistanceToShip = 0;
            # 初始化变量 H_Hour 为 0，表示小时
            int H_Hour = 0;
            # 初始化变量 A1_Angle 为 0，表示角度
            int A1_Angle = 0;
            # 初始化变量 D1_DistanceForDetonation 为 0，表示爆炸距离
            int D1_DistanceForDetonation = 0;
            # 初始化变量 T 为 0
            int T = 0;
            # 初始化变量 C_ExplosionDistance 为 0，表示爆炸距离

            # 生成一个随机角度 A_AngleToShip
            A_AngleToShip = Convert.ToInt32(360 * rand.NextDouble());
            # 生成一个随机距离 D_DistanceFromBombToShip
            D_DistanceFromBombToShip = Convert.ToInt32(200 * rand.NextDouble()) + 200;
            # 生成一个随机距离 R_DistanceToShip
            R_DistanceToShip = Convert.ToInt32(20 * rand.NextDouble()) + 10;

            # 当 H_Hour 小于 7 时执行循环
            while (H_Hour < 7)
            {
                # H_Hour 自增 1
                H_Hour++;

                # 输出空行
                Console.WriteLine();
                Console.WriteLine();
                # 提示用户输入角度
                Prompt = "This is hour " + H_Hour.ToString() + ", at what angle do you wish to send\nyour photon bomb? ";
                A1_Angle = PromptForNumber(Prompt);
                # 提示用户输入爆炸距离
                D1_DistanceForDetonation = PromptForNumber("How far out do you wish to detonate it? ");
// 输出空行
Console.WriteLine();
Console.WriteLine();

// 将 A_AngleToShip 增加 R_DistanceToShip 的值
A_AngleToShip += R_DistanceToShip;
// 如果 A_AngleToShip 大于等于 360，则减去 360
if (A_AngleToShip >= 360)
    A_AngleToShip -= 360;

// 计算 T 的值，T 等于 A_AngleToShip 和 A1_Angle 的差的绝对值
T = Math.Abs(A_AngleToShip - A1_Angle);
// 如果 T 大于等于 180，则将 T 设为 360 减去 T
if (T >= 180)
    T = 360 - T;

// 计算 C_ExplosionDistance 的值，使用三角函数计算两点之间的距离
C_ExplosionDistance = Math.Sqrt(D_DistanceFromBombToShip * D_DistanceFromBombToShip + D1_DistanceForDetonation * 
                                D1_DistanceForDetonation - 2 * D_DistanceFromBombToShip * D1_DistanceForDetonation * 
                                Math.Cos(T * 3.14159 / 180));
                
// 输出爆炸距离的信息
Console.WriteLine("Your photon bomb exploded {0:N3}*10^2 miles from the", C_ExplosionDistance);
Console.WriteLine("Romulan ship.");

// 如果爆炸距离小于等于 50
if (C_ExplosionDistance <= 50)
                {
                    Console.WriteLine("You have successfully completed your mission.");  # 打印消息，表示成功完成任务
                    return;  # 结束函数的执行
                }
            }

            Console.WriteLine("You allowed the Romulans to escape.");  # 打印消息，表示罗穆兰人逃脱了
            return;  # 结束函数的执行
 
        }

        public void Play()  # 定义一个名为Play的公共函数
        {
            bool ContinuePlay = true;  # 声明一个布尔变量ContinuePlay，并赋值为true

            DisplayIntro();  # 调用DisplayIntro函数

            do  # 开始一个do-while循环
            {
                PlayOneRound();  # 调用PlayOneRound函数
                Console.WriteLine("Another Romulan ship has gone in to orbit.");  # 输出信息，指示另一艘罗穆兰星舰进入了轨道
                ContinuePlay = PromptYesNo("Do you wish to try to destroy it? ");  # 调用PromptYesNo函数询问用户是否想尝试摧毁罗穆兰星舰，并将结果赋值给ContinuePlay变量
            }
            while (ContinuePlay);  # 当ContinuePlay为True时，继续执行循环
            
            Console.WriteLine("Good bye.");  # 输出信息，表示游戏结束
        }
    }
    class Program
    {
        static void Main(string[] args)
        {

            new Orbit().Play();  # 创建Orbit对象并调用其Play方法

        }
    }
}
```