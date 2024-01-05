# `d:/src/tocomm/basic-computer-games\42_Gunner\java\Gunner.java`

```
import java.util.Random;  // 导入 Random 类，用于生成随机数
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入

public class Gunner {

    public static final int MAX_ROUNDS = 6;  // 定义最大回合数为 6
    public static final int MAX_ENEMIES = 4;  // 定义最大敌人数为 4
    public static final int ERROR_DISTANCE = 100;  // 定义误差距离为 100

    private static Scanner scanner = new Scanner(System.in);  // 创建一个 Scanner 对象，用于接收用户输入
    private static Random random = new Random();  // 创建一个 Random 对象，用于生成随机数

    public static void main(String[] args) {  // 主函数
        println("                              GUNNER");  // 打印标题
        println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印创意计算机的信息
        println();  // 打印空行
        println();  // 打印空行
        println();  // 打印空行
        println("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN");  // 打印提示信息
        println("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE");  // 打印提示信息
        # 打印信息，将在目标上放置一个抛射物。在目标范围内的命中将摧毁目标。
        println("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN " + ERROR_DISTANCE + " YARDS");
        println("OF THE TARGET WILL DESTROY IT.");
        println();
        # 进入无限循环
        while (true) {
            # 生成一个随机数，表示最大射程
            int maxRange = random.nextInt(40000) + 20000;
            # 敌人数量初始化为0
            int enemyCount = 0;
            # 总回合数初始化为0
            int totalRounds = 0;
            # 打印信息，显示枪的最大射程
            println("MAXIMUM RANGE OF YOUR GUN IS " + maxRange + " YARDS.\n");

            # 进入内部无限循环
            while (true) {
                # 调用fightEnemy函数，传入最大射程参数，返回战斗回合数
                int rounds = fightEnemy(maxRange);
                # 总回合数累加
                totalRounds += rounds;

                # 如果敌人数量等于最大敌人数量，或者回合数大于等于最大回合数
                if (enemyCount == MAX_ENEMIES || rounds >= MAX_ROUNDS) {
                    # 如果回合数小于最大回合数
                    if (rounds < MAX_ROUNDS) {
                        # 打印总回合数
                        println("\n\n\nTOTAL ROUNDS EXPENDED WERE:" + totalRounds);
                    }
                    # 如果总回合数大于18，或者回合数大于等于最大回合数
                    if (totalRounds > 18 || rounds >= MAX_ROUNDS) {
                        # 打印信息，需要返回Fort Sill进行进一步训练
                        println("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!");
                    } else {
                        println("NICE SHOOTING !!");  # 打印“射击成功！”的消息

                    }
                    println("\nTRY AGAIN (Y OR N)");  # 打印“再试一次（Y或N）”的消息
                    String tryAgainResponse = scanner.nextLine();  # 从用户输入中获取再试一次的响应
                    if ("Y".equals(tryAgainResponse) || "y".equals(tryAgainResponse)) {  # 如果用户输入是Y或y
                        break;  # 跳出循环
                    }
                    println("\nOK.  RETURN TO BASE CAMP.");  # 打印“好的。返回基地营。”的消息
                    return;  # 返回
                }
                enemyCount++;  # 敌人数量加一
                println("\nTHE FORWARD OBSERVER HAS SIGHTED MORE ENEMY ACTIVITY...");  # 打印“前方观察员发现更多敌人活动…”的消息
            }
        }
    }

    private static int fightEnemy(int maxRange) {  # 定义一个名为fightEnemy的私有静态方法，参数为maxRange
        int rounds = 0;  # 初始化rounds为0
        long target = Math.round(maxRange * (random.nextDouble() * 0.8 + 0.1));  # 计算目标距离
        println("      DISTANCE TO THE TARGET IS " + target + " YARDS.");  # 打印目标距离的消息
        while (true) {  # 创建一个无限循环，直到条件不满足时退出循环
            println("\nELEVATION?");  # 打印提示信息要求输入仰角
            double elevation = Double.parseDouble(scanner.nextLine());  # 从用户输入中获取仰角并转换为浮点数
            if (elevation > 89.0) {  # 如果仰角大于89度
                println("MAXIMUM ELEVATION IS 89 DEGREES.");  # 打印最大仰角为89度的提示信息
                continue;  # 继续下一次循环
            }
            if (elevation < 1.0) {  # 如果仰角小于1度
                println("MINIMUM ELEVATION IS ONE DEGREE.");  # 打印最小仰角为1度的提示信息
                continue;  # 继续下一次循环
            }
            rounds++;  # 回合数加1
            if (rounds >= MAX_ROUNDS) {  # 如果回合数大于等于最大回合数
                println("\nBOOM !!!!   YOU HAVE JUST BEEN DESTROYED ");  # 打印被敌人摧毁的提示信息
                println("BY THE ENEMY.\n\n\n");  # 打印被敌人摧毁的提示信息
                break;  # 退出循环
            }

            long error = calculateError(maxRange, target, elevation);  # 调用calculateError函数计算误差
            if (Math.abs(error) < ERROR_DISTANCE) { // 如果误差的绝对值小于ERROR_DISTANCE
                println("*** TARGET DESTROYED ***  " + rounds + " ROUNDS OF AMMUNITION EXPENDED."); // 打印目标被摧毁，并输出射击的轮数
                break; // 跳出循环
            } else if (error > ERROR_DISTANCE) { // 如果误差大于ERROR_DISTANCE
                println("SHORT OF TARGET BY " + Math.abs(error) + " YARDS."); // 打印射击距离目标短
            } else { // 如果误差小于ERROR_DISTANCE且不大于ERROR_DISTANCE
                println("OVER TARGET BY " + Math.abs(error) + " YARDS."); // 打印射击距离目标远
            }

        }
        return rounds; // 返回射击的轮数
    }

    private static long calculateError(int maxRange, long target, double elevationInDegrees) { // 计算误差的方法
        double elevationInRadians = Math.PI * elevationInDegrees / 90.0; // 将角度转换为弧度
        double impact = maxRange * Math.sin(elevationInRadians); // 计算射击点的位置
        double error = target - impact; // 计算误差
        return Math.round(error); // 返回四舍五入后的误差值
    }
    # 定义一个私有的静态方法，用于打印输出字符串
    private static void println(String s) {
        System.out.println(s);
    }

    # 定义一个私有的静态方法，用于打印输出空行
    private static void println() {
        System.out.println();
    }
```

这段代码定义了两个私有的静态方法，一个用于打印输出字符串，另一个用于打印输出空行。
```