# `basic-computer-games\42_Gunner\java\Gunner.java`

```
# 导入 java.util 包中的 Random 和 Scanner 类
import java.util.Random;
import java.util.Scanner;

# 创建名为 Gunner 的类
public class Gunner {

    # 定义常量 MAX_ROUNDS，表示最大回合数为 6
    public static final int MAX_ROUNDS = 6;
    # 定义常量 MAX_ENEMIES，表示最大敌人数为 4
    public static final int MAX_ENEMIES = 4;
    # 定义常量 ERROR_DISTANCE，表示错误距离为 100
    public static final int ERROR_DISTANCE = 100;

    # 创建一个静态的 Scanner 对象，用于接收用户输入
    private static Scanner scanner = new Scanner(System.in);
    # 创建一个静态的 Random 对象，用于生成随机数
    private static Random random = new Random();
    // 主函数，程序入口
    public static void main(String[] args) {
        // 打印标题
        println("                              GUNNER");
        println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        println();
        println();
        println();
        println("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN");
        println("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE");
        println("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN " + ERROR_DISTANCE + " YARDS");
        println("OF THE TARGET WILL DESTROY IT.");
        println();
        // 进入循环
        while (true) {
            // 生成最大射程
            int maxRange = random.nextInt(40000) + 20000;
            int enemyCount = 0;
            int totalRounds = 0;
            println("MAXIMUM RANGE OF YOUR GUN IS " + maxRange + " YARDS.\n");

            // 进入内循环
            while (true) {
                // 与敌人交战，返回消耗的弹药数
                int rounds = fightEnemy(maxRange);
                totalRounds += rounds;

                // 判断敌人数量和总回合数
                if (enemyCount == MAX_ENEMIES || rounds >= MAX_ROUNDS) {
                    if (rounds < MAX_ROUNDS) {
                        println("\n\n\nTOTAL ROUNDS EXPENDED WERE:" + totalRounds);
                    }
                    if (totalRounds > 18 || rounds >= MAX_ROUNDS) {
                        println("BETTER GO BACK TO FORT SILL FOR REFRESHER TRAINING!");
                    } else {
                        println("NICE SHOOTING !!");
                    }
                    println("\nTRY AGAIN (Y OR N)");
                    // 获取用户输入
                    String tryAgainResponse = scanner.nextLine();
                    if ("Y".equals(tryAgainResponse) || "y".equals(tryAgainResponse)) {
                        break;
                    }
                    println("\nOK.  RETURN TO BASE CAMP.");
                    return;
                }
                enemyCount++;
                println("\nTHE FORWARD OBSERVER HAS SIGHTED MORE ENEMY ACTIVITY...");
            }
        }
    }
    # 战斗敌人的函数，参数为最大射程
    private static int fightEnemy(int maxRange) {
        # 初始化回合数
        int rounds = 0;
        # 计算目标距离，取值范围为最大射程的 10% 到 90%
        long target = Math.round(maxRange * (random.nextDouble() * 0.8 + 0.1));
        # 打印目标距离信息
        println("      DISTANCE TO THE TARGET IS " + target + " YARDS.");

        # 进入战斗循环
        while (true) {
            # 打印提示信息，要求输入仰角
            println("\nELEVATION?");
            # 读取输入的仰角
            double elevation = Double.parseDouble(scanner.nextLine());
            # 如果仰角大于 89 度，提示最大仰角为 89 度，重新输入
            if (elevation > 89.0) {
                println("MAXIMUM ELEVATION IS 89 DEGREES.");
                continue;
            }
            # 如果仰角小于 1 度，提示最小仰角为 1 度，重新输入
            if (elevation < 1.0) {
                println("MINIMUM ELEVATION IS ONE DEGREE.");
                continue;
            }
            # 回合数加一
            rounds++;
            # 如果回合数超过最大回合数，打印被敌人摧毁的信息，结束战斗
            if (rounds >= MAX_ROUNDS) {
                println("\nBOOM !!!!   YOU HAVE JUST BEEN DESTROYED ");
                println("BY THE ENEMY.\n\n\n");
                break;
            }

            # 计算射击误差
            long error = calculateError(maxRange, target, elevation);
            # 如果误差小于设定的距离误差，打印目标被摧毁的信息，结束战斗
            if (Math.abs(error) < ERROR_DISTANCE) {
                println("*** TARGET DESTROYED ***  " + rounds + " ROUNDS OF AMMUNITION EXPENDED.");
                break;
            } else if (error > ERROR_DISTANCE) {
                # 如果误差大于设定的距离误差，打印射击短了的信息
                println("SHORT OF TARGET BY " + Math.abs(error) + " YARDS.");
            } else {
                # 如果误差小于设定的距离误差，打印射击长了的信息
                println("OVER TARGET BY " + Math.abs(error) + " YARDS.");
            }

        }
        # 返回回合数
        return rounds;
    }

    # 计算射击误差的函数，参数为最大射程、目标距离、仰角
    private static long calculateError(int maxRange, long target, double elevationInDegrees) {
        # 将仰角从度转换为弧度
        double elevationInRadians = Math.PI * elevationInDegrees / 90.0; //convert degrees to radians
        # 计算射击落点距离目标的误差
        double impact = maxRange * Math.sin(elevationInRadians);
        double error = target - impact;
        # 返回误差的四舍五入值
        return Math.round(error);
    }

    # 打印字符串的函数
    private static void println(String s) {
        System.out.println(s);
    }

    # 打印空行的函数
    private static void println() {
        System.out.println();
    }
# 闭合前面的函数定义
```