# `basic-computer-games\79_Slalom\java\Slalom.java`

```
import java.util.Arrays;  // 导入 Arrays 类
import java.util.InputMismatchException;  // 导入 InputMismatchException 类
import java.util.Random;  // 导入 Random 类
import java.util.Scanner;  // 导入 Scanner 类

/**
 * Slalom
 * <p>
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 *
 * There is a bug in the original version where the data pointer doesn't reset after a race is completed. This causes subsequent races to error at
 * some future point on line "540    READ Q"
 */
public class Slalom {

    private static final int MAX_NUM_GATES = 25;  // 定义最大门数为 25
    private static final int[] MAX_SPEED = {  // 定义最大速度数组
            14, 18, 26, 29, 18,
            25, 28, 32, 29, 20,
            29, 29, 25, 21, 26,
            29, 20, 21, 20, 18,
            26, 25, 33, 31, 22
    };

    public static void main(String[] args) {  // 主函数
        var random = new Random();  // 创建 Random 对象

        printIntro();  // 调用打印介绍的函数
        Scanner scanner = new Scanner(System.in);  // 创建 Scanner 对象

        int numGates = readNumberOfGatesChoice(scanner);  // 读取门数选择

        printMenu();  // 打印菜单
        MenuChoice menuChoice;  // 定义菜单选择
        do {
            menuChoice = readMenuOption(scanner);  // 读取菜单选项
            switch (menuChoice) {  // 根据菜单选项进行选择
                case INS:  // 如果是 INS
                    printInstructions();  // 打印指令
                    break;
                case MAX:  // 如果是 MAX
                    printApproxMaxSpeeds(numGates);  // 打印最大速度
                    break;
                case RUN:  // 如果是 RUN
                    run(numGates, scanner, random);  // 运行比赛
                    break;
            }
        } while (menuChoice != MenuChoice.RUN);  // 当菜单选项不是 RUN 时循环
    }
}
    // 运行滑雪比赛
    private static void run(int numGates, Scanner scan, Random random) {
        // 读取滑雪者的评级
        int rating = readSkierRating(scan);
        // 标识比赛是否进行中
        boolean gameInProgress = true;
        // 创建奖牌对象，初始值为0
        var medals = new Medals(0, 0, 0);

        // 当比赛进行中时循环
        while (gameInProgress) {
            // 输出比赛开始倒计时
            System.out.println("THE STARTER COUNTS DOWN...5...4...3...2...1...GO!");
            System.out.println("YOU'RE OFF!");

            // 生成随机速度
            int speed = random.nextInt(18 - 9) + 9;

            // 初始化总耗时
            float totalTimeTaken = 0;
            try {
                // 进行通过门的比赛，并返回总耗时
                totalTimeTaken = runThroughGates(numGates, scan, random, speed);
                // 输出总耗时
                System.out.printf("%nYOU TOOK %.2f SECONDS.%n", totalTimeTaken + random.nextFloat());

                // 评估并更新奖牌情况
                medals = evaluateAndUpdateMedals(totalTimeTaken, numGates, rating, medals);
            } catch (WipedOutOrSnaggedAFlag | DisqualifiedException e) {
                // 比赛结束，打印总耗时并停止
                System.out.printf("%nYOU TOOK %.2f SECONDS.%n", totalTimeTaken + random.nextFloat());
            }

            // 读取是否继续比赛的选择
            gameInProgress = readRaceAgainChoice(scan);
        }

        // 输出感谢参与比赛的信息
        System.out.println("THANKS FOR THE RACE");
        // 如果获得金牌，则输出金牌数量
        if (medals.getGold() >= 1) System.out.printf("GOLD MEDALS: %d%n", medals.getGold());
        // 如果获得银牌，则输出银牌数量
        if (medals.getSilver() >= 1) System.out.printf("SILVER MEDALS: %d%n", medals.getSilver());
        // 如果获得铜牌，则输出铜牌数量
        if (medals.getBronze() >= 1) System.out.printf("BRONZE MEDALS: %d%n", medals.getBronze());
    }
    // 根据总耗时、门数、评分和奖牌情况，评估并更新奖牌情况
    private static Medals evaluateAndUpdateMedals(float totalTimeTaken, int numGates, int rating,
                                                  Medals medals) {
        // 计算每个门的平均耗时
        var m = totalTimeTaken;
        m = m / numGates;
        // 获取当前金牌、银牌、铜牌数量
        int goldMedals = medals.getGold();
        int silverMedals = medals.getSilver();
        int bronzeMedals = medals.getBronze();
        // 根据平均耗时和评分更新奖牌情况
        if (m < 1.5 - (rating * 0.1)) {
            System.out.println("YOU WON A GOLD MEDAL!");
            goldMedals++;
        } else if (m < 2.9 - rating * 0.1) {
            System.out.println("YOU WON A SILVER MEDAL");
            silverMedals++;
        } else if (m < 4.4 - rating * 0.01) {
            System.out.println("YOU WON A BRONZE MEDAL");
            bronzeMedals++;
        }
        // 返回更新后的奖牌情况
        return new Medals(goldMedals, silverMedals, bronzeMedals);
    }

    /**
     * @return the total time taken through all the gates.
     */
    // 检查并处理是否超过最大速度
    private static boolean checkAndProcessIfOverMaxSpeed(Random random, int speed, int maxSpeed) {
        boolean stillInRace = true;
        // 如果速度超过最大速度
        if (speed > maxSpeed) {
            // 根据随机数判断是否成功超速
            if (random.nextFloat() >= (speed - maxSpeed) * 0.1 + 0.2) {
                System.out.println("YOU WENT OVER THE MAXIMUM SPEED AND MADE IT!");
            } else {
                System.out.print("YOU WENT OVER THE MAXIMUM SPEED AND ");
                // 根据随机数判断是否摔倒或抓到旗子
                if (random.nextBoolean()) {
                    System.out.println("WIPED OUT!");
                } else {
                    System.out.println("SNAGGED A FLAG!");
                }
                stillInRace = false;
            }
        } else if (speed > maxSpeed - 1) {
            System.out.println("CLOSE ONE!");
        }
        // 返回是否仍在比赛中
        return stillInRace;
    }
    // 询问用户是否想再次参加比赛，并返回用户选择的布尔值
    private static boolean readRaceAgainChoice(Scanner scan) {
        // 打印提示信息，询问用户是否想再次参加比赛
        System.out.print("\nDO YOU WANT TO RACE AGAIN? ");
        // 初始化用户选择的字符串
        String raceAgain = "";
        // 初始化常量字符串 YES 和 NO
        final String YES = "YES";
        final String NO = "NO";
        // 当用户输入不是 YES 或 NO 时，循环继续询问用户输入
        while (!YES.equals(raceAgain) && !NO.equals(raceAgain)) {
            raceAgain = scan.nextLine();
            // 如果用户输入既不是 YES 也不是 NO，则提示用户重新输入
            if (!(YES.equals(raceAgain) || NO.equals(raceAgain))) {
                System.out.println("PLEASE TYPE 'YES' OR 'NO'");
            }
        }
        // 返回用户是否选择再次参加比赛的布尔值
        return raceAgain.equals(YES);
    }

    // 打印速度信息
    private static void printSpeed(int speed) {
        System.out.printf("%3d M.P.H.%n", speed);
    }

    // 打印比赛用时信息
    private static void printHowLong(float t, Random random) {
        System.out.printf("YOU'VE TAKEN %.2f SECONDS.%n", t + random.nextFloat());
    }

    // 读取用户输入的选项
    private static int readOption(Scanner scan) {
        // 初始化选项为 null
        Integer option = null;

        // 当用户输入不是数字或超出范围时，循环继续询问用户输入
        while (option == null) {
            System.out.print("OPTION? ");
            try {
                option = scan.nextInt();
            } catch (InputMismatchException ex) {
                System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE\n");
            }
            // 清空输入缓冲区
            scan.nextLine();
            // 如果选项不为空且超出范围，则提示用户重新输入
            if (option != null && (option > 8 || option < 0)) {
                System.out.println("WHAT?");
                option = null;
            }
        }
        // 返回用户输入的选项
        return option;
    }

    // 读取用户输入的滑雪者评级
    private static int readSkierRating(Scanner scan) {
        // 初始化评级为 0
        int rating = 0;

        // 当用户输入不在 1-3 范围内时，循环继续询问用户输入
        while (rating < 1 || rating > 3) {
            System.out.print("RATE YOURSELF AS A SKIER, (1=WORST, 3=BEST)? ");
            try {
                rating = scan.nextInt();
                // 如果评级不在 1-3 范围内，则提示用户重新输入
                if (rating < 1 || rating > 3) {
                    System.out.println("THE BOUNDS ARE 1-3");
                }
            } catch (InputMismatchException ex) {
                System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE\n");
            }
            // 清空输入缓冲区
            scan.nextLine();
        }
        // 返回用户输入的滑雪者评级
        return rating;
    }
    // 打印每个门的最大速度
    private static void printApproxMaxSpeeds(int numGates) {
        System.out.println("GATE MAX");
        System.out.println(" #  M.P.H.");
        System.out.println("---------");
        // 遍历每个门，打印门号和对应的最大速度
        for (int i = 0; i < numGates; i++) {
            System.out.println((i+1) + "  " + MAX_SPEED[i]);
        }
    }

    // 打印游戏说明
    private static void printInstructions() {
        System.out.println("\n*** SLALOM: THIS IS THE 1976 WINTER OLYMPIC GIANT SLALOM.  YOU ARE");
        System.out.println("            THE AMERICAN TEAM'S ONLY HOPE OF A GOLD MEDAL.");
        System.out.println();
        System.out.println("     0 -- TYPE THIS IS YOU WANT TO SEE HOW LONG YOU'VE TAKEN.");
        System.out.println("     1 -- TYPE THIS IF YOU WANT TO SPEED UP A LOT.");
        System.out.println("     2 -- TYPE THIS IF YOU WANT TO SPEED UP A LITTLE.");
        System.out.println("     3 -- TYPE THIS IF YOU WANT TO SPEED UP A TEENSY.");
        System.out.println("     4 -- TYPE THIS IF YOU WANT TO KEEP GOING THE SAME SPEED.");
        System.out.println("     5 -- TYPE THIS IF YOU WANT TO CHECK A TEENSY.");
        System.out.println("     6 -- TYPE THIS IF YOU WANT TO CHECK A LITTLE.");
        System.out.println("     7 -- TYPE THIS IF YOU WANT TO CHECK A LOT.");
        System.out.println("     8 -- TYPE THIS IF YOU WANT TO CHEAT AND TRY TO SKIP A GATE.");
        System.out.println();
        System.out.println(" THE PLACE TO USE THESE OPTIONS IS WHEN THE COMPUTER ASKS:");
        System.out.println();
        System.out.println("OPTION?");
        System.out.println();
        System.out.println("                GOOD LUCK!");
    }
    // 读取用户输入的菜单选项
    private static MenuChoice readMenuOption(Scanner scan) {
        // 提示用户输入命令
        System.out.print("COMMAND--? ");
        // 初始化菜单选项
        MenuChoice menuChoice = null;

        // 循环直到用户输入合法的选项
        while (menuChoice == null) {
            // 读取用户输入
            String choice = scan.next();
            // 检查用户输入是否为合法选项
            if (Arrays.stream(MenuChoice.values()).anyMatch(a -> a.name().equals(choice))) {
                menuChoice = MenuChoice.valueOf(choice);
            } else {
                // 提示用户输入非法命令，并要求重新输入
                System.out.print("\""+ choice + "\" IS AN ILLEGAL COMMAND--RETRY? ");
            }
            // 清空输入缓冲区
            scan.nextLine();
        }
        // 返回用户选择的菜单选项
        return menuChoice;
    }

    // 打印菜单选项
    private static void printMenu() {
        System.out.println("TYPE INS FOR INSTRUCTIONS");
        System.out.println("TYPE MAX FOR APPROXIMATE MAXIMUM SPEEDS");
        System.out.println("TYPE RUN FOR THE BEGINNING OF THE RACE");
    }

    // 读取用户输入的门数选择
    private static int readNumberOfGatesChoice(Scanner scan) {
        int numGates = 0;
        // 循环直到用户输入合法的门数
        while (numGates < 1) {
            // 提示用户输入门数
            System.out.print("HOW MANY GATES DOES THIS COURSE HAVE (1 TO 25)? ");
            // 读取用户输入的门数
            numGates = scan.nextInt();
            // 检查门数是否超过最大限制
            if (numGates > MAX_NUM_GATES) {
                // 提示用户门数超过最大限制，并将门数设置为最大限制
                System.out.println(MAX_NUM_GATES + " IS THE LIMIT.");
                numGates = MAX_NUM_GATES;
            }
        }
        // 返回用户输入的门数
        return numGates;
    }

    // 打印游戏介绍
    private static void printIntro() {
        System.out.println("                                SLALOM");
        System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("\n\n");
    }

    // 菜单选项枚举
    private enum MenuChoice {
        INS, MAX, RUN
    }

    // 淘汰异常类
    private static class DisqualifiedException extends Exception {
    }

    // 出界或撞旗异常类
    private static class WipedOutOrSnaggedAFlag extends Exception {
    }
    # 定义一个私有静态内部类 Medals，用于表示奖牌数量
    private static class Medals {
        # 初始化金牌数量为 0
        private int gold = 0;
        # 初始化银牌数量为 0
        private int silver = 0;
        # 初始化铜牌数量为 0
        private int bronze = 0;

        # 构造方法，用于初始化金牌、银牌、铜牌数量
        public Medals(int gold, int silver, int bronze) {
            this.gold = gold;
            this.silver = silver;
            this.bronze = bronze;
        }

        # 获取金牌数量的方法
        public int getGold() {
            return gold;
        }

        # 获取银牌数量的方法
        public int getSilver() {
            return silver;
        }

        # 获取铜牌数量的方法
        public int getBronze() {
            return bronze;
        }
    }
# 闭合前面的函数定义
```