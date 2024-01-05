# `79_Slalom\java\Slalom.java`

```
import java.util.Arrays;  // 导入 Arrays 类，用于操作数组
import java.util.InputMismatchException;  // 导入 InputMismatchException 类，用于处理输入不匹配异常
import java.util.Random;  // 导入 Random 类，用于生成随机数
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入

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
                case MAX:  # 如果选择了MAX
                    printApproxMaxSpeeds(numGates);  # 调用printApproxMaxSpeeds函数打印最大速度
                    break;  # 结束switch语句
                case RUN:  # 如果选择了RUN
                    run(numGates, scanner, random);  # 调用run函数，传入参数numGates, scanner, random
                    break;  # 结束switch语句
            }
        } while (menuChoice != MenuChoice.RUN);  # 当menuChoice不等于RUN时循环

    private static void run(int numGates, Scanner scan, Random random) {  # 定义run函数，接受参数numGates, scan, random
        int rating = readSkierRating(scan);  # 调用readSkierRating函数，将返回值赋给rating
        boolean gameInProgress = true;  # 初始化gameInProgress为true
        var medals = new Medals(0, 0, 0);  # 创建一个Medals对象，初始化为0

        while (gameInProgress) {  # 当gameInProgress为true时循环
            System.out.println("THE STARTER COUNTS DOWN...5...4...3...2...1...GO!");  # 打印倒计时信息
            System.out.println("YOU'RE OFF!");  # 打印出发信息

            int speed = random.nextInt(18 - 9) + 9;  # 生成一个9到18之间的随机数赋给speed
            float totalTimeTaken = 0; // 初始化总耗时为0
            try {
                totalTimeTaken = runThroughGates(numGates, scan, random, speed); // 调用runThroughGates方法，计算通过所有门所需的总时间
                System.out.printf("%nYOU TOOK %.2f SECONDS.%n", totalTimeTaken + random.nextFloat()); // 打印总时间

                medals = evaluateAndUpdateMedals(totalTimeTaken, numGates, rating, medals); // 调用evaluateAndUpdateMedals方法，评估并更新奖牌情况
            } catch (WipedOutOrSnaggedAFlag | DisqualifiedException e) {
                //end of this race! Print time taken and stop
                System.out.printf("%nYOU TOOK %.2f SECONDS.%n", totalTimeTaken + random.nextFloat()); // 打印总时间
            }

            gameInProgress = readRaceAgainChoice(scan); // 读取是否继续比赛的选择
        }

        System.out.println("THANKS FOR THE RACE"); // 打印感谢信息
        if (medals.getGold() >= 1) System.out.printf("GOLD MEDALS: %d%n", medals.getGold()); // 如果金牌数量大于等于1，则打印金牌数量
        if (medals.getSilver() >= 1) System.out.printf("SILVER MEDALS: %d%n", medals.getSilver()); // 如果银牌数量大于等于1，则打印银牌数量
        if (medals.getBronze() >= 1) System.out.printf("BRONZE MEDALS: %d%n", medals.getBronze()); // 如果铜牌数量大于等于1，则打印铜牌数量
    }
    // 评估并更新奖牌数量
    private static Medals evaluateAndUpdateMedals(float totalTimeTaken, int numGates, int rating,
                                                  Medals medals) {
        // 将总时间除以门数，得到平均时间
        var m = totalTimeTaken;
        m = m / numGates;
        // 获取当前金牌、银牌、铜牌数量
        int goldMedals = medals.getGold();
        int silverMedals = medals.getSilver();
        int bronzeMedals = medals.getBronze();
        // 根据平均时间和评分判断获得的奖牌
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
        // 返回更新后的奖牌数量
        return new Medals(goldMedals, silverMedals, bronzeMedals);
    }
    /**
     * @return the total time taken through all the gates.
     */
    private static float runThroughGates(int numGates, Scanner scan, Random random, int speed) throws DisqualifiedException, WipedOutOrSnaggedAFlag {
        float totalTimeTaken = 0.0f; // 初始化总时间为0
        for (int i = 0; i < numGates; i++) { // 循环遍历每个门
            var gateNum = i + 1; // 获取门的编号
            boolean stillInRace = true; // 初始化仍在比赛中为true
            boolean gateCompleted = false; // 初始化门未完成为false
            while (!gateCompleted) { // 当门未完成时循环
                System.out.printf("%nHERE COMES GATE # %d:%n", gateNum); // 打印门的编号
                printSpeed(speed); // 打印速度

                var tmpSpeed = speed; // 临时保存速度

                int chosenOption = readOption(scan); // 从输入中读取选项
                switch (chosenOption) { // 根据选项进行不同的操作
                    case 0:
                        //how long
                    // 打印总共花费的时间和随机数
                    printHowLong(totalTimeTaken, random);
                    // 跳出循环
                    break;
                    // 速度大幅提升
                    speed = speed + random.nextInt(10 - 5) + 5;
                    // 速度略微提升
                    speed = speed + random.nextInt(5 - 3) + 3;
                    // 速度微微提升
                    speed = speed + random.nextInt(4 - 1) + 1;
                    // 保持当前速度不变
                    // 速度微微减少
                    speed = speed - random.nextInt(4 - 1) + 1;
                    case 6:
                        // 检查一点
                        speed = speed - random.nextInt(5 - 3) + 3;
                        break;
                    case 7:
                        // 检查很多
                        speed = speed - random.nextInt(10 - 5) + 5;
                        break;
                    case 8:
                        // 作弊
                        System.out.println("***作弊");
                        if (random.nextFloat() < 0.7) {
                            System.out.println("有官员抓到你了！");
                            stillInRace = false;
                        } else {
                            System.out.println("你成功了！");
                            totalTimeTaken = totalTimeTaken + 1.5f;
                        }
                        break;
                }

                if (stillInRace) {  # 如果仍然在比赛中
                    printSpeed(speed);  # 打印当前速度
                    stillInRace = checkAndProcessIfOverMaxSpeed(random, speed, MAX_SPEED[i]);  # 检查并处理是否超过最大速度
                    if (!stillInRace) throw new WipedOutOrSnaggedAFlag();  # 如果不再比赛中，则抛出 WipedOutOrSnaggedAFlag 异常
                } else {
                    throw new DisqualifiedException();  # 否则，抛出 DisqualifiedException 异常，表示被取消资格
                }

                if (speed < 7) {  # 如果速度小于7
                    System.out.println("LET'S BE REALISTIC, OK?  LET'S GO BACK AND TRY AGAIN...");  # 打印提示信息
                    speed = tmpSpeed;  # 速度恢复到之前的值
                    gateCompleted = false;  # 设置gateCompleted为false
                } else {
                    totalTimeTaken = totalTimeTaken + (MAX_SPEED[i] - speed + 1);  # 总共花费的时间增加
                    if (speed > MAX_SPEED[i]) {  # 如果速度超过最大速度
                        totalTimeTaken = totalTimeTaken + 0.5f;  # 总共花费的时间再增加0.5
                    }
                    gateCompleted = true;  # 设置gateCompleted为true
    }

        }
    }

    private static boolean checkAndProcessIfOverMaxSpeed(Random random, int speed, int maxSpeed) {
        // 检查并处理是否超过最大速度
        boolean stillInRace = true;
        if (speed > maxSpeed) {
            // 如果速度超过最大速度
            if (random.nextFloat() >= (speed - maxSpeed) * 0.1 + 0.2) {
                // 如果随机数大于等于计算得到的值
                System.out.println("YOU WENT OVER THE MAXIMUM SPEED AND MADE IT!");
            } else {
                // 如果随机数小于计算得到的值
                System.out.print("YOU WENT OVER THE MAXIMUM SPEED AND ");
                if (random.nextBoolean()) {
                    // 如果随机布尔值为真
                    System.out.println("WIPED OUT!");
                } else {
                    // 如果随机布尔值为假
                    System.out.println("SNAGGED A FLAG!");
                }
                stillInRace = false;
            }
        }
        return stillInRace;
    }
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
    // 打印速度
    private static void printSpeed(int speed) {
        System.out.printf("%3d M.P.H.%n", speed);
    }

    // 打印花费的时间
    private static void printHowLong(float t, Random random) {
        System.out.printf("YOU'VE TAKEN %.2f SECONDS.%n", t + random.nextFloat());
    }

    // 读取用户输入的选项
    private static int readOption(Scanner scan) {
        Integer option = null;

        // 循环直到用户输入有效选项
        while (option == null) {
            System.out.print("OPTION? ");
            try {
                option = scan.nextInt(); // 读取用户输入的整数选项
            } catch (InputMismatchException ex) {
                System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE\n"); // 捕获输入不是整数的异常
            }
            scan.nextLine(); // 清空输入缓冲区
        }
            if (option != null && (option > 8 || option < 0)) {  # 检查选项是否不为空且不在0到8的范围内
                System.out.println("WHAT?");  # 输出错误信息
                option = null;  # 将选项设置为null
            }
        }
        return option;  # 返回选项值
    }

    private static int readSkierRating(Scanner scan) {  # 定义一个静态方法readSkierRating，接受一个Scanner对象作为参数
        int rating = 0;  # 初始化评分为0

        while (rating < 1 || rating > 3) {  # 当评分小于1或大于3时循环
            System.out.print("RATE YOURSELF AS A SKIER, (1=WORST, 3=BEST)? ");  # 输出提示信息
            try {
                rating = scan.nextInt();  # 从输入中读取评分
                if (rating < 1 || rating > 3) {  # 如果评分不在1到3的范围内
                    System.out.println("THE BOUNDS ARE 1-3");  # 输出错误信息
                }
            } catch (InputMismatchException ex) {  # 捕获输入不匹配异常
                System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE\n");  # 输出错误信息
    }
    scan.nextLine();  // 读取下一行输入
}
return rating;  // 返回评分

private static void printApproxMaxSpeeds(int numGates) {
    System.out.println("GATE MAX");  // 打印标题
    System.out.println(" #  M.P.H.");  // 打印表头
    System.out.println("---------");  // 打印分隔线
    for (int i = 0; i < numGates; i++) {  // 循环打印每个门的最大速度
        System.out.println((i+1) + "  " + MAX_SPEED[i]);
    }
}

private static void printInstructions() {
    System.out.println("\n*** SLALOM: THIS IS THE 1976 WINTER OLYMPIC GIANT SLALOM.  YOU ARE");  // 打印比赛介绍
    System.out.println("            THE AMERICAN TEAM'S ONLY HOPE OF A GOLD MEDAL.");
    System.out.println();
    System.out.println("     0 -- TYPE THIS IS YOU WANT TO SEE HOW LONG YOU'VE TAKEN.");  // 打印指令
        System.out.println("     1 -- TYPE THIS IF YOU WANT TO SPEED UP A LOT.");  // 打印提示信息
        System.out.println("     2 -- TYPE THIS IF YOU WANT TO SPEED UP A LITTLE.");  // 打印提示信息
        System.out.println("     3 -- TYPE THIS IF YOU WANT TO SPEED UP A TEENSY.");  // 打印提示信息
        System.out.println("     4 -- TYPE THIS IF YOU WANT TO KEEP GOING THE SAME SPEED.");  // 打印提示信息
        System.out.println("     5 -- TYPE THIS IF YOU WANT TO CHECK A TEENSY.");  // 打印提示信息
        System.out.println("     6 -- TYPE THIS IF YOU WANT TO CHECK A LITTLE.");  // 打印提示信息
        System.out.println("     7 -- TYPE THIS IF YOU WANT TO CHECK A LOT.");  // 打印提示信息
        System.out.println("     8 -- TYPE THIS IF YOU WANT TO CHEAT AND TRY TO SKIP A GATE.");  // 打印提示信息
        System.out.println();  // 打印空行
        System.out.println(" THE PLACE TO USE THESE OPTIONS IS WHEN THE COMPUTER ASKS:");  // 打印提示信息
        System.out.println();  // 打印空行
        System.out.println("OPTION?");  // 打印提示信息
        System.out.println();  // 打印空行
        System.out.println("                GOOD LUCK!");  // 打印提示信息
    }

    private static MenuChoice readMenuOption(Scanner scan) {  // 定义一个静态方法，接受一个Scanner对象作为参数
        System.out.print("COMMAND--? ");  // 打印提示信息
        MenuChoice menuChoice = null;  // 初始化menuChoice变量为null
        while (menuChoice == null) {  # 当menuChoice为空时，进入循环
            String choice = scan.next();  # 从输入中读取一个字符串
            if (Arrays.stream(MenuChoice.values()).anyMatch(a -> a.name().equals(choice))) {  # 检查输入是否在MenuChoice枚举中
                menuChoice = MenuChoice.valueOf(choice);  # 将输入转换为MenuChoice枚举类型并赋值给menuChoice
            } else {
                System.out.print("\""+ choice + "\" IS AN ILLEGAL COMMAND--RETRY? ");  # 如果输入不在枚举中，则打印错误信息
            }
            scan.nextLine();  # 读取下一行输入
        }
        return menuChoice;  # 返回menuChoice
    }

    private static void printMenu() {  # 打印菜单选项
        System.out.println("TYPE INS FOR INSTRUCTIONS");
        System.out.println("TYPE MAX FOR APPROXIMATE MAXIMUM SPEEDS");
        System.out.println("TYPE RUN FOR THE BEGINNING OF THE RACE");
    }

    private static int readNumberOfGatesChoice(Scanner scan) {  # 读取门数选择
        int numGates = 0;  # 初始化门数为0
        while (numGates < 1) {  # 当门数小于1时，进入循环
            System.out.print("HOW MANY GATES DOES THIS COURSE HAVE (1 TO 25)? ");  # 打印提示信息，询问门数范围
            numGates = scan.nextInt();  # 从用户输入中获取门数
            if (numGates > MAX_NUM_GATES) {  # 如果门数大于最大门数限制
                System.out.println(MAX_NUM_GATES + " IS THE LIMIT.");  # 打印最大门数限制
                numGates = MAX_NUM_GATES;  # 将门数设置为最大门数限制
            }
        }
        return numGates;  # 返回门数

    }

    private static void printIntro() {  # 定义打印介绍信息的函数
        System.out.println("                                SLALOM");  # 打印标题
        System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  # 打印创意计算机的地点
        System.out.println("\n\n");  # 打印空行
    }

    private enum MenuChoice {  # 定义菜单选项的枚举类型
        INS, MAX, RUN  # 三个选项：插入、最大、运行
    }
    private static class DisqualifiedException extends Exception {
        // 自定义异常类，表示被取消资格
    }

    private static class WipedOutOrSnaggedAFlag extends Exception {
        // 自定义异常类，表示被淘汰或抓住了旗帜
    }

    private static class Medals {
        private int gold = 0;
        private int silver = 0;
        private int bronze = 0;

        public Medals(int gold, int silver, int bronze) {
            this.gold = gold;
            this.silver = silver;
            this.bronze = bronze;
            // 构造方法，用于初始化金牌、银牌、铜牌的数量
        }

        public int getGold() {
            return gold;
            // 返回金牌数量
        }
        }

        public int getSilver() {
            return silver;  # 返回银牌数量
        }

        public int getBronze() {
            return bronze;  # 返回铜牌数量
        }
    }


}
```