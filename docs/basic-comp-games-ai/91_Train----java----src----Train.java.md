# `basic-computer-games\91_Train\java\src\Train.java`

```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Train
 * <p>
 * Based on the Basic program Train here
 * https://github.com/coding-horror/basic-computer-games/blob/main/91%20Train/train.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic program in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Train {

    private final Scanner kbScanner;

    public Train() {
        // 创建一个用于读取用户输入的 Scanner 对象
        kbScanner = new Scanner(System.in);
    }

    public void process() {

        // 显示游戏介绍
        intro();

        // 游戏是否结束的标志
        boolean gameOver = false;

        // 游戏循环
        do {
            // 生成随机的汽车速度、时间和火车速度
            double carMph = (int) (25 * Math.random() + 40);
            double hours = (int) (15 * Math.random() + 5);
            double train = (int) (19 * Math.random() + 20);

            // 显示问题
            System.out.println(" A CAR TRAVELING " + (int) carMph + " MPH CAN MAKE A CERTAIN TRIP IN");
            System.out.println((int) hours + " HOURS LESS THAN A TRAIN TRAVELING AT " + (int) train + " MPH.");

            // 获取用户输入的汽车行程时间
            double howLong = Double.parseDouble(displayTextAndGetInput("HOW LONG DOES THE TRIP TAKE BY CAR? "));

            // 计算正确答案并与用户输入进行比较
            double hoursAnswer = hours * train / (carMph - train);
            int percentage = (int) (Math.abs((hoursAnswer - howLong) * 100 / howLong) + .5);
            if (percentage > 5) {
                System.out.println("SORRY.  YOU WERE OFF BY " + percentage + " PERCENT.");
            } else {
                System.out.println("GOOD! ANSWER WITHIN " + percentage + " PERCENT.");
            }
            System.out.println("CORRECT ANSWER IS " + hoursAnswer + " HOURS.");

            System.out.println();
            // 询问用户是否继续游戏
            if (!yesEntered(displayTextAndGetInput("ANOTHER PROBLEM (YES OR NO)? "))) {
                gameOver = true;
            }

        } while (!gameOver);


    }
    // 打印介绍信息
    private void intro() {
        System.out.println("TRAIN");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("TIME - SPEED DISTANCE EXERCISE");
        System.out.println();
    }

    /*
     * 在屏幕上打印消息，然后从键盘接受输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * 检查玩家是否输入了 Y 或 YES。
     *
     * @param text 从键盘输入的字符串
     * @return 如果输入了 Y 或 YES，则返回 true，否则返回 false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * 检查字符串是否等于一系列变量值之一
     * 用于检查是否输入了 Y 或 YES 等
     * 比较不区分大小写。
     *
     * @param text   源字符串
     * @param values 要与源字符串进行比较的一系列值
     * @return 如果在传递的一系列字符串中找到了匹配，则返回 true
     */
    private boolean stringIsAnyValue(String text, String... values) {

        return Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text));

    }

    /**
     * 程序启动。
     *
     * @param args 未使用（来自命令行）。
     */
    public static void main(String[] args) {
        Train train = new Train();
        train.process();
    }
# 闭合之前的函数定义
```