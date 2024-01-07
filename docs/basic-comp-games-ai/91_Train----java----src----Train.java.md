# `basic-computer-games\91_Train\java\src\Train.java`

```

// 导入必要的类
import java.util.Arrays;
import java.util.Scanner;

/**
 * Train
 * <p>
 * 基于 Basic 程序 Train，参考链接 https://github.com/coding-horror/basic-computer-games/blob/main/91%20Train/train.bas
 * <p>
 * 注意：这个程序是为了在 Java 中创建 1970 年代 Basic 程序的版本，没有引入新特性 - 没有添加额外的文本、错误检查等。
 */
public class Train {

    private final Scanner kbScanner; // 创建 Scanner 对象

    public Train() {
        kbScanner = new Scanner(System.in); // 初始化 Scanner 对象
    }

    public void process() {

        intro(); // 调用 intro 方法

        boolean gameOver = false; // 初始化游戏结束标志

        do {
            // 生成随机数
            double carMph = (int) (25 * Math.random() + 40);
            double hours = (int) (15 * Math.random() + 5);
            double train = (int) (19 * Math.random() + 20);

            // 输出信息
            System.out.println(" A CAR TRAVELING " + (int) carMph + " MPH CAN MAKE A CERTAIN TRIP IN");
            System.out.println((int) hours + " HOURS LESS THAN A TRAIN TRAVELING AT " + (int) train + " MPH.");

            // 获取输入
            double howLong = Double.parseDouble(displayTextAndGetInput("HOW LONG DOES THE TRIP TAKE BY CAR? "));

            // 计算结果
            double hoursAnswer = hours * train / (carMph - train);
            int percentage = (int) (Math.abs((hoursAnswer - howLong) * 100 / howLong) + .5);
            if (percentage > 5) {
                System.out.println("SORRY.  YOU WERE OFF BY " + percentage + " PERCENT.");
            } else {
                System.out.println("GOOD! ANSWER WITHIN " + percentage + " PERCENT.");
            }
            System.out.println("CORRECT ANSWER IS " + hoursAnswer + " HOURS.");

            System.out.println();
            // 判断是否继续游戏
            if (!yesEntered(displayTextAndGetInput("ANOTHER PROBLEM (YES OR NO)? "))) {
                gameOver = true;
            }

        } while (!gameOver);


    }

    // 输出游戏介绍信息
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
     * @param text 要在屏幕上显示的消息。
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
     * @param text    源字符串
     * @param values  要与源字符串进行比较的一系列值
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
}

```