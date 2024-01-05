# `91_Train\java\src\Train.java`

```
import java.util.Arrays;  # 导入 java.util.Arrays 包，用于操作数组
import java.util.Scanner;  # 导入 java.util.Scanner 包，用于接收用户输入

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

    private final Scanner kbScanner;  # 声明一个私有的 Scanner 对象 kbScanner

    public Train() {  # 构造函数
        kbScanner = new Scanner(System.in);  # 初始化 kbScanner 对象，用于接收用户输入
    }
        public void process() {  // 定义一个名为process的公共方法

        intro();  // 调用intro方法

        boolean gameOver = false;  // 定义一个布尔变量gameOver，并初始化为false

        do {  // 开始一个do-while循环

            double carMph = (int) (25 * Math.random() + 40);  // 生成一个随机数赋值给carMph
            double hours = (int) (15 * Math.random() + 5);  // 生成一个随机数赋值给hours
            double train = (int) (19 * Math.random() + 20);  // 生成一个随机数赋值给train

            System.out.println(" A CAR TRAVELING " + (int) carMph + " MPH CAN MAKE A CERTAIN TRIP IN");  // 打印输出信息
            System.out.println((int) hours + " HOURS LESS THAN A TRAIN TRAVELING AT " + (int) train + " MPH.");  // 打印输出信息

            double howLong = Double.parseDouble(displayTextAndGetInput("HOW LONG DOES THE TRIP TAKE BY CAR? "));  // 调用displayTextAndGetInput方法获取输入并转换为double类型

            double hoursAnswer = hours * train / (carMph - train);  // 计算hoursAnswer的值
            int percentage = (int) (Math.abs((hoursAnswer - howLong) * 100 / howLong) + .5);  // 计算percentage的值
            if (percentage > 5) {  // 如果percentage大于5
                System.out.println("SORRY.  YOU WERE OFF BY " + percentage + " PERCENT.");  // 打印输出信息
            } else {
                // 如果答案在给定百分比内，则打印“GOOD! ANSWER WITHIN”和百分比
                System.out.println("GOOD! ANSWER WITHIN " + percentage + " PERCENT.");
            }
            // 打印“CORRECT ANSWER IS”和hoursAnswer的值
            System.out.println("CORRECT ANSWER IS " + hoursAnswer + " HOURS.");

            // 打印空行
            System.out.println();
            // 如果用户输入的不是“YES”，则将gameOver设置为true
            if (!yesEntered(displayTextAndGetInput("ANOTHER PROBLEM (YES OR NO)? "))) {
                gameOver = true;
            }

        } while (!gameOver);


    }

    private void intro() {
        // 打印“TRAIN”
        System.out.println("TRAIN");
        // 打印“CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY”
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        // 打印空行
        System.out.println();
        // 打印“TIME - SPEED DISTANCE EXERCISE”
        System.out.println("TIME - SPEED DISTANCE EXERCISE");
        System.out.println();  // 打印空行

    }

    /*
     * 在屏幕上打印一条消息，然后接受键盘输入。
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);  // 在屏幕上打印消息
        return kbScanner.next();  // 返回玩家输入的内容
    }

    /**
     * 检查玩家是否输入了Y或YES作为答案。
     *
     * @param text 玩家从键盘输入的字符串
     * @return 如果输入了Y或YES，则返回true，否则返回false
     */
    private boolean yesEntered(String text) {  // 定义一个名为yesEntered的私有方法，接受一个字符串参数text
        return stringIsAnyValue(text, "Y", "YES");  // 调用stringIsAnyValue方法，判断text是否等于"Y"或"YES"，并返回结果
    }

    /**
     * Check whether a string equals one of a variable number of values
     * Useful to check for Y or YES for example
     * Comparison is case insensitive.
     *
     * @param text   source string  // 参数text是源字符串
     * @param values a range of values to compare against the source string  // 参数values是要与源字符串进行比较的一系列值
     * @return true if a comparison was found in one of the variable number of strings passed  // 如果在传递的一系列字符串中找到了比较，则返回true
     */
    private boolean stringIsAnyValue(String text, String... values) {  // 定义一个名为stringIsAnyValue的私有方法，接受一个字符串参数text和一个可变数量的字符串参数values

        return Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text));  // 使用流处理values数组，检查是否有任何一个值与text相等（忽略大小写），并返回结果

    }

    /**
# 程序启动
# 参数 args 未被使用（来自命令行）
def main(args):
    # 创建 Train 对象
    train = Train()
    # 调用 Train 对象的 process 方法
    train.process()
```