# `d:/src/tocomm/basic-computer-games\58_Love\java\src\Love.java`

```
# 导入所需的模块
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Love
 * <p>
 * Based on the Basic game of Love here
 * https://github.com/coding-horror/basic-computer-games/blob/main/58%20Love/love.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */

public class Love {

    // This is actually defined in the data, but made it a const for readability
    public static final int ROW_LENGTH = 60;

    // Contains the data to draw the picture
```

这段代码是一个 Java 类的定义，包含了类的注释和常量的定义。在注释中说明了这个类是基于 1970 年代的 Basic 游戏的 Java 版本，没有引入新的功能，没有添加额外的文本、错误检查等。在代码中定义了一个常量 ROW_LENGTH，用于表示行的长度。接下来应该是该类的具体实现。
    private final ArrayList<Integer> data;  // 声明一个名为data的整数数组列表

    // 用于键盘输入
    private final Scanner kbScanner;  // 声明一个名为kbScanner的Scanner对象

    public Love() {  // Love类的构造函数
        data = storeData();  // 调用storeData方法并将返回的数组列表赋值给data
        kbScanner = new Scanner(System.in);  // 创建一个新的Scanner对象，用于从键盘输入
    }

    /**
     * 显示介绍，接受一条消息，然后绘制图片。
     */
    public void process() {  // process方法
        intro();  // 调用intro方法，显示介绍

        int rowLength = data.get(0);  // 从data数组列表中获取第一个元素并赋值给rowLength

        String message = displayTextAndGetInput("YOUR MESSAGE, PLEASE ");  // 调用displayTextAndGetInput方法，显示提示消息并接受用户输入的消息
        // 确保字符串至少有60个字符
        while (message.length() < rowLength) {
            message += message;
        }

        // 剪掉多余的字符，使其长度恰好为ROW_LENGTH
        if (message.length() > ROW_LENGTH) {
            message = message.substring(0, ROW_LENGTH);
        }

        // 打印标题
        System.out.println(message);

        int pos = 1;  // 不读取第一个元素位置的行长度值

        int runningLineTotal = 0;
        StringBuilder lineText = new StringBuilder();
        boolean outputChars = true;
        while (true) {
            int charsOrSpacesLength = data.get(pos);
            if (charsOrSpacesLength == ROW_LENGTH) {
                // 如果读取的字符或空格长度等于行长度，则表示已经到达文件末尾，退出循环
                break;
            }
            if (outputChars) {
                // 如果需要输出字符，则从消息字符串中添加 charsOrSpacesLength 个字符到 lineText 中
                for (int i = 0; i < charsOrSpacesLength; i++) {
                    lineText.append(message.charAt(i + runningLineTotal));
                    // 切换到下一个元素中输出空格
                    outputChars = false;
                }
            } else {
                // 如果需要输出空格，则向字符串中添加 charsOrSpacesLength 个空格
                lineText.append(addSpaces(charsOrSpacesLength));
                // 切换到下一个循环中输出字符
                outputChars = true;
            }

            // 记录已经读取的字符或空格的总数，用于判断何时打印字符串
            runningLineTotal += charsOrSpacesLength;
            // 如果当前行的字符总数达到了指定的行长度，就打印当前行的文本内容，并重置为下一行
            if (runningLineTotal >= ROW_LENGTH) {
                System.out.println(lineText);
                lineText = new StringBuilder();
                runningLineTotal = 0;
                outputChars = true;
            }

            // 移动到下一个数组列表元素
            pos++;
        }

        // 打印页脚
        System.out.println(message);

    }

    private void intro() {
        System.out.println(addSpaces(33) + "LOVE");
```
        // 打印一行空格和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"在屏幕上
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        // 打印空行
        System.out.println();
        // 打印"A TRIBUTE TO THE GREAT AMERICAN ARTIST, ROBERT INDIANA."在屏幕上
        System.out.println("A TRIBUTE TO THE GREAT AMERICAN ARTIST, ROBERT INDIANA.");
        // 打印"HIS GREATEST WORK WILL BE REPRODUCED WITH A MESSAGE OF"在屏幕上
        System.out.println("HIS GREATEST WORK WILL BE REPRODUCED WITH A MESSAGE OF");
        // 打印"YOUR CHOICE UP TO 60 CHARACTERS.  IF YOU CAN'T THINK OF"在屏幕上
        System.out.println("YOUR CHOICE UP TO 60 CHARACTERS.  IF YOU CAN'T THINK OF");
        // 打印"A MESSAGE, SIMPLE TYPE THE WORD 'LOVE'"在屏幕上
        System.out.println("A MESSAGE, SIMPLE TYPE THE WORD 'LOVE'");
        // 打印空行
        System.out.println();
    }

    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        // 在屏幕上打印消息
        System.out.print(text);
        // 从键盘接受输入并返回
        return kbScanner.nextLine();
    }
    /**
     * Return a string of x spaces
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    private String addSpaces(int spaces) {
        // 创建一个包含指定数量空格的字符数组
        char[] spacesTemp = new char[spaces];
        // 用空格填充字符数组
        Arrays.fill(spacesTemp, ' ');
        // 将字符数组转换为字符串并返回
        return new String(spacesTemp);
    }

    /**
     * Original Basic program had the data in DATA format.  We're importing all the data into an array for ease of
     * processing.
     * Format of data is
     * FIRST int of data is 60, which is the number of characters per line.
     * LAST int of data is same as FIRST above.
     * Then the data alternates between how many characters to print and how many spaces to print
     * You need to keep a running total of the count of ints read and once this hits 60, its time to
     */
    # 创建一个名为 storeData 的私有方法，返回一个整数类型的 ArrayList
    def storeData():
        # 创建一个名为 theData 的 ArrayList
        theData = []
        
        # 将整数列表添加到 theData 中
        theData.extend([60, 1, 12, 26, 9, 12, 3, 8, 24, 17, 8, 4, 6, 23, 21, 6, 4, 6, 22, 12, 5, 6, 5])
        theData.extend([4, 6, 21, 11, 8, 6, 4, 4, 6, 21, 10, 10, 5, 4, 4, 6, 21, 9, 11, 5, 4])
        theData.extend([4, 6, 21, 8, 11, 6, 4, 4, 6, 21, 7, 11, 7, 4, 4, 6, 21, 6, 11, 8, 4])
        theData.extend([4, 6, 19, 1, 1, 5, 11, 9, 4, 4, 6, 19, 1, 1, 5, 10, 10, 4, 4, 6, 18, 2, 1, 6, 8, 11, 4])
        theData.extend([4, 6, 17, 3, 1, 7, 5, 13, 4, 4, 6, 15, 5, 2, 23, 5, 1, 29, 5, 17, 8])
        theData.extend([1, 29, 9, 9, 12, 1, 13, 5, 40, 1, 1, 13, 5, 40, 1, 4, 6, 13, 3, 10, 6, 12, 5, 1])
        theData.extend([5, 6, 11, 3, 11, 6, 14, 3, 1, 5, 6, 11, 3, 11, 6, 15, 2, 1])
        theData.extend([6, 6, 9, 3, 12, 6, 16, 1, 1, 6, 6, 9, 3, 12, 6, 7, 1, 10])
        theData.extend([7, 6, 7, 3, 13, 6, 6, 2, 10, 7, 6, 7, 3, 13, 14, 10, 8, 6, 5, 3, 14, 6, 6, 2, 10])
        theData.extend([8, 6, 5, 3, 14, 6, 7, 1, 10, 9, 6, 3, 3, 15, 6, 16, 1, 1])
        theData.extend([9, 6, 3, 3, 15, 6, 15, 2, 1, 10, 6, 1, 3, 16, 6, 14, 3, 1, 10, 10, 16, 6, 12, 5, 1])
        theData.extend([11, 8, 13, 27, 1, 11, 8, 13, 27, 1, 60])
        
        # 返回 theData
        return theData
        return theData;
    }
    # 定义一个静态方法，名称为main，参数为字符串数组args
    public static void main(String[] args) {
        # 创建一个Love对象
        Love love = new Love();
        # 调用Love对象的process方法
        love.process();
    }
}
```