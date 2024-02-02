# `basic-computer-games\58_Love\java\src\Love.java`

```py
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
    private final ArrayList<Integer> data;

    // Used for keyboard input
    private final Scanner kbScanner;

    public Love() {
        // 初始化数据列表
        data = storeData();
        // 初始化键盘输入扫描器
        kbScanner = new Scanner(System.in);
    }

    /**
     * Show an intro, accept a message, then draw the picture.
     */
    // 显示游戏介绍
    private void intro() {
        System.out.println(addSpaces(33) + "LOVE");
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("A TRIBUTE TO THE GREAT AMERICAN ARTIST, ROBERT INDIANA.");
        System.out.println("HIS GREATEST WORK WILL BE REPRODUCED WITH A MESSAGE OF");
        System.out.println("YOUR CHOICE UP TO 60 CHARACTERS.  IF YOU CAN'T THINK OF");
        System.out.println("A MESSAGE, SIMPLE TYPE THE WORD 'LOVE'");
        System.out.println();
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    // 显示文本并获取玩家输入
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.nextLine();
    }

    /**
     * Return a string of x spaces
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    // 返回指定数量的空格字符串
    // 创建一个方法，用于在字符串中添加指定数量的空格
    private String addSpaces(int spaces) {
        // 创建一个包含指定数量空格的字符数组
        char[] spacesTemp = new char[spaces];
        // 用空格填充字符数组
        Arrays.fill(spacesTemp, ' ');
        // 将字符数组转换为字符串并返回
        return new String(spacesTemp);
    }

    /**
     * 原始的基本程序中的数据以DATA格式存在。我们将所有数据导入数组以便于处理。
     * 数据的格式是：
     * 数据的第一个整数是60，表示每行的字符数。
     * 数据的最后一个整数与上面的第一个整数相同。
     * 然后数据在指定的字符数和空格数之间交替。
     * 您需要保持对读取的整数计数的累加，一旦达到60，就该打印并重置计数为零。
     *
     * @return 包含数据的整数类型的ArrayList
     */
    // 存储数据的方法
    private ArrayList<Integer> storeData() {

        // 创建一个整数类型的数组列表
        ArrayList<Integer> theData = new ArrayList<>();

        // 将给定的整数数组转换为列表，并添加到 theData 中
        theData.addAll(Arrays.asList(60, 1, 12, 26, 9, 12, 3, 8, 24, 17, 8, 4, 6, 23, 21, 6, 4, 6, 22, 12, 5, 6, 5));
        theData.addAll(Arrays.asList(4, 6, 21, 11, 8, 6, 4, 4, 6, 21, 10, 10, 5, 4, 4, 6, 21, 9, 11, 5, 4));
        theData.addAll(Arrays.asList(4, 6, 21, 8, 11, 6, 4, 4, 6, 21, 7, 11, 7, 4, 4, 6, 21, 6, 11, 8, 4));
        theData.addAll(Arrays.asList(4, 6, 19, 1, 1, 5, 11, 9, 4, 4, 6, 19, 1, 1, 5, 10, 10, 4, 4, 6, 18, 2, 1, 6, 8, 11, 4));
        theData.addAll(Arrays.asList(4, 6, 17, 3, 1, 7, 5, 13, 4, 4, 6, 15, 5, 2, 23, 5, 1, 29, 5, 17, 8));
        theData.addAll(Arrays.asList(1, 29, 9, 9, 12, 1, 13, 5, 40, 1, 1, 13, 5, 40, 1, 4, 6, 13, 3, 10, 6, 12, 5, 1));
        theData.addAll(Arrays.asList(5, 6, 11, 3, 11, 6, 14, 3, 1, 5, 6, 11, 3, 11, 6, 15, 2, 1));
        theData.addAll(Arrays.asList(6, 6, 9, 3, 12, 6, 16, 1, 1, 6, 6, 9, 3, 12, 6, 7, 1, 10));
        theData.addAll(Arrays.asList(7, 6, 7, 3, 13, 6, 6, 2, 10, 7, 6, 7, 3, 13, 14, 10, 8, 6, 5, 3, 14, 6, 6, 2, 10));
        theData.addAll(Arrays.asList(8, 6, 5, 3, 14, 6, 7, 1, 10, 9, 6, 3, 3, 15, 6, 16, 1, 1));
        theData.addAll(Arrays.asList(9, 6, 3, 3, 15, 6, 15, 2, 1, 10, 6, 1, 3, 16, 6, 14, 3, 1, 10, 10, 16, 6, 12, 5, 1));
        theData.addAll(Arrays.asList(11, 8, 13, 27, 1, 11, 8, 13, 27, 1, 60));

        // 返回存储数据的数组列表
        return theData;
    }

    // 主方法
    public static void main(String[] args) {

        // 创建 Love 对象
        Love love = new Love();
        // 调用 process 方法
        love.process();
    }
# 闭合前面的函数定义
```