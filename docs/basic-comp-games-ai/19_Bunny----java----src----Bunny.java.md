# `basic-computer-games\19_Bunny\java\src\Bunny.java`

```
// 导入必要的类
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Bunny
 * <p>
 * 基于 Basic 程序 Bunny
 * https://github.com/coding-horror/basic-computer-games/blob/main/19%20Bunny/bunny.bas
 * <p>
 * 注意：这个想法是在 Java 中创建一个 1970 年代 Basic 游戏的版本，没有引入新功能 - 没有添加额外的文本、错误检查等。
 */
public class Bunny {

    // 第一个 4 个元素是文本 BUNNY，所以跳过这些
    public static final int REAL_DATA_START_POS = 5;

    // 字符的数据不代表三个 ASCII 字符，所以我们需要添加 64，按照原始程序设计。
    public static final int CONVERT_TO_ASCII = 64;

    public static final int EOF = 4096; // 文件结束
    public static final int EOL = -1;  // 行结束

    // 包含绘制图片的数据
    private final ArrayList<Integer> data;

    public Bunny() {
        // 加载数据
        data = loadData();
    }

    /**
     * 显示介绍，然后绘制图片。
     */
    private void intro() {
        System.out.println(addSpaces(33) + "BUNNY");
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
    }

    /**
     * 返回 x 个空格的字符串
     *
     * @param spaces 所需的空格数
     * @return 包含空格数的字符串
     */
    private String addSpaces(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }
}
    /**
     * Original Basic program had the data in DATA format.
     * We're importing all the data into an array for ease of processing.
     * Format of data is
     * characters 0-4 is the letters that will be used in the output. 64 + the value represents the ASCII character
     * ASCII code 65 = A, 66 = B, etc.  so 2+64=66 (B), 21+64=85 (U) and so on.
     * Then we next have pairs of numbers.
     * Looking at the this data
     * 1,2,-1,0,2,45,50,-1
     * That reads as
     * 1,2 = draw characters - in this case BU
     * -1 = go to a new line
     * 0,2 = DRAW BUN
     * 45,50 = DRAW BUNNYB starting at position 45
     * and so on.
     * 4096 is EOF
     *
     * @return ArrayList of type Integer containing the data
     */
    private ArrayList<Integer> loadData() {
        // 创建一个整数类型的 ArrayList 对象
        ArrayList<Integer> theData = new ArrayList<>();

        // 这是从原始基本程序中忠实添加的数据。
        // 注意：
        // 前5个整数是 ASCII 字符（加上64以使它们成为我们可以输出的 ASCII 字符）。
        theData.addAll(Arrays.asList(2, 21, 14, 14, 25));
        theData.addAll(Arrays.asList(1, 2, -1, 0, 2, 45, 50, -1, 0, 5, 43, 52, -1, 0, 7, 41, 52, -1));
        theData.addAll(Arrays.asList(1, 9, 37, 50, -1, 2, 11, 36, 50, -1, 3, 13, 34, 49, -1, 4, 14, 32, 48, -1));
        theData.addAll(Arrays.asList(5, 15, 31, 47, -1, 6, 16, 30, 45, -1, 7, 17, 29, 44, -1, 8, 19, 28, 43, -1));
        theData.addAll(Arrays.asList(9, 20, 27, 41, -1, 10, 21, 26, 40, -1, 11, 22, 25, 38, -1, 12, 22, 24, 36, -1));
        theData.addAll(Arrays.asList(13, 34, -1, 14, 33, -1, 15, 31, -1, 17, 29, -1, 18, 27, -1));
        theData.addAll(Arrays.asList(19, 26, -1, 16, 28, -1, 13, 30, -1, 11, 31, -1, 10, 32, -1));
        theData.addAll(Arrays.asList(8, 33, -1, 7, 34, -1, 6, 13, 16, 34, -1, 5, 12, 16, 35, -1));
        theData.addAll(Arrays.asList(4, 12, 16, 35, -1, 3, 12, 15, 35, -1, 2, 35, -1, 1, 35, -1));
        theData.addAll(Arrays.asList(2, 34, -1, 3, 34, -1, 4, 33, -1, 6, 33, -1, 10, 32, 34, 34, -1));
        theData.addAll(Arrays.asList(14, 17, 19, 25, 28, 31, 35, 35, -1, 15, 19, 23, 30, 36, 36, -1));
        theData.addAll(Arrays.asList(14, 18, 21, 21, 24, 30, 37, 37, -1, 13, 18, 23, 29, 33, 38, -1));
        theData.addAll(Arrays.asList(12, 29, 31, 33, -1, 11, 13, 17, 17, 19, 19, 22, 22, 24, 31, -1));
        theData.addAll(Arrays.asList(10, 11, 17, 18, 22, 22, 24, 24, 29, 29, -1));
        theData.addAll(Arrays.asList(22, 23, 26, 29, -1, 27, 29, -1, 28, 29, -1, 4096));

        // 返回包含数据的 ArrayList 对象
        return theData;
    }

    public static void main(String[] args) {
        // 创建 Bunny 对象
        Bunny bunny = new Bunny();
        // 调用 Bunny 对象的 process 方法
        bunny.process();
    }
# 闭合前面的函数定义
```