# `09_Battle\java\Input.java`

```
import java.io.BufferedReader;  // 导入用于读取输入的 BufferedReader 类
import java.io.InputStreamReader;  // 导入用于读取输入的 InputStreamReader 类
import java.io.IOException;  // 导入用于处理输入输出异常的 IOException 类
import java.text.NumberFormat;  // 导入用于解析数字格式的 NumberFormat 类

// This class handles reading input from the player
// Each input is an x and y coordinate
// e.g. 5,3
public class Input {
    private BufferedReader reader;  // 创建一个用于读取输入的 BufferedReader 对象
    private NumberFormat parser;  // 创建一个用于解析数字格式的 NumberFormat 对象
    private int scale;             // size of the sea, needed to validate input
    private boolean isQuit;        // whether the input has ended
    private int[] coords;          // the last coordinates read

    public Input(int seaSize) {  // 创建一个构造函数，参数为海域大小
        scale = seaSize;  // 将海域大小赋值给 scale 变量
        reader = new BufferedReader(new InputStreamReader(System.in));  // 初始化 BufferedReader 对象，用于读取控制台输入
        parser = NumberFormat.getIntegerInstance();  // 初始化 NumberFormat 对象，用于解析整数格式
    }
    public boolean readCoordinates() throws IOException {
        while (true) {
            // 写入提示
            System.out.print("\n目标 x,y\n> ");
            // 读取用户输入的坐标
            String inputLine = reader.readLine();
            if (inputLine == null) {
                // 如果输入流结束，无法继续游戏
                System.out.println("\n游戏退出\n");
                isQuit = true;
                return false;
            }

            // 将输入分割成两个字段
            String[] fields = inputLine.split(",");
            if (fields.length != 2) {
                // 必须恰好是两个坐标
                System.out.println("需要用','分隔的两个坐标");
                continue;
            }
            // 创建一个长度为2的整型数组来存储坐标
            coords = new int[2];
            // 初始化错误标志为false
            boolean error = false;
            // 每个字段应该包含一个从1到海域大小的整数
            try {
                // 遍历每个字段
                for (int c = 0 ; c < 2; ++c ) {
                    // 将字段转换为整数并去除空格
                    int val = Integer.parseInt(fields[c].strip());
                    // 如果值不在1到海域大小的范围内
                    if ((val < 1) || (val > scale)) {
                        // 输出错误信息
                        System.out.println("Coordinates must be from 1 to " + scale);
                        // 设置错误标志为true
                        error = true;
                    } else {
                        // 将坐标存入数组
                        coords[c] = val;
                    }
                }
            }
            // 捕获字段不是有效数字的异常
            catch (NumberFormatException ne) {
                // 输出错误信息
                System.out.println("Coordinates must be numbers");
                // 设置错误标志为true
                error = true;
            }
            if (!error) return true;  # 如果没有错误发生，则返回 true
        }  # 结束 while 循环
    }  # 结束 try 块
    public int x() { return coords[0]; }  # 定义一个返回 x 坐标的方法
    public int y() { return coords[1]; }  # 定义一个返回 y 坐标的方法
}  # 结束类定义
```