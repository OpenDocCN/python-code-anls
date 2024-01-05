# `78_Sine_Wave\java\src\SineWave.java`

```
/**
 * Sine Wave
 *
 * Based on the Sine Wave program here
 * https://github.com/coding-horror/basic-computer-games/blob/main/78%20Sine%20Wave/sinewave.bas
 *
 * Note:  The idea was to create a version of the 1970's Basic program in Java, without introducing
 *        new features - no additional text, error checking, etc has been added.
 */
public class SineWave {

    public static void main(String[] args) {
        System.out.println("""
           SINE WAVE
           CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
           """);
        var isCreative = true;
        for(var t = 0d; t<40; t += .25) {
            // 计算缩进量，根据正弦函数的值确定缩进的空格数
            var indentations = 26 + (int) (25 * Math.sin(t));
            System.out.print(" ".repeat(indentations));  // 在控制台输出一定数量的空格，用于缩进
            //Change output every iteration  // 每次迭代更改输出
            var word = isCreative ? "CREATIVE" : "COMPUTING";  // 根据 isCreative 的值选择输出 "CREATIVE" 或 "COMPUTING"
            System.out.println(word);  // 在控制台输出 word 变量的值
            isCreative = !isCreative ;  // 切换 isCreative 变量的值，用于下一次迭代
        }
    }
}
```