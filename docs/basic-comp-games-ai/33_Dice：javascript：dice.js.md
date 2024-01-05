# `d:/src/tocomm/basic-computer-games\33_Dice\javascript\dice.js`

```
// 定义一个名为print的函数，用于向页面输出字符串
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个名为input的函数，用于获取用户输入
function input()
{
    var input_element;
    var input_str;

    // 返回一个Promise对象，表示异步操作的最终完成或失败
    return new Promise(function (resolve) {
                       // 创建一个input元素
                       input_element = document.createElement("INPUT");

                       // 在页面上输出问号提示
                       print("? ");
                       // 设置input元素的类型为文本
                       input_element.setAttribute("type", "text");
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析并返回输入字符串
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器的添加
});
# 结束函数定义

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串

}

// Main program
async function main()
{
    print(tab(34) + "DICE\n");  // 打印带有34个空格的"DICE"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有15个空格的"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    f = [];  // 初始化空数组f
    // Danny Freidus
    print("THIS PROGRAM SIMULATES THE ROLLING OF A\n");  // 打印"This program simulates the rolling of a"并换行
    print("PAIR OF DICE.\n");  // 打印"PAIR OF DICE."并换行
    print("YOU ENTER THE NUMBER OF TIMES YOU WANT THE COMPUTER TO\n");  // 打印"YOU ENTER THE NUMBER OF TIMES YOU WANT THE COMPUTER TO"并换行
    print("'ROLL' THE DICE.  WATCH OUT, VERY LARGE NUMBERS TAKE\n");  // 打印"'ROLL' THE DICE.  WATCH OUT, VERY LARGE NUMBERS TAKE"并换行
    print("A LONG TIME.  IN PARTICULAR, NUMBERS OVER 5000.\n");  // 打印"A LONG TIME.  IN PARTICULAR, NUMBERS OVER 5000."并换行
    do {
        for (q = 1; q <= 12; q++)
            f[q] = 0;  // 初始化数组 f，将每个元素的值都设为 0

        print("\n");  // 打印换行符
        print("HOW MANY ROLLS");  // 打印提示信息，询问用户要投掷多少次骰子
        x = parseInt(await input());  // 从用户输入中获取投掷次数，并转换为整数赋值给变量 x

        for (s = 1; s <= x; s++) {  // 循环投掷骰子 x 次
            a = Math.floor(Math.random() * 6 + 1);  // 生成随机数 a，模拟掷骰子得到的点数
            b = Math.floor(Math.random() * 6 + 1);  // 生成随机数 b，模拟掷骰子得到的点数
            r = a + b;  // 计算两次投掷的点数之和
            f[r]++;  // 将点数之和对应的数组 f 的元素加一
        }

        print("\n");  // 打印换行符
        print("TOTAL SPOTS\tNUMBER OF TIMES\n");  // 打印表头

        for (v = 2; v <= 12; v++) {  // 遍历点数之和的可能取值
            print("\t" + v + "\t" + f[v] + "\n");  // 打印点数之和和对应的次数
        }

        print("\n");  // 打印换行符
        print("\n");  // 再次打印换行符
        print("TRY AGAIN");  // 提示用户再次尝试
        str = await input();  // 获取用户输入并赋值给变量 str
    } while (str.substr(0, 1) == "Y") ;
```
这是一个 do-while 循环的结束标志，表示当输入的字符串以 "Y" 开头时继续循环。

```python
main();
```
这是调用名为 main 的函数，用于执行程序的主要逻辑。
```