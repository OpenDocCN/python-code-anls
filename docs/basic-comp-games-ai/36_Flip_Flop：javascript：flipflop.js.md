# `36_Flip_Flop\javascript\flipflop.js`

```
# FLIPFLOP
# 
# 由Oscar Toledo G. (nanochess)将BASIC转换为Javascript
#

# 打印函数，将字符串添加到输出元素中
def print(str):
    document.getElementById("output").appendChild(document.createTextNode(str))

# 输入函数，返回一个Promise对象
def input():
    var input_element
    var input_str

    return new Promise(function (resolve):
                       # 创建一个输入元素
                       input_element = document.createElement("INPUT")

                       # 打印提示符
                       print("? ")
                       # 设置输入元素类型为文本
                       input_element.setAttribute("type", "text")
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
    # 如果按下的键是回车键
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
    # 当 space 大于 0 时执行循环
    while (space-- > 0)
```
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串

}

var as = [];  // 创建一个空数组

// Main program
async function main()
{
    print(tab(32) + "FLIPFLOP\n");  // 打印带有制表符的字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有制表符的字符串
    print("\n");  // 打印空行
    // *** Created by Michael Cass
    print("THE OBJECT OF THIS PUZZLE IS TO CHANGE THIS:\n");  // 打印提示信息
    print("\n");  // 打印空行
    print("X X X X X X X X X X\n");  // 打印一行X
    print("\n");  // 打印空行
    print("TO THIS:\n");  // 打印提示信息
    print("\n");  // 打印空行
    print("O O O O O O O O O O\n");  // 打印一行O
    # 打印空行
    print("\n")
    # 打印提示信息
    print("BY TYPING THE NUMBER CORRESPONDING TO THE POSITION OF THE\n")
    print("LETTER ON SOME NUMBERS, ONE POSITION WILL CHANGE, ON\n")
    print("OTHERS, TWO WILL CHANGE.  TO RESET LINE TO ALL X'S, TYPE 0\n")
    print("(ZERO) AND TO START OVER IN THE MIDDLE OF A GAME, TYPE \n")
    print("11 (ELEVEN).\n")
    print("\n")
    # 进入无限循环
    while (1):
        # 初始化变量start为1
        start = 1
        # 进入do-while循环
        do:
            # 初始化变量z为1
            z = 1
            # 如果start为1
            if (start == 1):
                # 初始化变量m为0
                m = 0
                # 生成一个随机数赋值给变量q
                q = Math.random()
                # 打印提示信息
                print("HERE IS THE STARTING LINE OF X'S.\n")
                print("\n")
                # 初始化变量c为0
                c = 0
                # 将start赋值为2
                start = 2
            # 如果start为2
            if (start == 2):
# 打印数字1到10和字母X，用于显示游戏界面
print("1 2 3 4 5 6 7 8 9 10\n");
print("X X X X X X X X X X\n");
print("\n");
# 初始化数组as，将每个元素都赋值为"X"
for (x = 1; x <= 10; x++)
    as[x] = "X";
# 初始化start变量为0
start = 0;
# 打印提示信息，要求输入一个数字
print("INPUT THE NUMBER");
# 进入循环，等待用户输入
while (1) {
    # 将用户输入的内容转换为整数赋值给变量n
    n = parseInt(await input());
    # 如果输入的数字在0到11之间，则跳出循环
    if (n >= 0 && n <= 11)
        break;
    # 如果输入的数字不在0到11之间，则打印提示信息要求重新输入
    print("ILLEGAL ENTRY--TRY AGAIN.\n");
}
# 如果输入的数字为11，则将start变量赋值为1，并继续循环
if (n == 11) {
    start = 1;
    continue;
}
# 如果输入的数字为0，则将start变量赋值为2
if (n == 0) {
    start = 2;
                continue;  # 继续下一次循环，跳过本次循环后面的代码
            }
            if (m != n) {  # 如果 m 不等于 n
                m = n;  # 将 m 的值设为 n
                as[n] = (as[n] == "O" ? "X" : "O");  # 如果 as[n] 的值为 "O"，则将其设为 "X"，否则设为 "O"
                do {  # 执行循环
                    r = Math.tan(q + n / q - n) - Math.sin(q / n) + 336 * Math.sin(8 * n);  # 计算 r 的值
                    n = r - Math.floor(r);  # 将 n 的值设为 r 减去 r 的向下取整
                    n = Math.floor(10 * n);  # 将 n 的值设为 10 倍的 n 的向下取整
                    as[n] = (as[n] == "O" ? "X" : "O");  # 如果 as[n] 的值为 "O"，则将其设为 "X"，否则设为 "O"
                } while (m == n) ;  # 当 m 等于 n 时继续循环
            } else {  # 如果 m 等于 n
                as[n] = (as[n] == "O" ? "X" : "O");  # 如果 as[n] 的值为 "O"，则将其设为 "X"，否则设为 "O"
                do {  # 执行循环
                    r = 0.592 * (1 / Math.tan(q / n + q)) / Math.sin(n * 2 + q) - Math.cos(n);  # 计算 r 的值
                    n = r - Math.floor(r);  # 将 n 的值设为 r 减去 r 的向下取整
                    n = Math.floor(10 * n);  # 将 n 的值设为 10 倍的 n 的向下取整
                    as[n] = (as[n] == "O" ? "X" : "O");  # 如果 as[n] 的值为 "O"，则将其设为 "X"，否则设为 "O"
                } while (m == n) ;  # 当 m 等于 n 时继续循环
            }
# 打印数字1到10
print("1 2 3 4 5 6 7 8 9 10\n");
# 遍历数组as的元素并打印出来
for (z = 1; z <= 10; z++)
    print(as[z] + " ");
# 增加计数器c的值
c++;
# 打印换行符
print("\n");
# 遍历数组as的元素，如果不等于"O"则跳出循环
for (z = 1; z <= 10; z++) {
    if (as[z] != "O")
        break;
}
# 当z小于等于10时执行循环
} while (z <= 10) ;
# 如果猜测次数小于等于12，则打印"VERY GOOD.  YOU GUESSED IT IN ONLY " + c + " GUESSES.\n"，否则打印"TRY HARDER NEXT TIME.  IT TOOK YOU " + c + " GUESSES.\n"
if (c <= 12) {
    print("VERY GOOD.  YOU GUESSED IT IN ONLY " + c + " GUESSES.\n");
} else {
    print("TRY HARDER NEXT TIME.  IT TOOK YOU " + c + " GUESSES.\n");
}
# 打印"Do you want to try another puzzle"
print("DO YOU WANT TO TRY ANOTHER PUZZLE");
# 等待输入
str = await input();
# 如果输入的第一个字符是"N"，则跳出循环
if (str.substr(0, 1) == "N")
    break;
    print("\n");  # 打印一个换行符

}

main();  # 调用名为main的函数
```