# `44_Hangman\javascript\hangman.js`

```
// HANGMAN
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 定义一个打印函数，用于将字符串输出到页面上
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，用于获取用户输入的字符串
function input()
{
    var input_element;
    var input_str;

    // 返回一个 Promise 对象，用于处理异步操作
    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 在页面上打印提示符
                       print("? ");
                       // 设置输入框类型为文本
                       input_element.setAttribute("type", "text");
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 为输入元素添加键盘按下事件监听器
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键（keyCode 为 13）
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析输入字符串
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器的定义
});
# 结束函数定义

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
```
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回修改后的字符串

print(tab(32) + "HANGMAN\n");  # 打印带有32个空格的字符串和"HANGMAN"，并换行
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印带有15个空格的字符串和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并换行
print("\n");  # 打印一个空行
print("\n");  # 打印一个空行
print("\n");  # 打印一个空行

var pa = [];  # 创建一个空数组pa
var la = [];  # 创建一个空数组la
var da = [];  # 创建一个空数组da
var na = [];  # 创建一个空数组na
var ua = [];  # 创建一个空数组ua

var words = ["GUM","SIN","FOR","CRY","LUG","BYE","FLY",  # 创建一个包含多个单词的数组
             "UGLY","EACH","FROM","WORK","TALK","WITH","SELF",
             "PIZZA","THING","FEIGN","FIEND","ELBOW","FAULT","DIRTY",
             "BUDGET","SPIRIT","QUAINT","MAIDEN","ESCORT","PICKAX",
// 创建一个包含单词的数组
var words = ["EXAMPLE","TENSION","QUININE","KIDNEY","REPLICA","SLEEPER",
             "TRIANGLE","KANGAROO","MAHOGANY","SERGEANT","SEQUENCE",
             "MOUSTACHE","DANGEROUS","SCIENTIST","DIFFERENT","QUIESCENT",
             "MAGISTRATE","ERRONEOUSLY","LOUDSPEAKER","PHYTOTOXIC",
             "MATRIMONIAL","PARASYMPATHOMIMETIC","THIGMOTROPISM"];

// 主控制部分
async function main()
{
    // 初始化变量
    c = 1; // 设置 c 的初始值为 1
    n = 50; // 设置 n 的初始值为 50
    while (1) { // 进入无限循环
        for (i = 1; i <= 20; i++) // 循环设置数组 da 的前 20 个元素为 "-"
            da[i] = "-";
        for (i = 1; i <= n; i++) // 循环设置数组 ua 的前 n 个元素为 0
            ua[i] = 0;
        m = 0; // 设置 m 的初始值为 0
        ns = ""; // 设置字符串 ns 的初始值为空
        for (i = 1; i <= 12; i++) { // 循环初始化数组 pa 的前 12 个元素为一个空数组
            pa[i] = [];
            for (j = 1; j <= 12; j++) {  // 循环遍历每一列，从1到12
                pa[i][j] = " ";  // 将二维数组pa的第i行第j列的元素赋值为空格
            }
        }
        for (i = 1; i <= 12; i++) {  // 循环遍历每一行，从1到12
            pa[i][1] = "X";  // 将二维数组pa的第i行第1列的元素赋值为X
        }
        for (i = 1; i <= 7; i++) {  // 循环遍历前7行，从1到7
            pa[1][i] = "X";  // 将二维数组pa的第1行第i列的元素赋值为X
        }
        pa[2][7] = "X";  // 将二维数组pa的第2行第7列的元素赋值为X
        if (c >= n) {  // 如果c大于等于n
            print("YOU DID ALL THE WORDS!!\n");  // 打印提示信息
            break;  // 跳出循环
        }
        do {
            q = Math.floor(n * Math.random()) + 1;  // 生成一个随机数q，范围在1到n之间
        } while (ua[q] == 1) ;  // 当ua[q]等于1时，继续循环
        ua[q] = 1;  // 将ua[q]赋值为1
        c++;  // c自增1
        t1 = 0;  // 初始化变量 t1 为 0
        as = words[q - 1];  // 从数组 words 中取出第 q-1 个元素赋值给变量 as
        l = as.length;  // 获取变量 as 的长度并赋值给变量 l
        for (i = 1; i <= as.length; i++)  // 循环遍历变量 as 的每个字符
            la[i] = as[i - 1];  // 将变量 as 的每个字符赋值给数组 la
        while (1) {  // 进入无限循环
            while (1) {  // 进入内层无限循环
                print("HERE ARE THE LETTERS YOU USED:\n");  // 打印提示信息
                print(ns + "\n");  // 打印变量 ns 的值
                print("\n");  // 打印空行
                for (i = 1; i <= l; i++) {  // 循环遍历变量 l
                    print(da[i]);  // 打印数组 da 的第 i 个元素
                }
                print("\n");  // 打印空行
                print("\n");  // 打印空行
                print("WHAT IS YOUR GUESS");  // 打印提示信息
                str = await input();  // 等待用户输入并赋值给变量 str
                if (ns.indexOf(str) != -1) {  // 判断变量 ns 是否包含变量 str
                    print("YOU GUESSED THAT LETTER BEFORE!\n");  // 如果包含则打印提示信息
                } else {
# 结束当前循环，跳出循环体
                    break;
                }
            }
            # 将字符串 str 添加到字符串 ns 的末尾
            ns += str;
            # t1 自增1
            t1++;
            # 初始化变量 r 为 0
            r = 0;
            # 遍历列表 la 中的元素，如果元素等于 str，则将列表 da 中对应位置的元素设为 str，并将 r 自增1
            for (i = 1; i <= l; i++) {
                if (la[i] == str) {
                    da[i] = str;
                    r++;
                }
            }
            # 如果 r 为 0，则执行以下操作
            if (r == 0) {
                # m 自增1
                m++;
                # 打印两个换行符
                print("\n");
                print("\n");
                # 打印提示信息
                print("SORRY, THAT LETTER ISN'T IN THE WORD.\n");
                # 根据 m 的值执行不同的操作
                switch (m) {
                    # 如果 m 为 1，则打印提示信息
                    case 1:
                        print("FIRST, WE DRAW A HEAD\n");
                    break;  # 结束当前的 switch 语句
                case 2:  # 如果变量的值为 2
                    print("NOW WE DRAW A BODY.\n");  # 打印“现在我们画一个身体。”
                    break;  # 结束当前的 switch 语句
                case 3:  # 如果变量的值为 3
                    print("NEXT WE DRAW AN ARM.\n");  # 打印“接下来我们画一只手臂。”
                    break;  # 结束当前的 switch 语句
                case 4:  # 如果变量的值为 4
                    print("THIS TIME IT'S THE OTHER ARM.\n");  # 打印“这次是另一只手臂。”
                    break;  # 结束当前的 switch 语句
                case 5:  # 如果变量的值为 5
                    print("NOW, LET'S DRAW THE RIGHT LEG.\n");  # 打印“现在，让我们画右腿。”
                    break;  # 结束当前的 switch 语句
                case 6:  # 如果变量的值为 6
                    print("THIS TIME WE DRAW THE LEFT LEG.\n");  # 打印“这次我们画左腿。”
                    break;  # 结束当前的 switch 语句
                case 7:  # 如果变量的值为 7
                    print("NOW WE PUT UP A HAND.\n");  # 打印“现在我们举起一只手。”
                    break;  # 结束当前的 switch 语句
                case 8:  # 如果变量的值为 8
                        # 打印“接下来是另一只手。”
                        print("NEXT THE OTHER HAND.\n");
                        # 跳出循环
                        break;
                    # 如果m的值为9
                    case 9:
                        # 打印“现在我们画一只脚。”
                        print("NOW WE DRAW ONE FOOT.\n");
                        # 跳出循环
                        break;
                    # 如果m的值为10
                    case 10:
                        # 打印“这是另一只脚 - 你被绞死了！！”
                        print("HERE'S THE OTHER FOOT -- YOU'RE HUNG!!\n");
                        # 跳出循环
                        break;
                }
                # 根据m的值进行不同的操作
                switch (m) {
                    # 如果m的值为1
                    case 1:
                        # 修改数组pa的特定位置的值
                        pa[3][6] = "-";
                        pa[3][7] = "-";
                        pa[3][8] = "-";
                        pa[4][5] = "(";
                        pa[4][6] = ".";
                        pa[4][8] = ".";
                        pa[4][9] = ")";
                        pa[5][6] = "-";
                        pa[5][7] = "-";
# 设置二维数组 pa 中的特定位置的值为 "-"
pa[5][8] = "-";
# 结束当前的 case 分支
break;
# 当 case 为 2 时，设置二维数组 pa 中的特定位置的值为 "X"
for (i = 6; i <= 9; i++)
    pa[i][7] = "X";
# 结束当前的 case 分支
break;
# 当 case 为 3 时，设置二维数组 pa 中的特定位置的值为 "\"
for (i = 4; i <= 7; i++)
    pa[i][i - 1] = "\\";
# 结束当前的 case 分支
break;
# 当 case 为 4 时，设置二维数组 pa 中的特定位置的值为 "/"
pa[4][11] = "/";
pa[5][10] = "/";
pa[6][9] = "/";
pa[7][8] = "/";
# 结束当前的 case 分支
break;
# 当 case 为 5 时，设置二维数组 pa 中的特定位置的值为 "/"
pa[10][6] = "/";
pa[11][5] = "/";
# 在第6个case中，将pa数组中第10行第8列和第11行第9列的元素赋值为"\"
# 在第7个case中，将pa数组中第3行第11列的元素赋值为"\"
# 在第8个case中，将pa数组中第3行第3列的元素赋值为"/"
# 在第9个case中，将pa数组中第12行第10列和第12行第11列的元素分别赋值为"\", "-"
# 在第10个case中，将pa数组中第12行第3列和第12行第4列的元素分别赋值为"-", "/"
# 循环遍历i从1到12
# 初始化一个空字符串
str = "";
# 遍历数组 pa 的第 i 个元素的第 1 到 12 个字符，将其拼接到字符串 str 上
for (j = 1; j <= 12; j++)
    str += pa[i][j];
# 打印字符串 str 和一个换行符
print(str + "\n");
# 打印两个换行符
print("\n");
print("\n");
# 如果 m 等于 10，则打印一条消息并跳出循环
if (m == 10) {
    print("SORRY, YOU LOSE.  THE WORD WAS " + as + "\n");
    print("YOU MISSED THAT ONE.  DO YOU ");
    break;
} else {
    # 遍历数组 da，如果找到了一个值为 "-" 的元素，则跳出循环
    for (i = 1; i <= l; i++)
        if (da[i] == "-")
            break;
    # 如果 i 大于 l，则打印一条消息并跳出循环
    if (i > l) {
        print("YOU FOUND THE WORD!\n");
        break;
    }
# 打印换行
print("\n");
# 遍历数组 da 中的元素并打印
for (i = 1; i <= l; i++)
    print(da[i]);
# 打印两个换行
print("\n");
print("\n");
# 打印提示信息
print("WHAT IS YOUR GUESS FOR THE WORD");
# 等待用户输入并将输入值赋给变量 bs
bs = await input();
# 如果用户猜对了单词
if (as == bs) {
    # 打印提示信息和猜测次数
    print("RIGHT!!  IT TOOK YOU " + t1 + " GUESSES!\n");
    # 跳出循环
    break;
}
# 如果用户猜错了单词
print("WRONG.  TRY ANOTHER LETTER.\n");
# 打印两个换行
print("\n");
print("\n");
# 打印提示信息
print("WANT ANOTHER WORD");
# 等待用户输入并将输入值赋给变量 str
str = await input();
# 如果用户输入的不是 "YES"
if (str != "YES")
    # 跳出循环
    break;
    print("\n");  # 打印一个空行
    print("IT'S BEEN FUN!  BYE FOR NOW.\n");  # 打印一条结束语
    // Lines 620 and 990 unused in original  # 注释掉原始代码中未使用的代码行
}

main();  # 调用主函数
```