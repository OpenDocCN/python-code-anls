# `d:/src/tocomm/basic-computer-games\15_Boxing\javascript\boxing.js`

```
// 创建一个名为print的函数，用于将字符串输出到指定的HTML元素中
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 创建一个名为input的函数，用于获取用户输入
function input() {
    var input_element;
    var input_str;

    // 返回一个Promise对象，表示异步操作的最终完成或失败
    return new Promise(function (resolve) {
        // 创建一个input元素
        input_element = document.createElement("INPUT");

        // 在HTML元素中输出提示符
        print("? ");

        // 设置input元素的类型为文本
        input_element.setAttribute("type", "text");
```
在这个示例中，我们为JavaScript代码添加了注释，解释了每个函数和操作的作用。这样可以帮助其他程序员更容易地理解和使用这些代码。
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
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串
}

// Main program
async function main()
{
    print(tab(33) + "BOXING\n");  // 打印标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印创意计算的信息
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("BOXING OLYMPIC STYLE (3 ROUNDS -- 2 OUT OF 3 WINS)\n");  // 打印拳击比赛的信息
    j = 0;  // 初始化变量 j
    l = 0;  // 初始化变量 l
    print("\n");  // 打印空行
    print("WHAT IS YOUR OPPONENT'S NAME");  // 打印提示信息，要求输入对手的名字
    js = await input();  // 从用户输入中获取对手的名字
    print("INPUT YOUR MAN'S NAME");  // 打印提示信息，要求输入自己的名字
    ls = await input();  // 从用户输入中获取自己的名字
}
    # 打印不同的拳击方式
    print("DIFFERENT PUNCHES ARE: (1) FULL SWING; (2) HOOK; (3) UPPERCUT; (4) JAB.\n");
    # 打印提示信息，询问用户的最佳拳击方式
    print("WHAT IS YOUR MANS BEST");
    # 获取用户输入的最佳拳击方式并转换为整数
    b = parseInt(await input());
    # 打印提示信息，询问用户对手的弱点
    print("WHAT IS HIS VULNERABILITY");
    # 获取用户输入的对手的弱点并转换为整数
    d = parseInt(await input());
    # 使用随机数生成对手的拳击方式和弱点，直到两者不相等
    do {
        b1 = Math.floor(4 * Math.random() + 1);
        d1 = Math.floor(4 * Math.random() + 1);
    } while (b1 == d1) ;
    # 打印对手的优势和弱点
    print(js + "'S ADVANTAGE IS " + b1 + " AND VULNERABILITY IS SECRET.\n");
    # 初始化被击倒的次数
    knocked = 0;
    # 进行三轮拳击比赛
    for (r = 1; r <= 3; r++) {
        # 如果j大于等于2，则跳出循环
        if (j >= 2)
            break;
        # 如果l大于等于2，则跳出循环
        if (l >= 2)
            break;
        # 初始化x和y的值为0
        x = 0;
        y = 0;
        # 打印当前是第几轮比赛
        print("ROUND " + r + " BEGIN...\n");
        for (r1 = 1; r1 <= 7; r1++) {  # 循环7次
            i = Math.floor(10 * Math.random() + 1);  # 生成1到10之间的随机整数
            if (i <= 5) {  # 如果随机数小于等于5
                print(ls + "'S PUNCH");  # 打印字符串
                p = parseInt(await input());  # 将输入的字符串转换为整数
                if (p == b)  # 如果输入的整数等于b
                    x += 2;  # x增加2
                if (p == 1) {  # 如果输入的整数等于1
                    print(ls + " SWINGS AND ");  # 打印字符串
                    x3 = Math.floor(30 * Math.random() + 1);  # 生成1到30之间的随机整数
                    if (d1 == 4 || x3 < 10) {  # 如果d1等于4或者x3小于10
                        print("HE CONNECTS!\n");  # 打印字符串
                        if (x > 35) {  # 如果x大于35
                            r = 3;  # r赋值为3
                            break;  # 跳出循环
                        }
                        x += 15;  # x增加15
                    } else {
                        print("HE MISSES \n");  # 打印字符串
                        if (x != 1)  # 如果x不等于1
# 打印两个换行符
print("\n\n");
# 如果 p 等于 1
if (p == 1) {
    # 打印 ls 和 " THROWS A JAB... "
    print(ls + " THROWS A JAB... ");
    # 生成一个 1 或 2 的随机整数并赋值给 d1
    d1 = Math.floor(2 * Math.random() + 1);
    # 如果 d1 等于 1
    if (d1 == 1) {
        # 打印 "AND HE CONNECTS!\n"
        print("AND HE CONNECTS!\n");
        # x 增加 3
        x += 3;
    } else {
        # 打印 "BUT HE MISSES!\n"
        print("BUT HE MISSES!\n");
    }
# 如果 p 等于 2
} else if (p == 2) {
    # 打印 ls 和 " GIVES THE HOOK... "
    print(ls + " GIVES THE HOOK... ");
    # 生成一个 1 或 2 的随机整数并赋值给 h1
    h1 = Math.floor(2 * Math.random() + 1);
    # 如果 d1 等于 2
    if (d1 == 2) {
        # x 增加 7
        x += 7;
    # 如果 h1 不等于 1
    } else if (h1 != 1) {
        # 打印 "CONNECTS...\n"
        print("CONNECTS...\n");
        # x 增加 7
        x += 7;
    } else {
        # 打印 "BUT IT'S BLOCKED!!!!!!!!!!!!!\n"
        print("BUT IT'S BLOCKED!!!!!!!!!!!!!\n");
    }
# 如果 p 等于 3
} else if (p == 3) {
    # 打印 ls 和 " TRIES AN UPPERCUT "
    print(ls + " TRIES AN UPPERCUT ");
    # 生成一个 1 到 100 的随机整数并赋值给 d5
    d5 = Math.floor(100 * Math.random() + 1);
    # 如果 d1 等于 3 或者 d5 小于 51
    if (d1 == 3 || d5 < 51) {
        # 打印 "AND HE CONNECTS!\n"
        print("AND HE CONNECTS!\n");
        # x 增加 4
        x += 4;
    } else {
                    print("AND IT'S BLOCKED (LUCKY BLOCK!)\n");
```
这行代码是一个打印语句，用于输出特定的文本信息。

```python
                }
            } else {
```
这是一个else语句，用于在前一个if语句条件不满足时执行其中的代码块。

```python
                j7 = Math.random(4 * Math.random() + 1);
```
这行代码是一个数学运算，使用Math.random()函数生成一个0到1之间的随机数，并进行数学运算。

```python
                if (j7 == b1)
```
这是一个if语句，用于检查j7是否等于b1。

```python
                    y += 2;
```
这行代码是一个数学运算，将y的值增加2。

```python
                if (j7 == 1) {
```
这是一个if语句，用于检查j7是否等于1。

```python
                    print(js + " TAKES A FULL SWING AND");
```
这行代码是一个打印语句，用于输出特定的文本信息。

```python
                    r6 = Math.floor(60 * Math.random() + 1);
```
这行代码是一个数学运算，使用Math.random()函数生成一个0到1之间的随机数，并进行数学运算。

```python
                    if (d == 1 || r6 < 30) {
```
这是一个if语句，用于检查d是否等于1或r6是否小于30。

```python
                        print(" POW!!!!! HE HITS HIM RIGHT IN THE FACE!\n");
```
这行代码是一个打印语句，用于输出特定的文本信息。
# 如果 y 大于 35，则将 knocked 设为 1，r 设为 3，并跳出循环
if (y > 35) {
    knocked = 1;
    r = 3;
    break;
}
# 将 y 增加 15
y += 15;
# 如果 j7 等于 2 或者 3
} else {
    # 打印 " IT'S BLOCKED!\n"
    print(" IT'S BLOCKED!\n");
}
# 如果 j7 等于 2 或者 3
} else if (j7 == 2 || j7 == 3) {
    # 如果 j7 等于 2
    if (j7 == 2) {
        # 打印 js + " GETS " + ls + " IN THE JAW (OUCH!)\n"
        print(js + " GETS " + ls + " IN THE JAW (OUCH!)\n");
        # 将 y 增加 7
        y += 7;
        # 打印 "....AND AGAIN!\n"
        print("....AND AGAIN!\n");
        # 将 y 增加 5
        y += 5;
        # 如果 y 大于 35，则将 knocked 设为 1，r 设为 3，并跳出循环
        if (y > 35) {
            knocked = 1;
            r = 3;
            break;
        }
                        print("\n");  # 打印空行
                        // From original, it goes over from handling 2 to handling 3  # 注释，说明代码从原始状态转移到处理2到处理3
                    }
                    print(ls + " IS ATTACKED BY AN UPPERCUT (OH,OH)...\n");  # 打印被上勾拳攻击的信息
                    q4 = Math.floor(200 * Math.random() + 1);  # 生成一个1到200之间的随机数
                    if (d == 3 || q4 <= 75):  # 如果d等于3或者q4小于等于75
                        print("AND " + js + " CONNECTS...\n");  # 打印连接信息
                        y += 8  # y增加8
                    else:
                        print(" BLOCKS AND HITS " + js + " WITH A HOOK.\n");  # 打印阻挡并用钩拳击中的信息
                        x += 5  # x增加5
                else:
                    print(js + " JABS AND ");  # 打印快速出拳的信息
                    z4 = Math.floor(7 * Math.random() + 1);  # 生成一个1到7之间的随机数
                    if (d == 4):  # 如果d等于4
                        y += 5  # y增加5
                    elif (z4 > 4):  # 否则如果z4大于4
                        print(" BLOOD SPILLS !!!\n");  # 打印血溅的信息
                        y += 5  # y增加5
                    } else {  # 如果条件不成立
                        print("IT'S BLOCKED!\n");  # 打印出"IT'S BLOCKED!"
                    }
                }
            }
        }
        if (x > y) {  # 如果x大于y
            print("\n");  # 打印一个空行
            print(ls + " WINS ROUND " + r + "\n");  # 打印出ls和r的值以及"WINS ROUND"的提示
            l++;  # l加1
        } else {  # 如果条件不成立
            print("\n");  # 打印一个空行
            print(js + " WINS ROUND " + r + "\n");  # 打印出js和r的值以及"WINS ROUND"的提示
            j++;  # j加1
        }
    }
    if (j >= 2) {  # 如果j大于等于2
        print(js + " WINS (NICE GOING, " + js + ").\n");  # 打印出js和"(NICE GOING, "以及js的值
    } else if (l >= 2) {  # 如果j不大于等于2，但l大于等于2
        print(ls + " AMAZINGLY WINS!!\n");  # 打印出ls和"AMAZINGLY WINS!!"
    } else if (knocked) {  # 如果被击倒了
        print(ls + " IS KNOCKED COLD AND " + js + " IS THE WINNER AND CHAMP!\n");  # 打印被击倒的人和获胜者
    } else {  # 否则
        print(js + " IS KNOCKED COLD AND " + ls + " IS THE WINNER AND CHAMP!\n");  # 打印被击倒的人和获胜者
    }
    print("\n");  # 打印空行
    print("\n");  # 打印空行
    print("AND NOW GOODBYE FROM THE OLYMPIC ARENA.\n");  # 打印告别语
    print("\n");  # 打印空行
}

main();  # 调用主函数
```