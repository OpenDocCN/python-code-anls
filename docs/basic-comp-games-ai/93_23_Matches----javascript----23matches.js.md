# `93_23_Matches\javascript\23matches.js`

```
// 创建一个新的 Promise 对象，用于处理输入操作
// 创建一个 INPUT 元素，用于接收用户输入
// 在页面上打印提示符 "? "
// 设置 INPUT 元素的类型为文本输入
// ... (接下来的代码未提供，需要继续添加注释)
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 为输入元素添加按键按下事件监听器
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
# 结束添加事件监听器的函数
});
}

# 定义一个 tab 函数，接受一个 space 参数
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串
}

// Main control section
async function main()
{
    print(tab(31) + "23 MATCHES\n");  // 在指定位置打印字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 在指定位置打印字符串
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print(" THIS IS A GAME CALLED '23 MATCHES'.\n");  // 打印游戏介绍
    print("\n");  // 打印空行
    print("WHEN IT IS YOUR TURN, YOU MAY TAKE ONE, TWO, OR THREE\n");  // 打印游戏规则
    print("MATCHES. THE OBJECT OF THE GAME IS NOT TO HAVE TO TAKE\n");  // 打印游戏规则
    print("THE LAST MATCH.\n");  // 打印游戏规则
    print("\n");  // 打印空行
    print("LET'S FLIP A COIN TO SEE WHO GOES FIRST.\n");  // 打印游戏开始提示
    print("IF IT COMES UP HEADS, I WILL WIN THE TOSS.\n");  // 打印游戏开始提示
}
    # 打印空行
    print("\n");
    # 初始化变量n为23
    n = 23;
    # 生成一个0或1的随机数
    q = Math.floor(2 * Math.random());
    # 如果随机数不等于1，则执行以下代码块
    if (q != 1) {
        # 打印"Tails! You go first."并换行
        print("TAILS! YOU GO FIRST. \n");
        # 打印空行
        print("\n");
    } else {
        # 打印"Heads! I win! Ha! Ha!"并换行
        print("HEADS! I WIN! HA! HA!\n");
        # 打印"Prepare to lose, meatball-nose!!"并换行
        print("PREPARE TO LOSE, MEATBALL-NOSE!!\n");
        # 打印空行
        print("\n");
        # 打印"I take 2 matches"
        print("I TAKE 2 MATCHES\n");
        # 变量n减去2
        n -= 2;
    }
    # 进入循环，条件为永远为真
    while (1) {
        # 如果q等于1，则执行以下代码块
        if (q == 1) {
            # 打印"The number of matches is now " + n，并换行
            print("THE NUMBER OF MATCHES IS NOW " + n + "\n");
            # 打印空行
            print("\n");
            # 打印"Your turn -- you may take 1, 2 or 3 matches."
            print("YOUR TURN -- YOU MAY TAKE 1, 2 OR 3 MATCHES.\n");
        }
        # 打印"How many do you wish to remove "
        print("HOW MANY DO YOU WISH TO REMOVE ");
        while (1):  # 进入无限循环
            k = parseInt(await input())  # 从输入中获取一个整数并赋值给变量k
            if (k <= 0 || k > 3):  # 如果k小于等于0或者大于3
                print("VERY FUNNY! DUMMY!\n")  # 打印"VERY FUNNY! DUMMY!"
                print("DO YOU WANT TO PLAY OR GOOF AROUND?\n")  # 打印"DO YOU WANT TO PLAY OR GOOF AROUND?"
                print("NOW, HOW MANY MATCHES DO YOU WANT ")  # 打印"NOW, HOW MANY MATCHES DO YOU WANT "
            else:  # 否则
                break  # 跳出循环
        n -= k  # 将n减去k的值
        print("THERE ARE NOW " + n + " MATCHES REMAINING.\n")  # 打印"THERE ARE NOW " + n + " MATCHES REMAINING."
        if (n == 4):  # 如果n等于4
            z = 3  # 将z赋值为3
        elif (n == 3):  # 否则如果n等于3
            z = 2  # 将z赋值为2
        elif (n == 2):  # 否则如果n等于2
            z = 1  # 将z赋值为1
        elif (n > 1):  # 否则如果n大于1
            z = 4 - k  # 将z赋值为4减去k的值
        } else {
            # 如果条件不满足，打印以下内容
            print("YOU WON, FLOPPY EARS !\n");
            print("THINK YOU'RE PRETTY SMART !\n");
            print("LETS PLAY AGAIN AND I'LL BLOW YOUR SHOES OFF !!\n");
            # 跳出循环
            break;
        }
        # 打印以下内容
        print("MY TURN ! I REMOVE " + z + " MATCHES\n");
        # 减去 z 的值
        n -= z;
        # 如果 n 小于等于 1，打印以下内容并跳出循环
        if (n <= 1) {
            print("\n");
            print("YOU POOR BOOB! YOU TOOK THE LAST MATCH! I GOTCHA!!\n");
            print("HA ! HA ! I BEAT YOU !!!\n");
            print("\n");
            print("GOOD BYE LOSER!\n");
            break;
        }
        # 将 q 赋值为 1
        q = 1;
    }

}
# 调用名为main的函数，但是在给定的代码中并没有定义这个函数，所以这行代码会导致错误。
```