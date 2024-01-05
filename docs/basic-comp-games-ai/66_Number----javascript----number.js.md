# `66_Number\javascript\number.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
// 创建一个 INPUT 元素，用于用户输入
// 在页面上打印问号，提示用户输入
// 设置 INPUT 元素的类型为文本输入
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时，获取输入字符串，移除输入元素，打印输入字符串并解析输入字符串
input_element.addEventListener("keydown", function (event) {
    if (event.keyCode == 13) {
        input_str = input_element.value;
        document.getElementById("output").removeChild(input_element);
        print(input_str);
        print("\n");
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器
});
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环减少 space 并在 str 后面添加一个空格
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串

}

// Main program
async function main()
{
    print(tab(33) + "NUMBER\n");  // 在第33列打印"NUMBER"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 在第15列打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("YOU HAVE 100 POINTS.  BY GUESSING NUMBERS FROM 1 TO 5, YOU\n");  // 打印提示信息
    print("CAN GAIN OR LOSE POINTS DEPENDING UPON HOW CLOSE YOU GET TO\n");  // 打印提示信息
    print("A RANDOM NUMBER SELECTED BY THE COMPUTER.\n");  // 打印提示信息
    print("\n");  // 打印空行
    print("YOU OCCASIONALLY WILL GET A JACKPOT WHICH WILL DOUBLE(!)\n");  // 打印提示信息
    print("YOUR POINT COUNT.  YOU WIN WHEN YOU GET 500 POINTS.\n");  // 打印提示信息
    print("\n");  // 打印空行
    p = 0;  // 初始化变量p为0
}
    while (1):  # 进入无限循环
        do {
            print("GUESS A NUMBER FROM 1 TO 5");  # 打印提示信息
            g = parseInt(await input());  # 从输入中获取一个整数并赋值给变量 g
        } while (g < 1 || g > 5) ;  # 如果 g 不在 1 到 5 的范围内，则继续循环
        r = Math.floor(5 * Math.random() + 1);  # 生成一个 1 到 5 之间的随机整数并赋值给变量 r
        s = Math.floor(5 * Math.random() + 1);  # 生成一个 1 到 5 之间的随机整数并赋值给变量 s
        t = Math.floor(5 * Math.random() + 1);  # 生成一个 1 到 5 之间的随机整数并赋值给变量 t
        u = Math.floor(5 * Math.random() + 1);  # 生成一个 1 到 5 之间的随机整数并赋值给变量 u
        v = Math.floor(5 * Math.random() + 1);  # 生成一个 1 到 5 之间的随机整数并赋值给变量 v
        if (g == r):  # 如果用户猜中了 r
            p -= 5;  # 从 p 中减去 5
        elif (g == s):  # 如果用户猜中了 s
            p += 5;  # 在 p 中加上 5
        elif (g == t):  # 如果用户猜中了 t
            p += p;  # p 的值翻倍
            print("YOU HIT THE JACKPOT!!!\n");  # 打印中奖信息
        elif (g == u):  # 如果用户猜中了 u
            p += 1;  # 在 p 中加上 1
        elif (g == v):  # 如果用户猜中了 v
            p -= p * 0.5;  # 减去当前分数的一半
        }
        if (p <= 500) {  # 如果分数小于等于500
            print("YOU HAVE " + p + " POINTS.\n");  # 打印当前分数
            print("\n");  # 打印空行
        } else {  # 如果分数大于500
            print("!!!!YOU WIN!!!! WITH " + p + " POINTS.\n");  # 打印获胜信息和当前分数
            break;  # 跳出循环
        }
    }
}

main();  # 调用主函数
```