# `47_Hi-Lo\javascript\hi-lo.js`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
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
        str += " "; // 将一个空格添加到字符串末尾
    return str; // 返回修改后的字符串
}

// Main program
async function main()
{
    print(tab(34) + "HI LO\n"); // 打印带有34个空格的字符串，然后是"HI LO"，并换行
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n"); // 打印带有15个空格的字符串，然后是"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并换行
    print("\n"); // 打印一个空行
    print("\n"); // 打印一个空行
    print("\n"); // 打印一个空行
    print("THIS IS THE GAME OF HI LO.\n"); // 打印"This is the game of HI LO."，并换行
    print("\n"); // 打印一个空行
    print("YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE\n"); // 打印"You will have 6 tries to guess the amount of money in the"，并换行
    print("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU\n"); // 打印"HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU"，并换行
    print("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!\n"); // 打印"Guess the amount, you win all the money in the jackpot!"，并换行
    print("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,\n"); // 打印"Then you get another chance to win more money.  However,"，并换行
    print("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.\n"); // 打印"If you do not guess the amount, the game ends."，并换行
    print("\n"); // 打印一个空行
}
    r = 0;  // 初始化变量 r 为 0
    while (1) {  // 进入无限循环
        b = 0;  // 初始化变量 b 为 0
        print("\n");  // 打印换行符
        y = Math.floor(100 * Math.random());  // 生成一个 0 到 100 之间的随机整数并赋值给变量 y
        for (b = 1; b <= 6; b++) {  // 进入循环，b 从 1 到 6
            print("YOUR GUESS");  // 打印提示信息
            a = parseInt(await input());  // 从用户输入中获取一个整数并赋值给变量 a
            if (a < y) {  // 如果 a 小于 y
                print("YOUR GUESS IS TOO LOW.\n");  // 打印提示信息
            } else if (a > y) {  // 如果 a 大于 y
                print("YOUR GUESS IS TOO HIGH.\n");  // 打印提示信息
            } else {  // 如果 a 等于 y
                break;  // 跳出循环
            }
            print("\n");  // 打印换行符
        }
        if (b > 6) {  // 如果 b 大于 6
            print("YOU BLEW IT...TOO BAD...THE NUMBER WAS " + y + "\n");  // 打印提示信息和 y 的值
            r = 0;  // 将变量 r 设为 0
        } else {
            # 如果猜对了，打印出获胜的消息和赢得的金额
            print("GOT IT!!!!!!!!!!   YOU WIN " + y + " DOLLARS.\n");
            # 将赢得的金额加到总金额上
            r += y;
            # 打印出当前总的赢得金额
            print("YOUR TOTAL WINNINGS ARE NOW " + r + " DOLLARS.\n");
        }
        # 打印空行
        print("\n");
        # 询问玩家是否想再玩一次
        print("PLAY AGAIN (YES OR NO)");
        # 等待玩家输入
        str = await input();
        # 将输入转换为大写
        str = str.toUpperCase();
        # 如果输入不是"YES"，跳出循环
        if (str != "YES")
            break;
    }
    # 打印空行
    print("\n");
    # 打印结束游戏的消息
    print("SO LONG.  HOPE YOU ENJOYED YOURSELF!!!\n");
}

main();
```