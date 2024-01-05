# `76_Russian_Roulette\javascript\russianroulette.js`

```
# RUSSIAN ROULETTE
# 
# 由 Oscar Toledo G. (nanochess) 从 BASIC 转换为 Javascript
#

# 定义一个打印函数，将字符串添加到输出元素中
def print(str):
    document.getElementById("output").appendChild(document.createTextNode(str))

# 定义一个输入函数，返回一个 Promise 对象
def input():
    var input_element
    var input_str

    return new Promise(function (resolve):
                       # 创建一个输入元素
                       input_element = document.createElement("INPUT")

                       # 打印提示符
                       print("? ")
                       # 设置输入元素的类型为文本
                       input_element.setAttribute("type", "text")
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
# 结束输入元素的添加事件监听器
});
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串
}

// Main program
async function main()
{
    print(tab(28) + "RUSSIAN ROULETTE\n");  // 在指定位置打印字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 在指定位置打印字符串
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("THIS IS A GAME OF >>>>>>>>>>RUSSIAN ROULETTE.\n");  // 打印游戏介绍
    restart = true;  // 设置重启标志为真
    while (1) {  // 进入无限循环
        if (restart) {  // 如果需要重启
            restart = false;  // 将重启标志设置为假
            print("\n");  // 打印空行
            print("HERE IS A REVOLVER.\n");  // 打印提示信息
        }
        # 打印提示信息，要求用户输入1来旋转弹膛并扣动扳机
        print("TYPE '1' TO SPIN CHAMBER AND PULL TRIGGER.\n");
        # 打印提示信息，要求用户输入2来放弃
        print("TYPE '2' TO GIVE UP.\n");
        # 打印"GO"
        print("GO");
        # 初始化变量n为0
        n = 0;
        # 进入循环，等待用户输入
        while (1) {
            # 将用户输入的值转换为整数并赋值给变量i
            i = parseInt(await input());
            # 如果用户输入2，打印"CHICKEN!!!!!"并跳出循环
            if (i == 2) {
                print("     CHICKEN!!!!!\n");
                break;
            }
            # 增加变量n的值
            n++;
            # 如果随机数大于0.833333，打印"BANG!!!!!   YOU'RE DEAD!"和"CONDOLENCES WILL BE SENT TO YOUR RELATIVES."，并跳出循环
            if (Math.random() > 0.833333) {
                print("     BANG!!!!!   YOU'RE DEAD!\n");
                print("CONDOLENCES WILL BE SENT TO YOUR RELATIVES.\n");
                break;
            }
            # 如果变量n大于10，打印"YOU WIN!!!!!"和"LET SOMEONE ELSE BLOW HIS BRAINS OUT."，并将restart设置为true
            if (n > 10) {
                print("YOU WIN!!!!!\n");
                print("LET SOMEONE ELSE BLOW HIS BRAINS OUT.\n");
                restart = true;
                break;  # 结束当前循环，跳出循环体
            }
            print("- CLICK -\n");  # 打印“- CLICK -”并换行
            print("\n");  # 打印一个空行
        }
        print("\n");  # 打印一个空行
        print("\n");  # 打印一个空行
        print("\n");  # 打印一个空行
        print("...NEXT VICTIM...\n");  # 打印“...NEXT VICTIM...”并换行
    }
}

main();  # 调用名为main的函数
```