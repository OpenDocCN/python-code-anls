# `d:/src/tocomm/basic-computer-games\41_Guess\javascript\guess.js`

```
# GUESS
# 
# Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
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
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
        str += " ";  // 将空格字符添加到字符串末尾
    return str;  // 返回修改后的字符串
}

function make_space()  // 定义名为make_space的函数
{
    for (h = 1; h <= 5; h++)  // 循环5次
        print("\n");  // 打印换行符
}

// Main control section
async function main()  // 定义名为main的异步函数
{
    while (1) {  // 当条件为真时循环执行
        print(tab(33) + "GUESS\n");  // 打印带有33个空格的字符串和"GUESS"，并换行
        print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有15个空格的字符串和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并换行
        print("\n");  // 打印换行符
        print("\n");  // 打印换行符
        print("\n");  // 打印换行符
        print("THIS IS A NUMBER GUESSING GAME. I'LL THINK\n");  // 打印"This is a number guessing game. I'll think"
        # 打印游戏规则提示
        print("OF A NUMBER BETWEEN 1 AND ANY LIMIT YOU WANT.\n");
        print("THEN YOU HAVE TO GUESS WHAT IT IS.\n");
        print("\n");

        # 获取用户输入的数字上限
        print("WHAT LIMIT DO YOU WANT");
        l = parseInt(await input());
        print("\n");
        
        # 计算猜测次数的上限
        l1 = Math.floor(Math.log(l) / Math.log(2)) + 1;
        
        # 游戏开始，猜测数字
        while (1) {
            # 提示玩家猜测的数字范围
            print("I'M THINKING OF A NUMBER BETWEEN 1 AND " + l + "\n");
            g = 1;
            print("NOW YOU TRY TO GUESS WHAT IT IS.\n");
            
            # 生成随机数作为答案
            m = Math.floor(l * Math.random() + 1);
            
            # 玩家猜测数字
            while (1) {
                n = parseInt(await input());
                
                # 处理玩家输入小于等于0的情况
                if (n <= 0) {
                    make_space();
                    break;
                }
                
                # 判断玩家猜测是否正确
                if (n == m) {
                    print("THAT'S IT! YOU GOT IT IN " + g + " TRIES.\n");  // 打印玩家猜对了答案，并输出猜测次数
                    if (g == l1) {  // 如果猜测次数等于答案
                        print("GOOD.\n");  // 打印GOOD
                    } else if (g < l1) {  // 如果猜测次数小于答案
                        print("VERY GOOD.\n");  // 打印VERY GOOD
                    } else {  // 如果猜测次数大于答案
                        print("YOU SHOULD HAVE BEEN TO GET IT IN ONLY " + l1 + "\n");  // 打印正确答案
                    }
                    make_space();  // 调用make_space函数
                    break;  // 退出循环
                }
                g++;  // 猜测次数加1
                if (n > m)  // 如果猜测的数字大于答案
                    print("TOO HIGH. TRY A SMALLER ANSWER.\n");  // 打印猜测数字太大
                else  // 如果猜测的数字小于答案
                    print("TOO LOW. TRY A BIGGER ANSWER.\n");  // 打印猜测数字太小
            }
            if (n <= 0)  // 如果猜测次数小于等于0
                break;  // 退出循环
        }
    }
}
```
这部分代码是一个函数的结束和一个程序的结束。在Python中，函数的结束需要使用"}"来表示，而程序的结束需要调用main()函数来执行程序的主要逻辑。
```