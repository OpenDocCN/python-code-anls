# `45_Hello\javascript\hello.js`

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
# 添加键盘按下事件监听器，当按下的键是回车键时，获取输入字符串，移除输入元素，打印输入字符串并换行，然后解析输入字符串
input_element.addEventListener("keydown", function (event) {
    if (event.keyCode == 13) {
        input_str = input_element.value;
        document.getElementById("output").removeChild(input_element);
        print(input_str);
        print("\n");
        resolve(input_str);
    }
});
# 结束事件监听器的添加
});

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回修改后的字符串

}

// Main control section
async function main()
{
    print(tab(33) + "HELLO\n");  # 在第33列打印"HELLO"并换行
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 在第15列打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"并换行
    print("\n");  # 打印一个空行
    print("\n");  # 打印一个空行
    print("\n");  # 打印一个空行
    print("HELLO.  MY NAME IS CREATIVE COMPUTER.\n");  # 打印"HELLO.  MY NAME IS CREATIVE COMPUTER."并换行
    print("\n");  # 打印一个空行
    print("\n");  # 打印一个空行
    print("WHAT'S YOUR NAME");  # 打印"WHAT'S YOUR NAME"
    ns = await input();  # 等待用户输入并将输入存储在变量ns中
    print("\n");  # 打印一个空行
    print("HI THERE, " + ns + ", ARE YOU ENJOYING YOURSELF HERE");  # 打印"HI THERE, "后跟用户输入的名字，再跟上", ARE YOU ENJOYING YOURSELF HERE"
    while (1) {  # 进入无限循环
# 从输入中获取字符串
bs = await input();
# 打印空行
print("\n");
# 如果输入为"YES"，则打印对应的问候语并跳出循环
if (bs == "YES") {
    print("I'M GLAD TO HEAR THAT, " + ns + ".\n");
    print("\n");
    break;
# 如果输入为"NO"，则打印对应的问候语并跳出循环
} else if (bs == "NO") {
    print("OH, I'M SORRY TO HEAR THAT, " + ns + ". MAYBE WE CAN\n");
    print("BRIGHTEN UP YOUR VISIT A BIT.\n");
    break;
# 如果输入既不是"YES"也不是"NO"，则提示用户重新输入
} else {
    print("PLEASE ANSWER 'YES' OR 'NO'.  DO YOU LIKE IT HERE");
}
# 打印空行
print("\n");
# 打印对话内容
print("SAY, " + ns + ", I CAN SOLVED ALL KINDS OF PROBLEMS EXCEPT\n");
print("THOSE DEALING WITH GREECE.  WHAT KIND OF PROBLEMS DO\n");
print("YOU HAVE (ANSWER SEX, HEALTH, MONEY, OR JOB)");
# 进入循环，从输入中获取字符串
while (1) {
    cs = await input();
        # 打印一个空行
        print("\n");
        # 如果输入的cs不是"SEX"、"HEALTH"、"MONEY"、"JOB"中的任何一个
        if (cs != "SEX" && cs != "HEALTH" && cs != "MONEY" && cs != "JOB") {
            # 打印一条提示信息，包括输入的cs和ns
            print("OH, " + ns + ", YOUR ANSWER OF " + cs + " IS GREEK TO ME.\n");
        # 如果输入的cs是"JOB"
        } else if (cs == "JOB") {
            # 打印一条关于工作的提示信息，包括输入的ns
            print("I CAN SYMPATHIZE WITH YOU " + ns + ".  I HAVE TO WORK\n");
            print("VERY LONG HOURS FOR NO PAY -- AND SOME OF MY BOSSES\n");
            print("REALLY BEAT ON MY KEYBOARD.  MY ADVICE TO YOU, " + ns + ",\n");
            print("IS TO OPEN A RETAIL COMPUTER STORE.  IT'S GREAT FUN.\n");
        # 如果输入的cs是"MONEY"
        } else if (cs == "MONEY") {
            # 打印一条关于金钱的提示信息，包括输入的ns
            print("SORRY, " + ns + ", I'M BROKE TOO.  WHY DON'T YOU SELL\n");
            print("ENCYCLOPEADIAS OR MARRY SOMEONE RICH OR STOP EATING\n");
            print("SO YOU WON'T NEED SO MUCH MONEY?\n");
        # 如果输入的cs是"HEALTH"
        } else if (cs == "HEALTH") {
            # 打印一条关于健康的提示信息，包括输入的ns
            print("MY ADVICE TO YOU " + ns + " IS:\n");
            print("     1.  TAKE TWO ASPRIN\n");
            print("     2.  DRINK PLENTY OF FLUIDS (ORANGE JUICE, NOT BEER!)\n");
            print("     3.  GO TO BED (ALONE)\n");
        # 如果输入的cs不是以上任何一个
        } else {
            # 打印一条提示信息
            print("IS YOUR PROBLEM TOO MUCH OR TOO LITTLE");
            # 进入一个无限循环
            while (1) {
# 从输入中获取数据
ds = await input();
# 打印空行
print("\n");
# 如果输入为"TOO MUCH"，则打印相应信息并跳出循环
if (ds == "TOO MUCH"):
    print("YOU CALL THAT A PROBLEM?!!  I SHOULD HAVE SUCH PROBLEMS!\n");
    print("IF IT BOTHERS YOU, " + ns + ", TAKE A COLD SHOWER.\n");
    break;
# 如果输入为"TOO LITTLE"，则打印相应信息并跳出循环
elif (ds == "TOO LITTLE"):
    print("WHY ARE YOU HERE IN SUFFERN, " + ns + "?  YOU SHOULD BE\n");
    print("IN TOKYO OR NEW YORK OR AMSTERDAM OR SOMEPLACE WITH SOME\n");
    print("REAL ACTION.\n");
    break;
# 如果输入既不是"TOO MUCH"也不是"TOO LITTLE"，则打印相应信息
else:
    print("DON'T GET ALL SHOOK, " + ns + ", JUST ANSWER THE QUESTION\n");
    print("WITH 'TOO MUCH' OR 'TOO LITTLE'.  WHICH IS IT");
# 打印空行
print("\n");
# 打印询问是否还有其他问题需要解决
print("ANY MORE PROBLEMS YOU WANT SOLVED, " + ns);
# 从输入中获取数据
es = await input();
        print("\n");  # 打印空行
        if (es == "YES"):  # 如果es变量的值为"YES"
            print("WHAT KIND (SEX, MONEY, HEALTH, JOB)");  # 打印提示信息
        elif (es == "NO"):  # 如果es变量的值为"NO"
            print("THAT WILL BE $5.00 FOR THE ADVICE, " + ns + ".\n");  # 打印收费信息
            print("PLEASE LEAVE THE MONEY ON THE TERMINAL.\n");  # 提示留下钱
            print("\n");  # 打印空行
//            d = new Date().valueOf();  # 获取当前时间的时间戳
//            while (new Date().valueOf() - d < 2000) ;  # 等待2秒
            print("\n");  # 打印空行
            print("\n");  # 再次打印空行
            while (1):  # 进入无限循环
                print("DID YOU LEAVE THE MONEY");  # 提示是否留下了钱
                gs = await input();  # 获取输入值
                print("\n");  # 打印空行
                if (gs == "YES"):  # 如果gs变量的值为"YES"
                    print("HEY, " + ns + "??? YOU LEFT NO MONEY AT ALL!\n");  # 打印未留下钱的信息
                    print("YOU ARE CHEATING ME OUT OF MY HARD-EARNED LIVING.\n");  # 打印被骗的信息
                    print("\n");  # 打印空行
                    print("WHAT A RIP OFF, " + ns + "!!!\n");  # 打印被骗的信息
                    print("\n");  # 打印空行
                    break;  # 跳出循环
                } else if (gs == "NO") {  # 如果输入的内容是"NO"
                    print("THAT'S HONEST, " + ns + ", BUT HOW DO YOU EXPECT\n");  # 打印提示信息
                    print("ME TO GO ON WITH MY PSYCHOLOGY STUDIES IF MY PATIENT\n");  # 打印提示信息
                    print("DON'T PAY THEIR BILLS?\n");  # 打印提示信息
                    break;  # 跳出循环
                } else {  # 如果输入的内容不是"YES"也不是"NO"
                    print("YOUR ANSWER OF '" + gs + "' CONFUSES ME, " + ns + ".\n");  # 打印提示信息
                    print("PLEASE RESPOND WITH 'YES' OR 'NO'.\n");  # 打印提示信息
                }
            }
            break;  # 跳出循环
        }
    }
    print("\n");  # 打印空行
    print("TAKE A WALK, " + ns + ".\n");  # 打印提示信息
    print("\n");  # 打印空行
    print("\n");  # 打印空行
    // Line 390 not used in original  # 注释：原始代码中未使用第390行
}

# 调用主函数
main();
```