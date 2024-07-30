# `.\comic-translate\modules\utils\archives.py`

```py
# 导入必要的库
import os  # 导入操作系统模块
import zipfile, tarfile, py7zr, rarfile  # 导入处理不同压缩格式的模块
import math  # 导入数学模块
import img2pdf  # 导入将图像转换为PDF的模块
import ebooklib  # 导入电子书处理库
from ebooklib import epub  # 从电子书处理库中导入EPUB格式支持

# 检查文件是否为图像文件
def is_image_file(filename):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    return filename.lower().endswith(image_extensions)

# 解压归档文件
def extract_archive(file_path, extract_to):
    image_paths = []  # 初始化图像文件路径列表

    # 如果文件是ZIP或EPUB格式
    if file_path.lower().endswith(('.cbz', '.zip', '.epub')):
        archive = zipfile.ZipFile(file_path, 'r')  # 打开ZIP文件
        archive.extractall(extract_to)  # 解压所有文件到指定目录
        # 获取所有不是封面且是图像文件的文件路径
        image_paths = [os.path.join(extract_to, file) for file in archive.namelist() if is_image_file(file) and 'cover' not in file.lower()]
        archive.close()  # 关闭ZIP文件

    # 如果文件是CBR格式
    elif file_path.lower().endswith('.cbr'):
        archive = rarfile.RarFile(file_path, 'r')  # 打开RAR文件
        archive.extractall(extract_to)  # 解压所有文件到指定目录
        # 获取所有图像文件的文件路径
        image_paths = [os.path.join(extract_to, file) for file in archive.namelist() if is_image_file(file)]
        archive.close()  # 关闭RAR文件

    # 如果文件是CBT格式
    elif file_path.lower().endswith('.cbt'):
        archive = tarfile.open(file_path, 'r')  # 打开TAR文件
        archive.extractall(extract_to)  # 解压所有文件到指定目录
        # 获取所有图像文件的文件路径
        image_paths = [os.path.join(extract_to, file.name) for file in archive.getmembers() if file.isfile() and is_image_file(file.name)]
        archive.close()  # 关闭TAR文件

    # 如果文件是CB7格式
    elif file_path.lower().endswith('.cb7'):
        with py7zr.SevenZipFile(file_path, 'r') as archive:  # 打开7z文件
            archive.extractall(extract_to)  # 解压所有文件到指定目录
            # 获取所有图像文件的文件路径
            image_paths = [os.path.join(extract_to, entry) for entry in archive.getnames() if is_image_file(entry)]

    # 如果文件是PDF格式
    elif file_path.lower().endswith('.pdf'):
        import fitz  # 导入PyMuPDF库用于处理PDF文件
        pdf_file = fitz.open(file_path)  # 打开PDF文件
        total_images = sum(len(page.get_images(full=True)) for page in pdf_file)  # 获取PDF中的所有图像数量
        digits = math.floor(math.log10(total_images)) + 1  # 计算图像索引的位数

        index = 0  # 初始化图像索引
        # 遍历PDF的每一页
        for page_num in range(len(pdf_file)):
            page = pdf_file[page_num]
            image_list = page.get_images(full=True)  # 获取当前页的所有图像
            # 遍历当前页的每个图像
            for img_index, img in enumerate(image_list, start=1):
                index += 1  # 增加图像索引
                xref = img[0]  # 获取图像的XRef索引
                base_image = pdf_file.extract_image(xref)  # 提取图像数据
                image_bytes = base_image["image"]  # 获取图像的字节数据
                image_ext = base_image["ext"]  # 获取图像的文件扩展名
                image_filename = f"{index:0{digits}d}.{image_ext}"  # 构造图像文件名
                image_path = os.path.join(extract_to, image_filename)  # 构造图像文件路径
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)  # 将图像字节数据写入文件
                image_paths.append(image_path)  # 将图像文件路径添加到列表中
        pdf_file.close()  # 关闭PDF文件

    else:
        raise ValueError("Unsupported file format")  # 抛出不支持的文件格式异常
    
    return image_paths  # 返回所有图像文件的路径列表

def make_cbz(input_dir, output_path="", output_dir="", output_base_name=""):
    if not output_path:
        output_path = os.path.join(output_dir, f"{output_base_name}_translated.cbz")
    # 使用 zipfile 库创建一个 ZIP 文件，并以写模式打开，同时将其赋值给变量 archive
    with zipfile.ZipFile(output_path, 'w') as archive:
        # 遍历指定目录及其子目录中的所有文件和文件夹
        for root, dirs, files in os.walk(input_dir):
            # 遍历当前目录下的所有文件
            for file in files:
                # 检查文件是否是图像文件（假设 is_image_file 是一个函数用于判断）
                if is_image_file(file):
                    # 构建文件的完整路径
                    file_path = os.path.join(root, file)
                    # 将文件添加到 ZIP 归档文件中，并使用相对路径添加到 ZIP 中（相对于输入目录）
                    archive.write(file_path, arcname=os.path.relpath(file_path, input_dir))
# 根据输入目录和输出路径（或目录和基本文件名），创建一个.cb7格式的压缩文件
def make_cb7(input_dir, output_path="", output_dir="", output_base_name=""):
    # 如果没有提供输出路径，则默认为输出目录下的"{output_base_name}_translated.cb7"
    if not output_path:
        output_path = os.path.join(output_dir, f"{output_base_name}_translated.cb7")
    
    # 使用py7zr库创建一个新的7z格式压缩文件
    with py7zr.SevenZipFile(output_path, 'w') as archive:
        # 遍历输入目录下的所有文件和子目录
        for root, dirs, files in os.walk(input_dir):
            # 对于每个文件，如果它是图片文件，则添加到压缩文件中
            for file in files:
                if is_image_file(file):
                    file_path = os.path.join(root, file)
                    archive.write(file_path, arcname=os.path.relpath(file_path, input_dir))

# 根据输入目录中的图片文件，创建一个PDF文件
def make_pdf(input_dir, output_path="", output_dir="", output_base_name=""):
    # 如果没有提供输出路径，则默认为输出目录下的"{output_base_name}_translated.pdf"
    if not output_path:
        output_path = os.path.join(output_dir, f"{output_base_name}_translated.pdf")

    image_paths = []
    # 遍历输入目录下的所有文件和子目录
    for root, dirs, files in os.walk(input_dir):
        # 对于每个文件，如果它是图片文件，则将其路径添加到列表中
        for file in files:
            if is_image_file(file):
                image_paths.append(os.path.join(root, file))

    # 使用img2pdf库将图片路径列表转换为PDF并写入输出文件
    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(image_paths))

# 根据输入目录中的图片文件，创建一个EPUB电子书
def make_epub(input_dir, lang, output_path="", output_dir="", output_base_name=""):
    # 如果没有提供输出路径，则默认为输出目录下的"{output_base_name}_translated.epub"
    if not output_path:
        output_path = os.path.join(output_dir, f"{output_base_name}_translated.epub")

    # 定义图片文件扩展名对应的MIME类型
    mime = {
        '.jpeg': 'jpeg',
        '.jpg': 'jpg',
        '.png': 'png',
        '.webp': 'webp',
        '.bmp': 'bmp'
    }
    # 创建一个新的EPUBBook对象
    book = epub.EpubBook()
    # 设置电子书的标题为输出路径的基本名称（不包含扩展名）
    book.set_title(os.path.splitext(os.path.basename(output_path))[0])
    # 设置电子书的语言
    book.set_language(lang)

    content = [u'<html> <head></head> <body>']

    image_paths = []
    # 遍历输入目录下的所有文件和子目录
    for root, dirs, files in os.walk(input_dir):
        # 对于每个文件，如果它是图片文件，则将其路径添加到列表中
        for file in files:
            if is_image_file(file):
                image_paths.append(os.path.join(root, file))
    
    # 使用第一个图片作为封面图像
    cover_image_path = image_paths[0]
    cover_ext = os.path.splitext(cover_image_path)[1]
    # 设置电子书的封面，使用封面图片的扩展名作为文件名后缀
    book.set_cover("cover" + cover_ext, open(cover_image_path, 'rb').read())

    # 遍历所有图片路径，为每个图片创建EPUBItem对象并添加到电子书中
    for image_path in image_paths:
        file_name = os.path.basename(image_path)
        ext = os.path.splitext(image_path)[1]
        # 创建EPUBItem对象，表示电子书中的一个项目（图片）
        epub_image = epub.EpubItem(file_name= "images/" + file_name, content=open(image_path, 'rb').read(), media_type=f"image/{mime[ext]}")
        book.add_item(epub_image)
        # 将图片的HTML标签添加到内容列表中
        content.append(f'<img src="{epub_image.file_name}"/>')

    content.append('</body> </html>')
    c1 = epub.EpubHtml(title='Images', file_name='images.xhtml', lang=lang)
    c1.content = ''.join(content)

    # 将HTML内容添加到电子书中
    book.add_item(c1)
    # 设置电子书的主内容
    book.spine = ['nav', c1]

    # 使用epub库将EPUBBook对象写入输出路径的EPUB文件
    epub.write_epub(output_path, book, {})

# 根据输入目录中的文件，根据保存扩展名生成相应类型的文件
def make(input_dir, output_path="", save_as_ext="", output_dir="", output_base_name="", trg_lng=""):
    # 如果既没有提供输出路径，也没有提供输出目录或基本文件名，则抛出值错误
    if not output_path and (not output_dir or not output_base_name):
        raise ValueError("Either output_path or both output_dir and output_base_name must be provided")
    
    # 如果提供了输出路径，则根据输出路径获取保存扩展名
    if output_path:
        save_as_ext = os.path.splitext(output_path)[1]

    # 根据保存扩展名调用相应的文件生成函数
    if save_as_ext in ['.cbz', '.zip']:
        make_cbz(input_dir, output_path, output_dir, output_base_name)
    # 如果保存的文件扩展名是 '.cb7'，则调用 make_cb7 函数处理文件
    elif save_as_ext == '.cb7':
        make_cb7(input_dir, output_path, output_dir, output_base_name)
    # 如果保存的文件扩展名是 '.pdf'，则调用 make_pdf 函数处理文件
    elif save_as_ext == '.pdf':
        make_pdf(input_dir, output_path, output_dir, output_base_name)
    # 如果保存的文件扩展名是 '.epub'，则调用 make_epub 函数处理文件
    elif save_as_ext == '.epub':
        make_epub(input_dir, trg_lng, output_path, output_dir, output_base_name)
    # 如果保存的文件扩展名不是以上支持的类型，则抛出 ValueError 异常
    else:
        raise ValueError(f"Unsupported save_as_ext: {save_as_ext}")
```