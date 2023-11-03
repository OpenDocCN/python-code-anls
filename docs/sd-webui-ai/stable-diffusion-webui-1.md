# SDWebUI源码解析 1

# `/opt/to-comment/stable-diffusion-webui/modules/images.py`

这段代码的主要作用是定义了一个名为 `image_grid` 的函数，用于将给定的图像列表（`imgs`）加载到内存中，并按指定批次大小（`batch_size`）加载到网格中，最后返回网格对象（`grid`）。

具体来说，函数首先通过 `opts.n_rows` 检查是否设置了行数（也可以通过 `opts.n_cols` 设置），如果没有设置行数，则将批次大小（`batch_size`）赋值为批次大小，否则通过 `math.sqrt(len(imgs))` 计算出每行可放置的图像数量，然后将 `batch_size` 和计算得到的每行图像数量取最大值，得到每行需要放置的列数（`cols`）。

接下来，定义了一个 `Image.new` 函数，创建一个和输入图像相同大小的黑色背景的 `Image` 对象，并将每个输入图像拉伸至适应网格的大小，然后将所有拉伸的图像放置在网格中。

函数的参数包括：

* `imgs`：需要加载的图像列表
* `batch_size`：批次大小，可以是图像数量的两倍，用于控制一次性加载多少图像到内存中
* `rows`：行数，用于指定批次的列数

函数返回一个和输入图像大小相同的 `Image` 对象，其中包含按批次大小加载的图像的 `Image.new` 对象。


```
import math
import os
from collections import namedtuple
import re

import numpy as np
from PIL import Image, ImageFont, ImageDraw, PngImagePlugin

import modules.shared
from modules.shared import opts

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)


def image_grid(imgs, batch_size=1, rows=None):
    if rows is None:
        if opts.n_rows > 0:
            rows = opts.n_rows
        elif opts.n_rows == 0:
            rows = batch_size
        else:
            rows = math.sqrt(len(imgs))
            rows = round(rows)

    cols = math.ceil(len(imgs) / rows)

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color='black')

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid


```

这段代码定义了一个名为Grid的命名元组类型，包含了一些和网格相关的参数。这个网格可以用来在平面上进行渲染和显示，例如在Pygame等游戏引擎中。

split_grid函数，接受一个图像对象和一个 tile_w 和 tile_h 参数，对传入的图像进行分割，并将分割出来的图像组成一个列表，返回这个列表。

具体来说，这个函数首先计算出每个网格的列数和行数。以每个网格所需的非覆盖区域(也就是 tile_w 和 tile_h 减去 overlap 的值)作为网格的列宽和行高。然后，从 0 到 cols-1 进行步长计算，从 0 到 rows-1 进行步长计算。接下来，在网格中按行和列构建一个列表，包含该网格中所有像素点的信息。如果当前列数小于列宽，或者当前行数小于行高，就通过在四周扩展(dx 和 dy)来填充任何剩余的空间，使得所有网格都连接在一起。最后，返回由所有网格列表组成的元组。


```
Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])


def split_grid(image, tile_w=512, tile_h=512, overlap=64):
    w = image.width
    h = image.height

    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    cols = math.ceil((w - overlap) / non_overlap_width)
    rows = math.ceil((h - overlap) / non_overlap_height)

    dx = (w - tile_w) / (cols-1) if cols > 1 else 0
    dy = (h - tile_h) / (rows-1) if rows > 1 else 0

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = int(row * dy)

        if y + tile_h >= h:
            y = h - tile_h

        for col in range(cols):
            x = int(col * dx)

            if x+tile_w >= w:
                x = w - tile_w

            tile = image.crop((x, y, x + tile_w, y + tile_h))

            row_images.append([x, tile_w, tile])

        grid.tiles.append([y, tile_h, row_images])

    return grid


```

这段代码定义了一个名为 `combine_grid` 的函数，它接收一个二维网格（grid）作为参数，并返回一个新的图像对象，该图像对象由一个具有相同大小但更高分辨率的图像和一个低分辨率图像合成而成。

函数内部首先定义了一个名为 `make_mask_image` 的函数，它接收一个二维数组（grid）和一个参数 `r`，计算并返回一个二进制掩码图像。这个函数将输入的 `r` 值进行下采样操作，然后将其转换为一个二进制形式，最后创建一个 NumPy 数组并返回一个 Image 对象。

接下来，代码定义了一个 `mask_w` 和一个 `mask_h`，它们分别是网格的中间部分的高和低分辨率版本，用来制作合成图像的遮罩。

函数内部接着创建一个名为 `combined_image` 的 Image 对象，并使用一个循环来遍历网格中的每个元素，对于每个元素，首先创建一个高分辨率版本的合成图像，然后从低分辨率版本中下载该元素并将其下载到合成图像中。具体来说，对于每个元素，函数会将合成图像中的对应位置下载一个高分辨率版本的元素，然后下载一个低分辨率版本的元素，这两个版本会在合成图像中合并为一个元素。最后，函数返回生成的合成图像。


```
def combine_grid(grid):
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, 'L')

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image


```

这是一个Python的函数，可以将一个文本图像（如原始图像）中的文本框从图像中分离出来，得到每个文本框的矩形边界框、内容文本以及图片的背景颜色。

python
def separate_text_boxes(image_path, text_thickness, max_width, min_width, rows, cols, pad_left, pad_top, width):
   img = Image.open(image_path).resize((width, height))
   result = Image.new("RGB", (img.width + pad_left, img.height + pad_top))
   result.paste(img)

   for col in range(cols):
       x = pad_left + width * col + width / 2
       y = pad_top / 2 - text_thickness * rows + row_spacing * col

       draw_rect = ImageDraw.Draw(result)
       draw_text = ImageDraw.Draw(result)

       for text in horiz_texts[col]:
           draw_rect.rectangle([(x, y), (x + max_width, y + text_thickness)], fill="black")
           draw_text.text((x + max_width, y + text_thickness), text, font=fnt)

       draw_rect.ellipse((x - min_width, y - min_width, x + max_width, y + text_thickness), fill="white")
       draw_text.ellipse((x - min_width, y - min_width, x + max_width, y + text_thickness), fill="white")

   return result


这个函数接受一个图片文件路径、文本框的垂直和水平宽度、图像的行数和列数，以及背景文本和字体大小。它返回原始图像、每个文本框的矩形边界框、内容文本以及图片的背景颜色。

这个函数需要一个前提条件：原始图像已经被处理过，提取出了所有文本框。这个函数处理文本框的方法是：对于每个文本框，首先绘制文本框的边框和内容文本，然后根据给定的文本框宽度计算出阴影的范围，最后将阴影部分和文本部分都变成白色。


```
class GridAnnotation:
    def __init__(self, text='', is_active=True):
        self.text = text
        self.is_active = is_active
        self.size = None


def draw_grid_annotations(im, width, height, hor_texts, ver_texts):
    def wrap(drawing, text, font, line_length):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if drawing.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return lines

    def draw_texts(drawing, draw_x, draw_y, lines):
        for i, line in enumerate(lines):
            drawing.multiline_text((draw_x, draw_y + line.size[1] / 2), line.text, font=fnt, fill=color_active if line.is_active else color_inactive, anchor="mm", align="center")

            if not line.is_active:
                drawing.line((draw_x - line.size[0]//2, draw_y + line.size[1]//2, draw_x + line.size[0]//2, draw_y + line.size[1]//2), fill=color_inactive, width=4)

            draw_y += line.size[1] + line_spacing

    fontsize = (width + height) // 25
    line_spacing = fontsize // 2
    fnt = ImageFont.truetype(opts.font, fontsize)
    color_active = (0, 0, 0)
    color_inactive = (153, 153, 153)

    pad_left = width * 3 // 4 if len(ver_texts) > 0 else 0

    cols = im.width // width
    rows = im.height // height

    assert cols == len(hor_texts), f'bad number of horizontal texts: {len(hor_texts)}; must be {cols}'
    assert rows == len(ver_texts), f'bad number of vertical texts: {len(ver_texts)}; must be {rows}'

    calc_img = Image.new("RGB", (1, 1), "white")
    calc_d = ImageDraw.Draw(calc_img)

    for texts, allowed_width in zip(hor_texts + ver_texts, [width] * len(hor_texts) + [pad_left] * len(ver_texts)):
        items = [] + texts
        texts.clear()

        for line in items:
            wrapped = wrap(calc_d, line.text, fnt, allowed_width)
            texts += [GridAnnotation(x, line.is_active) for x in wrapped]

        for line in texts:
            bbox = calc_d.multiline_textbbox((0, 0), line.text, font=fnt)
            line.size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    hor_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing for lines in hor_texts]
    ver_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing * len(lines) for lines in ver_texts]

    pad_top = max(hor_text_heights) + line_spacing * 2

    result = Image.new("RGB", (im.width + pad_left, im.height + pad_top), "white")
    result.paste(im, (pad_left, pad_top))

    d = ImageDraw.Draw(result)

    for col in range(cols):
        x = pad_left + width * col + width / 2
        y = pad_top / 2 - hor_text_heights[col] / 2

        draw_texts(d, x, y, hor_texts[col])

    for row in range(rows):
        x = pad_left / 2
        y = pad_top + height * row + height / 2 - ver_text_heights[row] / 2

        draw_texts(d, x, y, ver_texts[row])

    return result


```

This is a function that modifies an image using the given `src_ratio`. If `src_ratio` is less than 1, the image will be resized to maintain the aspect ratio with the minimum possible loss of image quality. If `src_ratio` is greater than 1, the image will be resized to maintain the aspect ratio with the maximum possible loss of image quality.

The function takes an input image and returns a modified version of the same image. The `box` parameter is used to specify the position of the crop operation.

Here is the code in Python:

def modify_image(image, src_ratio):
   if src_ratio < 1:
       res, bbox = image.resize((int(image.width * src_ratio), int(image.height * 1.25)), resample=LANCZOS)
   else:
       res, bbox = image.resize((int(image.height * 1.25), int(image.width * src_ratio)), resample=LANCZOS)
   return res, bbox

Note that this function modifies the original image by resizing it, and does not return the new image.


```
def draw_prompt_matrix(im, width, height, all_prompts):
    prompts = all_prompts[1:]
    boundary = math.ceil(len(prompts) / 2)

    prompts_horiz = prompts[:boundary]
    prompts_vert = prompts[boundary:]

    hor_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_horiz)] for pos in range(1 << len(prompts_horiz))]
    ver_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_vert)] for pos in range(1 << len(prompts_vert))]

    return draw_grid_annotations(im, width, height, hor_texts, ver_texts)


def resize_image(resize_mode, im, width, height):
    if resize_mode == 0:
        res = im.resize((width, height), resample=LANCZOS)
    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res


```

This is a Python function that generates a PNG image and saves it in a directory. The image is generated by iterating through all files in a directory, processing them to extract information about their file extensions and then using this information to generate a PNG image with a specified quality and exporting it to a file.

The function takes four arguments:

* `path`: The directory where the files are located.
* `extension`: The extension of the files to be processed.
* `opts`: A dictionary containing options for the generate process.
	+ `export_for_4chan`: Whether to save the image to a file in the 4chan MIME type.
	+ `target_side_length`: The target side length for the image.
	+ `oversize`: Whether to oversample the image to reach the target side length.
* `info`: If it is not None, information about the files to be processed.

The function first checks if the directory exists and creates it if it doesn't using the `os.makedirs` function. It then sets the directory structure and generates the PNG image according to the options passed.


```
invalid_filename_chars = '<>:"/\\|?*\n'


def sanitize_filename_part(text):
    return text.replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]


def save_image(image, path, basename, seed=None, prompt=None, extension='png', info=None, short_filename=False, no_prompt=False):
    if short_filename or prompt is None or seed is None:
        file_decoration = ""
    elif opts.save_to_dirs:
        file_decoration = f"-{seed}"
    else:
        file_decoration = f"-{seed}-{sanitize_filename_part(prompt)[:128]}"

    if extension == 'png' and opts.enable_pnginfo and info is not None:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", info)
    else:
        pnginfo = None

    if opts.save_to_dirs and not no_prompt:
        words = re.findall(r'\w+', prompt or "")
        if len(words) == 0:
            words = ["empty"]

        dirname = " ".join(words[0:opts.save_to_dirs_prompt_len])
        path = os.path.join(path, dirname)

    os.makedirs(path, exist_ok=True)

    filecount = len([x for x in os.listdir(path) if os.path.splitext(x)[1] == '.' + extension])
    fullfn = "a.png"
    fullfn_without_extension = "a"
    for i in range(500):
        fn = f"{filecount+i:05}" if basename == '' else f"{basename}-{filecount+i:04}"
        fullfn = os.path.join(path, f"{fn}{file_decoration}.{extension}")
        fullfn_without_extension = os.path.join(path, f"{fn}{file_decoration}")
        if not os.path.exists(fullfn):
            break

    image.save(fullfn, quality=opts.jpeg_quality, pnginfo=pnginfo)

    target_side_length = 4000
    oversize = image.width > target_side_length or image.height > target_side_length
    if opts.export_for_4chan and (oversize or os.stat(fullfn).st_size > 4 * 1024 * 1024):
        ratio = image.width / image.height

        if oversize and ratio > 1:
            image = image.resize((target_side_length, image.height * target_side_length // image.width), LANCZOS)
        elif oversize:
            image = image.resize((image.width * target_side_length // image.height, target_side_length), LANCZOS)

        image.save(f"{fullfn_without_extension}.jpg", quality=opts.jpeg_quality, pnginfo=pnginfo)

    if opts.save_txt and info is not None:
        with open(f"{fullfn_without_extension}.txt", "w", encoding="utf8") as file:
            file.write(info + "\n")


```

这段代码定义了一个名为 "Upscaler" 的类，用于对一张图片进行放大处理。

在类中，有两个方法，一个是 "do_upscale"，另一个是 "upscale"。

"do_upscale" 方法接收一张图片对象作为参数，并返回这张图片。

"upscale" 方法需要两个参数，一张图片对象以及图片的宽度和高度。它通过迭代算法来检查图片的宽度和高度是否大于给定的宽度和高度，如果是，就执行 "do_upscale" 方法，获取放大后的图片。否则，它会尝试将图片的大小调整为符合宽度和高度的大小，然后返回这张图片。

在 "upscale" 方法中，如果图片的宽度和高度不匹配给定的宽度和高度，它会使用 "resize" 方法将图片的大小调整为符合宽度和高度的大小。这里的 "resize" 方法使用了 "LANCZOS" 模式来拉伸图片，这是一种基于约束的拉伸算法，可以平滑地拉伸图片以保持其质量。


```
class Upscaler:
    name = "Lanczos"

    def do_upscale(self, img):
        return img

    def upscale(self, img, w, h):
        for i in range(3):
            if img.width >= w and img.height >= h:
                break

            img = self.do_upscale(img)

        if img.width != w or img.height != h:
            img = img.resize((w, h), resample=LANCZOS)

        return img


```

这段代码定义了一个名为 UpscalerNone 的类，其作用是重写父类 Upscaler 的 upscale 方法，并对传入的图像进行放大。

在代码中，首先定义了一个名为 UpscalerNone 的类，与父类 Upscaler 相比，这个类的名字被定义为 "None"，因为在 Python 中，类名的第一个字母要大写。

接着定义了一个名为 upscale 的方法，这个方法接收一个图像对象、宽度和高度参数，返回放大后的图像。

在 UpscalerNone 中，通过直接重写父类的 upscale 方法，实现了对父类的方法进行修改，以便在需要时调用。

在代码的末尾，将两个 UpscalerNone 对象添加到 shared.sd_upscalers 列表中。

通过这段代码，创建了一个 UpscalerNone 类，用于对图像进行放大。在代码中，通过直接重写父类的 upscale 方法，实现了创建一个副本并在需要时调用该方法。同时，在代码的末尾，将两个 UpscalerNone 对象添加到 shared.sd_upscalers 列表中，以便在需要时动态分配 UpscalerNone 对象。


```
class UpscalerNone(Upscaler):
    name = "None"

    def upscale(self, img, w, h):
        return img


modules.shared.sd_upscalers.append(UpscalerNone())
modules.shared.sd_upscalers.append(Upscaler())

```

# `/opt/to-comment/stable-diffusion-webui/modules/img2img.py`

This script appears to be a command-line interface for upscaling images using a tesseract-based OCR model and a specified image upscaleer. The script takes several arguments:

* `-h` or `--height`: The height of the output image, in the format of the height, width, and number of channels.
* `-w` or `--width`: The width of the output image, in the format of the height, width, and number of channels.
* `-b` or `--batch_size`: The number of tiles that will be used to process the image in parallel.
* `-p` or `--path_to_psd`: The path to a directory containing the teslaScripts AI model files.
* `-p_es` or `--es_json_path`: The path to a teslaScripts AI model JSON file.
* `-q` or `--quiet`: Force the script to be quiet, without printing any output.
* `--opts` or `-o`: Options for the image upscaler.
	+ `-r` or `--resc`: Rescale the image by a specified factor.
	+ `-m` or `--quality`: Control the quality of the output image.
	+ `-n` or `--no_ progress_bar`: Disable the progress bar.
* `--grid_format`: The output format of the grid.
* `-c` or `--colormap_path`: The path to a colormap to be used for color mapping.
* `-i` or `--info`: Include detailed information about the image, including information about the original image and processed image.
* `--wrapped_in_psd`: Wrapped in a tesseractScripts AI model Faster R-CNN model.
* `--max_es_api_iterations`: The maximum number of teslaScripts API iterations.
* `--psd_es_iterations`: The maximum number of teslaScripts API iterations for the PSD model.

The script uses the `egsplitter` library to split the input image into tiles, and the `processing_images.py` and `psd_models.py` scripts from the `image_upscaleers` directory to perform the image upscaling. The `es_api.py` script from the `teslaScripts/models/` directory is used to train the teslaScripts AI model.


```
import math
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageChops

from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, state
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.images as images
import modules.scripts

def img2img(prompt: str, negative_prompt: str, init_img, init_img_with_mask, steps: int, sampler_index: int, mask_blur: int, inpainting_fill: int, restore_faces: bool, tiling: bool, mode: int, n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float, denoising_strength_change_factor: float, seed: int, height: int, width: int, resize_mode: int, upscaler_index: str, upscale_overlap: int, inpaint_full_res: bool, inpainting_mask_invert: int, *args):
    is_inpaint = mode == 1
    is_loopback = mode == 2
    is_upscale = mode == 3

    if is_inpaint:
        image = init_img_with_mask['image']
        alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
        mask = ImageChops.lighter(alpha_mask, init_img_with_mask['mask'].convert('L')).convert('RGBA')
        image = image.convert('RGB')
    else:
        image = init_img
        mask = None

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        sampler_index=sampler_index,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        inpaint_full_res=inpaint_full_res,
        inpainting_mask_invert=inpainting_mask_invert,
        extra_generation_params={
            "Denoising strength": denoising_strength,
            "Denoising strength change factor": denoising_strength_change_factor
        }
    )
    print(f"\nimg2img: {prompt}", file=shared.progress_print_out)

    if is_loopback:
        output_images, info = None, None
        history = []
        initial_seed = None
        initial_info = None

        state.job_count = n_iter

        do_color_correction = False
        try:
            from skimage import exposure
            do_color_correction = True
        except:
            print("Install scikit-image to perform color correction on loopback")


        for i in range(n_iter):

            if do_color_correction and i == 0:
                correction_target = cv2.cvtColor(np.asarray(init_img.copy()), cv2.COLOR_RGB2LAB)

            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True

            state.job = f"Batch {i + 1} out of {n_iter}"
            processed = process_images(p)

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info
            
            init_img = processed.images[0]

            if do_color_correction and correction_target is not None:
                init_img = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
                    cv2.cvtColor(
                        np.asarray(init_img),
                        cv2.COLOR_RGB2LAB
                    ),
                    correction_target,
                    channel_axis=2
                ), cv2.COLOR_LAB2RGB).astype("uint8"))

            p.init_images = [init_img]
            p.seed = processed.seed + 1
            p.denoising_strength = min(max(p.denoising_strength * denoising_strength_change_factor, 0.1), 1)
            history.append(processed.images[0])

        grid = images.image_grid(history, batch_size, rows=1)

        images.save_image(grid, p.outpath_grids, "grid", initial_seed, prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename)

        processed = Processed(p, history, initial_seed, initial_info)

    elif is_upscale:
        initial_seed = None
        initial_info = None

        upscaler = shared.sd_upscalers[upscaler_index]
        img = upscaler.upscale(init_img, init_img.width * 2, init_img.height * 2)

        processing.torch_gc()

        grid = images.split_grid(img, tile_w=width, tile_h=height, overlap=upscale_overlap)

        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        work = []
        work_results = []

        for y, h, row in grid.tiles:
            for tiledata in row:
                work.append(tiledata[2])

        batch_count = math.ceil(len(work) / p.batch_size)
        print(f"SD upscaling will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)} in a total of {batch_count} batches.")

        state.job_count = batch_count

        for i in range(batch_count):
            p.init_images = work[i*p.batch_size:(i+1)*p.batch_size]

            state.job = f"Batch {i + 1} out of {batch_count}"
            processed = process_images(p)

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info

            p.seed = processed.seed + 1
            work_results += processed.images

        image_index = 0
        for y, h, row in grid.tiles:
            for tiledata in row:
                tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
                image_index += 1

        combined_image = images.combine_grid(grid)

        if opts.samples_save:
            images.save_image(combined_image, p.outpath_samples, "", initial_seed, prompt, opts.grid_format, info=initial_info)

        processed = Processed(p, [combined_image], initial_seed, initial_info)

    else:

        processed = modules.scripts.scripts_img2img.run(p, *args)

        if processed is None:
            processed = process_images(p)

    shared.total_tqdm.clear()

    return processed.images, processed.js(), plaintext_to_html(processed.info)

```

# `/opt/to-comment/stable-diffusion-webui/modules/lowvram.py`

It looks like you're trying to build a neural network model using the PyTorch library and you're having trouble with the model's first and last stages, which are producing errors. Specifically, you're trying to use the "send\_me\_to\_gpu" function, which appears to be a function that sends the input data to the GPU.

One possible solution would be to check the source code of the model to see if there's anything that's causing the errors. You might also want to check that the inputs to the model are the correct shape and size. Additionally, you may want to try using the `torch.no_grad` function to disable gradient calculation to see if that resolves the issue.


```
import torch

module_in_gpu = None
cpu = torch.device("cpu")
if torch.has_cuda:
    device = gpu = torch.device("cuda")
elif torch.has_mps:
    device = gpu = torch.device("mps")
else:
    device = gpu = torch.device("cpu")

def setup_for_low_vram(sd_model, use_medvram):
    parents = {}

    def send_me_to_gpu(module, _):
        """send this module to GPU; send whatever tracked module was previous in GPU to CPU;
        we add this as forward_pre_hook to a lot of modules and this way all but one of them will
        be in CPU
        """
        global module_in_gpu

        module = parents.get(module, module)

        if module_in_gpu == module:
            return

        if module_in_gpu is not None:
            module_in_gpu.to(cpu)

        module.to(gpu)
        module_in_gpu = module

    # see below for register_forward_pre_hook;
    # first_stage_model does not use forward(), it uses encode/decode, so register_forward_pre_hook is
    # useless here, and we just replace those methods
    def first_stage_model_encode_wrap(self, encoder, x):
        send_me_to_gpu(self, None)
        return encoder(x)

    def first_stage_model_decode_wrap(self, decoder, z):
        send_me_to_gpu(self, None)
        return decoder(z)

    # remove three big modules, cond, first_stage, and unet from the model and then
    # send the model to GPU. Then put modules back. the modules will be in CPU.
    stored = sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.model
    sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.model = None, None, None
    sd_model.to(device)
    sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.model = stored

    # register hooks for those the first two models
    sd_model.cond_stage_model.transformer.register_forward_pre_hook(send_me_to_gpu)
    sd_model.first_stage_model.register_forward_pre_hook(send_me_to_gpu)
    sd_model.first_stage_model.encode = lambda x, en=sd_model.first_stage_model.encode: first_stage_model_encode_wrap(sd_model.first_stage_model, en, x)
    sd_model.first_stage_model.decode = lambda z, de=sd_model.first_stage_model.decode: first_stage_model_decode_wrap(sd_model.first_stage_model, de, z)
    parents[sd_model.cond_stage_model.transformer] = sd_model.cond_stage_model

    if use_medvram:
        sd_model.model.register_forward_pre_hook(send_me_to_gpu)
    else:
        diff_model = sd_model.model.diffusion_model

        # the third remaining model is still too big for 4 GB, so we also do the same for its submodules
        # so that only one of them is in GPU at a time
        stored = diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = None, None, None, None
        sd_model.model.to(device)
        diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = stored

        # install hooks for bits of third model
        diff_model.time_embed.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.input_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)
        diff_model.middle_block.register_forward_pre_hook(send_me_to_gpu)
        for block in diff_model.output_blocks:
            block.register_forward_pre_hook(send_me_to_gpu)

```

# `/opt/to-comment/stable-diffusion-webui/modules/paths.py`

这段代码是一个 Python 脚本，用于在给定的目录中查找一个名为 "stable-diffusion" 的数据科学模型（即 MDP）的稳定扩散（即 DDP）文件。它使用了一些 Python 标准库函数和第三方库（如 argparse 和 os）。

具体来说，这段代码首先获取当前工作目录（即包含脚本和一些依赖库的目录），并将其作为参数传递给 argparse 库的parser对象。然后，它使用 os.path.dirname 和 os.path.join 函数查找给定目录中是否存在名为 "stable-diffusion" 的目录，以及是否存在名为 "ddpm.py" 的文件。如果存在，它将记录 sd_path，否则不会执行任何操作。

这段代码可能是在一个名为 "find_stable_diffusion" 的命令行工具中使用的。这个工具可能将用户指定一个目录，然后搜索该目录中是否存在指定的数据科学模型和相关的依赖文件。


```
import argparse
import os
import sys

script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, script_path)

# search for directory of stable diffsuion in following palces
sd_path = None
possible_sd_paths = ['.', os.path.dirname(script_path), os.path.join(script_path, 'repositories/stable-diffusion')]
for possible_sd_path in possible_sd_paths:
    if os.path.exists(os.path.join(possible_sd_path, 'ldm/models/diffusion/ddpm.py')):
        sd_path = os.path.abspath(possible_sd_path)

assert sd_path is not None, "Couldn't find Stable Diffusion in any of: " + possible_sd_paths

```

这段代码的作用是创建一个字典 `path_dirs`，其中包含 Stable Diffusion 项目的各个子目录。这些子目录的路径被存储在列表 `path_dirs` 中。

具体来说，这段代码首先定义了一个包含四个元素的列表 `path_dirs`，然后定义了一个字典 `paths`。接着，代码遍历 `path_dirs` 中的每个元素，其中每个元素包含一个路径名、一个必须存在的文件名和一个文件或项目名称。

对于每个 `path_dirs` 中的元素，代码首先尝试获取它的绝对路径。如果路径不存在，代码会打印一个警告消息。否则，代码会将 `path_dirs` 中的元素目录或文件名添加到 `paths` 字典中。这样，`paths` 字典中就会存储每个 `path_dirs` 元素所属的项目的路径。


```
path_dirs = [
    (sd_path, 'ldm', 'Stable Diffusion'),
    (os.path.join(sd_path, '../taming-transformers'), 'taming', 'Taming Transformers'),
    (os.path.join(sd_path, '../CodeFormer'), 'inference_codeformer.py', 'CodeFormer'),
]

paths = {}

for d, must_exist, what in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(script_path, d, must_exist))
    if not os.path.exists(must_exist_path):
        print(f"Warning: {what} not found at path {must_exist_path}", file=sys.stderr)
    else:
        d = os.path.abspath(d)
        sys.path.append(d)
        paths[what] = d

```

# `/opt/to-comment/stable-diffusion-webui/modules/processing.py`

这段代码的作用是实现一个图像处理工具，用于对图像进行各种处理，包括图像增强、图像压缩、图像分割等。

具体来说，这段代码实现了一个基于PyTorch的图像处理工具，可以对图像进行一些常见的操作，如调整图像大小、旋转图像、翻转图像、对图像进行二值化、对图像进行去噪等。

此外，这段代码还实现了一个图像复原工具，用于从噪声图像中恢复出原始图像。

整个工具箱包含以下几个模块：

1. PIL（Python Imaging Library）模块：用于处理图像，包括图像调整、图像处理等。
2. NumPy模块：用于支持PyTorch中的数学操作，如矩阵运算等。
3. Torch模块：用于支持PyTorch中的图像处理和机器学习操作，包括图像增强、图像分割等。
4. PIL Image模块：用于支持PIL中的图像处理操作，如调整图像大小、旋转图像等。
5. NumPy Image模块：用于支持NumPy中的图像处理操作，如对图像进行二值化、对图像进行去噪等。
6. Image模块：用于支持PyTorch中的图像处理操作，如对图像进行增强、对图像进行去噪等。
7. Common module：用于支持SD-Hijack模型的训练和评估，包括模型设置、损失函数等。
8. Utils module：用于支持工具箱中的其他工具类，如图像复原工具、图像查看工具等。


```
import contextlib
import json
import math
import os
import sys

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import random

import modules.sd_hijack
from modules.sd_hijack import model_hijack
from modules.sd_samplers import samplers, samplers_for_img2img
from modules.shared import opts, cmd_opts, state
```

这段代码的作用是定义了几个变量，包括 opt_C 和 opt_f，以及一个名为 torch_gc 的函数。函数内部使用了一些选项，但有一些选项被认为是错误的，因此被从函数中省略了。 opt_C 和 opt_f 变量可能是用来在训练过程中对图像进行处理和优化设置的。而 torch_gc 函数则是在图像生成时对图像进行一些预处理操作，以提高模型的效果。


```
import modules.shared as shared
import modules.face_restoration
import modules.images as images

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


```

这段代码定义了一个名为 `StableDiffusionProcessing` 的类，用于处理静态图像生成过程中的扩散模型。该类包含了许多方法，用于设置和控制生成的图像的一些方面，例如：

1. `__init__` 方法：用于初始化生成的配置参数，包括扩散模型、存储样本的文件路径、存储网格的文件路径、提示信息、随机种子等。
2. `sample` 方法：该方法负责生成图像。在样本生成过程中，需要提供扩散模型、条件图像和条件方程。该方法实现了样本生成的功能，但具体实现了一个抽象方法，需要子类实现该方法。

该类的作用是，定义了如何设置生成图像的参数，包括扩散模型、生成的样本数量、采样步长等，以及如何生成图像。通过调用 `StableDiffusionProcessing` 类中的 `sample` 方法，可以生成具有良好结构和连贯性的图像。


```
class StableDiffusionProcessing:
    def __init__(self, sd_model=None, outpath_samples=None, outpath_grids=None, prompt="", seed=-1, sampler_index=0, batch_size=1, n_iter=1, steps=50, cfg_scale=7.0, width=512, height=512, restore_faces=False, tiling=False, do_not_save_samples=False, do_not_save_grid=False, extra_generation_params=None, overlay_images=None, negative_prompt=None):
        self.sd_model = sd_model
        self.outpath_samples: str = outpath_samples
        self.outpath_grids: str = outpath_grids
        self.prompt: str = prompt
        self.prompt_for_display: str = None
        self.negative_prompt: str = (negative_prompt or "")
        self.seed: int = seed
        self.sampler_index: int = sampler_index
        self.batch_size: int = batch_size
        self.n_iter: int = n_iter
        self.steps: int = steps
        self.cfg_scale: float = cfg_scale
        self.width: int = width
        self.height: int = height
        self.restore_faces: bool = restore_faces
        self.tiling: bool = tiling
        self.do_not_save_samples: bool = do_not_save_samples
        self.do_not_save_grid: bool = do_not_save_grid
        self.extra_generation_params: dict = extra_generation_params
        self.overlay_images = overlay_images
        self.paste_to = None

    def init(self, seed):
        pass

    def sample(self, x, conditioning, unconditional_conditioning):
        raise NotImplementedError()


```

这段代码定义了一个名为 `Processed` 的类，用于表示在 `StableDiffusionProcessing` 类中进行图像处理时所需的参数。

在 `__init__` 方法中，首先创建一个包含图像列表、种子、处理信息以及图像宽度和高度的变量。然后，根据 `StableDiffusionProcessing` 类中指定的采样器和配置比例，创建一个 `js` 函数，将上述参数转换为 JSON 格式并返回。

具体来说，`self.images` 变量存储了输入的图像列表，`self.prompt` 变量存储了在 `StableDiffusionProcessing` 中使用的提示信息，`self.seed` 变量存储了用于种子选择的整数，`self.width` 和 `self.height` 变量用于存储图像的宽度和高度，`self.sampler` 变量用于指定图像采样器，`self.cfg_scale` 变量用于指定图像缩放的比例，`self.steps` 变量用于指定图像处理步骤。

在 `js` 函数中，将这些变量转换为字典类型，并将它们存储在一个名为 `obj` 的字典中。最后，使用 `json.dumps` 方法将 `obj` 对象转换为 JSON 格式的字符串，并返回这个字符串。这样，`Processed` 类就可以作为一个具有 `js` 函数的接口，用于在 `StableDiffusionProcessing` 中调用图像处理操作。


```
class Processed:
    def __init__(self, p: StableDiffusionProcessing, images_list, seed, info):
        self.images = images_list
        self.prompt = p.prompt
        self.seed = seed
        self.info = info
        self.width = p.width
        self.height = p.height
        self.sampler = samplers[p.sampler_index].name
        self.cfg_scale = p.cfg_scale
        self.steps = p.steps

    def js(self):
        obj = {
            "prompt": self.prompt if type(self.prompt) != list else self.prompt[0],
            "seed": int(self.seed if type(self.seed) != list else self.seed[0]),
            "width": self.width,
            "height": self.height,
            "sampler": self.sampler,
            "cfg_scale": self.cfg_scale,
            "steps": self.steps,
        }

        return json.dumps(obj)


```

这段代码定义了两个函数，create_random_tensors() 和 set_seed()。

create_random_tensors()函数的作用是创建一组随机的二维张量，其中每个张量的形状和大小由参数shape和seeds参数指定。函数的实现是通过循环遍历每个seeds并创建一个包含 shape 个随机实数的张量，然后将这些张量堆叠成一个大的张量x。最后，函数返回x。

set_seed()函数的作用是在给定一个种子（整数或浮点数）后，将其设置为当前设备的种子，并在创建随机张量时使用该种子。这样，每次运行函数时都会从同一个种子生成随机张量，使得不同的设备生成相同的张量。

综合来看，这段代码的作用是创建一组随机的二维张量，并使用给定的种子来控制每次运行时生成的随机张量。


```
def create_random_tensors(shape, seeds):
    xs = []
    for seed in seeds:
        torch.manual_seed(seed)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this so I do not dare change it for now because
        # it will break everyone's seeds.
        xs.append(torch.randn(shape, device=shared.device))
    x = torch.stack(xs)
    return x


def set_seed(seed):
    return int(random.randrange(4294967294)) if seed is None or seed == -1 else seed


```

This is a Python implementation of a sample processor that processes a list of images. It does this by first opening the image file, then it converts it to a NumPy array, and then it resizing it using the `Image.resize_image` method from the `PIL` module.

Next, it converts the image to an `Image` object and then it adds the image to the list of output images. If the `Image.paste_to` method is defined, it is then applied to the image, either by resizing it using the `Image.resize_image` method or by pasting the image onto the resized image.

Finally, it checks if the `opts.samples_save` option is True and if not, it converts the image to an RGB color and saves it using the `images.save_image` method from the `PIL` module. Additionally, it also checks if the `opts.grid_only_if_multiple` option is True or not and if not, it converts the image to a grid and saves it using the `images.image_grid` method from the `PIL` module.


```
def process_images(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    prompt = p.prompt

    assert p.prompt is not None
    torch_gc()

    seed = set_seed(p.seed)

    os.makedirs(p.outpath_samples, exist_ok=True)
    os.makedirs(p.outpath_grids, exist_ok=True)

    modules.sd_hijack.model_hijack.apply_circular(p.tiling)

    comments = []

    if type(prompt) == list:
        all_prompts = prompt
    else:
        all_prompts = p.batch_size * p.n_iter * [prompt]

    if type(seed) == list:
        all_seeds = seed
    else:
        all_seeds = [int(seed + x) for x in range(len(all_prompts))]

    def infotext(iteration=0, position_in_batch=0):
        generation_params = {
            "Steps": p.steps,
            "Sampler": samplers[p.sampler_index].name,
            "CFG scale": p.cfg_scale,
            "Seed": all_seeds[position_in_batch + iteration * p.batch_size],
            "Face restoration": (opts.face_restoration_model if p.restore_faces else None),
            "Batch size": (None if p.batch_size < 2 else p.batch_size),
            "Batch pos": (None if p.batch_size < 2 else position_in_batch),
        }

        if p.extra_generation_params is not None:
            generation_params.update(p.extra_generation_params)

        generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])

        return f"{p.prompt_for_display or prompt}\n{generation_params_text}".strip() + "".join(["\n\n" + x for x in comments])

    if os.path.exists(cmd_opts.embeddings_dir):
        model_hijack.load_textual_inversion_embeddings(cmd_opts.embeddings_dir, p.sd_model)

    output_images = []
    precision_scope = torch.autocast if cmd_opts.precision == "autocast" else contextlib.nullcontext
    ema_scope = (contextlib.nullcontext if cmd_opts.lowvram else p.sd_model.ema_scope)
    with torch.no_grad(), precision_scope("cuda"), ema_scope():
        p.init(seed=all_seeds[0])

        if state.job_count == -1:
            state.job_count = p.n_iter

        for n in range(p.n_iter):
            if state.interrupted:
                break

            prompts = all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            seeds = all_seeds[n * p.batch_size:(n + 1) * p.batch_size]

            uc = p.sd_model.get_learned_conditioning(len(prompts) * [p.negative_prompt])
            c = p.sd_model.get_learned_conditioning(prompts)

            if len(model_hijack.comments) > 0:
                comments += model_hijack.comments

            # we manually generate all input noises because each one should have a specific seed
            x = create_random_tensors([opt_C, p.height // opt_f, p.width // opt_f], seeds=seeds)

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            samples_ddim = p.sample(x=x, conditioning=c, unconditional_conditioning=uc)
            if state.interrupted:

                # if we are interruped, sample returns just noise
                # use the image collected previously in sampler loop
                samples_ddim = shared.state.current_latent

            x_samples_ddim = p.sd_model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.restore_faces:
                    torch_gc()

                    x_sample = modules.face_restoration.restore_faces(x_sample)

                image = Image.fromarray(x_sample)

                if p.overlay_images is not None and i < len(p.overlay_images):
                    overlay = p.overlay_images[i]

                    if p.paste_to is not None:
                        x, y, w, h = p.paste_to
                        base_image = Image.new('RGBA', (overlay.width, overlay.height))
                        image = images.resize_image(1, image, w, h)
                        base_image.paste(image, (x, y))
                        image = base_image

                    image = image.convert('RGBA')
                    image.alpha_composite(overlay)
                    image = image.convert('RGB')

                if opts.samples_save and not p.do_not_save_samples:
                    images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i))

                output_images.append(image)

            state.nextjob()

        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            return_grid = opts.return_grid

            grid = images.image_grid(output_images, p.batch_size)

            if return_grid:
                output_images.insert(0, grid)

            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", seed, all_prompts[0], opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename)

    torch_gc()
    return Processed(p, output_images, seed, infotext())


```



这段代码定义了一个名为 `StableDiffusionProcessingTxt2Img` 的类，其父类为 `StableDiffusionProcessing` 类。这个类的目的是在给定条件下的图像中进行采样，并将采样结果返回。

具体来说，这个类包含以下方法：

- `init`: 初始化方法，用于设置随机数种子并生成随机数数签。
- `sample`: 采样方法，用于根据给定的条件下的图像区域和 unconditional_conditioning 变量，从给定的 sampler 中采样一个整数。
- `get_crop_region`: 插件方法，用于根据给定的掩模图像和 crop 区域，返回一个裁剪区域。

这里 `StableDiffusionProcessingTxt2Img` 类中的方法都使用了给定的 sampler，因此这个类需要定义一个可以返回一个 sampler 的函数。在这个类中，`init` 方法用于设置 sampler,`sample` 方法用于返回采样结果，而 `get_crop_region` 方法则返回一个裁剪区域。


```
class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    sampler = None

    def init(self, seed):
        self.sampler = samplers[self.sampler_index].constructor(self.sd_model)

    def sample(self, x, conditioning, unconditional_conditioning):
        samples_ddim = self.sampler.sample(self, x, conditioning, unconditional_conditioning)
        return samples_ddim


def get_crop_region(mask, pad=0):
    h, w = mask.shape

    crop_left = 0
    for i in range(w):
        if not (mask[:, i] == 0).all():
            break
        crop_left += 1

    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:, i] == 0).all():
            break
        crop_right += 1

    crop_top = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        crop_top += 1

    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        crop_bottom += 1

    return (
        int(max(crop_left-pad, 0)),
        int(max(crop_top-pad, 0)),
        int(min(w - crop_right + pad, w)),
        int(min(h - crop_bottom + pad, h))
    )


```

这段代码定义了一个名为 `fill` 的函数，它接受两个参数 `image` 和 `mask`，并将它们转换成一个新的 `Image` 对象。接下来，函数创建一个新的 `Image` 对象 `image_mod`，并创建一个新的 `Image` 对象 `image_masked`，它使用 `image.convert("RGBA").convert("RGBa")` 将图像转换为 `RGBa` 格式，并使用给定的掩码图像（也就是 `mask`）的 `convert("L")` 方法从图像中提取像素。

然后，函数使用 `image_masked.convert("RGBa")` 将图像转换为 `RGBa` 格式。接下来，函数遍历一系列不同的半径值，每次使用 `image_masked.filter(ImageFilter.GaussianBlur).convert("RGBA")` 将图像模糊化，并将模糊后的图像与之前的图像做逐像素混合，通过 `image_mod.alpha_composite()` 方法将透明度设置为 0（完全透明）。

最后，函数将 `image_mod.convert("RGB")` 并将图像返回。


```
def fill(image, mask):
    image_mod = Image.new('RGBA', (image.width, image.height))

    image_masked = Image.new('RGBa', (image.width, image.height))
    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert('L')))

    image_masked = image_masked.convert('RGBa')

    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert('RGBA')
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)

    return image_mod.convert("RGB")


```

This is a PyTorch implementation of an image reinforcement learning (IRL) algorithm for generating realistic images. It uses a hierarchical dynamic sleep- wakefulness (SDW) model, where the encoder is in the high-dimensional space, and the decoder is in the low-dimensional space. The image is processed in the low-dimensional space, and the sleep-wake state is represented by a binary mask.

The SDW model consists of multiple stages. The first stage has an encoder that maps the input image to a latent space, and a decoder that maps the latent space to a fixed-size image. The second stage has an encoder that maps the input image to a lower-dimensional latent space, and a decoder that maps the lower-dimensional latent space to a fixed-size image. The third stage has an encoder that maps the input image to a higher-dimensional latent space, and a decoder that maps the higher-dimensional latent space to a fixed-size image.

The image is processed in the first and second stages by the encoders, and in the third stage by the decoder. The SDW model takes a batch of images as input, and generates a batch of images as output. The batch size, image dimensions, and other hyperparameters are set in the `ImageReinforcementLearning` class constructor.

The `ImageReinforcementLearning` class also has a method called `sample`, which applies the SDW model to an input image and generates a sample image. This sample image can be used for further processing or evaluation.

The code also includes some additional details for visualization and debugging, such as the display of the original image and the binary mask, and the use of numpy arrays instead of PIL Image objects.


```
class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    sampler = None

    def __init__(self, init_images=None, resize_mode=0, denoising_strength=0.75, mask=None, mask_blur=4, inpainting_fill=0, inpaint_full_res=True, inpainting_mask_invert=0, **kwargs):
        super().__init__(**kwargs)

        self.init_images = init_images
        self.resize_mode: int = resize_mode
        self.denoising_strength: float = denoising_strength
        self.init_latent = None
        self.image_mask = mask
        #self.image_unblurred_mask = None
        self.latent_mask = None
        self.mask_for_overlay = None
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.inpaint_full_res = inpaint_full_res
        self.inpainting_mask_invert = inpainting_mask_invert
        self.mask = None
        self.nmask = None

    def init(self, seed):
        self.sampler = samplers_for_img2img[self.sampler_index].constructor(self.sd_model)
        crop_region = None

        if self.image_mask is not None:
            self.image_mask = self.image_mask.convert('L')

            if self.inpainting_mask_invert:
                self.image_mask = ImageOps.invert(self.image_mask)

            #self.image_unblurred_mask = self.image_mask

            if self.mask_blur > 0:
                self.image_mask = self.image_mask.filter(ImageFilter.GaussianBlur(self.mask_blur))

            if self.inpaint_full_res:
                self.mask_for_overlay = self.image_mask
                mask = self.image_mask.convert('L')
                crop_region = get_crop_region(np.array(mask), opts.upscale_at_full_resolution_padding)
                x1, y1, x2, y2 = crop_region

                mask = mask.crop(crop_region)
                self.image_mask = images.resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2-x1, y2-y1)
            else:
                self.image_mask = images.resize_image(self.resize_mode, self.image_mask, self.width, self.height)
                np_mask = np.array(self.image_mask)
                np_mask = np.clip((np_mask.astype(np.float)) * 2, 0, 255).astype(np.uint8)
                self.mask_for_overlay = Image.fromarray(np_mask)

            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else self.image_mask

        imgs = []
        for img in self.init_images:
            image = img.convert("RGB")

            if crop_region is None:
                image = images.resize_image(self.resize_mode, image, self.width, self.height)

            if self.image_mask is not None:
                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            if crop_region is not None:
                image = image.crop(crop_region)
                image = images.resize_image(2, image, self.width, self.height)

            if self.image_mask is not None:
                if self.inpainting_fill != 1:
                    image = fill(image, latent_mask)

            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)

            imgs.append(image)

        if len(imgs) == 1:
            batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
            if self.overlay_images is not None:
                self.overlay_images = self.overlay_images * self.batch_size
        elif len(imgs) <= self.batch_size:
            self.batch_size = len(imgs)
            batch_images = np.array(imgs)
        else:
            raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

        image = torch.from_numpy(batch_images)
        image = 2. * image - 1.
        image = image.to(shared.device)

        self.init_latent = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(image))

        if self.image_mask is not None:
            init_mask = latent_mask
            latmask = init_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float64), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (4, 1, 1))

            self.mask = torch.asarray(1.0 - latmask).to(shared.device).type(self.sd_model.dtype)
            self.nmask = torch.asarray(latmask).to(shared.device).type(self.sd_model.dtype)

            if self.inpainting_fill == 2:
                self.init_latent = self.init_latent * self.mask + create_random_tensors(self.init_latent.shape[1:], [seed + x + 1 for x in range(self.init_latent.shape[0])]) * self.nmask
            elif self.inpainting_fill == 3:
                self.init_latent = self.init_latent * self.mask

    def sample(self, x, conditioning, unconditional_conditioning):
        samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning)

        if self.mask is not None:
            samples = samples * self.nmask + self.init_latent * self.mask

        return samples

```

# `/opt/to-comment/stable-diffusion-webui/modules/realesrgan_model.py`

这段代码的作用是实现了一个图像生成模型，属于Realesrgan系列。以下是对代码的详细解释：

1. 导入需要用到的库：`sys`、`traceback`、`collections`、`namedtuple`、`numpy`、`PIL`。

2. 从`collections`中导入`namedtuple`，创建了一个名为`RealesrganModelInfo`的类，包含以下属性：`name`、`location`、`model`、`netscale`。

3. 从`PIL`中导入`Image`，以便于操作图像。

4. 导入另一个名为`RealesrganModelInfo`的类，可能用于在运行时设置选项。

5. 定义了一个名为`realesrgan_models`的列表，用于存储Realesrgan模型的实例。

6. 定义了一个名为`have_realesrgan`的布尔值，初始值为False。

7. 定义了一个名为`RealESRGANer_constructor`的函数，该函数可能用于创建Realesrgan模型实例。

8. 由于在之前定义了`have_realesrgan`为False，所以这个函数不会被调用。

9. 在代码的最后部分，可能还有一些其他代码，但无法确定，因为本回答只提供了部分代码。


```
import sys
import traceback
from collections import namedtuple
import numpy as np
from PIL import Image

import modules.images
from modules.shared import cmd_opts, opts

RealesrganModelInfo = namedtuple("RealesrganModelInfo", ["name", "location", "model", "netscale"])

realesrgan_models = []
have_realesrgan = False
RealESRGANer_constructor = None


```

It seems like the code is trying to import the Real-ESRGAN model and its variants. The code uses the SD-upscalers library to improve the performance of the models.

The first part of the code imports the Real-ESRGAN model and defines some constants. The netscale parameter is set to 4 for the first model, and the model is defined as a functional block with 3 input channels, 3 output channels, and 64 feature points. The number of blocks and the growth factor are also defined.

The next part of the code imports the Realesrgan model and its information. The name and location of the Realesrgan model are defined, and the model is defined as a functional block with 3 input channels, 3 output channels, and 64 feature points.

Finally, the have\_realesrgan variable is set to True, indicating that the Realesrgan model has been imported. Also, the RealESRGANer class is defined, which seems to be the constructor for the Real-ESRGAN model.

It's worth noting that there are several versions of the Real-ESRGAN model defined in the code, such as "4x plus anime 6B", "2x plus", and "anime 6B". It seems that these models have different specifications, such as the number of input and output channels, the feature points, and the number of blocks.


```
class UpscalerRealESRGAN(modules.images.Upscaler):
    def __init__(self, upscaling, model_index):
        self.upscaling = upscaling
        self.model_index = model_index
        self.name = realesrgan_models[model_index].name

    def do_upscale(self, img):
        return upscale_with_realesrgan(img, self.upscaling, self.model_index)


def setup_realesrgan():
    global realesrgan_models
    global have_realesrgan
    global RealESRGANer_constructor

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact

        realesrgan_models = [
            RealesrganModelInfo(
                name="Real-ESRGAN 4x plus",
                location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                netscale=4, model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            ),
            RealesrganModelInfo(
                name="Real-ESRGAN 4x plus anime 6B",
                location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                netscale=4, model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            ),
            RealesrganModelInfo(
                name="Real-ESRGAN 2x plus",
                location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                netscale=2, model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            ),
        ]
        have_realesrgan = True
        RealESRGANer_constructor = RealESRGANer

        for i, model in enumerate(realesrgan_models):
            modules.shared.sd_upscalers.append(UpscalerRealESRGAN(model.netscale, i))

    except Exception:
        print("Error importing Real-ESRGAN:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

        realesrgan_models = [RealesrganModelInfo('None', '', 0, None)]
        have_realesrgan = False


```

这段代码定义了一个名为 `upscale_with_realesrgan` 的函数，它接受一个 `image` 图像和一个 `RealESRGAN_upscaling` 参数，以及一个 `RealESRGAN_model_index` 参数。

函数首先检查是否支持 RealESRGAN 和 RealESRGANer，如果其中任何一个不支持，那么函数将直接返回输入的 `image` 图像。

接下来，函数将访问 RealESRGAN 模型信息，并从该模型中获取一个模型对象。然后，函数使用获取的模型对象创建一个名为 `upsampler` 的对象，该对象使用传入的 `scale`、`model_path`、`model` 和 `half` 参数进行调整。接着，函数使用 `enhance` 方法对输入的 `image` 图像进行增强，并将其返回值存储在 `upsampled` 变量中。

最后，函数创建一个名为 `image` 的新的 `Image` 对象，并将 `upsampled` 图像转换为 `image` 对象，返回 `image`。


```
def upscale_with_realesrgan(image, RealESRGAN_upscaling, RealESRGAN_model_index):
    if not have_realesrgan or RealESRGANer_constructor is None:
        return image

    info = realesrgan_models[RealESRGAN_model_index]

    model = info.model()
    upsampler = RealESRGANer_constructor(
        scale=info.netscale,
        model_path=info.location,
        model=model,
        half=not cmd_opts.no_half,
        tile=opts.ESRGAN_tile,
        tile_pad=opts.ESRGAN_tile_overlap,
    )

    upsampled = upsampler.enhance(np.array(image), outscale=RealESRGAN_upscaling)[0]

    image = Image.fromarray(upsampled)
    return image

```

# `/opt/to-comment/stable-diffusion-webui/modules/scripts.py`



这是一个Python脚本，其作用是定义一个名为`Script`的类，该类用于实现图像处理和机器学习相关任务。具体来说，该类包含以下方法：

- `__init__`方法，用于初始化该类的实例变量。
- `import os`、`import sys`和`import traceback`方法，用于导入操作系统、系统和调试信息的包。
- `import gradio`方法，用于导入机器学习库`gradio`。
- `from modules.ui import ui`和`from modules.processing import StableDiffusionProcessing`方法的引用，表明该类从`modules.ui`和`modules.processing`类中获取相关的方法。
- `__len__`方法，用于返回该类实例对象的个数。
- `__getitem__`方法，用于返回该类实例对象的一个元素，例如索引为0的元素。
- `__setitem__`方法，用于设置该类实例对象的某个元素，例如索引为0的元素。
- `__repr__`方法，用于返回该类实例对象的表示。
- `__str__`方法，用于返回该类实例对象的格式的字符串表示。
- `__len__`方法，用于返回该类实例对象的内容数量。
- `__getitem__`方法，用于返回该类实例对象的一个元素，例如索引为0的元素。
- `__setitem__`方法，用于设置该类实例对象的某个元素，例如索引为0的元素。
- `__repr__`方法，用于返回该类实例对象的格式的字符串表示。
- `__str__`方法，用于返回该类实例对象的格式的字符串表示。
- `__len__`方法，用于返回该类实例对象的内容数量。
- `__getitem__`方法，用于返回该类实例对象的一个元素，例如索引为0的元素。
- `__setitem__`方法，用于设置该类实例对象的某个元素，例如索引为0的元素。


```
import os
import sys
import traceback

import modules.ui as ui
import gradio as gr

from modules.processing import StableDiffusionProcessing
from modules import shared

class Script:
    filename = None
    args_from = None
    args_to = None

    def title(self):
        raise NotImplementedError()

    def ui(self, is_img2img):
        pass

    def show(self, is_img2img):
        return True

    def run(self, *args):
        raise NotImplementedError()

    def describe(self):
        return ""


```

这段代码定义了一个名为 `load_scripts` 的函数，它接受一个参数 `basedir`，表示要加载的文件所在的目录。

该函数首先检查传入的目录是否存在，如果不存在，则返回。

接下来，该函数遍历目录中的所有文件名，对于每个文件，它首先检查文件是否存在，如果文件存在，则尝试读取文件内容。

如果文件内容是可读的，该函数将文件路径和文件名存储在一个名为 `scripts_data` 的列表中。如果文件内容是二进制文件，该函数将抛出异常并打印错误消息。

此外，该函数使用 `compile` 函数将文件内容编译成可执行代码，并使用 `ModuleType` 类将模块定义为 `ModuleType` 类，以便可以访问模块中的属性。在循环中，该函数使用 `__dict__` 获取模块中的所有属性，并使用 `type` 函数检查给定的属性类型是否为 `Script` 类。如果是，该函数将该属性值存储在 `scripts_data` 列表中。


```
scripts_data = []


def load_scripts(basedir):
    if not os.path.exists(basedir):
        return

    for filename in os.listdir(basedir):
        path = os.path.join(basedir, filename)

        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf8") as file:
            text = file.read()

        try:
            from types import ModuleType
            compiled = compile(text, path, 'exec')
            module = ModuleType(filename)
            exec(compiled, module.__dict__)

            for key, script_class in module.__dict__.items():
                if type(script_class) == type and issubclass(script_class, Script):
                    scripts_data.append((script_class, path))

        except Exception:
            print(f"Error loading script: {filename}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)


```

This appears to be a Python script that allows users to select and run scripts for processing image data. It has a dropdown menu for selecting a script and a button for running the selected script.

When a user selects a script, the script's code is executed on the image data with the processing specified in the script. The processed image data is then shown using the `gr.Dropdown` and `wrap_call` functions.

The script also handles command line arguments passed in when the script is run.


```
def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        res = func(*args, **kwargs)
        return res
    except Exception:
        print(f"Error calling: {filename}/{funcname}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

    return default


class ScriptRunner:
    def __init__(self):
        self.scripts = []

    def setup_ui(self, is_img2img):
        for script_class, path in scripts_data:
            script = script_class()
            script.filename = path

            if not script.show(is_img2img):
                continue

            self.scripts.append(script)

        titles = [wrap_call(script.title, script.filename, "title") or f"{script.filename} [error]" for script in self.scripts]

        dropdown = gr.Dropdown(label="Script", choices=["None"] + titles, value="None", type="index")
        inputs = [dropdown]

        for script in self.scripts:
            script.args_from = len(inputs)

            controls = wrap_call(script.ui, script.filename, "ui", is_img2img)

            if controls is None:
                continue

            for control in controls:
                control.visible = False

            inputs += controls
            script.args_to = len(inputs)

        def select_script(script_index):
            if 0 < script_index <= len(self.scripts):
                script = self.scripts[script_index-1]
                args_from = script.args_from
                args_to = script.args_to
            else:
                args_from = 0
                args_to = 0

            return [ui.gr_show(True if i == 0 else args_from <= i < args_to) for i in range(len(inputs))]

        dropdown.change(
            fn=select_script,
            inputs=[dropdown],
            outputs=inputs
        )

        return inputs


    def run(self, p: StableDiffusionProcessing, *args):
        script_index = args[0]

        if script_index == 0:
            return None

        script = self.scripts[script_index-1]

        if script is None:
            return None

        script_args = args[script.args_from:script.args_to]
        processed = script.run(p, *script_args)

        shared.total_tqdm.clear()

        return processed


```

这段代码是在运行两个相同的函数ScriptRunner()，并将返回值分别存储在变量Scripts_txt2img和Scripts_img2img中。这样做是为了创建一个副本，以便在需要时可以再次使用之前定义的函数。

在这个例子中，ScriptRunner()是一个装饰器，用于运行一段Python代码并返回其结果。由于你需要运行相同的代码，因此你可以将上述函数赋予一个名称（如generate_images），并将其结果存储在需要时通过调用该函数获取。


```
scripts_txt2img = ScriptRunner()
scripts_img2img = ScriptRunner()

```