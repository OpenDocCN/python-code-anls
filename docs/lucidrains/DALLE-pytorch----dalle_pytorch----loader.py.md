# `.\lucidrains\DALLE-pytorch\dalle_pytorch\loader.py`

```py
from pathlib import Path
from random import randint, choice
import PIL
from torch.utils.data import Dataset
from torchvision import transforms as T

class TextImageDataset(Dataset):
    def __init__(self,
                 folder,
                 text_len=256,
                 image_size=128,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 transparent=False,
                 tokenizer=None,
                 shuffle=False
                 ):
        """
        @param folder: 包含图像和文本文件的文件夹，它们通过其路径的相应“stem”匹配
        @param truncate_captions: 如果标题太长，将截断标题而不是抛出异常
        """
        super().__init__()
        self.shuffle = shuffle
        path = Path(folder)

        # 获取所有文本文件和图像文件的路径
        text_files = [*path.glob('**/*.txt')]
        image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ]

        # 将文本文件和图像文件的stem作为key，文件路径作为value存储在字典中
        text_files = {text_file.stem: text_file for text_file in text_files}
        image_files = {image_file.stem: image_file for image_file in image_files}

        # 获取文本文件和图像文件stem的交集作为keys
        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer

        # 根据是否透明设置图像模式
        image_mode = 'RGBA' if transparent else 'RGB'

        # 图像转换操作
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert(image_mode)
            if img.mode != image_mode else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.keys[ind]

        text_file = self.text_files[key]
        image_file = self.image_files[key]

        # 读取文本文件内容并按换行符分割
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # 对文本进行标记化处理
        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # 成功返回标记化的文本和图像张量
        return tokenized_text, image_tensor
```