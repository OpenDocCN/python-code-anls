# `.\comic-translate\modules\ocr\pororo\pororo\utils.py`

```py
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from requests import get


def postprocess_span(tagger, text: str) -> str:
    """
    Postprocess NOUN span to remove unnecessary character

    Args:
        text (str): NOUN span to be processed

    Returns:
        (str): post-processed NOUN span

    Examples:
        >>> postprocess_span("강감찬 장군은")
        '강감찬 장군'
        >>> postprocess_span("그녀에게")
        '그녀'

    """

    # 首先，去除标点符号
    text = text.strip("""!"\#$&'()*+,\-./:;<=>?@\^_‘{|}~《》""")

    # 完成不平衡的括号对
    if text.count("(") == text.count(")") + 1:
        text += ")"
    elif text.count("(") + 1 == text.count(")"):
        text = "(" + text

    # 保留开头的标记，因为我们只想提取最后一个어절的名词短语
    noun_phrase = " ".join(text.rsplit(" ", 1)[:-1])
    tokens = text.split(" ")
    eojeols = list()
    for token in tokens:
        eojeols.append(tagger.pos(token))
    last_eojeol = eojeols[-1]

    # 逆向迭代以去除不必要的后缀
    i = 0
    for i, token in enumerate(last_eojeol[::-1]):
        _, pos = token
        # 1. 当遇到名词时，循环中断
        # 2. 当遇到XSN时（例如 8/SN+일/NNB LG/SL 전/XSN），循环也会中断
        if (pos[0] in ("N", "S")) or pos.startswith("XSN"):
            break
    idx = len(last_eojeol) - i

    # 从最后一个어절中提取名词短语，并将其追加到开头的标记
    ext_last_eojeol = "".join(morph for morph, _ in last_eojeol[:idx])
    noun_phrase += " " + ext_last_eojeol
    return noun_phrase.strip()


@contextmanager
def control_temp(file_path: str):
    """
    Download temporary file from web, then remove it after some context

    Args:
        file_path (str): web file path

    """
    # yapf: disable
    assert file_path.startswith("http"), "File path should contain `http` prefix !"
    # yapf: enable

    ext = file_path[file_path.rfind("."):]

    with NamedTemporaryFile("wb", suffix=ext, delete=True) as f:
        response = get(file_path, allow_redirects=True)
        f.write(response.content)
        yield f.name
```