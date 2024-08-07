# `.\DB-GPT-src\dbgpt\util\splitter_utils.py`

```py
from typing import Callable, List


def split_text_keep_separator(text: str, separator: str) -> List[str]:
    """Split text with separator and keep the separator at the end of each split."""
    # Split the text into parts using the specified separator
    parts = text.split(separator)
    # Ensure that each part retains the separator at the end (except the first part)
    result = [separator + s if i > 0 else s for i, s in enumerate(parts)]
    # Filter out any empty strings from the result list
    return [s for s in result if s]


def split_by_sep(sep: str, keep_sep: bool = True) -> Callable[[str], List[str]]:
    """Split text by separator."""
    if keep_sep:
        # Return a lambda function that splits text while preserving the separator
        return lambda text: split_text_keep_separator(text, sep)
    else:
        # Return a lambda function that splits text without preserving the separator
        return lambda text: text.split(sep)


def split_by_char() -> Callable[[str], List[str]]:
    """Split text by character."""
    # Return a lambda function that converts text into a list of characters
    return lambda text: list(text)


def split_by_sentence_tokenizer() -> Callable[[str], List[str]]:
    import os
    import nltk
    from llama_index.utils import get_cache_dir

    cache_dir = get_cache_dir()
    nltk_data_dir = os.environ.get("NLTK_DATA", cache_dir)

    # Update nltk path to include the specified data directory
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        # Download the 'punkt' data if not found
        nltk.download("punkt", download_dir=nltk_data_dir)

    # Create a PunktSentenceTokenizer instance for sentence tokenization
    tokenizer = nltk.tokenize.PunktSentenceTokenizer()

    # Define a function to split text into sentences using the tokenizer
    def split(text: str) -> List[str]:
        # Get spans of sentences in the text
        spans = list(tokenizer.span_tokenize(text))
        sentences = []
        for i, span in enumerate(spans):
            start = span[0]
            if i < len(spans) - 1:
                end = spans[i + 1][0]
            else:
                end = len(text)
            # Extract each sentence using the start and end spans
            sentences.append(text[start:end])

        return sentences

    return split


def split_by_regex(regex: str) -> Callable[[str], List[str]]:
    """Split text by regex."""
    import re

    # Return a lambda function that splits text using the provided regex
    return lambda text: re.findall(regex, text)


def split_by_phrase_regex() -> Callable[[str], List[str]]:
    """Split text by phrase regex.

    This regular expression will split the sentences into phrases,
    where each phrase is a sequence of one or more non-comma,
    non-period, and non-semicolon characters, followed by an optional comma,
    period, or semicolon. The regular expression will also capture the
    delimiters themselves as separate items in the list of phrases.
    """
    regex = "[^,.;。]+[,.;。]?"
    # Return the result of calling split_by_regex with the defined phrase regex
    return split_by_regex(regex)
```