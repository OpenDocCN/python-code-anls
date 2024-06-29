# `D:\src\scipysrc\pandas\scripts\sort_whatsnew_note.py`

```
"""
Sort whatsnew note blocks by issue number.

NOTE: this assumes that each entry is on its own line, and ends with an issue number.
If that's not the case, then an entry might not get sorted. However, virtually all
recent-enough whatsnew entries follow this pattern. So, although not perfect, this
script should be good enough to significantly reduce merge conflicts.

For example:

- Fixed bug in resample (:issue:`321`)
- Fixed bug in groupby (:issue:`123`)

would become

- Fixed bug in groupby (:issue:`123`)
- Fixed bug in resample (:issue:`321`)

The motivation is to reduce merge conflicts by reducing the chances that multiple
contributors will edit the same line of code.

You can run this manually with

    pre-commit run sort-whatsnew-items --all-files
"""

from __future__ import annotations  # Enable annotations for forward references

import argparse  # Module for parsing command-line arguments
import re  # Regular expression operations
import sys  # System-specific parameters and functions
from typing import TYPE_CHECKING  # Type hinting support

if TYPE_CHECKING:
    from collections.abc import Sequence  # Import for type hinting

# Check line starts with `-` and ends with e.g. `(:issue:`12345`)`,
# possibly with a trailing full stop.
pattern = re.compile(r"-.*\(:issue:`(\d+)`\)\.?$")


def sort_whatsnew_note(content: str) -> int:
    new_lines = []  # Initialize an empty list to store sorted lines
    block: list[str] = []  # Initialize an empty list to collect lines of each block
    lines = content.splitlines(keepends=True)  # Split content into lines keeping line endings
    for line in lines:
        if line.startswith("- ") and pattern.search(line) is not None:
            block.append(line)  # Add lines matching the pattern to the current block
        else:
            block = sorted(block)  # Sort the block by issue number
            new_lines.extend(block)  # Extend new_lines with sorted block
            new_lines.append(line)  # Add current line after sorted block
            block = []  # Reset block for the next iteration
    if sorted(new_lines) != sorted(lines):  # pragma: no cover
        # Defensive check - this script should only reorder lines, not modify any
        # content.
        raise AssertionError(
            "Script modified content of file. Something is wrong, please don't "
            "trust it."
        )
    return "".join(new_lines)  # Return sorted content as a single string


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()  # Create an argument parser object
    parser.add_argument("paths", nargs="*")  # Define positional argument 'paths'
    args = parser.parse_args(argv)  # Parse command-line arguments
    ret = 0  # Initialize return code variable
    for path in args.paths:  # Iterate over paths from command-line arguments
        with open(path, encoding="utf-8") as fd:  # Open file for reading
            content = fd.read()  # Read entire file content
        new_content = sort_whatsnew_note(content)  # Sort whatsnew notes in the file
        if content != new_content:  # If content has changed
            ret |= 1  # Set return code to indicate changes
            with open(path, "w", encoding="utf-8") as fd:  # Open file for writing
                fd.write(new_content)  # Write sorted content back to file
    return ret  # Return final return code


if __name__ == "__main__":
    sys.exit(main())  # Run main function and exit with its return code
```