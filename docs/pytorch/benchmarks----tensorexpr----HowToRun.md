From the root of pytorch repo, run:
```py
python -m benchmarks.tensorexpr --help
```
to show documentation.

An example of an actual command line that one might use as a starting point:
```py
python -m benchmarks.tensorexpr --device gpu --mode fwd --jit-mode trace --cuda-fuser=te
```
