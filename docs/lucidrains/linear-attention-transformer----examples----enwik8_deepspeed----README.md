# Linear Attention Transformer with Deepspeed for Enwik8

Deepspeed is the framework Microsoft used to train the world's largest Attention model (17GB) to date.

1. First install Deepspeed following instructions from their official repository https://github.com/microsoft/DeepSpeed

2. Run the following command in this folder

```py
$ deepspeed train.py --deepspeed --deepspeed_config ds_config.json
```
