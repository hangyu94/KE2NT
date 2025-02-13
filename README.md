# KE2NT

## Requirements

We test the source code in the following environments:

- Python==3.7.16
- PyTorch==1.8.1
- Torchvision==0.9.1


## Dataset

Change the path to RAF-DB like the following:

```
├── data
│   └── raf_db
│       ├── Image
│       ├── train_label.txt
│       └── test_label.txt
```


## Training

To train on RAF-DB:

```bash
CUDA_VISIBLE_DEVICES=0 python swint_raf_main.py
```
