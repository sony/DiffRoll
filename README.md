- __Demo__: https://sony.github.io/DiffRoll/
- __Paper__: https://arxiv.org/abs/2210.05148

# Table of Content
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=4 orderedList=false} -->

<!-- code_chunk_output -->
- [Installation](#installation)
- [Table of Content](#table-of-content)
- [Installation](#installation)
- [Training](#training)
  - [Supervised training](#supervised-training)
  - [Unsupervised pretraining](#unsupervised-pretraining)
    - [Step 1: Pretraining on MAESTRO using only piano rolls](#step-1-pretraining-on-maestro-using-only-piano-rolls)
    - [Step 2](#step-2)
      - [Option A: pre-DiffRoll (p=0.1)](#option-a-pre-diffroll-p01)
      - [Option B: pre-DiffRoll (p=0+1)](#option-b-pre-diffroll-p01)
      - [Option C: MAESTRO 0.1](#option-c-maestro-01)
- [Sampling](#sampling)
  - [Transcription](#transcription)
  - [Inpainting](#inpainting)
  - [Generation](#generation)

<!-- /code_chunk_output -->


# Installation
This repo is developed using `python==3.8.10`, so it is recommended to use `python>=3.8.10`.

To install all dependencies
```
pip install -r requirements.txt
```

# Training

## Supervised training
```
python train_spec_roll.py gpus=[0] model.args.kernel_size=9 model.args.spec_dropout=0.1 dataset=MAESTRO dataloader.train.num_workers=4 epochs=2500 download=True
```


- `gpus` sets which GPU to use. `gpus=[k]` means `device='cuda:k'`, `gpus=2` means [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) (DDP) is used with two GPUs.
- `model.args.kernel_size` sets the kernel size for the ResNet layers in DiffRoll. `model.args.kernel_size=9` performs the best according to our experiments.
- `model.args.spec_dropout` sets the dropout rate ($p$ in the paper)
- `dataset` sets the dataset to be trained on. Can be `MAESTRO` or `MAPS`.
- `dataloader.train.num_workers` sets the number of workers for train loader.
- `download` should be set to `True` if you are running the script for the first time to download and setup the dataset automatically. You can set it to `False` if you already have the dataset downloaded.

The checkpoints and training logs are avaliable at `outputs/YYYY-MM-DD/HH-MM-SS/`. 

To check the progress of training using TensorBoard, you can use the command below
```
tensorboard --logdir='./outputs'
```

## Unsupervised pretraining
### Step 1: Pretraining on MAESTRO using only piano rolls
```
python train_spec_roll.py gpus=[0] model.args.kernel_size=9 model.args.spec_dropout=1 dataset=MAESTRO dataloader.train.num_workers=4 epochs=2500
```

- `model.args.spec_dropout` sets the dropout rate ($p$ in the paper). When it is set to `1`, it means no spectrograms will be used (all spectrograms dropped to `-1`)
- other arguments are same as [Supervised Training](#supervised-training).

The pretrained checkpoints are avaliable at `outputs/YYYY-MM-DD/HH-MM-SS/ClassifierFreeDiffRoll/version_1/checkpoints`.

After this, you can choose one of the options ([2A](#option-a-pre-diffroll-p01), [2B](#option-b-pre-diffroll-p01), or [2C](#option-c-maestro-01)) to continue training below.


### Step 2
Choose one of the options below ([A](#option-a-pre-diffroll-p01), [B](#option-b-pre-diffroll-p01), or [C](#option-c-maestro-01)).
#### Option A: pre-DiffRoll (p=0.1)

```
python continue_train_single.py gpus=[0] model.args.kernel_size=9 model.args.spec_dropout=0.1 dataset=MAPS dataloader.train.num_workers=4 epochs=10000 pretrained_path='path_to_your_weights' 
```

- `pretrained_path` specifies the location of pretrained weights obtained in [Step 1](#step-1-pretraining-on-maestro-using-only-piano-rolls)
- other arguments are same as [Supervised Training](#supervised-training).


#### Option B: pre-DiffRoll (p=0+1)

```
python continue_train_both.py gpus=[0] model.args.kernel_size=9 model.args.spec_dropout=0 dataset=Both dataloader.train.num_workers=4epochs=10000 pretrained_path='path_to_your_weights' 
```

- `pretrained_path` specifies the location of pretrained weights obtained in [Step 1](#step-1-pretraining-on-maestro-using-only-piano-rolls)
- `model.args.spec_dropout` controls the dropout for the MAPS dataset. The MAESTRO dataset is always set to p=-1. 
- other arguments are same as [Supervised Training](#supervised-training).

#### Option C: MAESTRO 0.1
This option is not reported in the paper, but it is the best.

```
python continue_train_single.py gpus=[0] model.args.kernel_size=9 model.args.spec_dropout=0 dataset=MAESTRO dataloader.train.num_workers=4 epochs=2500 pretrained_path='path_to_your_weights' 
```

- `pretrained_path` specifies the location of pretrained weights obtained in [Step 1](#step-1-pretraining-on-maestro-using-only-piano-rolls)
- other arguments are same as [Supervised Training](#supervised-training).

# Testing
The training script above already includes the testing. This section is for you to re-run the test set and get the transcription score.

First, open `config/test.yaml`, and then specify the weight to use in `checkpoint_path`.

For example, if you want to use `Pretrain_MAESTRO-retrain_Both-k=9.ckpt`, then set  `checkpoint_path='weights/Pretrain_MAESTRO-retrain_Both-k=9.ckpt'`.

You can download pretrained weights from [Zenodo](https://zenodo.org/record/7214252#.Y00_xUzP260). After downloading, put them inside the folder `weights`.

```
python test.py gpus=[0] dataset=MAPS
```

- `dataset` sets the dataset to be trained on. Can be `MAESTRO` or `MAPS`.

# Sampling
You can download pretrained weights from [Zenodo](https://zenodo.org/record/7214252#.Y00_xUzP260). After downloading, put them inside the folder `weights`.

The folder `my_audio` already includes four samples as a demonstration. You can put your own audio clips inside this folder.

## Transcription
This script supports only transcribing music from either MAPS or MAESTRO.

TODO: add support for transcribing any music

First, open `config/test.yaml`, and then specify the weight to use in `checkpoint_path`.

For example, if you want to use `Pretrain_MAESTRO-retrain_Both-k=9.ckpt`, then set  `checkpoint_path='weights/Pretrain_MAESTRO-retrain_Both-k=9.ckpt'`.

```
python sampling.py task=transcription dataloader.batch_size=4 dataset=Custom dataset.args.audio_ext=mp3 dataset.args.max_segment_samples=327680 gpus=[0]
```

- `dataloader.batch_size` sets the batch size. You can set a higher number if your GPU has enough memory.
- `dataset` when setting to `Custom`, it load audio clips from the folder `my_audio`.
- `dataset.args.audio_ext` sets the file extension to be loaded. The default extension is `mp3`.
- `dataset.args.max_segment_samples` sets length of audio segment to be loaded. If `dataset.args.max_segment_samples` is smaller than the actual audio clip duration, the first `dataset.args.max_segment_samples` samples of the audio clip would be loaded. If `dataset.args.max_segment_samples` is larger than the actual audio clip, the audio clip will be padded to `dataset.args.max_segment_samples` with 0. The default value is `327680` which is around 10 seconds when `sample_rate=16000`.
- `gpus` sets which GPU to use. `gpus=[k]` means `device='cuda:k'`, `gpus=2` means [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) (DDP) is used with two GPUs.

## Inpainting
This script supports only transcribing music from either MAPS or MAESTRO.

TODO: add support for transcribing any music

First, open `config/sampling.yaml`, and then specify the weight to use in `checkpoint_path`.

For example, if you want to use `Pretrain_MAESTRO-retrain_Both-k=9.ckpt`, then set  `checkpoint_path='weights/Pretrain_MAESTRO-retrain_Both-k=9.ckpt'`.

```
python sampling.py task=inpainting task.inpainting_t=[0,100] dataloader.batch_size=4 dataset=Custom dataset.args.audio_ext=mp3 dataset.args.max_segment_samples=327680 gpus=[0]
```

- `gpus` sets which GPU to use. `gpus=[k]` means `device='cuda:k'`, `gpus=2` means [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) (DDP) is used with two GPUs.
- `task.inpainting_t` sets the frames to be masked to -1 in the spectrogram. `[0,100]` means that frame 0-99 will be masked to -1.
- `dataloader.batch_size` sets the batch size. You can set a higher number if your GPU has enough memory.
- `dataset` when setting to `Custom`, it load audio clips from the folder `my_audio`.
- `dataset.args.audio_ext` sets the file extension to be loaded. The default extension is `mp3`.
- `dataset.args.max_segment_samples` sets length of audio segment to be loaded. If `dataset.args.max_segment_samples` is smaller than the actual audio clip duration, the first `dataset.args.max_segment_samples` samples of the audio clip would be loaded. If `dataset.args.max_segment_samples` is larger than the actual audio clip, the audio clip will be padded to `dataset.args.max_segment_samples` with 0. The default value is `327680` which is around 10 seconds when `sample_rate=16000`.


## Generation
First, open `config/sampling.yaml`, and then specify the weight to use in `checkpoint_path`.

For example, if you want to use `Pretrain_MAESTRO-retrain_Both-k=9.ckpt`, then set  `checkpoint_path='weights/Pretrain_MAESTRO-retrain_Both-k=9.ckpt'`.

```
python sampling.py task=generation dataset.num_samples=8 dataloader.batch_size=4

```

- `generation dataset.num_sample` sets the number of piano rolls to be generated.
- `dataloader.batch_size` sets the batch size of the dataloader. If you have enough GPU memory, you can set `dataloader.batch_size` to be equal to `dataset.num_samples` to generate everything in one go.