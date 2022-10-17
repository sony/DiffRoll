# Installation

# Training

## Supervised training
```
python train_spec_roll.py gpus=[0] model.args.kernel_size=9 model.args.spec_dropout=0.1 dataset=MAESTRO dataloader.train.num_workers=4 epochs=2500
```

## Unsupervised pretraining
### Step 1: Pretraining on MAESTRO using only piano rolls
```
python train_spec_roll.py gpus=[0] model.args.kernel_size=9 model.args.spec_dropout=1 dataset=MAESTRO dataloader.train.num_workers=4 epochs=2500
```



### Step 2: Continue training on MAPS using both spectrograms and piano rolls

```
python train_both.py gpus=[0] model.args.kernel_size=9 dataset=Both epochs=10000 model.args.spec_dropout=0
```



# Sampling
## Transcription
```python test.py gpus=[1]```


## Inpainting
```python sampling.py task=inpaintingpython sampling.py task=inpainting```

## Generation
```python generate_music.py```