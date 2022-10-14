# Installation

# Training
```python train_both.py gpus=[1] model.args.kernel_size=9 dataset=Both epochs=10000 model.args.spec_dropout=0 ```



# Sampling
## Transcription
```python test.py gpus=[1]```


## Inpainting
```python sampling.py task=inpaintingpython sampling.py task=inpainting```

## Generation
```python generate_music.py```