Proteins
========

## Training a model example

```sh
CUDA_VISIBLE_DEVICES=3 python train.py --inputFile resnet34/proteins.yaml\ 
                                               --fold 4 \
                                               --dir /media/1t/protein/data \
                                               --gpus 1 \
                                               --workers 8
```

## Custom variables to change data folders or read glued images

```python
DIR =                os.getenv('MAIN_DIR', 'D:/cells')
GLUED_IMAGES =       os.getenv('GLUED_IMAGES', False)
TRAIN_SUBDIR =       os.getenv('TRAIN_SUBDIR', 'train')
EXTRA_TRAIN_SUBDIR = os.getenv('EXTRA_TRAIN_SUBDIR', 'train2')
TRAIN_CSV =          os.getenv('TRAIN_CSV', 'train.csv')
EXTRA_TRAIN_CSV =    os.getenv('EXTRA_TRAIN_CSV', 'train2.csv')
TEST_SUBDIR =        os.getenv('TEST_SUBDIR', 'test')
```

## Eval model example

```sh
GLUED_IMAGES=True TRAIN_SUBDIR=train_glued_cv2 EXTRA_TRAIN_SUBDIR=external_data_512_glued_cv2 CUDA_VISIBLE_DEVICES=2 python eval.py --inp
utFile ./resnet34/proteins.yaml --fold 3 --dir /media/1t/protein/data
```
