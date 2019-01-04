import argparse
from classification_pipeline import classification
import loaders
import musket_core.datasets as ds
import pandas as pd
import os
import numpy as np


def main():
    """
    Example for a single GPU:
    
        CUDA_VISIBLE_DEVICES=3 python train.py --inputFile resnet34/proteins.yaml \
                                               --fold 4 \
                                               --dir /media/1t/protein/data \
                                               --gpus 1 \
                                               --workers 8 \
                                               --gluedImages True \
                                               --trainImagesDir train_glued_cv2 \
                                               --extraTrainImagesDir external_data_512_glued_cv2 \
                                               --ql 256
    """
    parser = argparse.ArgumentParser(description='Training for proteins')
    
    parser.add_argument('--inputFile', type=str, default="./proteins.yaml",
                        help='learner config')
    parser.add_argument('--fold', type=int, default=0,
                        help='fold number')
    parser.add_argument('--stage', type=int, default=0,
                        help='stage')
    parser.add_argument('--dir', type=str, default="",
                        help='directory with data')
    parser.add_argument('--gpus', type=int, default=1,
                        help='stage')
    parser.add_argument('--workers', type=int, default=0,
                        help='stage')
    parser.add_argument('--ql', type=int, default=20,
                        help='stage')
    
    parser.add_argument('--gluedImages', type=bool, required=True)
    parser.add_argument('--trainImagesDir', type=str)
    parser.add_argument('--extraTrainImagesDir', type=str)

    args = parser.parse_args()
    if args.workers>0:
        ds.USE_MULTIPROCESSING=True
        ds.NB_WORKERS=args.workers
        ds.AUGMENTER_QUEUE_LIMIT = args.ql
    if args.dir!="":
        loaders.DIR=args.dir


    paths, labels = loaders.getTrainDataset(images_dir=args.trainImagesDir)
    paths2, labels2 = loaders.getTrainDataset2(images_dir=args.extraTrainImagesDir)
    paths = np.concatenate([paths, paths2])
    labels = np.concatenate([labels, labels2])

    foldIndexes = calculate_fold_indexes(paths)

    tg = loaders.ProteinDataGenerator(paths, labels, glued_images=args.gluedImages)
    tg.folds = foldIndexes;

#     paths, labels = loaders.getTrainDataset(images_dir=args.trainImagesDir, 
#                                             pd_file=args.trainPdFile)

#     trainD, testD = split_data.split(paths, labels)
#     tg = loaders.ProteinDataGenerator(trainD[0], trainD[1], glued_images=args.gluedImages)

#     paths2, labels2 = loaders.getTrainDataset2(images_dir=args.extraTrainImagesDir, 
#                                                pd_file=args.extraTrainPdFile)
#     extra_data = loaders.ProteinDataGenerator(paths2, labels2, glued_images=args.gluedImages)
#     classification.extra_train["train2"] = extra_data

    cfg = classification.parse(args.inputFile)

    cfg.gpus = args.gpus
    cfg.setAllowResume(True)

    cfg.fit(tg, foldsToExecute=[args.fold], start_from_stage=args.stage)


def calculate_fold_indexes(paths):
    foldSets = []
    foldIndexes = []
    for i in range(1, 6):
        train = pd.read_csv("./folds/fold" + str(i) + "_train.csv")
        test = pd.read_csv("./folds/fold" + str(i) + "_val.csv")
        foldSets.append((set(train["Id"].values), set(test["Id"].values)))
        foldIndexes.append([[], []])
    for i in range(len(paths)):
        bn = os.path.basename(paths[i])
        for j in range(len(foldSets)):
            if bn in foldSets[j][0]:
                foldIndexes[j][0].append(i)
            if bn in foldSets[j][1]:
                foldIndexes[j][1].append(i)
    return foldIndexes

if __name__ == '__main__':
    main()
