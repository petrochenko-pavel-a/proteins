import argparse
from classification_pipeline import classification
import split_data
import loaders

def main():
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

    args = parser.parse_args()
    if args.dir!="":
        loaders.DIR=args.dir
    paths, labels = loaders.getTrainDataset()

    trainD, testD = split_data.split(paths, labels)
    tg = loaders.ProteinDataGenerator(trainD[0], trainD[1])

    paths2, labels2 = loaders.getTrainDataset2()
    extra_data = loaders.ProteinDataGenerator(paths2, labels2)
    classification.extra_train["train2"] = extra_data
    cfg = classification.parse(args.inputFile)
    cfg.gpus = args.gpus
    cfg.setAllowResume(True)
    cfg.fit(tg, foldsToExecute=[args.fold], start_from_stage=args.stage)

if __name__ == '__main__':
    main()