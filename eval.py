import argparse
from classification_pipeline import classification
import loaders
from sklearn.metrics.classification import f1_score
import musket_core.datasets as ds
import tresholds
import os
import utils
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
    parser.add_argument('--workers', type=int, default=0,
                        help='stage')
    parser.add_argument('--ql', type=int, default=20,
                        help='stage')

    args = parser.parse_args()
    if args.workers>0:
        ds.USE_MULTIPROCESSING=True
        ds.NB_WORKERS=args.workers
        ds.AUGMENTER_QUEUE_LIMIT = args.ql
    if args.dir!="":
        loaders.DIR=args.dir
    if args.fold==100:
        for i in range(5):
            args.fold=i
            doEval(args)
    else:
        doEval(args)


def doEval(args):
    test, correct_labels = loaders.createHoldoutDataSet()
    t = get_or_calculate_tresholds(args)
    predictions = get_or_calculate_holdout(args, test)
    for i in range(28):
        print(i, f1_score(correct_labels[:, i], predictions[:, i] > t[0][i]))
    print("Macro F1:" + str(f1_score(correct_labels, predictions > t[0], average="macro")))
    print("Val Macro F1:", t[1])


def get_or_calculate_holdout(args, test):
    ps = args.inputFile + ".hold_out_pred." + str(args.fold) + "." + str(args.stage)
    if not os.path.exists(ps):
        cfg = classification.parse(args.inputFile)
        cfg.gpus = args.gpus
        cfg.setAllowResume(True)
        predictions = cfg.predict_all_to_array(test, fold=args.fold, stage=args.stage, ttflips=True)
        utils.save(ps, predictions)
    predictions = utils.load(ps);
    return predictions

def get_or_calculate_tresholds(args):
    ps = args.inputFile + ".tresholds." + str(args.fold) + "." + str(args.stage)
    if not os.path.exists(ps):
        cfg = classification.parse(args.inputFile)
        cfg.gpus = args.gpus
        cfg.setAllowResume(True)
        train = loaders.createDataSet();
        pred, labels = cfg.evaluate_all_to_arrays(train, fold=args.fold, stage=args.stage, ttflips=True)
        v = tresholds.getOptimalT(pred, labels)
        utils.save(ps, v)
        utils.save(args.inputFile + ".validation_pred." + str(args.fold) + "." + str(args.stage), [pred,labels])
    t = utils.load(ps)
    return t

if __name__ == '__main__':
    main()