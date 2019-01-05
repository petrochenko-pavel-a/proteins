import argparse
from classification_pipeline import classification
import loaders
from sklearn.metrics.classification import f1_score
import musket_core.datasets as ds
import tresholds
import os
import utils
import eval
import numpy as np

from tqdm import tqdm
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
    cfg = classification.parse(args.inputFile)

    predictions = get_or_calculate_test_predictions(args, cfg)

    tresholds=eval.get_or_calculate_tresholds(args)[0]
    create_submission(predictions, tresholds)


def create_submission(predictions, tresholds):
    prediction = []
    submit = loaders.getSubmitSample()
    for row in tqdm(range(submit.shape[0])):
        str_label = ''
        for col in range(predictions.shape[1]):

            if predictions[row, col] < tresholds[col]:
                str_label += ''
            else:
                str_label += str(col) + ' '
        prediction.append(str_label.strip())
    submit['Predicted'] = np.array(prediction)
    submit.to_csv('4channels_cnn_from_scratch.csv', index=False)


def get_or_calculate_test_predictions(args, cfg):
    ps = args.inputFile + ".test_pred." + str(args.fold) + "." + str(args.stage)
    if not os.path.exists(ps):
        predictions = cfg.predict_all_to_array(loaders.getTestDataSet(), args.fold, args.stage, ttflips=True, batch_size=64)
        utils.save(ps, predictions)
    predictions = utils.load(ps);
    return predictions


if __name__ == '__main__':
    main()