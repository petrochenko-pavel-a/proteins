import argparse
from classification_pipeline import classification
import loaders
import musket_core.datasets as ds


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
    parser.add_argument('--loader_workers', type=int, default=1,
                        help='loader workers')
    parser.add_argument('--loader_threaded', type=bool, default=True,
                        help='loader is threaded')
    parser.add_argument('--loader_size', type=int, default=50,
                        help='loader quee size')

    args = parser.parse_args()
    if args.workers>0:
        ds.USE_MULTIPROCESSING=True
        ds.NB_WORKERS=args.workers
        ds.AUGMENTER_QUEUE_LIMIT = args.ql
        ds.NB_WORKERS_IN_LOADER =args.loader_workers
        ds.LOADER_SIZE = args.loader_size
        ds.LOADER_THREADED = args.loader_threaded
    if args.dir!="":
        loaders.DIR=args.dir

    tg = loaders.createDataSet()

    cfg = classification.parse(args.inputFile)

    cfg.gpus = args.gpus
    cfg.setAllowResume(True)

    cfg.fit(tg, foldsToExecute=[args.fold], start_from_stage=args.stage)


if __name__ == '__main__':
    main()