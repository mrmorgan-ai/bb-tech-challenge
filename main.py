import argparse
import sys

from src.models.train import train_all
from src.models.evaluate import evaluate
from src.prediction_logging.batch_inference import run_batch_inference
from src.monitoring.drift_monitor import main as run_monitoring

DATA_PATH = "data/raw/storedata_total.xlsx"


def train():
    train_all(DATA_PATH)

def eval():
    evaluate(DATA_PATH)

def infer():
    run_batch_inference(DATA_PATH, n_samples=200)

def monitor():
    run_monitoring(DATA_PATH)


STEPS = [
    ("train", train),
    ("evaluate", eval),
    ("infer", infer),
    ("monitor", monitor),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=[s for s, _ in STEPS], default=None)
    args = parser.parse_args()

    if args.step:
        dict(STEPS)[args.step]()
    else:
        for name, fn in STEPS:
            fn()
