from ruletaker.cli.train import run_training


if __name__ == "__main__":
    run_training(
        prog="python -m ruletaker.cli.baseline",
        description="Train transformer baseline on RuleTaker only.",
    )
