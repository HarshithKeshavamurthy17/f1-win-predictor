.PHONY: env data features train eval interpret bet

env:
	python -m ipykernel install --user --name f1-win --display-name "Python (f1-win)"

data:
	python -m src.collect --start 1995 --end 2024 --out data/raw

features:
	python -m src.preprocess --in data/raw --out data/curated && \
	python -m src.features --in data/curated --out data/curated

train:
	python -m src.train --in data/curated/f1_training.parquet --out data/curated/models --model xgb

eval:
	python -m src.evaluate --pred data/curated/models/preds.parquet --races data/curated/races.parquet

interpret:
	python -m src.interpret --pred data/curated/models/preds.parquet --out data/curated/models

bet:
	python -m src.bet_backtest --pred data/curated/models/preds.parquet --odds data/curated/odds.csv --out data/curated/models

