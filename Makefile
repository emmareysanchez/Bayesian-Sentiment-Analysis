.PHONY: help install dataset slda train evaluate business demo all clean

help:
	@echo "Targets:"
	@echo "  make install    - install python dependencies"
	@echo "  make dataset    - build dataset (IMDb + OOD + noise)"
	@echo "  make slda       - train amortized sLDA"
	@echo "  make train      - train all 5 model variants x 3 seeds"
	@echo "  make evaluate   - run E1..E5 evaluation suite"
	@echo "  make business   - cost-benefit analysis of rejection threshold"
	@echo "  make demo       - launch Streamlit demo"
	@echo "  make all        - run the full pipeline in order"
	@echo "  make clean      - remove generated data and results"

install:
	pip install -r requirements.txt

dataset:
	python scripts/01_build_dataset.py --imdb-size 20000 --ood-size-per-source 2000

slda:
	python scripts/02_train_slda.py --n-topics 10 --epochs 30

train:
	python scripts/03_train_all_models.py --seeds 42 43 44 --epochs 25

evaluate:
	python scripts/04_evaluate.py

business:
	python scripts/05_business_analysis.py --cost-fp 1.0 --cost-fn 5.0 --cost-review 0.2

demo:
	streamlit run app/streamlit_app.py

all: dataset slda train evaluate business
	@echo ""
	@echo "=== Full pipeline finished ==="
	@echo "Run 'make demo' to launch the interactive app."

clean:
	rm -rf data/processed data/ood experiments/results
	@echo "Cleaned generated artifacts."
