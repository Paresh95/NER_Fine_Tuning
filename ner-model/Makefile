init-local-env:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry env use 3.10.5
	poetry shell
	poetry install
	mkdir logs

preprocess-data:
	poetry run python src/data_utils/preprocess_raw_data.py

run-training-locally:
	poetry run python src/training/main/run_training.py

run-inference-locally:
	poetry run python src/inference/main/run_inference.py

run-api-locally:
	poetry run python src/inference/main/run_api.py

api-test:
	curl -X 'POST' 'http://0.0.0.0:8000/test/'

api-predict:
	curl -X 'POST' 'http://0.0.0.0:8000/predict/' \
     -H 'Content-Type: application/json' \
     -d '{"text":"England has a capital called London. On wednesday, the Prime Minister will give a presentation"}'

build:
	docker build -t my_ner_app:latest .

run:
	docker run -p 8000:8000 my_ner_app:latest

build-training:
	docker build --target trainer -t my_ner_app:latest-training .

build-inference:
	docker build --target inferencer -t my_ner_app:latest-inference .

deploy-docker-compose:
	docker-compose up

delete-local-env:
	exit
	poetry env remove 3.10
