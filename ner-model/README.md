


```
poetry run uvicorn src.run_api:app --host 0.0.0.0 --port 8000 --reload


curl -X 'POST' 'http://0.0.0.0:8000/predict/' \
     -H 'Content-Type: application/json' \
     -d '{"text":"England has a capital called London. On wednesday, the Prime Minister will give a presentation"}'
```




Run on docker


```
docker build -t my_ner_model:latest .
docker run -p 8000:8000 my_ner_model:latest

# test request works
curl -X 'POST' 'http://0.0.0.0:8000/test/'


# However, when running below curl request in docker
# my container crashes and I get error - curl: (52) Empty reply from server
# Stems from manual_inference_pipeline when running 'outputs = model(ids, mask)'
# I think there is a compute M1 mac issue or memory incompatibility issue - WIP

curl -X 'POST' 'http://0.0.0.0:8000/predict/' \
     -H 'Content-Type: application/json' \
     -d '{"text":"England has a capital called London. On wednesday, the Prime Minister will give a presentation"}'
```

- Options tried to fix above - different docker base image, expand memory on docker hub, retrain model with cpu



```
docker-compose up
```
