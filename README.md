# Train & Deploy Named Entity Recognition Model

<p align="center">
  <img src="images/ner_example.png" />
</p>


## What is this repo about?
This repo:
- Fine tunes a BERT model's classification head for Named Entity Recognition
- Creates a model API
- Deploys a Docker Image via a CI/CD pipeline

## How to run this code

1. To set up the local environment

```
cd ner-model
make init-local-env
```

2. To create model training data

```
make preprocess-data
```

3. To run training script locally
```
make run-training-locally
```

4. To run inference locally
```
make run-inference-locally
```

5. To run the API locally
```
make run-api-locally
# open new terminal
make api-test
make api-predict
```

6. To deploy the dockerfile
- Note the `make api-predict` command crashes the container with error `curl: (52) Empty reply from server`. This stems from the code `outputs = model(ids, mask)`. The API works as the test runs, however, I think there is a compute M1 mac issue or memory incompatibility issue. 
```
make deploy-dockerfile
# open new terminal
make api-test
make api-predict 
```

7. To deploy the dockerfile via docker compose
```
make deploy-docker-compose
# open new terminal
make api-test
make api-predict 
```

8. To delete the local poetry environment
```
make delete-local-env
```