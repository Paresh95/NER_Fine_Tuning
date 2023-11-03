# Train & Deploy Named Entity Recognition Model

<p align="center">
  <img src="images/ner_example.jpeg" />
</p>


## What is this repo about?
This repo contains:

- Scripts to fine tune a BERT model for Named Entity Recognition with the traditional approach (train model head weights) or LoRA.
- The option of using accelerate for distributed training
- Model inference scripts
- FastAPI script to create model endpoint
- Dockerfile to deploy model
- GitHub actions CI/CD pipeline


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
- Note, to improve speed the default set-up only trains with a sample of the data. Change the `df_sample_size` parameter in `ner-model/src/static.yaml` if you wish to train with more data.
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

Note, these commands default to using the traditional approach for fine-tuning (training the model head). To use LoRA change the filepaths specified in the `Makefile` and `Dockerfile`.

## Future ideas
- Store model in AWS S3 bucket
- Create separate docker image for training or run CI/CD for automated retraining
- Could add accelerate config file to set parameters
