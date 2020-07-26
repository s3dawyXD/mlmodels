
```bash

Ressources to find information



https://github.com/arita37/mlmodels/blob/adata2/README_testing.md

https://github.com/arita37/mlmodels/blob/adata2/README_usage_CLI.md


https://github.com/arita37/mlmodels/blob/adata2/README_addmodel.md


https://github.com/arita37/mlmodels/issues?q=is%3Aopen+is%3Aissue+label%3Adev-documentation


https://github.com/arita37/mlmodels/issues?q=is%3Aopen+is%3Aissue+label%3Adev-documentation






```




## How to install mlmodels ?
<details>


</details>


## How to check if mlmodels works ?




## How to check if  one model works ?





## How to develop using Colab ?


## How to develop using Gitpod ?









## How to add  a model ?
<details>
To add new model fork the repo. Inside the mlmodels directory we have multiple subdirectories named like model_keras, model_sklearn and so on the idea is to use **model_** before the type of framework you want to use. Now once you have decided the frame work create appripriately named model file and config file as described in the read me doc [README_addmodel.md](docs\README_docs\README_addmodel.md). The same model structure and config allows us to do the testing of all the models easily.
</details>



## How to check test log after commit ?
<details>
Once the model is added we can do testing on it with commands like this, where model_framework is a placeholder for your selected framework and model_file.json is the config file for your model.

```
ml_models --do fit     --config_file model_framework/model_file.json --config_mode "test" 
```
Here the fit method is tested, you can check the predict fucntionality of the model like this.
```
ml_models --do predict --config_file model_tf/1_lstm.json --config_mode "test"
```
But this is individual testing that we can do to debug our model when we find an error in automatic the test logs.

We have automated testing in our repo and the results are stored in here https://github.com/arita37/mlmodels_store We havemultiple level logs and they are put under different directories as you can see here, log folders have **logs_** at the start.
![Mlmodels Store](imgs/test_repo.PNG?raw=true "Mlmodels Store")
We can focus on the error_list directory to debug our testing errors. Inside the error_list directory we can find the logs of all test cases in directories named at the time they are created
![Error List](imgs/error_list.PNG?raw=true "Error List")
Inside we can see separate files for each test cases which will have the details of the errors.
![Error Logs](imgs/error_logs.PNG?raw=true "Error logs")
For example we can look at the errors for test cli cases named as list_log_test_cli_20200610.md
![Error](imgs/test_cli_error.PNG?raw=true "Error")
We see multiple erros and we can click on the traceback for error 1 which will take us to the line 421 of the log file.
![Error Line](imgs/error_line.PNG?raw=true "Error Line")
We can see that while running the test case at line 418 caused the error, and we can see the error. 
```
ml_models --do fit  --config_file dataset/json/benchmark_timeseries/gluonts_m4.json --config_mode "deepar" 
```
So we fix the erorr by launch the git pod and test the test case again and see it works correctly after that we can commit teh changes and submit the pull request.
</details>


## How to debug the repo >


## How to find information ?













## How dataloader works ?
### 1. What is dataloader?
Dataloader is helper module which load data from configuration and prepare for training/evaluation task.
### 2. Dataloader's configuration: store in json file with key "data_pars" have 2 main parts:
- **data_info**: conntain common data info such as data_path, dataset_name, batch_size when processing and dataset_type like training and testing
- **preprocessors**: list of data preprocessing which could be a function or a class object specified by "uri". Argument for those function or class object init passed by "args". These preprocessor will be processed one by one in sequence (TBD: how to process data with more dynamic ways with complicated combination like sequence and parallel)
Example:
```
"data_pars": {
    "data_info": {
            "data_path"  : "dataset/recommender/",
            "dataset"    : "IMDB_sample.txt",
            "data_type"  : "csv_dataset",
            "batch_size" : 64,
            "train"      : true
        },
    "preprocessors": [
        {"uri"  : "mlmodels.model_tch.textcnn:split_train_valid",
         "args" : {
                    "frac": 0.99
                    }
        },
        {"uri"  : "mlmodels.model_tch.textcnn:create_tabular_dataset",
         "args" : {
                    "lang": "en",
                    "pretrained_emb": "glove.6B.300d"
                    }

        }
        ]
},
```
### 3. Dataloader workflow
- Dataload output contain 2 parts: 
   + dataset output: memory object for next step like training/validating
   + internal_state: which is dictionary to store extra-data if need
- To get output, just init dataloader and invoke dataloader **compute** function which return tuple (dataset_outout, internal_state)
During computing, dataloader init all objects in list of preprocessor and invoke function to preprocess data

**Note**
- Class object in dataloader configuration need to implement 3 interface:
```buildoutcfg
def __init__(self, **args):
    # initialization
def compute(self, input_tmp):
    # self.data = xxx

def get_data(self):
    return self.data
```    

## How configuation JSON works ?





## How to improve the test process >





## How to debug the repo >





## How to find information ?














