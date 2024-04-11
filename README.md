# MCLP

This is the PyTorch implementation of the paper "Going Where, by Whom, and at What Time: Next Location Prediction Considering User Preference and Temporal Regularity"

## Parameter Analysis
This is a supplementary parameter analysis conducted on the Traffic Camera Dataset, exploring the effects of various dimensions, including location/time, user, and the number of Transformer encoder layers.

![Effect of Location Dimension](fig/lt_embedding.png)
![Effect of User Dimension](fig/user_embedding.png)
![Effect of num of layer](fig/num_layer.png)

## Requirements
The runtime environment can be viewed in requirements.txt or by executing the following command:
```shell
pip install -r requirements.txt
```

## Hyperparameters
All hyperparameter settings are saved in the `.yml` files under the respective dataset folder under `saved_models/`. \
\
For example, `saved_models/TC/settings.yml` contains hyperparameter settings of MCLP for Traffic Camera Dataset. 

## Run
#### The following is a run of Traffic Camera Dataset (Mobile Phone Dataset is similarly provided):
- Unzip `data/TC.zip` to `dataset/TC`. The two files are training data and testing data.

- For MCLP model:
  ```shell
  python ./model/run.py --dataset TC --dim 16 --topic 400 --at attn
  ```
- For MCLP(LSTM) model:
  ```shell
  python ./model/run.py --dataset TC --dim 16 --topic 400 --at attn --encoder lstm
  ```
#### Study of Different Variants:
- For Base model:
  ```shell
  python ./model/run.py --dataset TC --dim 16
  ```
- For +Pre model:
  ```shell
  python ./model/run.py --dataset TC --dim 16 --topic 400
  ```
- For +At model :
  ```shell
  python ./model/run.py --dataset TC --dim 16 --at attn
  ```

#### Study of Different Arrival Time Estimators:
- For Static model :
  ```shell
  python ./model/run.py --dataset TC --dim 16 --at static
  ```
- For True model :
  ```shell
  python ./model/run.py --dataset TC --dim 16 --at truth
  ```
