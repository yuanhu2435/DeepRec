# MMOE

- [MMOE](#mmoe)
  - [Model Structure](#model-structure)
  - [Usage](#usage)
    - [Stand-alone Training](#stand-alone-training)
    - [Distribute Training](#distribute-training)
  - [Benchmark](#benchmark)
    - [Stand-alone Training](#stand-alone-training-1)
      - [Test Environment](#test-environment)
      - [Performance Result](#performance-result)
  - [Dataset](#dataset)
    - [Prepare](#prepare)
    - [Fields](#fields)
    - [Processing](#processing)
  - [TODO LIST](#todo-list)

## Model Structure
[Multi-gate Mixture-of-Experts]MMoE is Multi-Tower model in which each tower gets weighted inputs from multiple exprets
The model structure in this folder is as follow:
```
output:                Probability of click                      Probability of buy
                                ▲                                        ▲
                                │                                        │
model:                     ┌────┴────┐                              ┌────┴────┐
                           │         │                              │         │
                           │  click  │                              │   buy   │
                           │         │                              │         │
                           │   DNN   │                              │   DNN   │
                           │         │                              │         │
                           └─────────┘                              └─────────┘
                                ▲                                        ▲
                                │                                        │
                              ┌─┴─┐                                    ┌─┴─┐
             ┌───────────────►│ x │                                    │ x │◄───────────────┐
             │                └───┘                                    └───┘                │
             │                  ▲                                        ▲                  │
             │                  │                                        │                  │
             │                  ├────────────────────┬───────────────────┤                  │
             │                  │                    │                   │                  │
             │                  │                    │                   │                  │
             │             ┌────┴────┐          ┌────┴────┐         ┌────┴────┐             │
             │             │         │          │         │         │         │             │
             │             │         │          │         │         │         │             │
       ┌─────┴──────┐      │ Expert  │          │ Expert  │         │ Expert  │      ┌──────┴─────┐
       │            │      │         │          │         │         │         │      │            │
       │ click gate │      │    1    │          │   ...   │         │    N    │      │  buy gate  │
       │    DNN     │      │         │          │         │         │         │      │    DNN     │
       └────────────┘      │   DNN   │          │   DNN   │         │   DNN   │      └────────────┘
             ▲             │         │          │         │         │         │             ▲
             │             │         │          │         │         │         │             │
             │             └─────────┘          └─────────┘         └─────────┘             │
             │                  ▲                    ▲                   ▲                  │
             │                  │                    │                   │                  │
             │                  │                    │                   │                  │
             │                  └────────────────────┼───────────────────┘                  │
             │                                       │                                      │
             │                                       │                                      │
             │                                       │                                      │
             │                                  ┌────┴────┐                                 │
             │                                  │         │                                 │
             └──────────────────────────────────┤  input  ├─────────────────────────────────┘
                                                │         │
                                                └─────────┘
```
## Usage

### Stand-alone Training
1.  Please prepare the [data set](#prepare) first.

2.  Create a docker image by DockerFile.   
    Choose DockerFile corresponding to DeepRec(Pending) or Google tensorflow.
    ```
    docker build -t DeepRec_Model_Zoo_MMOE_training:v1.0 .
    ```

3.  Run a docker container.
    ```
    docker run -it DeepRec_Model_Zoo_MMOE_training:v1.0 /bin/bash
    ```

4.  Training.  
    ```
    cd /root/
    python train.py
    ```
    Use argument `--bf16` to enable DeepRec BF16 in deep model.
    ```
    python train.py --bf16
    ```
    Use arguments to set up a custom configuation:
    - `--data_location`: Full path of train & eval data, default is `./data`.
    - `--output_dir`: Full path to output directory for logs and saved model, default is `./result`.
    - `--checkpoint`: Full path to checkpoints input/output directory, default is `$(OUTPUT_DIR)/model_$(MODEL_NAME)_$(TIMESTAMPS)`
    - `--steps`: Set the number of steps on train dataset. Default will be set to 10 epoch.
    - `--batch_size`: Batch size to train. Default is 512.
    - `--timeline`: Save steps of profile hooks to record timeline, zero to close, defualt to 0.
    - `--save_steps`: Set the number of steps on saving checkpoints, zero to close. Default will be set to 0.
    - `--keep_checkpoint_max`: Maximum number of recent checkpoint to keep. Default is 1.
    - `--learning_rate`: Learning rate for network. Default is 0.1.
    - `--bf16`: Enable DeepRec BF16 feature in DeepRec. Use FP32 by default.
    - `--no_eval`: Do not evaluate trained model by eval dataset.
    - `--protocol`: Set the protocol("grpc", "grpc++", "star_server") used when starting server in distributed training. Default to grpc. 


### Distribute Training
1. Prepare a K8S cluster and shared storage volume.
2. Create a PVC(PeritetVolumeClaim) for storage volumn in cluster.
3. Prepare docker image by DockerFile.
4. Edit k8s yaml file
- `replicas`: numbers of cheif, worker, ps.
- `image`: where nodes can pull the docker image.
- `claimName`: PVC name.

## Benchmark
### Stand-alone Training

#### Test Environment
TBD

#### Performance Result
<table>
    <tr>
        <td colspan="1"></td>
        <td>Framework</td>
        <td>DType</td>
        <td>Accuracy</td>
        <td>AUC</td>
        <td>Globalsetp/Sec</td>
    </tr>
    <tr>
        <td rowspan="3">MMOE</td>
        <td>Community TensorFlow</td>
        <td>FP32</td>
        <td>0.97378</td>
        <td>0.74330</td>
        <td>70.4714 (baseline)</td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32</td>
        <td>0.97378</td>
        <td>0.74426</td>
        <td>92.1696 (+1.30x)</td>
    </tr>
    <tr>
        <td>DeepRec w/ oneDNN</td>
        <td>FP32+BF16</td>
        <td>0.97378</td>
        <td>0.74607</td>
        <td>102.4703 (+1.45x)</td>
    </tr>
</table>

- Community TensorFlow version is v1.15.5.

## Dataset
Taobao dataset from [EasyRec](https://github.com/AlibabaPAI/EasyRec) is used.
### Prepare
Put data file **taobao_train_data & taobao_test_data** into ./data/    
For details of Data download, see [EasyRec](https://github.com/AlibabaPAI/EasyRec/#GetStarted)

### Fields
The dataset contains 20 columns, details as follow:
| Name | clk      | buy      | pid       | adgroup_id | cate_id   | campaign_id | customer  | brand     | user_id   | cms_segid | cms_group_id | final_gender_code | age_level | pvalue_level | shopping_level | occupation | new_user_class_level | tag_category_list | tag_brand_list | price    |
| ---- | -------- | -------- | --------- | ---------- | --------- | ----------- | --------- | --------- | --------- | --------- | ------------ | ----------------- | --------- | ------------ | -------------- | ---------- | -------------------- | ----------------- | -------------- | -------- |
| Type | tf.int32 | tf.int32 | tf.string | tf.string  | tf.string | tf.string   | tf.string | tf.string | tf.string | tf.string | tf.string    | tf.string         | tf.string | tf.string    | tf.string      | tf.string  | tf.string            | tf.string         | tf.string      | tf.int32 |


The data in `tag_category_list` and `tag_brand_list` column are separated by `'|'` 

### Processing
The 'clk' ans 'buy' columns are` used as labels.  
Input feature columns are as follow:
| Column name          | Hash bucket size | Embedding dimension |
| -------------------- | ---------------- | ------------------- |
| pid                  | 10               | 16                  |
| adgroup_id           | 100000           | 16                  |
| cate_id              | 10000            | 16                  |
| campaign_id          | 100000           | 16                  |
| customer             | 100000           | 16                  |
| brand                | 100000           | 16                  |
| user_id              | 100000           | 16                  |
| cms_segid            | 100              | 16                  |
| cms_group_id         | 100              | 16                  |
| age_level            | 10               | 16                  |
| pvalue_level         | 10               | 16                  |
| shopping_level       | 10               | 16                  |
| occupation           | 10               | 16                  |
| new_user_class_level | 10               | 16                  |
| tag_category_list    | 100000           | 16                  |
| tag_brand_list       | 100000           | 16                  |
| -------------------- | Num Buckets      | ------------------- |
| price                | 50               | 16                  |

## TODO LIST
- Distribute training model
- Benchmark
- DeepRec DockerFile