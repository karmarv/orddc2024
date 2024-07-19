


## MMYolo - RTMDet

- Installation: https://github.com/open-mmlab/mmyolo/blob/main/docs/en/get_started/installation.md 
  ```bash
    pip install wandb future tensorboard
    # After running wandb login, enter the API Keys obtained above, and the login is successful.
    wandb login 
  ```

#### Exp 1
- Run
```
# Multi 2 GPU # XXX hours for b4 100 epochs
CUDA_VISIBLE_DEVICES=4,5 PORT=29601 ./tools/dist_train.sh rtmdet_l_syncbn_fast_8xb32-150e_rdd2022.py 2
```
> 

