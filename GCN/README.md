# Graph Convolutional Networks

## Reproduce

Use the following command to train a GCN model trained on Cora dataset with default parameters to reproduce the result in the paper: [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://openreview.net/pdf?id=SJU4ayYgl):

```
$ python train.py accuracy --dataset Cora
```

![](./images/gcn.png)

## 

Use the following command to train a multi-layer GCN model with or without residual connections:

```
$ python train.py layers --dataset Cora
```

![](./images/multi_gcn.png)