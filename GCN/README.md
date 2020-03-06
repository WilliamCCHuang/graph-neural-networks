# Graph Convolutional Networks

![](./images/layer.png)

## How does GCN work?

![](./images/gcn.gif)

1. Given a central node (blue node), perform an affine transformation on all node feature vecots of its neighbors (gray nodes).
2. Aggregate these new node feature vectors by weighted sum. The weights used in the weighted sum come from the topology of the given graph.
3. Send the result of weighted sum into an activation funciton, for example, ReLU.
4. Assign the result of the activation function to the central node.
4. Repeat the step 1 to step 3 for all nodes in the graph.
5. Finish when all nodes are updated.

The algorithm of GCN can be written as an simple equation:

<img src=./images/equation.png
 width=200>

The weights used in the weighted sum performed at the step 2 are determined by the graph topology, as the following figure shown, and hence are not learnable, which limit GCN model to capture the relations between nodes.

<img src=./images/weight.png
 width=300>

## Results

| Dataset | Test Accuracy | Reported Accuracy |
| :-: | :-: | :-: |
| Cora | 81.1 +- 0.6% | 81.5 |
| Citeseer | 71.3 +- 0.5% | 70.3 |
| Pubmed | 79.0 +- 0.6% | 79.0 |

## Reproduce

Use the following command to train a GCN model trained on Cora dataset with default parameters to reproduce the result in the paper: [Semi-supervised Classification With Graph Convolutional Networks](https://openreview.net/pdf?id=SJU4ayYgl).

```
$ python train.py accuracy --dataset=cora  --gpu=True
```

You can set different dataset in the argument `--dataset` to reproduce other results. See help for more setting.

Once the training is completed, there would be a model file called `gcn_{dataset}.pth`, and an image called `gcn_{dataset}.png` which visualizes loss and accuracy during the training. The accuracy on test set can be seen at the tail of output on your terminal.

![](./images/gcn_cora.png)

## Multi-layer GCN

Use the following command to train a multi-layer GCN model with or without residual connections:

```
$ python train.py layers --dataset=cora  --gpu=True
```

It will train two multi-layer GCN models, one with residual connections and the other without residual connections.

Once the training is done, an image showing the different between these two models will be saved as `multi_gcn.png`.

![](./images/multi_gcn.png)

The experiment shows that the original GCN can not be stacked to get too many layers. It is even worse when layers are too many. Adding residual connections can decrease the effect, however, no increase is observed.
