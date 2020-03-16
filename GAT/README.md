# Graph Attention Networks

![](./images/layer.jpg)

## How does GAT work?

![](./images/gat.gif)

1. Given a central node (blue node), perform an affine transformation on all node feature vectors of its neighbors (gray nodes).
2. Concatenate the new node feature vectors of the central node and one of its neighbors,  perform a dot product with a learnable vector to get a scalar number, and pass this number through an activation function Leaky ReLU.
3. Repeat the step 2 for all neighbors of the given central node.
4. Use softmax function to normalize the scalar numbers produced in the step 3. These normalized numbers are the weights of weighted sum.
5. Aggregate these new node feature vectors by weighted sum. The weights used in the weighted sum come from the step 4.
6. Send the result of weighted sum into another activation funciton, for example, ELU.
7. Assign the result to the central node as its new node feature vector.
8. Repeat the step 1 to step 7 for all nodes in the graph.
9. Finish when all nodes are updated.

<img src=./images/weight.png width=300>

Unlike GCN, in GAT, the weights used in the weighted sum performed at the step 2 are determined by node features, as the above figure shown, and they are learnable. The weights between nodes not only have something to do with the edges but also node features, so in general, GAT has more capacity than GCN.

## Results

| Dataset | Test Accuracy | Reported Accuracy |
| :-: | :-: | :-: |
| Cora | 83.4 ± 0.3% | 83.0 ± 0.7% |
| Citeseer | 71.6 ± 0.6% | 72.5 ± 0.7% |
| Pubmed | 78.6 ± 0.3% | 79.0  ± 0.3% |

## Reproduce

Use the following command to train a GAT model trained on Cora dataset with default parameters to reproduce the result in the paper: [Graph Attention Network](https://arxiv.org/pdf/1710.10903.pdf).

```
$ python train.py accuracy --dataset=cora  --gpu=True
```

You can set different dataset in the argument `--dataset` to reproduce other results. See help for more setting.

Once the training is completed, there would be a model file called `gat_{dataset}.pth`, and an image called `gat_{dataset}.png` which visualizes loss and accuracy during the training. The accuracy on test set can be seen at the tail of output on your terminal.

![](./images/gat_cora.png)

## Compare with GCN

Type the following command to run the comparison between GCN and GAT at different model parameters.

```
$ python train.py parameters --dataset=cora --gpu=True
```

![](./images/gcn:gat.png)

The above figure shows that GAT is almost better than GCN.

## Multi-layer GAT

Use the following command to train a multi-layer GAT model with or without residual connections:

```
$ python train.py layers --dataset=cora  --gpu=True
```

It will train two GAT models, one with residual connections and the other without residual connections.

Once the training is done, an image showing the different between these two models will be saved as `multi_gat.png`.

![](./images/multi_gat.png)

Like multi-layer GCN, multi-layer GAT can not get benefit by increasing its depth. However, the decrease of multi-layer GAT is smaller than that of multi-layer GCN. It may be caused by attention mechanism. Again, there is no improvement when adding residual connections.
