python './train.py' \
    citation \
    --model GCN \
    --dataset cora \
    --n_layers 16 \
    --hidden_dim 128 \
    --lr 1e-3 \
    --epochs 2000 \
    --dropout 0.2 \
    --edge_dropout 0.0 \
    --trials=5