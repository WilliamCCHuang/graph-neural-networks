python './train.py' \
    citation \
    --model GCN \
    --dataset cora \
    --n_layers 8 \
    --hidden_dim 128 \
    --lr 5e-3 \
    --epochs 400 \
    --dropout 0.0 \
    --edge_dropout 0.8 \
    --trials=5