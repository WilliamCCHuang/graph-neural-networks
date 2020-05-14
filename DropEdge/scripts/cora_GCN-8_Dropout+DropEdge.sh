python './train.py' \
    citation \
    --model GCN \
    --dataset cora \
    --n_layers 8 \
    --hidden_dim 128 \
    --lr 5e-3 \
    --epochs 400 \
    --dropout 0.5 \
    --edge_dropout 0.5 \
    --trials=5