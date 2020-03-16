import argparse

import torch

from models import GCN, GAT, MultiGAT
from utils import (
    load_dataset,
    normalize_features,
    train_for_accuracy,
    train_for_parameters,
    train_for_layers,
    visualize_training,
    plot_acc_vs_parameters,
    plot_acc_vs_layers
)


def build_parser():
    parser = argparse.ArgumentParser()
    subcmd = parser.add_subparsers(dest='subcmd', help='training scheme')
    subcmd.required = True

    accuracy_parser = subcmd.add_parser('accuracy', help='reproduce the accuracy reported in papaer.')
    accuracy_parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    accuracy_parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    accuracy_parser.add_argument('--heads_1', type=int, default=8, help='number heads of the first attention layer')
    accuracy_parser.add_argument('--heads_2', type=int, default=1, help='number heads of the second attention layer')
    accuracy_parser.add_argument('--att_dropout', type=float, default=0.6, help='dropout rate of attention')
    accuracy_parser.add_argument('--input_dropout', type=float, default=0.6, help='dropout rate of input')
    accuracy_parser.add_argument('--trials', type=int, default=10, help='number of experiments')
    accuracy_parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    accuracy_parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    accuracy_parser.add_argument('--l2', type=float, default=5e-4, help='weight decay')
    accuracy_parser.add_argument('--gpu', type=bool, default=True, help='whether use GPU or not')

    parameter_parser = subcmd.add_parser('parameters', help='compare GAT with GCN in different number of parameters.')
    parameter_parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parameter_parser.add_argument('--trials', type=int, default=10, help='number of experiments')
    parameter_parser.add_argument('--gpu', type=bool, default=True, help='whether use GPU or not')
    parameter_parser.add_argument('--num_hidden_dim', nargs='+', default=[8, 16, 32, 64, 128])

    layers_parser = subcmd.add_parser('layers', help='train on different layers')
    layers_parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    layers_parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    layers_parser.add_argument('--heads_1', type=int, default=8, help='number heads of the first attention layer')
    layers_parser.add_argument('--heads_2', type=int, default=1, help='number heads of the second attention layer')
    layers_parser.add_argument('--att_dropout', type=float, default=0.6, help='dropout rate of attention')
    layers_parser.add_argument('--input_dropout', type=float, default=0.6, help='dropout rate of input')
    layers_parser.add_argument('--trials', type=int, default=5, help='number of experiments')
    layers_parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    layers_parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    layers_parser.add_argument('--l2', type=float, default=5e-4, help='weight decay')
    layers_parser.add_argument('--gpu', type=bool, default=True, help='whether use GPU or not')
    layers_parser.add_argument('--num_layers', nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset.title() not in ['Citeseer', 'Cora', 'Pubmed']:
        raise ValueError('Only \'Citeseer\', \'Cora\', \'Pubmed\' datasets are available.')

    dataset = load_dataset(args.dataset)
    data = dataset[0]
    data = normalize_features(data)

    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    if args.subcmd == 'accuracy':
        if args.hidden_dim % args.heads_1 != 0:
            raise ValueError('The value of the argument `hidden_dim` must be a multiple of the value of the argument \
                             `heads_1`.')

        hparams = {
            'input_dim': dataset.num_node_features,
            'hidden_dim': args.hidden_dim,
            'output_dim': dataset.num_classes,
            'heads_1': args.heads_1,
            'heads_2': args.heads_2,
            'att_dropout': args.att_dropout,
            'input_dropout': args.input_dropout,
        }

        histories = train_for_accuracy(model_class=GAT, hparams=hparams, data=data,
                                       epochs=args.epochs, lr=args.lr, l2=args.l2, trials=args.trials,
                                       device=device, model_path=f'models/gat_{args.dataset.lower()}.pth')
        visualize_training(histories, title=f'GAT / {args.dataset.title()}',
                           save_path=f'images/gat_{args.dataset.lower()}.png')

    elif args.subcmd == 'parameters':
        hidden_dim_list = [int(dim) for dim in args.num_hidden_dim]
        for i, dim in enumerate(hidden_dim_list):
            if dim < 1:
                raise ValueError(f'The {i+1}-th element of the argument `num_hidden_dim` should be a positive integer, \
                                 but get the value of {dim}.')
        
        # train GCN
        hparams = {
            'input_dim': dataset.num_node_features,
            'output_dim': dataset.num_classes,
            'dropout': 0.5
        }

        gcn_acc_list, gcn_params_list = \
            train_for_parameters(model_class=GCN, hparams=hparams, data=data,
                                 epochs=400, lr=0.01, hidden_dim_list=hidden_dim_list,
                                 trials=args.trials, device=device,
                                 model_path=f'models/gcn_{args.dataset.lower()}.pth')
        # train GAT
        hparams = {
            'input_dim': dataset.num_node_features,
            'output_dim': dataset.num_classes,
            'heads_1': 4,
            'heads_2': 1,
            'att_dropout': 0.6,
            'input_dropout': 0.6,
        }

        gat_acc_list, gat_params_list = \
            train_for_parameters(model_class=GAT, hparams=hparams, data=data,
                                 epochs=1000, lr=0.005, hidden_dim_list=hidden_dim_list,
                                 trials=args.trials, device=device,
                                 model_path=f'models/gat_{args.dataset.lower()}.pth')
        
        plot_acc_vs_parameters(gcn_acc_list, gcn_params_list, gat_acc_list, gat_params_list,
                               save_path=f'images/gcn_vs_gat_{args.dataset.lower()}.png')

    elif args.subcmd == 'layers':
        if args.hidden_dim % args.heads_1 != 0:
            raise ValueError('The value of the argument `hidden_dim` must be a multiple of the value of the argument \
                             `heads_1`.')

        num_layers = [int(layer) for layer in args.num_layers]
        for i, layer in enumerate(num_layers):
            if layer < 1:
                raise ValueError(f'The {i+1}-th element of the argument `num_layers` should be a positive integer, \
                                 but get the value of {layer}.')
        
        hparams = {
            'input_dim': dataset.num_node_features,
            'hidden_dim': args.hidden_dim,
            'output_dim': dataset.num_classes,
            'heads_1': args.heads_1,
            'heads_2': args.heads_2,
            'att_dropout': args.att_dropout,
            'input_dropout': args.input_dropout,
        }

        hparams['residual'] = False
        multigcn_no_residual_train_acc_list, multigcn_no_residual_val_acc_list, multigcn_no_residual_test_acc_list = \
            train_for_layers(model_class=MultiGAT, hparams=hparams, data=data,
                             epochs=args.epochs, lr=args.lr, l2=args.l2, num_layers=num_layers,
                             trials=args.trials, device=device,
                             model_path=f'models/multigat_no_residual_{args.dataset.lower()}.pth')
        
        hparams['residual'] = True
        multigcn_residual_train_acc_list, multigcn_residual_val_acc_list, multigcn_residual_test_acc_list = \
            train_for_layers(model_class=MultiGAT, hparams=hparams, data=data,
                             epochs=args.epochs, lr=args.lr, l2=args.l2, num_layers=num_layers,
                             trials=args.trials, device=device,
                             model_path=f'models/multigat_with_residual_{args.dataset.lower()}.pth')

        plot_acc_vs_layers(multigcn_no_residual_train_acc_list, multigcn_no_residual_test_acc_list,
                           multigcn_residual_train_acc_list, multigcn_residual_test_acc_list,
                           num_layers=num_layers, title='Multi-GAT',
                           save_path=f'images/multigat_{args.dataset.lower()}.png')


if __name__ == "__main__":
    main()