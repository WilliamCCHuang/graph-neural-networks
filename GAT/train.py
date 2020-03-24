import argparse

import torch

from models import GCN, GAT, MultiGAT
from utils import (
    load_dataset,
    normalize_features,
    train_for_citation,
    train_for_ppi,
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

    citation_parser = subcmd.add_parser('citation', help='reproduce the accuracy reported in paper.')
    citation_parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    citation_parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    citation_parser.add_argument('--heads_1', type=int, default=8, help='number heads of the first attention layer')
    citation_parser.add_argument('--heads_2', type=int, default=1, help='number heads of the second attention layer')
    citation_parser.add_argument('--att_dropout', type=float, default=0.6, help='dropout rate of attention')
    citation_parser.add_argument('--input_dropout', type=float, default=0.6, help='dropout rate of input')
    citation_parser.add_argument('--trials', type=int, default=10, help='number of experiments')
    citation_parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    citation_parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    citation_parser.add_argument('--l2', type=float, default=5e-4, help='weight decay')
    citation_parser.add_argument('--gpu', type=bool, default=True, help='whether use GPU or not')

    ppi_parser = subcmd.add_parser('ppi', help='reproduce the result in PPI dataset.')
    ppi_parser.add_argument('--hidden_dim', type=int, default=1024, help='hidden dimension')
    ppi_parser.add_argument('--heads', nargs='+', default=[4, 4, 6], help='number of heads in each layer')
    ppi_parser.add_argument('--residual', type=bool, default=True, help='residual connection')
    ppi_parser.add_argument('--att_dropout', type=float, default=0.0, help='dropout rate of attention')
    ppi_parser.add_argument('--input_dropout', type=float, default=0.0, help='dropout rate of input')
    ppi_parser.add_argument('--trials', type=int, default=10, help='number of experiments')
    ppi_parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    ppi_parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    ppi_parser.add_argument('--l2', type=float, default=0.0, help='weight decay')
    ppi_parser.add_argument('--gpu', type=bool, default=True, help='whether use GPU or not')

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

    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    if args.subcmd == 'citation':
        if args.dataset.lower() not in ['citeseer', 'cora', 'pubmed']:
            raise ValueError('Only \'Citeseer\', \'Cora\', \'Pubmed\' datasets are available.')

        if args.hidden_dim % args.heads_1 != 0:
            raise ValueError('The value of the argument `hidden_dim` must be a multiple of the value of the argument \
                             `heads_1`.')

        dataset = load_dataset(args.dataset)

        hparams = {
            'input_dim': dataset.num_node_features,
            'hidden_dim': args.hidden_dim,
            'output_dim': dataset.num_classes,
            'heads_1': args.heads_1,
            'heads_2': args.heads_2,
            'att_dropout': args.att_dropout,
            'input_dropout': args.input_dropout,
        }

        histories = train_for_citation(model_class=GAT, hparams=hparams, dataset=dataset,
                                       epochs=args.epochs, lr=args.lr, l2=args.l2, trials=args.trials,
                                       device=device, model_path=f'pretrained_models/gat_{args.dataset.lower()}.pth')
        visualize_training(histories, title=f'GAT / {args.dataset.title()}',
                           metric_name='accuracy', save_path=f'images/gat_{args.dataset.lower()}.png')

    elif args.subcmd == 'ppi':
        for i, head in enumerate(args.heads):
            if i < len(args.heads)-1:
                if args.hidden_dim % head != 0:
                    raise ValueError('The value of the argument `hidden_dim` must be a multiple of the value of the argument \
                                     `heads`.')

        datasets = load_dataset('ppi')

        hparams = {
            'input_dim': datasets[0].num_node_features,
            'hidden_dim': args.hidden_dim,
            'output_dim': datasets[0].num_classes,
            'num_layer': len(args.heads)
            'heads': args.heads,
            'residual': args.residual,
            'att_dropout': args.att_dropout,
            'input_dropout': args.input_dropout,
        }

        histories = train_for_ppi(model_class=MultiGAT, hparams=hparams, datasets=datasets,
                                  epochs=args.epochs, lr=args.lr, l2=args.l2, trials=args.trials,
                                  device=device, model_path=f'pretrained_models/gat_ppi.pth')
        visualize_training(histories, title=f'GAT / PPI', metric_name='f1-score', save_path=f'images/gat_ppi.png')

    elif args.subcmd == 'parameters':
        hidden_dim_list = [int(dim) for dim in args.num_hidden_dim]
        for i, dim in enumerate(hidden_dim_list):
            if dim < 1:
                raise ValueError(f'The {i+1}-th element of the argument `num_hidden_dim` should be a positive integer, \
                                 but get the value of {dim}.')
        
        dataset = load_dataset(args.dataset)

        # train GCN
        hparams = {
            'input_dim': dataset.num_node_features,
            'output_dim': dataset.num_classes,
            'dropout': 0.5
        }

        gcn_acc_list, gcn_params_list = train_for_parameters(model_class=GCN, hparams=hparams, data=data,
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

        gat_acc_list, gat_params_list = train_for_parameters(model_class=GAT, hparams=hparams, data=data,
                                                             epochs=1000, lr=0.005, hidden_dim_list=hidden_dim_list,
                                                             trials=args.trials, device=device,
                                                             model_path=f'pretrained_models/gat_{args.dataset.lower()}.pth')
        
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

        dataset = load_dataset(args.dataset)
        
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
                             model_path=f'pretrained_models/multigat_no_residual_{args.dataset.lower()}.pth')
        
        hparams['residual'] = True
        multigcn_residual_train_acc_list, multigcn_residual_val_acc_list, multigcn_residual_test_acc_list = \
            train_for_layers(model_class=MultiGAT, hparams=hparams, data=data,
                             epochs=args.epochs, lr=args.lr, l2=args.l2, num_layers=num_layers,
                             trials=args.trials, device=device,
                             model_path=f'pretrained_models/multigat_with_residual_{args.dataset.lower()}.pth')

        plot_acc_vs_layers(multigcn_no_residual_train_acc_list, multigcn_no_residual_test_acc_list,
                           multigcn_residual_train_acc_list, multigcn_residual_test_acc_list,
                           num_layers=num_layers, title='Multi-GAT',
                           save_path=f'images/multigat_{args.dataset.lower()}.png')


if __name__ == "__main__":
    main()