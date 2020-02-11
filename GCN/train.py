import argparse

from models import GCN, MultiGCN
from utils import (
    load_dataset,
    normalize_features,
    train_for_accuracy,
    train_for_layers,
    visualize_training,
    plot_acc_vs_layers
)


def build_parser():
    parser = argparse.ArgumentParser()
    subcmd = parser.add_subparsers(dest='subcmd', help='training scheme')
    subcmd.required = True

    accuracy_parser = subcmd.add_parser('accuracy', help='reproduce the accuracy reported in papaer.')
    accuracy_parser.add_argument('--dataset', type=str, default='Cora', help='dataset')
    accuracy_parser.add_argument('--hidden_dim', type=int, default=16, help='hidden dimension')
    accuracy_parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    accuracy_parser.add_argument('--trials', type=int, default=10, help='number of experiments')
    accuracy_parser.add_argument('--epochs', type=int, default=400, help='number of epochs')
    accuracy_parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    
    layers_parser = subcmd.add_parser('layers', help='train on different layers')
    layers_parser.add_argument('--dataset', type=str, default='Cora', help='dataset')
    layers_parser.add_argument('--hidden_dim', type=int, default=16, help='hidden dimension')
    layers_parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    layers_parser.add_argument('--trials', type=int, default=5, help='number of experiments d')
    layers_parser.add_argument('--epochs', type=int, default=400, help='number of epochs')
    layers_parser.add_argument('--lr', type=float, default=0.01, help='learning rate') 
    layers_parser.add_argument('--num_layers', nargs='+')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset not in ['Citeseer', 'Cora', 'Pubmed']:
        raise ValueError('Only \'Citeseer\', \'Cora\', \'Pubmed\' datasets are available.')

    dataset = load_dataset(args.dataset)
    data = dataset[0]
    data = normalize_features(data)
    hparams = {
        'input_dim': dataset.num_node_features,
        'hidden_dim': args.hidden_dim,
        'output_dim': dataset.num_classes,
        'dropout': args.dropout
    }

    if args.subcmd == 'accuracy':
        histories = train_for_accuracy(model_class=GCN, hparams=hparams, data=data,
                                       epochs=args.epochs, lr=args.lr, trials=args.trials, model_path='gcn.pth')
        visualize_training(histories, title='GCN', save_path='gcn.png')
    elif args.subcmd == 'layers':
        num_layers = [int(layer) for layer in args.num_layers]
        for i, layer in enumerate(num_layers):
            if layer < 1:
                raise ValueError(f'The {i+1}-th element of the argument `num_layers` should be a positive integer, but get the value of {layer}.')
        
        hparams['residual'] = False
        multigcn_no_residual_train_acc_list, multigcn_no_residual_val_acc_list, multigcn_no_residual_test_acc_list = \
            train_for_layers(model_class=MultiGCN, hparams=hparams, data=data,
                             epochs=args.epochs, lr=args.lr, num_layers=num_layers,
                             trials=args.trials, model_path='multigcn_no_residual.pth')
        
        hparams['residual'] = True
        multigcn_residual_train_acc_list, multigcn_residual_val_acc_list, multigcn_residual_test_acc_list = \
            train_for_layers(model_class=MultiGCN, hparams=hparams, data=data,
                             epochs=args.epochs, lr=args.lr, num_layers=num_layers,
                             trials=args.trials, model_path='multigcn_with_residual.pth')

        plot_acc_vs_layers(multigcn_no_residual_train_acc_list, multigcn_no_residual_test_acc_list,
                           multigcn_residual_train_acc_list, multigcn_residual_test_acc_list,
                           num_layers=num_layers, title='Multi-GCN', save_path='multi_gcn.png')


if __name__ == "__main__":
    main()