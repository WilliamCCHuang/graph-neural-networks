import argparse

import torch

from utils import (
    str2bool,
    check_args,
    check_args_to_run,
    load_dataset,
    train_for_citation,
    plot_training,
)


def build_parser():
    parser = argparse.ArgumentParser()
    subcmd = parser.add_subparsers(dest='subcmd', help='training scheme')
    subcmd.required = True

    citation_parser = subcmd.add_parser('citation', help='reproduce the accuracy reported in paper.')
    citation_parser.add_argument('--model', type=str, default='GCN', help='model name')
    citation_parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    citation_parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
    citation_parser.add_argument('--n_layers', type=int, default=2, help='number of layers')
    citation_parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    citation_parser.add_argument('--edge_dropout', type=float, default=0.5, help='edge dropout rate')
    citation_parser.add_argument('--layer_wise_dropedge', type=str2bool, default=False, help='whether or not to use layer-wise DropEdge')
    citation_parser.add_argument('--trials', type=int, default=5, help='number of experiments')
    citation_parser.add_argument('--epochs', type=int, default=400, help='number of epochs')
    citation_parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    citation_parser.add_argument('--l2', type=float, default=5e-4, help='weight decay')
    citation_parser.add_argument('--gpu', type=bool, default=True, help='whether use GPU or not')

    reddit_parser = subcmd.add_parser('reddit', help='reproduce the result in PPI dataset.')
    reddit_parser.add_argument('--model', type=str, default='GCN', help='model name')
    reddit_parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    reddit_parser.add_argument('--n_layers', type=int, default=2, help='number of layers')
    reddit_parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    reddit_parser.add_argument('--edge_dropout', type=float, default=0.5, help='edge dropout rate')
    reddit_parser.add_argument('--layer_wise_dropedge', type=str2bool, default=False, help='whether or not to use layer-wise DropEdge')
    reddit_parser.add_argument('--trials', type=int, default=1, help='number of experiments')
    reddit_parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    reddit_parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    reddit_parser.add_argument('--l2', type=float, default=0.0, help='weight decay')
    reddit_parser.add_argument('--lr_scheduler', type=str2bool, default=False, help='learning rate scheduler')
    reddit_parser.add_argument('--gpu', type=bool, default=True, help='whether use GPU or not')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    check_args(args)
    check_args_to_run(args)

    device = torch.device('cuda') if args.gpu and torch.cuda.is_available() else torch.device('cpu')

    if args.subcmd == 'citation':
        dataset = load_dataset(args.dataset)

        hparams = {
            'input_dim': dataset.num_node_features,
            'hidden_dim': args.hidden_dim,
            'output_dim': dataset.num_classes,
            'n_layers': args.n_layers,
            'dropout': args.dropout,
            'edge_dropout': args.edge_dropout,
            'layer_wise_dropedge': args.layer_wise_dropedge
        }

        model_name = f'{args.model}-{args.n_layers}-hidden_dim={args.hidden_dim}-dropout={args.dropout}-edge_dropout={args.edge_dropout}-LW={args.layer_wise_dropedge}'
        model_path = f'pretrained_models/{model_name}_{args.dataset.lower()}' + '_{}.pth'

        histories = train_for_citation(model_name=args.model, hparams=hparams, dataset=dataset,
                                       epochs=args.epochs, lr=args.lr, l2=args.l2, trials=args.trials,
                                       device=device, model_path=model_path)
        
        plot_training(histories, title=f'{model_name} / {args.dataset.title()}',
                      metric_name='accuracy', save_path=f'images/{model_name}_{args.dataset.lower()}.png')

    # elif args.subcmd == 'reddit':  # TODO:
    #     for i, head in enumerate(args.heads):
    #         if i < len(args.heads)-1:
    #             if args.hidden_dim % head != 0:
    #                 raise ValueError('The value of the argument `hidden_dim` must be a multiple of the value of the argument \
    #                                  `heads`.')

    #     datasets = load_dataset('ppi')

    #     hparams = {
    #         'input_dim': datasets[0].num_node_features,
    #         'hidden_dim': args.hidden_dim,
    #         'output_dim': datasets[0].num_classes,
    #         'num_layer': len(args.heads),
    #         'heads': args.heads,
    #         'residual': args.residual,
    #         'att_dropout': args.att_dropout,
    #         'input_dropout': args.input_dropout,
    #     }

    #     histories = train_for_ppi(model_class=MultiGAT, hparams=hparams, datasets=datasets,
    #                               epochs=args.epochs, lr=args.lr, l2=args.l2, trials=args.trials,
    #                               device=device, model_path='pretrained_models/gat_ppi_{}.pth',
    #                               lr_scheduler=args.lr_scheduler)
    #     plot_training(histories, title=f'GAT / PPI', metric_name='f1-score', save_path='images/gat_ppi.png')


if __name__ == "__main__":
    main()
