import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid

from models import GCN, ResGCN, JKNet, IncepGCN


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_args(args):
    if args.model not in ['GCN', 'ResGCN', 'IncepGCN', 'JKNet']:
        raise ValueError('The argument `model` can only be one of GCN, ResGCN, IncepGCN, or JKNet')

    if args.dataset.lower() not in ['citeseer', 'cora', 'pubmed']:
        raise ValueError('Only \'Citeseer\', \'Cora\', \'Pubmed\' datasets are available.')
    
    
def check_args_to_run(args):
    print('\nConfiguration:')
    for arg in vars(args):
        print(f'* {arg} = {getattr(args, arg)}')

    result = input('\nDo you wanna run? (y/n): ')

    if result in ['y', 'Y']:
        print('Run!\n')
    elif result in ['n', 'N']:
        print('Exit!\n')
        exit()
    else:
        raise ValueError('You can only type in \'y\', \'Y\', \'n\', or \'N\'.')


def create_dirs(file_path):
    dir_name = os.path.dirname(file_path)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def pre_transform(data):
    # normalize_features
    row_sum = data.x.sum(axis=-1, keepdim=True)
    data.x = data.x / (row_sum + 1e-8)  # there are some data having features with all zero in Citeseer

    # modidy train_mask
    data.train_mask = ~(data.val_mask + data.test_mask)
    assert sum(data.train_mask).item() == 1208, sum(data.train_mask)

    return data


def load_dataset(name):
    name = name.lower()

    if name in ['cora', 'citeseer', 'pubmed']:
        return Planetoid(root=name, name=name, pre_transform=pre_transform)
    elif name == 'reddit':
        # TODO:
        raise NotImplementedError
        

def accuracy(logits, labels):
    _, pred = logits.max(dim=1)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    
    return correct / len(labels)


def f1_score(logits, labels):
    preds = logits > 0

    preds_p = preds.sum().double()
    labels_p = labels.sum().double()
    tp = (preds * labels).sum().double()

    recall = tp / labels_p
    precision = tp / preds_p

    return 2 * recall * precision / (recall + precision)


def compute_mean_error(array):
    mean = np.mean(array)
    std = np.std(array)

    return mean, std


def train_on_epoch(model, optimizer, dataloader, criterion, metric_func, device):
    model.train()
    optimizer.zero_grad()

    for data in dataloader:
        data = data.to(device)
        logits = model(data)
        y = data.y

        mask = getattr(data, 'train_mask', None)
        
        if mask is not None:  # for citation
            logits = logits[mask]
            y = y[mask]
    
        train_loss = criterion(logits, y)
        train_metric = metric_func(logits, y)

        train_loss.backward()
        optimizer.step()

    return train_loss.item(), train_metric.item()


def evaluate(model, dataloader, criterion, metric_func, mode, device):
    assert mode in ['val', 'test'], '`mode` can only be `val` or `test`'

    model.eval()
    data = next(iter(dataloader)).to(device)
    y = data.y

    mask_name = 'val_mask' if mode == 'val' else 'test_mask'
    mask = getattr(data, mask_name, None)

    with torch.no_grad():
        logits = model(data)

        if mask is not None:  # for citation
            logits = logits[mask]
            y = y[mask]

        loss = criterion(logits, y)
        metric = metric_func(logits, y)

    return loss.item(), metric.item()


def best_result(model, model_path, dataloader, criterion, metric_func, device):
    if os.path.exists(model_path.format('loss')):
        model.load_state_dict(torch.load(model_path.format('loss')))
        loss_, metric_ = evaluate(model, dataloader, criterion, metric_func, mode='test', device=device)
    
    model.load_state_dict(torch.load(model_path.format('metric')))
    loss, metric = evaluate(model, dataloader, criterion, metric_func, mode='test', device=device)

    loss, metric = (loss, metric) if metric > metric_ else (loss_, metric_)

    return loss, metric


def train(model, dataloaders, criterion, metric_func, epochs, lr, weight_decay, device, model_path, lr_scheduler=False, verbose=True):
    if model_path is None:
        print('Warning: you must assign `model_path` to save model.\n')
    
    train_dataloader, val_dataloader, _ = dataloaders
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if lr_scheduler:
        print('use lr scheduler...')
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=100, verbose=True)

    train_loss_values, train_metric_values = [], []
    val_loss_values, val_metric_values = [], []

    best_loss = np.inf
    best_metric = 0.0
    for epoch in tqdm(range(epochs), desc='Training', leave=verbose):
        if epoch == 0:
            print('       |    Trainging     |    Validation    |')
            print('       |------------------|------------------|')
            print(' Epoch |  loss    metric  |  loss    metric  |')
            print('-------|------------------|------------------|')

        train_loss, train_metric = train_on_epoch(model, optimizer, train_dataloader, criterion, metric_func, device)
        train_loss_values.append(train_loss)
        train_metric_values.append(train_metric)

        val_loss, val_metric = evaluate(model, val_dataloader, criterion, metric_func, mode='val', device=device)
        val_loss_values.append(val_loss)
        val_metric_values.append(val_metric)

        if lr_scheduler:
            scheduler.step(val_metric)

        log = '  {:3d}  | {:.4f}    {:.4f} | {:.4f}    {:.4f} |'
        log = log.format(epoch + 1, train_loss, train_metric, val_loss, val_metric)

        if val_loss_values[-1] < best_loss or val_metric_values[-1] > best_metric:
            create_dirs(model_path)

            if val_loss_values[-1] < best_loss:
                path = model_path.format('loss')
                best_loss = val_loss_values[-1]
            if val_metric_values[-1] > best_metric:
                path = model_path.format('metric')
                best_metric = val_metric_values[-1]
            
            torch.save(model.state_dict(), path)
            log += ' save model to {}'.format(path)
        if verbose:
            tqdm.write(log)

    print('-------------------------------------------------')

    history = {
        'train_loss': train_loss_values,
        'val_loss': val_loss_values,
        'train_metric': train_metric_values,
        'val_metric': val_metric_values
    }

    return history


def train_for_citation(model_name, hparams, dataset, epochs, lr, l2, trials, device, model_path):
    if model_name == 'GCN':
        model_class = GCN
    elif model_name == 'ResGCN':
        model_class == ResGCN
    elif model_name == 'JKNet':
        model_class = JKNet
        hparams.pop('layer_wise_dropedge')
    else:
        model_class = IncepGCN
        hparams.pop('layer_wise_dropedge')
    
    dataloaders = [
        DataLoader(dataset, batch_size=1),
        DataLoader(dataset, batch_size=1),
        DataLoader(dataset, batch_size=1)
    ]

    histories = []
    acc_values = []
    for trial in tqdm(range(trials), desc='Trials'):
        print(f'\n=== The {trial+1}-th experiment ===\n')

        model = model_class(**hparams).to(device)
        criterion = nn.CrossEntropyLoss()
        metric_func = accuracy

        history = train(model, dataloaders, criterion, metric_func, epochs, lr, l2, device, model_path)
        histories.append(history)

        best_loss, best_acc = best_result(model, model_path, dataloaders[-1], criterion, metric_func, device)
        acc_values.append(best_acc)

        print('\ntest_loss = {:.4f}, test_acc = {:.4f}\n'.format(best_loss, best_acc))
    
    mean, std = compute_mean_error(acc_values)
    print('=== Final result ===\n')
    print('{:.1f} +- {:.1f}%'.format(mean * 100.0, std * 100.0))

    return histories


def plot_training(histories, title, metric_name, save_path=None):
    plt.figure(figsize=(13, 4))
    for i, metric in enumerate(['loss', metric_name]):
        plt.subplot(1, 2, i + 1)

        train_key = 'train_loss' if metric == 'loss' else 'train_metric'
        val_key = 'val_loss' if metric == 'loss' else 'val_metric'
        
        train_values = np.array([history[train_key] for history in histories])
        val_values = np.array([history[val_key] for history in histories])
        
        for values, label, c in zip([train_values, val_values],
                                    ['training', 'validation'],
                                    ['b', 'r']):
            value_mean = np.mean(values, axis=0)
            value_std = np.std(values, axis=0)

            plt.plot(value_mean, c=c, label=label)
            plt.fill_between(x=np.arange(1, len(value_mean) + 1),
                             y1=value_mean - value_std,
                             y2=value_mean + value_std,
                             color=c, alpha=0.2)
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel(metric)
        if metric == 'loss':
            plt.ylim(0.0, 2.0)
        else:
            plt.ylim(0.0, 1.2)
        if title:
            plt.title(title)

        plt.tight_layout()

    if save_path:
        create_dirs(save_path)
        plt.savefig(save_path)


if __name__ == "__main__":
    dataset = Planetoid(root='./data', name='Cora')
    train_mask = dataset[0].train_mask
    print('Before')
    print(sum(train_mask))
    print(train_mask)
