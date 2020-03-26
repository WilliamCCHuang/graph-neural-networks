import os
import copy
import numpy as np
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid, PPI


def create_dirs(file_path):
    dir_name = os.path.dirname(file_path)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def normalize_features(data):
    row_sum = data.x.sum(axis=-1, keepdim=True)
    data.x = data.x / (row_sum + 1e-8) # there are some data having features with all zero in Citeseer

    return data


def load_dataset(name):
    name = name.lower()

    if name in ['cora', 'citeseer', 'pubmed']:
        return Planetoid(root=name, name=name, pre_transform=normalize_features)
    elif name == 'ppi':
        datasets = []
        for split in ['train', 'val', 'test']:
            dataset = PPI(root='PPI', split=split, pre_transform=normalize_features)

            datasets.append(dataset)
            
        return datasets
        

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
        
        if mask is not None: # for citation
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

        if mask is not None: # for citation
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


def train(model, dataloaders, criterion, metric_func, epochs, lr, weight_decay, device, model_path, verbose=True):
    if model_path is None:
        print('Warning: you must assign `model_path` to save model.\n')
    
    train_dataloader, val_dataloader, _ = dataloaders
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

        log = '  {:3d}  | {:.4f}    {:.4f} | {:.4f}    {:.4f} |'
        log = log.format(epoch+1, train_loss, train_metric, val_loss, val_metric)

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


def train_for_citation(model_class, hparams, dataset, epochs, lr, l2, trials, device, model_path):
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
        acc_values.append(acbest_accc.item())

        print('\ntest_loss = {:.4f}, test_acc = {:.4f}\n'.format(best_loss, best_acc))
    
    mean, std = compute_mean_error(acc_values)
    print('=== Final result ===\n')
    print('{:.1f} +- {:.1f}%'.format(mean * 100.0, std * 100.0))

    return histories


def train_for_ppi(model_class, hparams, datasets, epochs, lr, l2, trials, device, model_path):
    dataloaders = [
        DataLoader(datasets[0], batch_size=2, shuffle=True),
        DataLoader(datasets[1], batch_size=2, shuffle=False),
        DataLoader(datasets[2], batch_size=2, shuffle=False),
    ]

    histories = []
    f1_values = []
    for trial in tqdm(range(trials), desc='Trials'):
        print(f'\n=== The {trial+1}-th experiment ===\n')

        model = model_class(**hparams).to(device)
        criterion = nn.BCEWithLogitsLoss()
        metric_func = f1_score

        history = train(model, dataloaders, criterion, metric_func, epochs, lr, l2, device, model_path)
        histories.append(history)

        best_loss, best_f1 = best_result(model, model_path, dataloaders[-1], criterion, metric_func, device)
        f1_values.append(best_f1)

        print('\ntest_loss = {:.4f}, test_f1 = {:.4f}\n'.format(best_loss, best_f1))
    
    mean, std = compute_mean_error(f1_values)
    print('=== Final result ===\n')
    print('{:.1f} +- {:.1f}%'.format(mean * 100.0, std * 100.0))

    return histories


def train_for_parameters(model_class, hparams, data, epochs, lr, hidden_dim_list, trials, device, model_path):
    data = move_data(data, device)

    acc_list = []
    params_list = []
    for hidden_dim in tqdm(hidden_dim_list, desc='Hidden Dimension'):
        tqdm.write(f'\n=== Hidden dimension = {hidden_dim} ===')

        acc_values = []
        hparams['hidden_dim'] = hidden_dim
        for trial in tqdm(range(trials), desc='Trials', leave=False):
            model = model_class(**hparams).to(device)
            _ = train(model, data, epochs=epochs, lr=lr, model_path=model_path, verbose=False)

            model.load_state_dict(torch.load(model_path))
            loss, acc = evaluate(model, data, data.test_mask)
            acc_values.append(acc)

            tqdm.write('{:2d}-th run: test_loss = {:.4f} test_acc = {:.4f}\n'.format(trial+1, loss, acc))

        acc_list.append(acc_values)
        params_list.append(count_params(model))

    return acc_list, params_list


def train_for_layers(model_class, hparams, data, epochs, lr, l2, num_layers, trials, device, model_path):
    data = move_data(data, device)

    train_acc_list, val_acc_list, test_acc_list = [], [], []
    for num_layer in tqdm(num_layers, desc='Layers'):
        hparams['num_layer'] = num_layer
        model = model_class(**hparams)

        tqdm.write(f'\n=== Number of layer: {num_layer} ===')
        tqdm.write('\nModel Structure')
        tqdm.write(model.__str__())
        tqdm.write('\n' + '-'*72)

        train_acc_values = []
        val_acc_values = []
        test_acc_values = []
        for trial in tqdm(range(trials), desc='Trials', leave=False):
            model = model_class(**hparams).to(device)
            _ = train(model, data, epochs=epochs, lr=lr, weight_decay=l2, model_path=model_path, verbose=False)

            model.load_state_dict(torch.load(model_path))
            _, train_acc = evaluate(model, data, data.train_mask)
            train_acc_values.append(train_acc)
            _, val_acc = evaluate(model, data, data.val_mask)
            val_acc_values.append(val_acc)
            _, test_acc = evaluate(model, data, data.test_mask)
            test_acc_values.append(test_acc)

            log = '| {}-th run | train_acc = {:.4f} | val_acc = {:.4f} | test_acc = {:.4f} |'
            log = log.format(trial+1, train_acc, val_acc, test_acc)
            tqdm.write(log)

        print('-'*72)
        train_acc_list.append(train_acc_values)
        val_acc_list.append(val_acc_values)
        test_acc_list.append(test_acc_values)

    return train_acc_list, val_acc_list, test_acc_list


def plot_training(histories, title, metric_name, save_path=None):
    plt.figure(figsize=(13, 4))
    for i, metric in enumerate(['loss', metric_name]):
        plt.subplot(1, 2, i+1)

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
            plt.fill_between(x=np.arange(1, len(value_mean)+1),
                             y1=value_mean-value_std,
                             y2=value_mean+value_std,
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

    if save_path:
        create_dirs(save_path)
        plt.savefig(save_path)


def plot_acc_vs_parameters(gcn_acc_list, gcn_params_list, gat_acc_list, gat_params_list, save_path=None):
    gcn_mean_acc = np.mean(gcn_acc_list, axis=1)
    gat_mean_acc = np.mean(gat_acc_list, axis=1)
    gcn_std_acc = np.std(gcn_acc_list, axis=1)
    gat_std_acc = np.std(gat_acc_list, axis=1)

    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 15})
    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_facecolor(('w'))
    ax.spines['bottom'].set_color('k')
    ax.spines['top'].set_color('k') 
    ax.spines['right'].set_color('k')
    ax.spines['left'].set_color('k')

    plt.plot(gcn_params_list, gcn_mean_acc, c='b', marker='o', label='GCN')
    plt.plot(gat_params_list, gat_mean_acc, c='r', marker='o', label='GAT')
    plt.fill_between(x=gcn_params_list, y1=gcn_mean_acc-gcn_std_acc, y2=gcn_mean_acc+gcn_std_acc, color='b', alpha=0.2)
    plt.fill_between(x=gat_params_list, y1=gat_mean_acc-gat_std_acc, y2=gat_mean_acc+gat_std_acc, color='r', alpha=0.2)

    plt.ylim(0.79, 0.85)
    plt.title('GCN / GAT')
    plt.xlabel('Parameters')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right', framealpha=0.)
    plt.xticks(plt.xticks()[0], [str(int(p)//1000) + 'k' if p != 0 else '0' for p in plt.xticks()[0]])

    if save_path:
        create_dirs(save_path)
        plt.savefig(save_path)


def plot_acc_vs_layers(no_residual_train_acc_list, no_residual_test_acc_list,
                       residual_train_acc_list, residual_test_acc_list,
                       num_layers, title, save_path=None):
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 15})
    _, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.set_facecolor(('w'))
    ax.spines['bottom'].set_color('k')
    ax.spines['top'].set_color('k') 
    ax.spines['right'].set_color('k')
    ax.spines['left'].set_color('k')

    for i, (train_acc, test_acc) in enumerate([[no_residual_train_acc_list,
                                                no_residual_test_acc_list],
                                               [residual_train_acc_list,
                                                residual_test_acc_list]]):
        suffix = ' (Residual)' if i == 1 else ''

        train_mean = np.mean(train_acc, axis=1)
        train_std = np.std(train_acc, axis=1)
        test_mean = np.mean(test_acc, axis=1)
        test_std = np.std(test_acc, axis=1)

        train_color = 'r' if i == 0 else 'g'
        test_color = 'b' if i == 0 else 'purple'
        linestyle = '--' if i == 0 else '-'

        ax.plot(num_layers, train_mean, c=train_color, marker='o', linestyle=linestyle, label='Train'+suffix)
        ax.plot(num_layers, test_mean, c=test_color, marker='o', linestyle=linestyle, label='Test'+suffix)
        plt.fill_between(x=num_layers,
                        y1=train_mean-train_std,
                        y2=train_mean+train_std,
                        color=train_color, alpha=0.1)
        plt.fill_between(x=num_layers,
                        y1=test_mean-test_std,
                        y2=test_mean+test_std,
                        color=test_color, alpha=0.1)

    plt.title(title)
    plt.xlabel('Number of layers')
    plt.ylabel('Accuracy')
    plt.ylim(0.0, 1.01)
    plt.xticks(num_layers)
    plt.yticks(np.arange(0.0, 1.05, 0.1))
    plt.legend(framealpha=0.)

    if save_path:
        create_dirs(save_path)
        plt.savefig(save_path)