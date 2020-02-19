import os
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid


def create_dirs(file_path):
    dir_name = os.path.dirname(file_path)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_dataset(name):
    name = name.title()

    return Planetoid(root=name, name=name)


def move_data(data, device):
    if data.x.device == device:
        return data

    data = copy.deepcopy(data)
    
    data.x = data.x.to(device)
    data.y = data.y.to(device)
    data.edge_index = data.edge_index.to(device)

    return data


def normalize_features(data):
    data = copy.deepcopy(data)

    row_sum = data.x.sum(axis=-1, keepdim=True)
    data.x = data.x / (row_sum + 1e-8) # there are some data have features with all zero in Citeseer

    return data


def accuracy(output, labels):
    _, pred = output.max(dim=1)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    
    return correct / len(labels)


def compute_mean_error(array):
    mean = np.mean(array)
    std = np.std(array)

    return mean, std


def train_on_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    output = model(data)

    train_loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    train_acc = accuracy(output[data.train_mask], data.y[data.train_mask])

    train_loss.backward()
    optimizer.step()

    # print('check')
    # print(f'data: {data.x.device}')
    # print(f'model: {next(model.parameters()).device}')

    return train_loss, train_acc


def evaluate(model, data, mask):
    model.eval()

    with torch.no_grad():
        output = model(data)
        loss = F.nll_loss(output[mask], data.y[mask])
        acc = accuracy(output[mask], data.y[mask])

    return loss, acc


def train(model, data, epochs, lr, weight_decay=5e-4, model_path=None, verbose=True):
    if not model_path:
        print('Warning: you must assign `model_path` to save model.\n')
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss_values, train_acc_values = [], []
    val_loss_values, val_acc_values = [], []

    best = np.inf
    bad_counter = 0
    for epoch in tqdm(range(epochs), desc='Training', leave=verbose):
        if epoch == 0:
            print('       |     Trainging     |     Validation     |')
            print('       |-------------------|--------------------|')
            print(' Epoch |  loss    accuracy |  loss    accuracy  |')
            print('-------|-------------------|--------------------|')

        train_loss, train_acc = train_on_epoch(model, optimizer, data)
        train_loss_values.append(train_loss.item())
        train_acc_values.append(train_acc.item())

        val_loss, val_acc = evaluate(model, data, data.val_mask)
        val_loss_values.append(val_loss.item())
        val_acc_values.append(val_acc.item())

        if val_loss_values[-1] < best:
            bad_counter = 0
            log = '  {:3d}  | {:.4f}    {:.4f}  | {:.4f}    {:.4f}   |'.format(epoch+1,
                                                                               train_loss.item(),
                                                                               train_acc.item(),
                                                                               val_loss.item(),
                                                                               val_acc.item())
            
            if model_path:
                create_dirs(model_path)
                torch.save(model.state_dict(), model_path)
                log += ' save model to {}'.format(model_path)
            
            if verbose:
                tqdm.write(log)

            best = val_loss_values[-1]
        else:
            bad_counter += 1

    print('-------------------------------------------------')

    history = {
        'train_loss': train_loss_values,
        'val_loss': val_loss_values,
        'train_acc': train_acc_values,
        'val_acc': val_acc_values
    }

    return history


def train_for_accuracy(model_class, hparams, data, epochs, lr, trials, device, model_path):
    data = move_data(data, device)

    histories = []
    acc_values = []
    for trial in tqdm(range(trials), desc='Trials'):
        print(f'\n=== The {trial+1}-th experiment ===\n')

        model = model_class(**hparams).to(device)
        history = train(model, data, epochs, lr, model_path=model_path)
        histories.append(history)

        model.load_state_dict(torch.load(model_path))
        loss, acc = evaluate(model, data, data.test_mask)
        acc_values.append(acc.item())

        print('\ntest_loss = {:.4f}, test_acc = {:.4f}\n'.format(loss.item(), acc.item()))
    
    mean, std = compute_mean_error(acc_values)
    print('=== Final result ===\n')
    print('{:.1f} +- {:.1f}%'.format(mean * 100.0, std * 100.0))

    return histories


def train_for_layers(model_class, hparams, data, epochs, lr, num_layers, trials, device, model_path):
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
            _ = train(model, data, epochs, lr, model_path=model_path, verbose=False)

            model.load_state_dict(torch.load(model_path))
            _, train_acc = evaluate(model, data, data.train_mask)
            train_acc_values.append(train_acc.item())
            _, val_acc = evaluate(model, data, data.val_mask)
            val_acc_values.append(val_acc.item())
            _, test_acc = evaluate(model, data, data.test_mask)
            test_acc_values.append(test_acc.item())

            log = '| {}-th run | train_acc = {:.4f} | val_acc = {:.4f} | test_acc = {:.4f} |'
            tqdm.write(log.format(trial+1, train_acc.item(), val_acc.item(), test_acc.item()))

        print('-'*72)
        train_acc_list.append(train_acc_values)
        val_acc_list.append(val_acc_values)
        test_acc_list.append(test_acc_values)

    return train_acc_list, val_acc_list, test_acc_list


def visualize_training(histories, title, save_path=None):
    plt.figure(figsize=(13, 4))
    for i, metric in enumerate(['loss', 'acc']):
        plt.subplot(1, 2, i+1)

        train_key = 'train_loss' if metric == 'loss' else 'train_acc'
        val_key = 'val_loss' if metric == 'loss' else 'val_acc'
        
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