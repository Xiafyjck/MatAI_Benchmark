import csv
import deepchem as dc
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os

tasks = ['band_gap']
featurizer = dc.feat.CGCNNFeaturizer()
loader = dc.data.JsonLoader(
    tasks=tasks,
    feature_field='structure',
    label_field='band_gap',
    featurizer=featurizer,
)

input_file = 'data/bandgap_data_155351.json'

print(f'Loading data from {input_file}')
dataset = loader.create_dataset(input_file)
print(f'Dataset loaded with {len(dataset)} samples')

class fileSplitter(dc.splits.Splitter):
    def __init__(self, order_file):
        self.order_file = order_file

    def split(self, dataset, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=None, log_every_n=None):
        with open(self.order_file, 'r') as f:
            order = [int(line.strip()) for line in f.readlines()]
        np.testing.assert_almost_equal(sum([frac_train, frac_valid, frac_test]), 1.0)
        assert len(order) == len(dataset)
        num_datapoints = len(order)
        train_cutoff = int(num_datapoints * frac_train)
        valid_cutoff = int(num_datapoints * (frac_train + frac_valid))
        return (order[:train_cutoff], order[train_cutoff:valid_cutoff], order[valid_cutoff:])


mse_metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
mae_metric = dc.metrics.Metric(dc.metrics.mae_score)

# 4 fold cross validation
i = 4
print(f'Training on fold {i}')

splitter = fileSplitter(f'data/fold_{i}_train_val_test.txt')
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset, frac_train=0.6, frac_valid=0.2, frac_test=0.2)

model = dc.models.CGCNNModel(
    n_tasks=len(tasks),
    mode = 'regression',
    learning_rate=0.01
)

train_losses = []
valid_losses = []
test_losses = []
train_maes = []
valid_maes = []
test_maes = []

for epoch in tqdm.tqdm(range(100), desc=f'Fold {i}'):
    model.fit(train_dataset, nb_epoch=1)
    train_loss = model.evaluate(train_dataset, [mse_metric])['mean_squared_error']
    valid_loss = model.evaluate(valid_dataset, [mse_metric])['mean_squared_error']
    test_loss = model.evaluate(test_dataset, [mse_metric])['mean_squared_error']
    train_mae = model.evaluate(train_dataset, [mae_metric])['mae_score']
    valid_mae = model.evaluate(valid_dataset, [mae_metric])['mae_score']
    test_mae = model.evaluate(test_dataset, [mae_metric])['mae_score']
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    test_losses.append(test_loss)
    train_maes.append(train_mae)
    valid_maes.append(valid_mae)
    test_maes.append(test_mae)

# find the best valid loss
best_valid_loss = min(valid_losses)
best_epoch = valid_losses.index(best_valid_loss)
print(f'Best valid loss: {best_valid_loss} at epoch {best_epoch}')
print(f'test loss: {test_losses[best_epoch]}')
print(f'train loss: {train_losses[best_epoch]}')
print(f'valid mae: {valid_maes[best_epoch]}')
print(f'test mae: {test_maes[best_epoch]}')
print(f'train mae: {train_maes[best_epoch]}')

with open(f'results/cgcnn_bandgap_crossval_results.txt', 'a') as f:
    f.write(f'Fold {i}: Best valid loss: {best_valid_loss} at epoch {best_epoch}\n')
    f.write(f'test loss: {test_losses[best_epoch]}\n')
    f.write(f'train loss: {train_losses[best_epoch]}\n')
    f.write(f'valid mae: {valid_maes[best_epoch]}\n')
    f.write(f'test mae: {test_maes[best_epoch]}\n')
    f.write(f'train mae: {train_maes[best_epoch]}\n')

with open(f'results/fold_{i}.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'valid_loss', 'test_loss', 'train_mae', 'valid_mae', 'test_mae'])
    for j in range(len(train_losses)):
        writer.writerow([j, train_losses[j], valid_losses[j], test_losses[j], train_maes[j], valid_maes[j], test_maes[j]])