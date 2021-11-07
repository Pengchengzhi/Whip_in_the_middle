# Train Resnet Model
import torch
import numpy as np
import csv
import os
import time
from model import ResnetModel
from dataloader import get_train_val_test_loaders
from hierarchicalloss import HierarchicalLoss

def train_epoch(data_loader, model, loss_func, optimizer):
    for X, y in data_loader:
        optimizer.zero_grad()
        output = model.forward(X)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()

def evaluate_epoch(train_loader, val_loader, model, loss_func, epoch_num, result_path):
    # Evaluate model on validation set
    out_path = result_path + '/'
    val_preds, train_preds = [], [], []
    train_losses, val_losses = [], []
    num_correct_val = 0
    num_correct_train = 0

    for X,y in train_loader:
        with torch.no_grad():
            output_vector = model(X)
            prediction = model.vector_to_prediction(output_vector)
            loss = loss_func(output_vector, y)
            train_preds.append(prediction)
            train_losses.append(loss)
            num_correct_train += int((prediction == y))
    for X,y in val_loader:
        with torch.no_grad():
            output_vector = model(X)
            prediction = model.vector_to_prediction(output_vector)
            loss = loss_func(output_vector, y)
            val_preds.append(prediction)
            val_losses.append(loss)
            num_correct_val += int((prediction == y))
    
    train_loss = np.mean(train_losses)
    train_acc = num_correct_train/len(train_loader)
    val_loss = np.mean(val_losses)
    val_acc = num_correct_val/len(val_loader)

    print('Performance Metrics for epoch ' + str(epoch_num) + ':')
    print('train loss: ' + str(train_loss))
    print('val loss: ' + str(val_loss))
    print('train accuracy: ' + str(train_acc))
    print('val accuracy: ' + str(val_acc))

    with open(result_path + 'epoch' + str(epoch_num) + '.csv', 'w') as csvfile:
        fieldnames = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})


def main():

    # TODO: implement get train_val_test_loaders (dataloader.py)
    train_loader, val_loader, test_loader = get_train_val_test_loaders()

    # TODO: implement model
    model = ResnetModel()

    # TODO: implement loss function and probably new optimizer ()
    loss_func = HierarchicalLoss()
    optmizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    run_name = 'run-'+ str(time.time())
    result_path = 'results/' + run_name + '/'
    train_val_metrics_path = result_path + 'train_val_metrics/'
    model_path = result_path + 'saved_model/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        os.mkdir(train_val_metrics_path)
        os.mkedir(model_path)
        
    # TODO: define config for variables like num epochs
    for epoch in range(0, 10):
        print('Training epoch ' +  str(epoch))
        train_epoch(train_loader, model, loss_func, optmizer)
        evaluate_epoch(train_loader, val_loader, model, loss_func, epoch, train_val_metrics_path)

    torch.save(model, model_path)

if __name__ == '__main__':
    main()