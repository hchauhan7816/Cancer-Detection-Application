from Data_transformers import setUpDataLoaderTransformers
from Optimizer import Optimizer
from Classifier import Classifier
from Dataset import DataSet
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import csv
from pathlib import Path


def Dense_test(run_id, data_dir, model_name, output_classes, feature_extract,
               batch_size, num_epochs):
    print("Running independent test...")
    save_as_name = 'Models/Dense_net/densenetFeatureExtraction_' + run_id + '.pt'
    state = torch.load(save_as_name)
    densenetClassifier = Classifier(
        model_name, output_classes, batch_size, num_epochs, feature_extract)
    model = densenetClassifier.initPretrainedModel(224)
    model.load_state_dict(state['model_state_dict'])
    dataloaders_dict, dataset_sizes, class_names = DataSet.initDataLoaders(
        data_dir, batch_size)
    data_transforms = setUpDataLoaderTransformers()
    predictions = densenetClassifier.testModel(
        dataloaders_dict, model, class_names, dataset_sizes, batch_size=8)

    save_as_name = 'Models/Dense_net/predicted_labels/predictedLabelsDensenets_' + run_id + '.csv'
    np.savetxt(save_as_name, predictions, fmt='%s')


def Dense_train(data_dir, model_name, output_classes, feature_extract,
                batch_size, num_epochs, learningRate, momentum):
    run_id = 'l_' + str(learningRate) + '_m_' + str(momentum)

    densenetClassifier = Classifier(
        model_name, output_classes, batch_size, num_epochs, feature_extract)
    model = densenetClassifier.initPretrainedModel(224)

    dataloaders_dict, dataset_sizes, class_names = DataSet.initDataLoaders(
        data_dir, batch_size)
    data_transforms = setUpDataLoaderTransformers()

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'val', 'test']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sgdOptimizer = Optimizer(device)
    optimizer_ft = sgdOptimizer.optimize(
        model, feature_extract, learningRate, momentum)

    criterion = nn.CrossEntropyLoss()

    model, val_acc_history, per_epoch_loss, per_epoch_accuracy = densenetClassifier.train_model(model,
                                                                                                criterion,
                                                                                                optimizer_ft,
                                                                                                dataloaders_dict,
                                                                                                dataset_sizes)

    print(f"Validation Accuracy History:\n{val_acc_history}")
    print(f"\nPer epoch loss:\n{per_epoch_loss}")
    print(f"\nPer epoch accuracy:\n{per_epoch_accuracy}")

    print("Saving final model to disk...")
    save_as_name = 'Models/Dense_net/densenetFeatureExtraction_' + run_id + '.pt'
    torch.save({
        'name': 'densenet_feature_extraction_' + run_id,
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_ft.state_dict(),
    }, save_as_name)

    return run_id


def Alex_test(data_dir, model_name, output_classes):
    alexnetClassifier = Classifier(model_name, output_classes)
    model = alexnetClassifier.initPretrainedModel(224)
    batchSize = 8
    dataloaders_dict, dataset_sizes, class_names = DataSet.initDataLoaders(
        data_dir, batchSize)
    data_transforms = setUpDataLoaderTransformers()
    state = torch.load('Models/Alex_net/alexnetFeatureExtraction.pt')
    model.load_state_dict(state['model_state_dict'])
    predictions = alexnetClassifier.testModel(
        dataloaders_dict, model, class_names, dataset_sizes, batchSize)

    # save predicted values
    np.savetxt('Models/Alex_net/metrics/predicted_labels/predictedLabelsAlexNetEpoch40.csv',
               predictions, fmt='%s')


def Alex_train(data_dir, model_name, output_classes, feature_extract):
    paramsFile = open('PramsAlexnet.json')

    paramsMetricFilePath = Path(
        'modelEvaluationMetrics/alexnet/alexnetMetrics.csv')

    hyperparamsArray = json.load(paramsFile)
    fields = ['learningRate', 'momentum', 'epochs', 'batchSize', 'valAccuracy']
    metricsPath = 'modelEvaluationMetrics/alexnet/'

    best_accuracy_until_now = 0

    if not paramsMetricFilePath.exists():
        with open(paramsMetricFilePath, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    for ctr, hyperparams in enumerate(hyperparamsArray):
        learningRate = hyperparams['learningRate']
        epochs = hyperparams['epochs']
        momentum = hyperparams['momentum']
        batchSize = hyperparams['batchSize']

        alexnetClassifier = Classifier(model_name, output_classes)
        model = alexnetClassifier.initPretrainedModel(224)

        dataloaders_dict, dataset_sizes, class_names = DataSet.initDataLoaders(
            data_dir, batchSize)

        data_transforms = setUpDataLoaderTransformers()
        image_datasets = {x: datasets.ImageFolder(os.path.join(
            data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
        print(image_datasets)

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        sgdOptimizer = Optimizer(device)
        optimizer_ft = sgdOptimizer.optimize(
            model, feature_extract, learningRate, momentum)

        criterion = nn.CrossEntropyLoss()

        model, train_acc_history, val_acc_history, best_accuracy = alexnetClassifier.train_model(
            model, criterion, optimizer_ft, dataloaders_dict, dataset_sizes)

        with open(paramsMetricFilePath, 'a+') as f:
            writer = csv.writer(f)
            print("here")
            writer.writerow([learningRate, epochs, momentum,
                            batchSize, best_accuracy])

        print("here2")
        plt.title("Validation Accuracy vs. Number of Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.plot(range(1, epochs+1), val_acc_history)
        # plt.clf()
        # plt.cla()

        plt.savefig(metricsPath+'val_epoch_'+str(ctr)+'.png')
        np.save(metricsPath+'val_epoch_'+str(ctr)+'.npy', val_acc_history)

        if best_accuracy > best_accuracy_until_now:
            print(best_accuracy)
            best_accuracy_until_now = best_accuracy

            # save only the best model until now
            torch.save({
                'name': 'alexnetFeatureExtraction',
                'epoch': epochs,
                'learningRate': learningRate,
                'momentum': momentum,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_ft.state_dict(),
            }, 'Models/Alex_net/alexnetFeatureExtraction.pt')


def Dense_run():
    data_dir = 'data/'
    model_name = 'densenet'
    output_classes = 2
    feature_extract = True
    batch_size = 8
    num_epochs = 1
    learningRate = 0.001
    momentum = 0.9

    run_id = 'l_' + str(learningRate) + '_m_' + str(momentum)

    Dense_train(data_dir, model_name, output_classes, feature_extract,
                batch_size, num_epochs, learningRate, momentum)

    # Dense_test(run_id, data_dir, model_name, output_classes, feature_extract,
    #            batch_size, num_epochs)


def Alex_run():
    data_dir = 'data/'
    model_name = 'alexnet'
    output_classes = 2
    feature_extract = True

    Alex_train(data_dir, model_name, output_classes, feature_extract)
    # Alex_test(data_dir, model_name, output_classes)


if __name__ == '__main__':

    Alex_run()
    # Dense_run()
