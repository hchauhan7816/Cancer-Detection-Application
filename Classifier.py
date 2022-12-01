from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import sklearn.metrics


class Classifier:

    model_name = None
    output_classes = 2

    def __init__(self, model_name, output_classes=2, batch_size=8, num_epochs=1, feature_extract=True):
        self.model_name = model_name
        self.output_classes = output_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.feature_extract = feature_extract
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def train_model(self, model, criterion, optimizer, dataloaders, dataset_sizes):
        since = time.time()

        best_model_weights = copy.deepcopy(model.state_dict())
        best_accuracy = 0.0
        val_acc_history = []
        train_acc_history = []

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_accuracy:
                    best_accuracy = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                if phase == 'train':
                    train_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_accuracy))

        # load best model weights
        model.load_state_dict(best_model_weights)
        return model, train_acc_history, val_acc_history, best_accuracy

    def set_requires_grad(self, model):
        if self.feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def initPretrainedModel(self, inputSize):
        model = None
        input_size = 0
        if self.model_name == 'alexnet' and self.feature_extract:
            model = torchvision.models.alexnet(pretrained=True)
            self.set_requires_grad(model)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, self.output_classes)
            input_size = inputSize

        if self.model_name == 'densenet' and self.feature_extract:

            model = models.densenet121(pretrained=True)
            self.set_requires_grad(model)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, self.output_classes)
            input_size = inputSize

        return model

    def testModel(self, dataloaders, model, classes, dataset_sizes, batch_size):
        correct = 0
        total = dataset_sizes['test']
        predictions = []

        y_actual = []
        y_pred = []

        model.eval()
        with torch.no_grad():
            for index, (inputs, labels) in enumerate(dataloaders['test'], 0):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                samples = dataloaders['test'].dataset.samples[index *
                                                              batch_size: index*batch_size + batch_size]
                predicted_classes = [classes[predicted[j]]
                                     for j in range(predicted.size()[0])]
                sample_names = [s[0] for s in samples]

                predictions.extend(list(zip(sample_names, predicted_classes)))

                y_actual.extend(labels.numpy())
                y_pred.extend(predicted.numpy())

        try:
            print(
                f"Accuracy (Sklearn): {sklearn.metrics.accuracy_score(y_actual, y_pred)}")

            print(
                f"\nConfusion Matrix:\n{sklearn.metrics.confusion_matrix(y_actual, y_pred)}")

        except RuntimeError:
            print("Error computing metrics: \n", RuntimeError)

        print('\n\nAccuracy of the network on the test images: %d %%' %
              (100 * correct / total))

        return predictions
