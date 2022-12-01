from Data_transformers import setUpDataLoaderTransformers
import torch
from torchvision import datasets
import os


class DataSet:
    data_dir = None

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def initDataLoaders(data_dir, batch_size):
        data_transforms = setUpDataLoaderTransformers()
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val', 'test']}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                      shuffle=True, num_workers=4)
                       for x in ['train', 'val']}

        # test data is not shuffled to get proper correspondences between filename and predicted label
        dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size,
                                                          shuffle=False, num_workers=4)
        dataset_sizes = {x: len(image_datasets[x]) for x in [
            'train', 'val', 'test']}
        class_names = image_datasets['train'].classes
        return dataloaders, dataset_sizes, class_names
