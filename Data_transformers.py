from torchvision import transforms


def setUpDataLoaderTransformers(inputSize=224):

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(inputSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                                 0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(inputSize),
            transforms.CenterCrop(inputSize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                                 0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(inputSize),
            transforms.CenterCrop(inputSize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                                 0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms
