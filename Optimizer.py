import torch.optim as optim


class Optimizer:

    def __init__(self, device):
        self.device = device

    def optimize(self, model, feature_extract, learningRate, momentum):
        model = model.to(self.device)
        params_to_update = model.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        optimizer_ft = optim.SGD(
            params_to_update, lr=learningRate, momentum=momentum)

        return optimizer_ft
