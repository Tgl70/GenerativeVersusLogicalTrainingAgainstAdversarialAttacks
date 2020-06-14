from models import MnistNet, ResNet18
from torchvision import transforms
from getDatasets import MyDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import json
import numpy as np
from PIL import Image


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test(model, device, test_loader, epsilon, dataset, dtype):
    # Accuracy counter
    correct = 0
    adv_examples = []
    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        data = data.float()
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        # Calculate the loss
        loss = F.nll_loss(output, target)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # Re-classify the perturbed image
        output = model(perturbed_data)
        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

                if dataset == "cifar10":
                    img = Image.fromarray(np.uint8(adv_ex * 255), 'RGB')
                else:
                    img = Image.fromarray(np.uint8(adv_ex * 255), 'L')
                img.save(f"FGSM_images/{dataset}_{dtype}_e{epsilon}_i{init_pred.item()}_f{final_pred.item()}.png")

        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

                if dataset == "cifar10":
                    img = Image.fromarray(np.uint8(adv_ex * 255), 'RGB')
                else:
                    img = Image.fromarray(np.uint8(adv_ex * 255), 'L')
                img.save(f"FGSM_images/{dataset}_{dtype}_e{epsilon}_i{init_pred.item()}_f{final_pred.item()}.png")

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"        Epsilon: {epsilon}\tTest Accuracy: {final_acc} ({correct}/{len(test_loader)})")
    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


if __name__ == "__main__":
    datasets = ['fashion_mnist', 'mnist']
    model_names_dict = {
        'mnist': [
            'dl2/mnist_dl2_88.pth',
            'datasetA/mnist_datasetA_98.pth',
            'datasetB/mnist_datasetB_83.pth',
            'datasetC/mnist_datasetC_98.pth'
        ],
        'fashion_mnist': [
            'dl2/fashion_mnist_dl2_88.pth',
            'datasetA/fashion_mnist_datasetA_98.pth',
            'datasetB/fashion_mnist_datasetB_81.pth',
            'datasetC/fashion_mnist_datasetC_96.pth'
        ]
    }

    epsilons = [0, .01, .02, .03, .04, .05]

    report_file = f'reports/FGSM_Attack.json'
    data_dict = []

    for dataset in datasets:
        model_names = model_names_dict.get(dataset)

        for model_name in model_names:
            model_path = f'models/{dataset}/{model_name}'

            use_cuda = torch.cuda.is_available()
            if use_cuda:
                torch.cuda.empty_cache()
            device = torch.device("cuda" if use_cuda else "cpu")
            kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

            if dataset == 'mnist' or dataset == 'fashion_mnist':
                transform_train = transforms.Compose([transforms.ToTensor()])
                transform_test = transforms.Compose([transforms.ToTensor()])
                model = MnistNet().to(device)

            elif dataset == 'cifar10':
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                ])
                model = ResNet18().to(device)

            Xy_test = MyDataset(dataset=dataset, dtype='datasetA', train=False, transform=transform_test)
            test_loader = torch.utils.data.DataLoader(Xy_test, shuffle=True, batch_size=1, **kwargs)

            model.load_state_dict(torch.load(model_path))
            model.eval()

            print(f'{model_name}:')
            dtype = model_name.rpartition('/')[0]
            model_dict = {
                'dataset': dataset,
                'dtype': dtype,
                'attacks': []
            }

            for eps in epsilons:
                acc, ex = test(model, device, test_loader, eps, dataset, dtype)
                eps_dict = {
                    'epsilon': eps,
                    'accuracy': acc
                }
                model_dict['attacks'].append(eps_dict)
            
            data_dict.append(model_dict)

    with open(report_file, 'w') as fou:
        json.dump(data_dict, fou, indent=4)
