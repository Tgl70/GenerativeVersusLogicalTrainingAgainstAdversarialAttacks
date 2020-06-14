from models import MnistNet, ResNet18
from torchvision import transforms
from getDatasets import MyDataset
import torch
import json


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

    report_file = f'reports/ACGAN_Attack.json'
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

            Xy_test = MyDataset(dataset=dataset, dtype='testAdversarial', train=False, transform=transform_test)
            test_loader = torch.utils.data.DataLoader(Xy_test, shuffle=True, batch_size=128, **kwargs)

            model.load_state_dict(torch.load(model_path))
            model.eval()

            correct = 0

            for data, target in test_loader:
                data = data.float()
                x_batch, y_batch = data.to(device), target.to(device)
                n_batch = int(x_batch.size()[0])

                output = model(x_batch)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(y_batch.view_as(pred)).sum().item()

            acc = correct / len(test_loader.dataset)

            print(f'{model_name}: {acc} ({correct}/{len(test_loader.dataset)})')

            model_dict = {
                'dataset': dataset,
                'dtype': model_name.rpartition('/')[0],
                'accuracy': acc
            }
            data_dict.append(model_dict)

    with open(report_file, 'w') as fou:
        json.dump(data_dict, fou, indent=4)
