from models import MnistNet, ResNet18
from torchvision import transforms
from getDatasets import MyDataset
from oracles import DL2_Oracle
from constraints import *
import torch.optim as optim
import numpy as np
import args as dl2
import argparse
import torch
import json
import time
import os


def RobustnessT(eps1, eps2):
    return lambda model, use_cuda, network_output: RobustnessDatasetConstraint(model, eps1, eps2, use_cuda=use_cuda, network_output=network_output)


def RobustnessG(eps, delta):
    return lambda model, use_cuda, network_output: RobustnessConstraint(model, eps, delta, use_cuda, network_output=network_output)


def train(args, oracle, net, device, train_loader, optimizer, epoch):
    t1 = time.time()
    num_steps = 0
    avg_train_acc, avg_constr_acc = 0, 0
    avg_ce_loss, avg_dl2_loss = 0, 0
    ce_loss = torch.nn.CrossEntropyLoss()

    print('\nEpoch ', epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        x_batch, y_batch = data.to(device), target.to(device)
        n_batch = int(x_batch.size()[0])
        num_steps += 1

        x_outputs = net(x_batch)
        x_correct = torch.mean(torch.argmax(x_outputs, dim=1).eq(y_batch).float())
        ce_batch_loss = ce_loss(x_outputs, y_batch)
        
        if epoch <= args.delay or args.dl2_weight < 1e-7:
            net.train()
            optimizer.zero_grad()
            ce_batch_loss.backward()
            optimizer.step()

            avg_train_acc += x_correct.item()
            avg_ce_loss += ce_batch_loss.item()

            if batch_idx % args.print_freq == 0:
                print(f'[{batch_idx}] Train p_acc: {x_correct.item():.4f}, CE loss: {ce_batch_loss.item():.4f}')
            continue

        x_batches, y_batches = [], []
        k = n_batch // oracle.constraint.n_tvars
        assert n_batch % oracle.constraint.n_tvars == 0, 'Batch size must be divisible by number of train variables!'
        for i in range(oracle.constraint.n_tvars):
            x_batches.append(x_batch[i:(i + k)])
            y_batches.append(y_batch[i:(i + k)])

        net.eval()

        if oracle.constraint.n_gvars > 0:
            domains = oracle.constraint.get_domains(x_batches, y_batches)
            z_batches = oracle.general_attack(x_batches, y_batches, domains, num_restarts=1, num_iters=args.num_iters, args=args)
            _, dl2_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, z_batches, args)
        else:
            _, dl2_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, None, args)

        avg_train_acc += x_correct.item()
        avg_constr_acc += constr_acc.item()
        avg_ce_loss += ce_batch_loss.item()
        avg_dl2_loss += dl2_batch_loss.item()

        if batch_idx % args.print_freq == 0:
            print(f'[{batch_idx}] Train p_acc: {x_correct.item():.4f}, Train c_acc: {constr_acc.item():.4f}, CE loss: {ce_batch_loss.item():.4f}, DL2 loss: {dl2_batch_loss.item():.4f}')
        
        net.train()
        optimizer.zero_grad()
        tot_batch_loss = args.dl2_weight * dl2_batch_loss + ce_batch_loss
        tot_batch_loss.backward()
        optimizer.step()
    t2 = time.time()
        
    avg_train_acc /= float(num_steps)
    avg_constr_acc /= float(num_steps)
    avg_dl2_loss /= float(num_steps)
    avg_ce_loss /= float(num_steps)
    t = t2 - t1
    
    return avg_train_acc, avg_constr_acc, avg_dl2_loss, avg_ce_loss, t


def test(args, oracle, model, device, test_loader):
    loss = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct, constr, num_steps, pgd_ok = 0, 0, 0, 0
    
    for data, target in test_loader:
        num_steps += 1
        x_batch, y_batch = data.to(device), target.to(device)
        n_batch = int(x_batch.size()[0])

        x_batches, y_batches = [], []
        k = n_batch // oracle.constraint.n_tvars
        assert n_batch % oracle.constraint.n_tvars == 0, 'Batch size must be divisible by number of train variables!'

        for i in range(oracle.constraint.n_tvars):
            x_batches.append(x_batch[i:(i + k)])
            y_batches.append(y_batch[i:(i + k)])

        if oracle.constraint.n_gvars > 0:
            domains = oracle.constraint.get_domains(x_batches, y_batches)
            z_batches = oracle.general_attack(x_batches, y_batches, domains, num_restarts=1, num_iters=args.num_iters, args=args)
            _, dl2_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, z_batches, args)
        else:
            _, dl2_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, None, args)

        output = model(x_batch)
        test_loss += loss(output, y_batch).item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            
        correct += pred.eq(y_batch.view_as(pred)).sum().item()
        constr += constr_acc.item()

    test_loss /= len(test_loader.dataset)
    print(f"\nTest p_acc: {correct / len(test_loader.dataset):.4f}, Test c_acc: {(constr / float(num_steps)):.4f}, Average loss: {test_loss:.4f}")

    return correct / len(test_loader.dataset), constr / float(num_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NN with constraints.')
    parser = dl2.add_default_parser_args(parser)
    parser.add_argument('--batch-size', type=int, default=128, help='Number of samples in a batch.')
    parser.add_argument('--num-iters', type=int, default=50, help='Number of oracle iterations.')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs to train for.')
    parser.add_argument('--dl2-weight', type=float, default=0.0, help='Weight of DL2 loss.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset on which to train.')
    parser.add_argument('--delay', type=int, default=0, help='How many epochs to wait before training with constraints.')
    parser.add_argument('--print-freq', type=int, default=10, help='Print frequency.')
    parser.add_argument('--report-dir', type=str, default='reports', help='Directory where results should be stored')
    parser.add_argument('--constraint', type=str, default="RobustnessG(eps=0.3, delta=0.52)", help='the constraint to train with: RobustnessT(eps1, eps2), RobustnessG(eps, delta)')
    parser.add_argument('--network-output', type=str, choices=['logits', 'prob', 'logprob'], default='logits', help='Wether to treat the output of the network as logits, probabilities or log(probabilities) in the constraints.')
    parser.add_argument('--dtype', type=str, required=True, choices=['datasetA', 'datasetB', 'datasetC'])
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        model = MnistNet().to(device)

    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])  # meanstd transformation

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        model = ResNet18().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    Xy_train = MyDataset(dataset=args.dataset, dtype=args.dtype, train=True, transform=transform_train)
    Xy_test = MyDataset(dataset=args.dataset, dtype=args.dtype, train=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(Xy_train, shuffle=True, batch_size=args.batch_size, **kwargs)
    test_loader = torch.utils.data.DataLoader(Xy_test, shuffle=True, batch_size=args.batch_size, **kwargs)

    constraint = eval(args.constraint)(model, use_cuda, network_output=args.network_output)
    oracle = DL2_Oracle(learning_rate=0.01, net=model, constraint=constraint, use_cuda=use_cuda)

    tstime = int(time.time())
    dtype = args.dtype if args.dl2_weight < 1e-7 else 'dl2'
    report_file = os.path.join(args.report_dir, f"{args.dataset}_{dtype}_{constraint.name}_{tstime}.json")
    data_dict = {
        'dl2_weight': args.dl2_weight,
        'delay': args.delay,
        'name': constraint.name,
        'constraint_txt': args.constraint,
        'constraint_params': constraint.params(),
        'num_iters': args.num_iters,
        'ce_loss': [],
        'dl2_loss': [],
        'Train p_acc': [],
        'Train c_acc': [],
        'Test p_acc': [],
        'Test c_acc': [],
        'epoch_time': []
    }

    for epoch in range(1, args.num_epochs + 1):
        avg_train_acc, avg_constr_acc, avg_dl2_loss, avg_ce_loss, epoch_time = \
            train(args, oracle, model, device, train_loader, optimizer, epoch)
        data_dict['Train p_acc'].append(avg_train_acc)
        data_dict['Train c_acc'].append(avg_constr_acc)
        data_dict['ce_loss'].append(avg_ce_loss)
        data_dict['dl2_loss'].append(avg_dl2_loss)
        data_dict['epoch_time'].append(epoch_time)

        p, c = test(args, oracle, model, device, test_loader)
        data_dict['Test p_acc'].append(p)
        data_dict['Test c_acc'].append(c)
        print(f'Epoch Time [s]: {epoch_time:.4f}')

        if epoch > (args.num_epochs - 25):
            torch.save(model.state_dict(), f"models/{args.dataset}/{dtype}/{args.dataset}_{dtype}_{epoch}_{tstime}.pth")

    with open(report_file, 'w') as fou:
        json.dump(data_dict, fou, indent=4)

# global robustness also checks the same label not the correct label,
# modificare il programma per checkare la correct label
# maybe usare la loss solo se è correttamente classified

# man mano che gli esperimenti sono pronti metterli in una tabella subito

# salvare non solo l'ultimo modello ma gli ultimi tot, maybe 20-25 - OK
# modificare i nomi dei report (rimane solo 'dtype') - OK
# magari fare un method a parte per loadare i datasets e i modelli - OK
# pulire il codice e togliere i pezzi inutili


# args.py OK
# constraints.py da controllare la riga commentata
# diffsat.py OK
# domains.py OK
# getDataset da finire con datasetsC
# models.py OK
# oracles.py ancora da vedere
# results.py da vedere e aggiustare al mio caso





'''
Our global robustness constraint requires that for any input
x with a classification y, inputs in its [0,1] neighborhood which
are valid images (pixels are between 0 and 1), have a high
probability for y. For numerical stability, instead of directly
checking the probability, we check that the corresponding
log-probability is larger than a given threshold δ.


Here instead of the CLASSIFICATION y, we should use the LABEL y.      !!!!!




Sarebbe da sostituire (riga 31, constraints.py):
        z_outputs = [self.net(z_input) for z_input in z_inputs]
con le vere label per ogni z_input.




Bisogna vedere in (riga 68, constraints.py):
        def get_domains(self, x_batches, y_batches):
questo method crea un oggetto BOX e forse si può salvare anche y_batches come terzo parametro nell'oggetto BOX,
viene già passato ma non usato.

Bisogna capire cosa vuol dire (riga 69, constraints.py):
        assert len(x_batches) == 1
forse vuol dire che c'è solo un batch.

Bisogna capire la correlazione tra x_batches e il BOX e vedere se è possibile fare un pair con y_batches in BOX.
Se fosse possibile, allora quando z_batches viene creato in (riga 42, oracles.py):
        z_batches = [np.concatenate([np.expand_dims(domains[j][i].sample(), axis=0) for i in range(n_batch)], axis=0) for j in range(n_gvars)]
si può prendere y_batches corrispondente da BOX e matcharlo con z_batches.

Successisivamente bisgona modificare (riga 47, oracles.py):
        neg_losses, pos_losses, sat, z_inputs = self.constraint.loss(x_batches, y_batches, z_batches, args)
dando anche gli y_batches accoppiati con gli z_batches.




(può creare confusione con gli y_batches che già stiamo passando,
c'è da verificare che non vadano già bene,
in questo caso si può skippare quello fatto fino ad ora e modificare solo quello che segue)
DA CHECKARE (vedere le immagini nella riga di codice sopra citata e vedere se x_batches, y_batches e z_batches già corrispondono)




Infine (riga 41, constraints.py):
        z_inp, z_out = self.eval_z(z_batches)
deve passare anche gli y_batches che corrispondono agli z_batches
e (riga 31, constraints.py):
        z_outputs = [self.net(z_input) for z_input in z_inputs]
deve essere modificata come: --->   z_outputs = y_batches   <--- (controllando che siano dei floatTensor, se no farli con un loop come appena sopra in riga 25)



(riga 81, constraints.py):
        pred = z_out[np.arange(n_batch), y_batches[0]]
FORSE FA GIà QUELLO CHE VOGLIAMO, CIOè pred è L'OUTPUT CORRISPONDENTE ALLA CLASSE GIUSTA.      DA VERIFICARE MA CREDO SIA GIUSTO!!!
'''

