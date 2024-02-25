import torch
import tqdm
import os
from utils import accuracy, expected_calibration_error, predictive_entropy
from models import lenet5, resnet18
from datasets import mnist, cifar10
from adversarial import ufgsm_attack
from curvatures import Diagonal, BlockDiagonal, KFAC, EFB, INF


def eval_nn(model,
            dataset,
            device=torch.device('cuda'),
            verbose=False):
    model.eval()

    with torch.no_grad():
        logits_list = torch.Tensor().to(device)
        labels_list = torch.LongTensor()

        dataset = tqdm.tqdm(dataset, disable=not verbose or len(dataset) == 1)
        for images, labels in dataset:
            logits = model(images.to(device))
            logits_list = torch.cat([logits_list, logits])
            labels_list = torch.cat([labels_list, labels])

        predictions = torch.nn.functional.softmax(logits_list, dim=1).cpu().numpy()
        labels = labels_list.numpy()

    if verbose:
        print(f"Accuracy: {accuracy(predictions, labels):.4f}% | ECE: {expected_calibration_error(predictions, labels)[0]:.4f} | Entropy: {predictive_entropy(predictions, mean=True):.4f}")

    return predictions, labels

def train(model, train_loader, val_loader, optimizer, criterion, epochs, device=torch.device('cuda')):
    train_loss = 0
    for epoch in range(epochs):
        model.train()
        train_loader = tqdm.tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]")
        for batch, (images, labels) in enumerate(train_loader):
            train_loader.set_postfix({'Train loss': train_loss / ((batch + 1) + (epoch * len(train_loader))),
                                      'Train acc.': train_acc if batch > 10 else 0})

            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % (len(train_loader) // 10) == 0:
                train_loss += loss.detach().cpu().numpy()
                train_acc = accuracy(logits.detach().cpu().numpy(), labels.numpy())
        
        eval_nn(model, val_loader, device, verbose=True)

def train_script(model_name: str = 'lenet5', 
                 dataset_name: str = 'mnist', 
                 learning_rate: float = 0.01, 
                 epoch: int = 40,
                 device = torch.device('cuda')):
    if model_name == 'lenet5':
        model = lenet5(pretrained=True)
    else:
        model = resnet18(pretrained=True)
    model.to(device).train()
    if dataset_name == 'mnist':
        train_loader, val_loader = mnist(split='train'), mnist(split='test')
    else: 
        train_loader, val_loader = cifar10(split='train'), cifar10(split='test')
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train(model, train_loader, val_loader, optimizer, criterion, epoch, device)

    path = os.path.join(os.path.abspath(os.getcwd()), "weights", f"{model_name}_{dataset_name}.pt")

    torch.save(model.state_dict(), path) 

def calc_info_matrix(model, data, model_name, dataset_name, est_name = 'Diagonal', last_layer_mode = False, factors = None, version_suffix = 0, epochs = 1, sample_size = 20, device=torch.device('cuda')):
    if last_layer_mode:
        last_layer_suffix = 'll'
    else:
        last_layer_suffix = 'fl'

    criterion = torch.nn.CrossEntropyLoss().to(device)

    assert est_name in ['Diagonal', 'BlockDiagonal', 'KFAC', 'EFB'], "Estimator not in scope!"
    if est_name == 'Diagonal':
        estimator = Diagonal(model, last_layer_mode=last_layer_mode)
    elif est_name == 'BlockDiagonal':
        estimator = BlockDiagonal(model, last_layer_mode=last_layer_mode)
    elif est_name == 'KFAC':
        estimator = KFAC(model, last_layer_mode=last_layer_mode)
    else: 
        estimator = EFB(model, factors, last_layer_mode=last_layer_mode)
    
    for epoch in tqdm.tqdm(range(epochs), disable= False):
        data = tqdm.tqdm(data, desc=f"Epoch [{epoch + 1}/{epochs}]", disable=True)
        for batch, (images, labels) in enumerate(data):
            logits = model(images.to(device))

            '''
            ### data augmentation
            dist = torch.distributions.Categorical(logits=logits)

            for sample in range(sample_size):
                labels = dist.sample()

                loss = criterion(logits, labels)
                model.zero_grad()
                loss.backward(retain_graph=True)

                estimator.update(images.size(0))
            '''
            loss = criterion(logits, labels.to(device))
            model.zero_grad()
            loss.backward(retain_graph=True)

            estimator.update(images.size(0))

    filename = f"{model_name}_{dataset_name}_{last_layer_suffix}_{est_name}_v{version_suffix}.pt"
    state_path = os.path.join(os.path.abspath(os.getcwd()), "factors", filename)
    torch.save(estimator.state, state_path)

    return estimator

def calc_lr_im(model, model_name, dataset_name, last_layer_mode = False, diags= None, factors= None, lambdas= None, version_suffix = 0, rank = 100, device=torch.device('cuda')):

    if last_layer_mode:
        last_layer_suffix = 'll'
    else:
        last_layer_suffix = 'fl'

    ###  torch layer is compared using memory address, therefore the loaded state cannot be applied 
    '''
    factors_path = os.path.join(os.path.abspath(os.getcwd()), "factors", f"{model_name}_{dataset_name}_{last_layer_suffix}_KFAC_v{version_suffix}.pt")
    factors = torch.load(factors_path)

    lambdas_path = os.path.join(os.path.abspath(os.getcwd()), "factors", f"{model_name}_{dataset_name}_{last_layer_suffix}_EFB_v{version_suffix}.pt")
    lambdas = torch.load(lambdas_path)

    diags_path = os.path.join(os.path.abspath(os.getcwd()), "factors", f"{model_name}_{dataset_name}_{last_layer_suffix}_Diagonal_v{version_suffix}.pt")
    diags = torch.load(diags_path)
    '''

    # compute low rank information matrix
    inf = INF(model, diags, factors, lambdas, last_layer_mode=last_layer_mode)
    inf.update(rank)

    filename = f"{model_name}_{dataset_name}_{last_layer_suffix}_INF_r{rank}_v{version_suffix}.pt"
    state_path = os.path.join(os.path.abspath(os.getcwd()), "factors", filename)
    torch.save(inf.state, state_path)

    return inf

def eval_bnn(model,
             dataset,
             estimator,
             samples=50,
             device=torch.device('cuda'),
             verbose=True):
    
    model.eval()
    mean_predictions = 0

    with torch.no_grad():
        samples = tqdm.tqdm(range(samples), disable=not verbose)
        for sample in samples:
            estimator.sample_and_replace()
            predictions, labels = eval_nn(model, dataset, device)
            mean_predictions += predictions

        mean_predictions /= len(samples)
        
        acc = accuracy(mean_predictions, labels)
        ece = expected_calibration_error(mean_predictions, labels)[0]
        ent = predictive_entropy(mean_predictions, mean=True)

    if verbose:
        print(f"Accuracy: {acc:.4f}% | ECE: {ece:.4f} | Entropy: {ent:.4f}") 
    
    return mean_predictions, labels, acc, ece, ent
    
def eval_ufgsm(model, data, epsilon=0.1, device=torch.device('cuda'), verbose=True):
    model.eval()
    logits_list = torch.Tensor().to(device)
    labels_list = torch.LongTensor()

    data = tqdm.tqdm(data, disable=not verbose or len(data) == 1)
    for images, labels in data:
        adv_images = ufgsm_attack(model, images.to(device, non_blocking=True), labels.to(device, non_blocking=True),
                                   epsilon=epsilon)
        with torch.no_grad():
            adv_logits = model(adv_images)

        logits_list = torch.cat([logits_list, adv_logits])
        labels_list = torch.cat([labels_list, labels])

    adv_predictions = torch.nn.functional.softmax(logits_list, dim=1).detach().cpu().numpy()
    labels = labels_list.numpy()

    acc = accuracy(adv_predictions, labels)
    ece = expected_calibration_error(adv_predictions, labels)[0]
    ent = predictive_entropy(adv_predictions, mean=True)

    if verbose:
        print(f"Step: {epsilon:.2f} | Adv. Entropy: {ent:.4f} | Adv. Accuracy: {acc:.4f}%")

    return adv_predictions, labels, acc, ece, ent

def eval_ufgsm_bnn(model,
                  data,
                  estimator,
                  samples=50,
                  epsilon=0.1,
                  device=torch.device('cuda'),
                  verbose=True):

    model.eval()
    mean_predictions = 0

    samples = tqdm.tqdm(range(samples), disable=not verbose)
    for _ in samples:
        estimator.sample_and_replace()
        predictions, labels, _ , _, _= eval_ufgsm(model, data, epsilon, device=device, verbose=False)
        mean_predictions += predictions
    mean_predictions /= len(samples)
    
    acc = accuracy(mean_predictions, labels)
    ece = expected_calibration_error(mean_predictions, labels)[0]
    ent = predictive_entropy(mean_predictions, mean=True)

    if verbose:
        print(f"Step: {epsilon:.2f} | Adv. Entropy: {ent:.4f} | Adv. Accuracy: {acc:.4f}%")
    
    return mean_predictions, labels, acc, ece, ent

    

