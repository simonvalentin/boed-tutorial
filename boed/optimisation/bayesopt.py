import random
import numpy as np
from tqdm import tqdm as tqdm

# PyTorch stuff
import torch
import torch.utils.data
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# GPyTorch / BoTorch Stuff
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model

# Modelcomp stuff
from boed.networks.fullyconnected import FullyConnected
from boed.networks.summstats import NeuralSummStats, CAT_NSS
from boed.bounds.minef import minef_loss

# ----- TRAINING FUNCTIONS WITHOUT SUMM-STATS ----- #

def train(model, trainloader, validloader, optimizer, num_epochs, device, bar=True, valid=True):
    
    train_loss = list()
    valid_loss = list()
    for epoch in tqdm(range(num_epochs), leave=True, disable=not bar):

        for x, y in trainloader:
            
            # move to device
            x, y = x.to(device), y.to(device)

            # compute loss
            loss = minef_loss(x, y, model, device)

            # Zero grad the NN optimizer
            optimizer.zero_grad()

            # Back-Propagation
            loss.backward()

            # Perform opt steps for NN
            optimizer.step()

            # save a few things to lists
            train_loss.append(-loss.clone().detach())
            
        # validate every 5 epochs
        if valid and ((epoch + 1) % 5 == 0):
            with torch.no_grad():
                val = validate(model, validloader, device)
                valid_loss.append(float(val))
    
    # move list items to cpu
    train_loss = np.array([m.cpu().data.numpy() for m in train_loss])
    
    return model, train_loss, valid_loss


def validate(model, validloader, device):
    
    scores = list()
    for x, y in validloader:
        
        x, y = x.to(device), y.to(device)
        
        # compute loss and store in list
        loss = minef_loss(x, y, model, device)
        scores.append(loss.cpu().data.numpy())
        
    return np.mean(-np.array(scores))


def evaluate(designs, prior, device, modelparams, DatasetClass, bar=False, valid=True, **kwargs):
    
    # extract model parameters
    batchsize = modelparams['batchsize']
    L = modelparams['layers']
    H = modelparams['hidden']
    lr = modelparams['lr']
    num_epochs = modelparams['num_epochs']
    num_workers = modelparams['num_workers']
    
    # define data set
    dataset = DatasetClass(designs, prior, device, **kwargs)

    # define data loaders
    t_size = int(len(dataset) * 0.80)  # 80% training data
    v_size = int(len(dataset) * 0.20)  # 20% validation data
    trainset, validset = torch.utils.data.random_split(dataset, [t_size, v_size])
    train_batchsize = batchsize if batchsize < t_size else t_size
    valid_batchsize = batchsize if batchsize < v_size else v_size
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batchsize, drop_last=True, shuffle=True, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=valid_batchsize, drop_last=True, shuffle=True, num_workers=num_workers)
    
    # define neural network
    dim1, dim2 = len(dataset[0][0]), len(dataset[0][1])  # need to state the correct dimensions
    model = FullyConnected(var1_dim=dim1, var2_dim=dim2, L=L, H=H)
    model.to(device)
    
    # define optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=lr, amsgrad=True)
    
    # train model
    model_trained, train_loss, valid_loss = train(
        model, trainloader, validloader, optimizer, num_epochs, device, bar, valid=valid)
    
    # validate trained model
    val_score = validate(model_trained, validloader, device)
    
    return val_score, train_loss, valid_loss, model_trained


# ----- TRAINING FUNCTIONS WITHOUT SUMM-STATS ----- #


def train_summ(
    model, model_summ, trainloader, validloader, optimizer, scheduler,
    num_epochs, device, bar=True, valid=True, valid_num=5, plateau=False):
    
    # need to compute validation estimate if using ReduceLROnPlateau
    if valid==False and plateau==True:
        raise ValueError('You need to compute validation errors weith ReduceLROnPlateau.')
    
    train_loss = list()
    valid_loss = list()
    val = 0.0
    pbar = tqdm(range(num_epochs), leave=True, disable=not bar)
    for epoch in pbar:

        for x, y in trainloader:
            
            # move to device
            x, y = x.to(device), y.to(device)
            
            # compute summary statistics
            Sy = model_summ(y)

            # compute loss
            loss = minef_loss(x, Sy, model, device)

            # Zero grad the NN optimizer
            optimizer.zero_grad()

            # Back-Propagation
            loss.backward()

            # Perform opt steps for NN
            optimizer.step()

            # save a few things to lists
            train_loss.append(-loss.clone().detach())
            
        # validate every 5 epochs
        if valid and ((epoch + 1) % valid_num == 0):
            with torch.no_grad():
                val = validate_summ(model, model_summ, validloader, device)
                valid_loss.append(float(val))
                
        pbar.set_postfix(
            {'Train Loss': '{:.3f}'.format(-loss.data.cpu().numpy()),
             'Valid Loss': '{:.3f}'.format(float(val))})
        
        # call learning rate scheduler 
        if plateau:
            scheduler.step(val)
        else:
            scheduler.step()
    
    # move list items to cpu
    train_loss = np.array([m.cpu().data.numpy() for m in train_loss])
    
    return model, model_summ, train_loss, valid_loss


def validate_summ(model, model_summ, validloader, device):
    
    scores = list()
    for x, y in validloader:
        
        x, y = x.to(device), y.to(device)
        
        # compute summary statistics
        Sy = model_summ(y)
        
        # compute loss and store in list
        loss = minef_loss(x, Sy, model, device)
        scores.append(loss.cpu().data.numpy())
        
    return np.mean(-np.array(scores))


def evaluate_summ(designs, prior, device, modelparams, DatasetClass, bar=False, valid=True, valid_num=5, **kwargs):
    
    # extract model parameters
    batchsize = modelparams['batchsize']
    L = modelparams['layers']
    H = modelparams['hidden']
    lr = modelparams['lr']
    num_epochs = modelparams['num_epochs']
    num_workers = modelparams['num_workers']
    
    # define data set
    dataset = DatasetClass(designs, prior, device, **kwargs)

    # define data loaders
    t_size = int(len(dataset) * 0.80)  # 80% training data
    v_size = int(len(dataset) * 0.20)  # 20% validation data
    trainset, validset = torch.utils.data.random_split(dataset, [t_size, v_size])
    train_batchsize = batchsize if batchsize < t_size else t_size
    valid_batchsize = batchsize if batchsize < v_size else v_size
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batchsize, drop_last=True, shuffle=True, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=valid_batchsize, drop_last=True, shuffle=True, num_workers=num_workers)
    
    dim1, dim2 = len(dataset[0][0]), len(dataset[0][1][0])

    # For summary statistics
    summary_stats = modelparams['summary_stats']  # boolean

    if summary_stats:
        
        # extract data
        summ_L = modelparams['summ_L']
        summ_H = modelparams['summ_H']
        summ_out = modelparams['summ_out']
        num_measurements = modelparams['num_measurements']

        # define the neural summary stats
        model_summ = CAT_NSS(
            var_dim=dim2, L=summ_L, H=summ_H, Output_dim=summ_out, 
            num_measurements=num_measurements)
        model_summ.to(device)
    
        # define neural network
        model = FullyConnected(
            var1_dim=dim1, var2_dim=summ_out * num_measurements, L=L, H=H)
        model.to(device)
        
        # define optimizer
        optimizer = Adam(
            list(model_summ.parameters()) + list(model.parameters()),
            lr=lr, amsgrad=True, weight_decay=modelparams['weight_decay'])

    else:
        
        dim2 = dim2 * modelparams['num_measurements']
        dataset.Y = dataset.Y.reshape(-1, dim2)
        
        def model_summ(x):
            return x
        
        # define neural network
        model = FullyConnected(var1_dim=dim1, var2_dim=dim2, L=L, H=H)
        model.to(device)
        
        # define optimizer
        optimizer = Adam(
            model.parameters(),
            lr=lr, amsgrad=True, weight_decay=modelparams['weight_decay'])
    
    # define the scheduler
    pl = False
    if modelparams['scheduler'] == 'none':  # basically do nothing
        scheduler = StepLR(optimizer, step_size=10000, gamma=1)
    elif modelparams['scheduler'] == 'step':  # step scheduler; need appropriate params
        scheduler = StepLR(
            optimizer, step_size=modelparams['step_scheduler'],
            gamma=modelparams['gamma_scheduler'])
    elif modelparams['scheduler'] == 'plateau':  # reduce on plateau; need params
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max',
            factor=modelparams['plateau_factor'],
            patience=modelparams['plateau_patience'], verbose=bar)
        pl = True
    else:
        raise NotImplementedError('This scheduler is not implemented yet.')
        
    # train model  
    model_trained, model_summ_trained, train_loss, valid_loss = train_summ(
        model, model_summ, trainloader, validloader,
        optimizer, scheduler, num_epochs, device, bar, valid=valid, valid_num=valid_num, plateau=pl)

    # validate trained model
    val_score = validate_summ(
        model_trained, model_summ_trained, validloader, device)
    
    if not summary_stats:
        model_summ_trained = None
    
    return val_score, train_loss, valid_loss, model_trained, model_summ_trained


# ---- GP Training ---- #


def train_GP_torch(
    gp, likelihood, train_x, train_y,
    lr=0.01, train_iter=1000, trainbar=True):
    
    # set mode to train
    gp.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(gp.parameters(), lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, gp)

    for i in tqdm(range(train_iter), disable=not trainbar):
        optimizer.zero_grad()
        out = gp(train_x)
        loss = -mll(out, train_y)
        loss.backward()
        #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()
        
    return gp, likelihood, loss.detach().item()


def train_GP_scipy(
    gp, likelihood, train_x, train_y, 
    lr=0.01, train_iter=1000, trainbar=True, method='L-BFGS-B'):
    
    # set mode to train
    gp.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, gp)
    
    # Train the model with your favourite algorithm (default is L-BFGS-B)
    fit_gpytorch_model(mll, method=method);

    return gp, likelihood #, loss.detach().item()