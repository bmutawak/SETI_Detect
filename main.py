from datasets import  get_dataloaders
import argparse
import os
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from efficient_net_model import EfficientNetModel

# Relevant hyperparamters
hyp = {
       'image_size':256,
       'batch_size':32,
       'vertical_flip_prob':0.5,
       'horizontal_flip_prob':0.5,
       'rotation_degrees':25,
       'lr':0.001,
       'adam':True,
       'perc_train':0.8,
       'shuffle':True,
       'stratify':True,
       'num_epochs':300,
       'device': 'cuda'
       }


def combine_args(args):
    """
    Combines hyperparameters + supplied arguments

    Parameters
    ----------
    args : ArgumentParser
        parser supplied from argparse.

    Returns
    -------
    None.

    """
    
    # Iterate through hyperparmater keys
    for key in hyp.keys():
        hyp[key] = args.__dict__[key]
    
    return

def train_one_epoch(model, loader, opt, loss_func, epoch):
    """
    Trains a model for one epoch

    Parameters
    ----------
    model : torch.nn.Module
        Model class.
    loader : torch.utils.Dataloder
        dataset loader.
    opt : torch.optim
        optimizer.
    loss_func : TYPE
        Loss function.
    epoch : int
        Current epoch.

    Returns
    -------
    None.

    """
    
    # ITerate through each batch
    model.train()
    pbar = tqdm(loader, desc=f'Training Epoch: {epoch}')
    for batch_index, (images, targets) in enumerate(pbar):
        
        # Send to device
        images = images.to(hyp['device'])
        targets = targets.to(hyp['device'])
        
        # Forward
        opt.zero_grad()
        pred = model(images)
        
        # Backprop
        loss = loss_func(pred, targets.view(targets.shape[0], 1))
        loss.backward()
        opt.step()
    
    return

def validate(model, loader):
    """
    

    Parameters
    ----------
    model : toch.nn.Module
        Model class.
    loader : toch.utils.DataLoader
        dataset loader.

    Returns
    -------
    None.

    """
    
    model.eval()
    pbar = tqdm(loader, desc='Validating model...')
    predictions = []
    truths = []
    roc_score = 0
    with torch.no_grad():
            
        for batch_index, (images, targets) in enumerate(pbar):
            
            # Send to device
            images = images.to(hyp['device'])
            targets = targets.to(hyp['device'])
            
            # Forward
            pred = model(images)
            
            # Save for later
            predictions.extend(pred.detach().cpu().tolist())
            truths.extend(targets.detach().cpu().tolist())
            
        roc_score = roc_auc_score(truths, predictions)
    
    return roc_score * 100

            
            
            
    
def train(args):
    
    # Ensure train directory is acceptable
    if not os.path.exists(args.train_dir):
        print("ERROR: INCORRECT TRAINING DIRECTORY PATH SUPPLIED")
        return
    
    # Combine hyperparameters + supplied args
    combine_args(args)
    
    # Grab data loaders
    train_loader, val_loader = get_dataloaders(args.train_dir, hyp)
    
    # Init model, optimizer
    model = EfficientNetModel(6, 3, 1).to(hyp['device'])
    
    if hyp['adam']:
        optimizer = torch.optim.Adam(model.seq.parameters(), lr=hyp['lr'])
    else:
        optimizer = torch.optim.SGD(model.seq.parameters(), lr=hyp['lr'])
    
    # loss func
    loss_func = torch.nn.BCEWithLogitsLoss()
    
    # load saved model
    if not args.model_file is None:
        cp = torch.load(args.model_file)
        model.load_state_dict(cp['model_dict'])
        optimizer.load_state_dict(cp['opt_dict'])
        
    
    # Create models directory if it doesn't exist
    if not os.path.exists("./models"):
        os.mkdir("./models")
    
    # Iterate through all epochs
    current_best_score = 0
    for epoch in range(hyp['num_epochs']):
        
        # Train
        train_one_epoch(model, train_loader, optimizer, loss_func, epoch)
        
        # Val
        roc_score = validate(model, val_loader)
        
        print(f'Epoch: {epoch}\tScore: {roc_score}')
        # If better than what we've seen yet, save it
        if roc_score > current_best_score:
            model_path = os.path.join("./models", f'effnet_model_{int(roc_score)}_score_{epoch}_epoch.pt')
            torch.save({'epoch':epoch,
                        'model_dict':model.state_dict(),
                        'opt_dict':optimizer.state_dict(),
                        'roc_score':roc_score}, model_path)
    
    return    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', default=hyp['image_size'], type=int)
    parser.add_argument('--batch-size', default=hyp['batch_size'], type=int)
    parser.add_argument('--vertical-flip-prob', default=hyp['vertical_flip_prob'], type=float)
    parser.add_argument('--rotation-degrees', default=hyp['rotation_degrees'], type=float)
    parser.add_argument('--horizontal-flip-prob', default=hyp['horizontal_flip_prob'], type=float)
    parser.add_argument('--lr', default=hyp['lr'], type=float)
    parser.add_argument('--adam', default=hyp['adam'], action='store_true')
    parser.add_argument('--perc-train', default=hyp['perc_train'], type=float)
    parser.add_argument('--shuffle', default=hyp['shuffle'], action='store_true')
    parser.add_argument('--stratify', default=hyp['stratify'], action='store_true')
    parser.add_argument('--num-epochs', default=hyp['num_epochs'], type=int)
    parser.add_argument('--device', default=hyp['device'], type=str)
    parser.add_argument('--train-dir', default='./data/train', type=str)
    parser.add_argument('--model-file', type=str)
    args = parser.parse_args()    
    train(args)

    