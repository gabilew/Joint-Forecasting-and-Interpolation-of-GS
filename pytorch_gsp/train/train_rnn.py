### training code ####
### modified from https://github.com/zhiyongc/Graph_Convolutional_LSTM/blob/master/Code_V2/HGC_LSTM%20%26%20Experiments.ipynb
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable

toolbar_width=20



def Train(model, train_dataloader, valid_dataloader, learning_rate = 1e-5, epochs = 300, patience = 10, 
verbose=1, gpu = True, sample = None, optimizer = 'rmsprop'):


    if optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate )

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    batch_size = train_dataloader.batch_size
   
    if gpu: device='cuda' 
    else:  device= 'cpu'
 
    losses_epochs_train = []
    losses_epochs_valid = []
    time_epochs = []
    time_epochs_val = []

    is_best_model = 0
    patient_epoch = 0
    scheduler = model.schedule(optimizer)

    for epoch in range(epochs):
        pre_time = time.time()
      

        try:
            data_size=train_dataloader.dataset.data_size
        except: pass
        try:
            data_size=train_dataloader.dataset.tensors[0].shape[0]
        except: pass
        n_iter=data_size/train_dataloader.batch_size
        if verbose:
            count=0

            checkpoints=np.linspace(0,n_iter,toolbar_width).astype(np.int16)
            text='Epoch {:02d}: '.format(epoch)
            sys.stdout.write(text+"[%s]" % (" " * toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (toolbar_width+1))

        losses_train = []
        losses_valid = []

        for data in train_dataloader:            
            inputs, labels = data
            if inputs.shape[0] != batch_size:
                continue

            model.zero_grad()            
            outputs = model(inputs.to(device))
            outputs, y = torch.squeeze(outputs),  torch.squeeze(labels).to(device)              
            loss_train = model.loss(outputs,y)
            
            losses_train.append(loss_train.cpu().data.numpy())
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()            
            
            if verbose:
                if count in checkpoints:
                    sys.stdout.write('=')
                    sys.stdout.flush()
                count+=1
          
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']
        if learning_rate >1e-5:
            scheduler.step()
        time_epochs.append(time.time()-pre_time)

        pre_time = time.time()        
        
        losses_valid = []
        for data in valid_dataloader:            
            inputs, labels = data
            if inputs.shape[0] != batch_size:
                continue
       
            outputs= model(inputs.to(device))
            outputs, y = torch.squeeze(outputs),  torch.squeeze(labels).to(device)        
            losses_valid.append(model.loss(outputs, y).cpu().data.numpy())
     
        time_epochs_val.append(time.time()-pre_time)
        losses_epochs_train.append(np.mean(losses_train))
        losses_epochs_valid.append(np.mean(losses_valid))
      
        avg_losses_epoch_train = losses_epochs_train[-1]
        avg_losses_epoch_valid = losses_epochs_valid[-1] 
   
  
        if avg_losses_epoch_valid >100000000000:
            print("Diverged")
            return (None,None)
        if epoch == 0:
            is_best_model = True
            best_model = model
            min_loss = avg_losses_epoch_valid
        else:
            if min_loss - avg_losses_epoch_valid > 1e-6:
                is_best_model = True
                best_model = model
                min_loss = avg_losses_epoch_valid
                patient_epoch = 0
            else:
                is_best_model = False
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break
      
        if verbose:
            sys.stdout.write("]")
  
        print(' train loss: {}, valid loss: {}, time: {}, lr: {}'.format( \
                np.around(avg_losses_epoch_train, 6),\
                np.around(avg_losses_epoch_valid, 6),\
                np.around([time_epochs[-1] ] , 2),\
                learning_rate) )
        

    return best_model, [losses_epochs_train ,
    losses_epochs_valid ,
    time_epochs ,
    time_epochs_val ]


def Evaluate(model, dataloader, scale=1, pred_len = 1, gpu = True):

    batch_size = dataloader.batch_size
    pre_time = time.time()

    gpu = torch.cuda.is_available()
    if gpu: device='cuda' 
    else:  device= 'cpu'

    tested_batch = 0
    losses_mse = []
    losses_l1 = []
    losses_mape = []

    for i,data in enumerate(dataloader):
        inputs, labels = data
        if inputs.shape[0] != batch_size:
            continue

        outputs = model(inputs.to(device))
        outputs, y = torch.squeeze(outputs),  torch.squeeze(labels).to(device)

      
        loss_mse = torch.nn.MSELoss()(outputs*scale, y*scale).cpu().data
        loss_l1 = torch.nn.L1Loss()(outputs*scale, y*scale).cpu().data
      
        outputs = outputs.cpu().data.numpy()
        y  = y.cpu().data.numpy()        
        outputs = outputs*scale
        y  = y*scale
        
        abs_diff =  np.abs((outputs-y))
        abs_y = np.abs(y)
        abs_diff=abs_diff[abs_y>1]
        abs_y=abs_y[abs_y>1]
        
        loss_mape = abs_diff/abs_y
        loss_mape = np.mean(loss_mape)*100              
        
        losses_mse.append(loss_mse)
        losses_l1.append(loss_l1)
        losses_mape.append(loss_mape)
    
    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    mean_l1 = np.mean(losses_l1, axis = 0) 
    rmse = np.mean(np.sqrt(losses_mse))
    print('Test: MAE: {},  RMSE : {}, MAPE : {}'.format(mean_l1, rmse,np.mean(losses_mape)))
  

    return [losses_l1, losses_mse, mean_l1,  np.mean(losses_mape), time.time()-pre_time]


