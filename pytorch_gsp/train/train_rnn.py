import sys
import time

import numpy as np
import torch
from torch.autograd import Variable

toolbar_width=20



def TrainModel(model, train_dataloader, valid_dataloader, learning_rate = 1e-5, num_epochs = 300, patience = 10, min_delta = 0.00001,
verbose=1, use_gpu = True, sample = None):

    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size() 


    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
    #optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate )
   
    if use_gpu: device='cuda' 
    else:  device= 'cpu'
 
    losses_epochs_train = []
    losses_epochs_valid = []
    mse_epochs_train = []
    mse_epochs_valid = []
    mae_epochs_train = []
    mae_epochs_valid = []
    time_epochs = []
    time_epochs_val = []


    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    scheduler = model.schedule(optimizer)

    for epoch in range(num_epochs):
        pre_time = time.time()
        trained_number = 0
        valid_dataloader_iter = iter(valid_dataloader)        

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

        
        mse_train = []
        mse_valid = []
        losses_train = []
        losses_valid = []
        mae_train = []
        mae_valid = []
        for data in train_dataloader:            
            inputs, labels = data
            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
            
            
            model.zero_grad()            
            outputs = model(inputs)
            outputs, y = torch.squeeze(outputs),  torch.squeeze(labels).to(device)              
            loss_train = model.loss(outputs,y)
            
            losses_train.append(loss_train.cpu().data.numpy())
            mse_train.append(loss_MSE(outputs, y).cpu().data.numpy())
            mae_train.append(loss_L1(outputs, y).cpu().data.numpy())
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
        

        for data in valid_dataloader:            
            inputs_val, labels_val = data
            if inputs.shape[0] != batch_size:
                continue
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else:
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            outputs_val= model(inputs_val).to(device)
            outputs_val, y = torch.squeeze(outputs_val),  torch.squeeze(labels_val).to(device)
        
            mse_valid.append(model.loss(outputs_val, y).cpu().data.numpy())
            mae_valid.append(loss_L1(outputs_val, y).cpu().data.numpy())

            
        time_epochs_val.append(time.time()-pre_time)
        mse_epochs_train.append(np.mean(mse_train))
        mse_epochs_valid.append(np.mean(mse_valid))
        mae_epochs_train.append(np.mean(mae_train))
        mae_epochs_valid.append(np.mean(mae_valid))

        avg_losses_epoch_train = mse_epochs_train[-1]
        avg_losses_epoch_valid = mse_epochs_valid[-1] 
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)
        
        # Early Stopping
  
        if avg_losses_epoch_valid >100000000000:
            print("Diverged")
            return (None,None)
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break

        # Print training parameters        
        if verbose:
            sys.stdout.write("]")

        if device == 'cuda':
            print(' train_loss: {}, valid_loss: {}, time: {}, lr: {}'.format( \
                    np.around(avg_losses_epoch_train, decimals=8),\
                    np.around(avg_losses_epoch_valid, decimals=8),\
                    np.around([time_epochs[-1] ] , decimals=2),\
                    learning_rate) )
        else:
            print(' train_loss: {}, valid_loss: {}, time: {}, lr: {}'.format( \
                    np.around(avg_losses_epoch_train, decimals=8),\
                    np.around(avg_losses_epoch_valid, decimals=8),\
                    np.around([time_epochs[-1] ] , decimals=2),\
                    learning_rate) )
        

    min_loss = []
    for data in valid_dataloader:
        outputs = best_model(inputs)
        outputs, y = torch.squeeze(outputs),  torch.squeeze(labels).to(device)
        min_loss.append(best_model.loss(outputs,y).cpu().data.numpy())   


    return best_model, [mse_epochs_train ,
    mse_epochs_valid ,
    mae_epochs_train ,
    mae_epochs_valid ,
    time_epochs ,
    time_epochs_val ]


def TestModel(model, test_dataloader, max_speed, pred_len = 1):

    inputs, labels = next(iter(test_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    cur_time = time.time()
    pre_time = time.time()

    use_gpu = torch.cuda.is_available()
    if use_gpu: device='cuda' 
    else:  device= 'cpu'


    tested_batch = 0
    losses_mse = []
    losses_l1 = []
    losses_mape = []


    for i,data in enumerate(test_dataloader):
        inputs, labels = data

        if inputs.shape[0] != batch_size:
            continue

        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # rnn.loop()


        outputs = None
        outputs = model(inputs)
        outputs, y = torch.squeeze(outputs),  torch.squeeze(labels).to(device)

        loss_MSE = torch.nn.MSELoss()
        loss_L1 = torch.nn.L1Loss()
        loss_mse = loss_MSE(outputs*max_speed, y*max_speed).cpu().data
        loss_l1 = loss_L1(outputs*max_speed, y*max_speed).cpu().data
      
        outputs = outputs.cpu().data.numpy()
        y  = y.cpu().data.numpy()        
        outputs = outputs*max_speed
        y  = y*max_speed
        
    
        #outputs = outputs*9/5 + 32
        #y = y*9/5 + 32
        
        abs_diff =  np.abs((outputs-y))
        abs_y = np.abs(y)
        abs_diff=abs_diff[abs_y>1]
        abs_y=abs_y[abs_y>1]

        
        loss_mape = abs_diff/abs_y
        loss_mape = np.mean(loss_mape)*100             
        
        
        losses_mse.append(loss_mse)
        losses_l1.append(loss_l1)
        losses_mape.append(loss_mape)
        tested_batch += 1

        if tested_batch % 1000 == 0:
           
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format( \
                  tested_batch * batch_size, \
                  np.around([loss_l1.data[0]], decimals=8), \
                  np.around([loss_mse.data[0]], decimals=8), \
                  np.around([time.time() - pre_time], decimals=8) ) )
            
        
    losses_l1 = np.array(losses_l1)

    losses_mse = np.array(losses_mse)
    mean_l1 = np.mean(losses_l1, axis = 0) 
    rmse = np.mean(np.sqrt(losses_mse))
    print('Tested: L1_mean: {},  RMSE : {}, MAPE : {}'.format(mean_l1, rmse,np.mean(losses_mape)))
  

    return [losses_l1, losses_mse, mean_l1,  np.mean(losses_mape), time.time()-pre_time]


