import time
import torch
import os
from torchvision.utils import make_grid
import ray
from ray import tune
import optuna

import misc
import loss
import models
import data

def computeLoss(predicted, ground_truth, loss_function, device=None):
    '''
    The model can either return its predictions as a list or as a tensor. This function handles either case.
    '''
    if isinstance(predicted, list):
        loss = 0.0
        loss_list = [0, 0, 0]
        for i in range(len(predicted)):
            instance_loss, instance_loss_list = loss_function(predicted[i].to(device), ground_truth[i].unsqueeze(0).to(device))
            for i in range(2):
                loss_list[i] += instance_loss_list[i]

            loss += instance_loss
        for i in range(2):
            loss_list[i] = loss_list[i] / len(predicted)
        return loss / len(predicted), loss_list
    else:
        loss = loss_function(predicted.to(device), ground_truth.to(device))
        return loss

def train_step(model, optimizer, train_loader, epoch_n, loss_fn, device, model_params=None, timer=None, mem=None, scaler=None):
    '''
    Iterates through the dataloader once, training the model for dataset.
    '''
    train_loss = []
    if scaler:
        autocast_enable = True
    else:
        autocast_enable = False

    for i, inputs in enumerate(train_loader):

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(autocast_enable):
            if model_params["parallel_settings"]["flag"]:
                outputs = model(inputs[0].float().to(device), inputs[3].float().to(device), inputs[4].float().to(device), inputs[2]) # Input the resized data image and its original image
            else:
                outputs = model(inputs[0].float().to(device), original_images=inputs[2])

            for output in outputs:
                if torch.isnan(output).any().item():
                    print(f"NaN values detected in output.")
                if torch.isinf(output).any().item():
                    print(f"Inf values detected in output.")
            for param_name, param in model.named_parameters():
                if torch.isnan(param).any().item():
                    print(f"NaN values detected in {param_name}.")
                if torch.isinf(param).any().item():
                    print(f"Inf values detected in {param_name}.")

            # calculate loss
            loss, loss_list = computeLoss(outputs, inputs[1], loss_fn, device)
        train_loss.append(loss.cpu())

        #scaler.scale(loss).backward()

        #loss.backward()

        if mem:
            print(torch.cuda.memory_summary(device=device, abbreviated=True))

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        '''if i % 50 == 0:
            for name, param in model.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}, grad={param.grad}")'''

            
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        #scaler.step(optimizer)
        #scaler.update()


        if i % 50 == 0:
            time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - timer))
            print('Epoch:{:2d} | Iter:{:5d} | Train Loss: {:.4f} | Dice: {:.4f} | BCE: {:.4f} | Time: {} '
                .format(epoch_n + 1 , i, loss, loss_list[0], loss_list[1],  time_lapse))
    return torch.mean(torch.stack(train_loss))

def train(model, loss_fn, train_dl, valid_dl, optim, config, scheduler=None, writer=None, mem=None, scaler=None):
    start_time = time.time()
    best_validation_loss = 99999
    device = config["device"]
    if scaler:
        scaler = torch.cuda.amp.GradScaler()
    if writer:
        samples = misc.initialize_samples(train_dl.dataset, valid_dl.dataset)

    for epoch in range(config["epochs"]):

        model.train()

        avg_train_loss = train_step(model, optim, train_dl, epoch, loss_fn, device, config["model_params"], start_time, mem, scaler) #, writer, samples)
        
        avg_validation_loss, individual_losses, precision, recall, f1= evaluate(model, valid_dl, loss_fn, device, config["model_params"]["parallel_settings"]["flag"], config["threshold"])
        
        ########
        if scheduler:
            scheduler.step()
            current_lr = optim.param_groups[0]['lr']
            print("Current Learning Rate:", current_lr)

        if writer:
            write(samples, writer, model, epoch, device, config['threshold'], type='train')
            write(samples, writer, model, epoch, device, config['threshold'], type='validation')

            '''writer.add_scalar('Avg. Train for the Epoch', avg_train_loss, global_step=epoch)
            writer.add_scalar('Avg. Validation for the Epoch', avg_validation_loss, global_step=epoch)
            writer.add_scalar('Avg. Dice Loss for the Epoch', individual_losses[0], global_step=epoch)
            writer.add_scalar('Avg. BCE Loss for the Epoch', individual_losses[1], global_step=epoch)

            train_imgs, valid_imgs = samples
            with torch.no_grad():
                for j in range(len(train_imgs)):
                    if config["model_params"]["parallel_settings"]["flag"]:
                        prediction = model(torch.unsqueeze(train_imgs[j][0], 0).float().to(device), torch.unsqueeze(train_imgs[j][3], 0).float().to(device), torch.unsqueeze(train_imgs[j][4], 0).float().to(device), train_imgs[j][1]).cpu()
                    else:
                        prediction = model(torch.unsqueeze(train_imgs[j][0], 0).float().to(device)).cpu()
                    
                    thresholding = torch.round(prediction) #torch.where(prediction > threshold, prediction-prediction+1, prediction - prediction)
                    grid_images = make_grid([(train_imgs[j][0]), (train_imgs[j][1]), torch.squeeze(prediction, 0).cpu(), torch.squeeze(thresholding, 0).cpu()], nrow=4, normalize=True, scale_each=True)
                    writer.add_image(f'TrainImages_Grid/{j}', grid_images, global_step=epoch+1)

                for j in range(len(valid_imgs)):
                    if config["model_params"]["parallel_settings"]["flag"]:
                        prediction = model(torch.unsqueeze(valid_imgs[j][0], 0).float().to(device), torch.unsqueeze(valid_imgs[j][3], 0).float().to(device), torch.unsqueeze(valid_imgs[j][4], 0).float().to(device), valid_imgs[j][1])
                    else:
                        prediction = model(torch.unsqueeze(valid_imgs[j][0], 0).float().to(device))
                    thresholding = torch.round(prediction) # torch.where(prediction > threshold, prediction-prediction+1, prediction - prediction)
                    grid_images = make_grid([(valid_imgs[j][0]), (valid_imgs[j][1]), torch.squeeze(prediction, 0).cpu(), torch.squeeze(thresholding, 0).cpu()], nrow=4, normalize=True, scale_each=True)
                    writer.add_image(f'ValImages_Grid/{j}', grid_images, global_step=epoch+1)'''

        print('Epoch:{:2d} | Average Train Loss: {:.4f} | Validation Loss: {:.4f} | Dice Loss: {:.4f} | BCE Loss: {:.4f} | Time: {} '
              .format(epoch + 1 , avg_train_loss, avg_validation_loss, individual_losses[0], individual_losses[1], time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))
        print("\tPrecision: {:.4f} | Recall: {:.4f} | F1: {:.4f}".format(precision, recall, f1))

        # Save the model upon beating a validation loss "record"
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            model_path = 'model_{}_{}.pth'.format(config["modelname"], best_validation_loss)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
            }, os.path.join(config["experiment_name"],"weights", model_path))

        model_path = 'recent.pth'
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
            }, os.path.join(config["experiment_name"],"weights", model_path))
        
    return

def write(samples, writer, model, epoch, device, threshold, type='train'):
    if type == 'train':
        text = 'Train Image'
    else:
        text = 'Valid Images'

    with torch.no_grad():
        for i in range(len(samples)):
            small, mask, original, medium, large = samples[i]
            small, medium, large = small.to(device), medium.to(device), large.to(device)
            prediction = model(small.unsqueeze(0), medium.unsqueeze(0), large.unsqueeze(0), original).cpu()

            prediction = (torch.sigmoid(prediction[0]) > threshold).to(mask.dtype)

            # Mask, Original, Predicted
            grid_images = make_grid([original, mask, torch.squeeze(prediction, 0).cpu()], nrow=3, normalize=True, scale_each=True)
            writer.add_image(f'{text} {i+1}', grid_images, global_step=epoch+1)
    return

def evaluate(model, validation_loader, loss_fn, device, parallel=True, threshold=0.01):
    '''
    '''
    model.eval()
    individual_losses = [0, 0, 0]
    total_validation_loss = 0.0
    all_true_labels = []
    all_predicted_labels = []
    with torch.no_grad():
        for i, validation_data in enumerate(validation_loader):
            if parallel:
                small, ground_truth, original, medium, large = validation_data
                # Tuples ^
                val_outputs = model(small.to(device), medium.to(device), large.to(device), original)
            else:
                small, ground_truth, original = validation_data
                val_outputs = model(small.to(device), original_images=original)

            # Threshold the predicted masks
            if isinstance(val_outputs, list):
                for i in range(len(val_outputs)):
                    val_outputs[i] = (torch.sigmoid(val_outputs[i]) > threshold).to(ground_truth[i].dtype)
            else:
                val_outputs = (torch.sigmoid(val_outputs) > threshold).to(ground_truth[i].dtype)

            #compileLabels(val_outputs, ground_truth, all_true_labels, all_predicted_labels, device)
            all_true_labels.extend(ground_truth)
            all_predicted_labels.extend(val_outputs)

            val_loss, loss_list = computeLoss(val_outputs, ground_truth, loss.CombinedLossUnlog(3), device)#loss_fn, device)

            total_validation_loss += val_loss
            for i in range(len(individual_losses)):
                individual_losses[i] += loss_list[i]
    #all_true_labels = torch.cat(all_true_labels)
    #all_predicted_labels = torch.cat(all_predicted_labels)

    avg_validation_loss = total_validation_loss / len(validation_loader)
    for i in range(len(individual_losses)):
        individual_losses[i] = individual_losses[i] / len(validation_loader)
    precision, recall, f1 = loss.precision_recall_f1(all_true_labels, all_predicted_labels, device)
    
    return avg_validation_loss, individual_losses, precision, recall, f1

def compileLabels(predicted, ground_truth, true_labels, predicted_labels, device):
    true_labels.append(ground_truth.view(-1).to(device))
    predicted_labels.append(predicted.view(-1).to(device))

def guitar(trial, root, epochs):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(trial.number % 2)
    config = misc.initialize_config(root)

    # 512x512
    config["model_params"]["parallel_settings"]["trunk_blocks"] = trial.suggest_int('trunk_blocks', 1, 3)
    uwu = trial.suggest_categorical('branch_out_channels', [32, 64])
    config["model_params"]["parallel_settings"]["branch_out_channels"] = uwu
    config["model_params"]["trunk/res_channels"] = uwu

    blocks = [trial.suggest_int('int1', 3, 5), trial.suggest_int('int2', 3, 5), trial.suggest_int('int3', 3, 6)]
    config["model_params"]["resnet_settings"]["blocks"] = blocks

    config["model_params"]["transformer_params"]["num_heads"] = trial.suggest_categorical('num_heads', [8, 12])
    config["model_params"]["transformer_params"]["num_layers"] = trial.suggest_int('num_layers', 10, 14)
    # config["model_params"]["transformer_params"]["hidden_dim"] = trial.suggest_categorical('hidden_dim', [768, 1024])

    config["epochs"] = epochs
    config["num_workers"] = 4

    train_dl, validity_dl = data.initialize_data(config)

    model = models.ModernTransUNetV2(config["model_params"]).to(config["device"])
    loss_fn = loss.CombinedLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["decay"])

    start_time = time.time()
    for epoch in range(config["epochs"]):
        model.train()
        avg_train_loss = train_step(model, optimizer, train_dl, epoch, loss_fn, config["device"], config["model_params"], start_time)

    avg_validation_loss, individual_losses = evaluate(model, validity_dl, loss_fn, config["device"], config["model_params"]["parallel_settings"]["flag"])
    
    print('Epoch:{:2d} | Average Train Loss: {:.4f} | Validation Loss: {:.4f} | Dice Loss: {:.4f} | BCE Loss: {:.4f} | Time: {} '
              .format(epoch + 1 , avg_train_loss, avg_validation_loss, individual_losses[0], individual_losses[1], time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))
    # print("\tPrecision: {:.4f} | Recall: {:.4f} | F1: {:.4f}".format(precision, recall, f1))
    return avg_validation_loss


def tuningRod(config):
    tuner = tune.Tuner(
    guitar,
    param_space=config,
    )
    results = tuner.fit()
    return results