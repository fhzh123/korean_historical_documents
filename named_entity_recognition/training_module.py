# Import modules
import os
import time
import math
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score

# Import PyTorch
import torch

def train_model(args, model, dataloader_dict, optimizer, criterion, scheduler):

    # Setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_f1 = None
    total_train_loss_list = list()
    total_test_loss_list = list()
    for e in range(args.num_epoch):
        start_time_e = time.time()
        print(f'Model Fitting: [{e+1}/{args.num_epoch}]')
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                model.eval()
                val_precision = 0
                val_recall = 0
                val_f1 = 0
                val_loss = 0
            for i, (src, trg, king_id) in enumerate(dataloader_dict[phase]):
                # Sourcen, Target sentence setting
                src = src.to(device)
                trg = trg.to(device)
                king_id = king_id.to(device)
                
                # Optimizer setting
                optimizer.zero_grad()

                # Model / Calculate loss
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(src, king_id)
                    output_flat = output.transpose(0,1)[1:].transpose(0,1).contiguous().view(-1, 10)
                    trg_flat = trg.transpose(0,1)[1:].transpose(0,1).contiguous().view(-1)
                    loss = criterion(output_flat, trg_flat)
                    if phase == 'valid':
                        val_loss += loss.item()
                        output_list = output_flat.max(dim=1)[1].tolist()
                        real_list = trg_flat.tolist()
                        val_precision += precision_score(real_list, output_list, average='macro')
                        val_recall += recall_score(real_list, output_list, average='macro')
                        val_f1 += f1_score(real_list, output_list, average='macro')
                # If phase train, then backward loss and step optimizer and scheduler
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    total_train_loss_list.append(loss.item())

                    # Print loss value only training
                    if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
                        total_loss = loss.item()
                        output_list = output_flat.max(dim=1)[1].tolist()
                        real_list = trg_flat.tolist()
                        precision_ = precision_score(real_list, output_list, average='macro')
                        recall_ = recall_score(real_list, output_list, average='macro')
                        f1_ = f1_score(real_list, output_list, average='macro')
                        print("[Epoch:%d][%d/%d] train_loss:%5.3f | train_pp:%5.3fS | train_precision:%5.3f | train_recall:%5.3f | train_f1:%5.3f | learning_rate:%3.8f | spend_time:%5.2fmin"
                                % (e+1, i, len(dataloader_dict['train']), 
                                total_loss, math.exp(total_loss), 
                                precision_, recall_, f1_, 
                                optimizer.param_groups[0]['lr'], 
                                (time.time() - start_time_e) / 60))
                        freq = 0
                    freq += 1

            # Finishing iteration
            if phase == 'valid':
                val_loss /= len(dataloader_dict['valid'])
                val_precision /= len(dataloader_dict['valid'])
                val_recall /= len(dataloader_dict['valid'])
                val_f1 /= len(dataloader_dict['valid'])
                total_test_loss_list.append(val_loss)
                print("[Epoch:%d]val_loss:%5.3f | val_pp:%5.2f | val_precision:%5.3f | val_recall:%5.3f | val_f1:%5.3f | learning_rate:%3.8f | spend_time:%5.2fmin"
                        % (e+1, val_loss, 
                        math.exp(val_loss), val_precision, val_recall,
                        val_f1, optimizer.param_groups[0]['lr'], 
                        (time.time() - start_time_e) / 60))
                if not best_val_f1 or val_f1 > best_val_f1:
                    print("[!] saving model...")
                    if not os.path.exists(args.save_path):
                        os.mkdir(args.save_path)
                    torch.save(model.state_dict(), 
                               os.path.join(args.save_path, f'ner_model_{args.baseline}.pt'))
                    best_val_f1 = val_f1

    pd.DataFrame(total_train_loss_list).to_csv(os.path.join(args.save_path, f'ner_train_loss_{args.baseline}.csv'), index=False)
    pd.DataFrame(total_test_loss_list).to_csv(os.path.join(args.save_path, f'ner_test_loss_{args.baseline}.csv'), index=False)
