import os
import time
import pandas as pd

import torch

from .utils import accuracy

def train_model(args, model, dataloader_dict, optimizer, criterion, scheduler, device):

    best_val_loss = None
    total_train_loss_list = list()
    total_test_loss_list = list()
    freq = 0

    for e in range(args.num_epoch):
        start_time_e = time.time()
        print(f'Model Fitting: [{e+1}/{args.num_epoch}]')
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                model.eval()
                val_loss = 0
                val_top1_acc = 0
                val_top5_acc = 0
                val_top10_acc = 0
            for i, (src, trg, king_id) in enumerate(dataloader_dict[phase]):
                # Sourcen, Target sentence setting
                label_sequences = trg.to(device, non_blocking=True)
                input_sequences = src.to(device, non_blocking=True)
                king_id = king_id.to(device, non_blocking=True)

                non_pad = label_sequences != args.pad_idx
                trg_sequences_target = label_sequences[non_pad].contiguous().view(-1)

                if args.model_setting == 'transformer':
                    # Target Masking
                    tgt_mask = model.generate_square_subsequent_mask(label_sequences.size(1))
                    tgt_mask = tgt_mask.to(device, non_blocking=True)
                    tgt_mask = tgt_mask.transpose(0, 1)

                # Optimizer setting
                optimizer.zero_grad()

                # Model / Calculate loss
                with torch.set_grad_enabled(phase == 'train'):
                    if args.model_setting == 'transformer':
                        predicted = model(input_sequences, label_sequences, king_id, tgt_mask, non_pad)
                        loss = criterion(predicted, trg_sequences_target)
                    if args.model_setting == 'rnn':
                        teacher_forcing_ratio_ = 0.5
                        input_sequences = input_sequences.transpose(0, 1)
                        label_sequences = label_sequences.transpose(0, 1)
                        predicted = model(input_sequences, label_sequences, king_id, 
                                        teacher_forcing_ratio=teacher_forcing_ratio_)
                        predicted = predicted.view(-1, trg_vocab_num)
                        trg_sequences_target = label_sequences.contiguous().view(-1)
                        loss = criterion(predicted, trg_sequences_target)
                    if phase == 'valid':
                        val_loss += loss.item()
                        top1_acc, top5_acc, top10_acc = accuracy(predicted, 
                                                                 trg_sequences_target, 
                                                                 topk=(1,5,10))
                        val_top1_acc += top1_acc.item()
                        val_top5_acc += top5_acc.item()
                        val_top10_acc += top10_acc.item()
                # If phase train, then backward loss and step optimizer and scheduler
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    total_train_loss_list.append(loss.item())

                    # Print loss value only training
                    if freq == args.print_freq:
                        total_loss = loss.item()
                        top1_acc, top5_acc, top10_acc = accuracy(predicted, 
                                                                 trg_sequences_target, 
                                                                 topk=(1,5,10))
                        print("[Epoch:%d][%d/%d] train_loss:%5.3f | top1_acc:%5.2f | top5_acc:%5.2f | spend_time:%5.2fmin"
                                % (e+1, i, len(dataloader_dict['train']), total_loss, top1_acc, top5_acc, (time.time() - start_time_e) / 60))
                        freq = 0
                    freq += 1

            # Finishing iteration
            if phase == 'valid':
                val_loss /= len(dataloader_dict['valid'])
                val_top1_acc /= len(dataloader_dict['valid'])
                val_top5_acc /= len(dataloader_dict['valid'])
                val_top10_acc /= len(dataloader_dict['valid'])
                total_test_loss_list.append(val_loss)
                print("[Epoch:%d] val_loss:%5.3f | top1_acc:%5.2f | top5_acc:%5.2f | top10_acc:%5.2f | spend_time:%5.2fmin"
                        % (e+1, val_loss, val_top1_acc, val_top5_acc, val_top10_acc, (time.time() - start_time_e) / 60))
                if not best_val_loss or val_loss > best_val_loss:
                    print("[!] saving model...")
                    if not os.path.exists(args.save_path):
                        os.mkdir(args.save_path)
                    torch.save(model.state_dict(), 
                               os.path.join(args.save_path, f'nmt_model_{args.model_setting}_{args.resume}_{args.baseline}_testing2.pt'))
                    best_val_loss = val_loss

        # Learning rate scheduler setting
        # scheduler.step()

    pd.DataFrame(total_train_loss_list).to_csv(os.path.join(args.save_path, f'train_loss_{args.baseline}_{args.model_setting}_{args.resume}.csv'), index=False)
    pd.DataFrame(total_test_loss_list).to_csv(os.path.join(args.save_path, f'test_loss_{args.baseline}_{args.model_setting}_{args.resume}.csv'), index=False)