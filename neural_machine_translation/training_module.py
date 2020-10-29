import os
import time
import pandas as pd
import sentencepiece as spm

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .utils import accuracy, CustomError

def model_training(args, model, dataloader_dict, optimizer, scheduler, device):

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

                # Optimizer setting
                optimizer.zero_grad()

                # Sourcen, Target sentence setting
                input_sequences = src.to(device)
                label_sequences = trg.to(device)
                king_id = king_id.to(device)

                non_pad = label_sequences != args.pad_idx
                trg_sequences_target = label_sequences[non_pad].contiguous().view(-1)

                if args.model_setting == 'transformer':
                    # Target Masking
                    tgt_mask = model.generate_square_subsequent_mask(label_sequences.size(1))
                    tgt_mask = tgt_mask.to(device)

                # Model / Calculate loss
                with torch.set_grad_enabled(phase == 'train'):
                    if args.model_setting == 'transformer':
                        predicted = model(input_sequences, label_sequences, king_id, 
                                          tgt_mask=tgt_mask, non_pad_position=non_pad)
                        # predicted = predicted.view(-1, predicted.size(-1))
                        loss = F.cross_entropy(predicted, trg_sequences_target, 
                                               ignore_index=model.pad_idx, reduction='mean')
                        # loss = criterion(predicted, trg_sequences_target)
                    if args.model_setting == 'rnn':
                        teacher_forcing_ratio_ = 0.5
                        input_sequences = input_sequences.transpose(0, 1)
                        label_sequences = label_sequences.transpose(0, 1)
                        predicted = model(input_sequences, label_sequences, king_id, 
                                        teacher_forcing_ratio=teacher_forcing_ratio_)
                        predicted = predicted.view(-1, trg_vocab_num)
                        trg_sequences_target = label_sequences.contiguous().view(-1)
                        loss = criterion(predicted, trg_sequences_target)

                    # If phase train, then backward loss and step optimizer and scheduler
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        clip_grad_norm_(model.parameters(), args.grad_clip)
                        total_train_loss_list.append(loss.item())
                        # Print loss value only training
                        if freq == args.print_freq or i == 0 or i == len(dataloader_dict['train']):
                            total_loss = loss.item()
                            top1_acc, top5_acc, top10_acc = accuracy(predicted, 
                                                                    trg_sequences_target, 
                                                                    topk=(1,5,10))
                            print("[Epoch:%d][%d/%d] train_loss:%5.3f | top1_acc:%5.2f | top5_acc:%5.2f | spend_time:%5.2fmin"
                                    % (e+1, i, len(dataloader_dict['train']), total_loss, top1_acc, top5_acc, (time.time() - start_time_e) / 60))
                            freq = 0
                        freq += 1
                    if phase == 'valid':
                        val_loss += loss.item()
                        top1_acc, top5_acc, top10_acc = accuracy(predicted, 
                                                                 trg_sequences_target, 
                                                                 topk=(1,5,10))
                        val_top1_acc += top1_acc.item()
                        val_top5_acc += top5_acc.item()
                        val_top10_acc += top10_acc.item()

            # Finishing iteration
            if phase == 'valid':
                val_loss /= len(dataloader_dict['valid'])
                val_top1_acc /= len(dataloader_dict['valid'])
                val_top5_acc /= len(dataloader_dict['valid'])
                val_top10_acc /= len(dataloader_dict['valid'])
                total_test_loss_list.append(val_loss)
                print("[Epoch:%d] val_loss:%5.3f | top1_acc:%5.2f | top5_acc:%5.2f | top10_acc:%5.2f | spend_time:%5.2fmin"
                        % (e+1, val_loss, val_top1_acc, val_top5_acc, val_top10_acc, (time.time() - start_time_e) / 60))
                if not best_val_loss or val_loss < best_val_loss:
                    print("[!] saving model...")
                    if not os.path.exists(args.save_path):
                        os.mkdir(args.save_path)
                    torch.save(model.state_dict(), 
                               os.path.join(args.save_path, f'nmt_model_{args.model_setting}_testing2.pt'))
                    best_val_loss = val_loss

        # Learning rate scheduler setting
        # scheduler.step()

    pd.DataFrame(total_train_loss_list).to_csv(os.path.join(args.save_path, f'train_loss_{args.src_baseline}_{args.trg_baseline}_{args.model_setting}2.csv'), index=False)
    pd.DataFrame(total_test_loss_list).to_csv(os.path.join(args.save_path, f'test_loss_{args.src_baseline}_{args.trg_baseline}_{args.model_setting}2.csv'), index=False)

def sentencepiece_training(lang, split_record, args):
    # 0) Pre-setting
    if lang == 'korean':
        vocab_size = args.kr_vocab_size
    elif lang == 'hanja':
        vocab_size = args.hj_vocab_size
    else:
        raise CustomError('Sorry; Language not supported')

    # 1) Make Korean text to train vocab
    with open(f'{args.save_path}/{lang}.txt', 'w') as f:
        for text in split_record['train']:
            f.write(f'{text}\n')

    # 2) SentencePiece model training
    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.save_path}/{lang}.txt --model_prefix={args.save_path}/m_{lang} --model_type=bpe '
        f'--vocab_size={vocab_size} --character_coverage=0.995 --split_by_whitespace=true '
        f'--pad_id={args.pad_idx} --unk_id={args.unk_idx} --bos_id={args.bos_idx} --eos_id={args.eos_idx}')

    # 3) Korean vocabulary setting
    lang_vocab = list()
    with open(f'{args.save_path}/m_{lang}.vocab') as f:
        for line in f:
            lang_vocab.append(line[:-1].split('\t')[0])
    lang_word2id = {w: i for i, w in enumerate(lang_vocab)}

    # 4) SentencePiece model load
    spm_ = spm.SentencePieceProcessor()
    spm_.Load(f"{args.save_path}/m_{lang}.model")

    # 5) Korean parsing by SentencePiece model
    train_lang_indices = [[args.bos_idx] + spm_.EncodeAsIds(korean) + [args.eos_idx] for korean in split_record['train']]
    valid_lang_indices = [[args.bos_idx] + spm_.EncodeAsIds(korean) + [args.eos_idx] for korean in split_record['valid']]
    test_lang_indices = [[args.bos_idx] + spm_.EncodeAsIds(korean) + [args.eos_idx] for korean in split_record['test']]

    # 6) Return
    return (train_lang_indices, valid_lang_indices, test_lang_indices), lang_word2id