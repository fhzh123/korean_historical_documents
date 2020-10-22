import fcntl
import struct
import random
import termios
import numpy as np
from tqdm import tqdm

def train_test_split(record_list1, record_list2, record_list3, split_percent=0.2):
    assert len(record_list1) == len(record_list2) # Check paired data
    assert len(record_list1) == len(record_list3) # Check paired data
    assert len(record_list2) == len(record_list3) # Check paired data

    # Paired data split
    valid_ratio = split_percent / 2
    test_ratio = split_percent / 2
    paired_data_len = len(record_list1)
    valid_num = int(paired_data_len * valid_ratio)
    test_num = int(paired_data_len * test_ratio)
    
    test_index = np.random.choice(paired_data_len, test_num, replace=False)
    valid_index = np.random.choice(list(set(range(paired_data_len)) - set(test_index)), 
                                   valid_num, replace=False)
    train_index = list(set(range(paired_data_len)) - set(valid_index) - set(test_index))
    random.shuffle(train_index)

    train_record_list1 = [record_list1[i] for i in train_index]
    train_record_list2 = [record_list2[i] for i in train_index]
    train_record_list3 = [record_list3[i] for i in train_index]
    valid_record_list1 = [record_list1[i] for i in valid_index]
    valid_record_list2 = [record_list2[i] for i in valid_index]
    valid_record_list3 = [record_list3[i] for i in valid_index]
    test_record_list1 = [record_list1[i] for i in test_index]
    test_record_list2 = [record_list2[i] for i in test_index]
    test_record_list3 = [record_list3[i] for i in test_index]

    split_record1 = {'train': train_record_list1, 
                     'valid': valid_record_list1,
                     'test': test_record_list1}
    split_record2 = {'train': train_record_list2, 
                     'valid': valid_record_list2,
                     'test': test_record_list2}
    split_record3 = {'train': train_record_list3, 
                     'valid': valid_record_list3,
                     'test': test_record_list3}

    return split_record1, split_record2, split_record3

def terminal_size():
    th, tw, hp, wp = struct.unpack('HHHH',
        fcntl.ioctl(0, termios.TIOCGWINSZ,
        struct.pack('HHHH', 0, 0, 0, 0)))
    return tw

def hj_encode_to_ids(records_, word2id, args):
    # Setting
    parsed_indices = list()
    # Loop
    for index in tqdm(records_):
        parsed_index = list()
        parsed_index.append(word2id['<s>']) # Start token add
        for ind in index:
            try:
                parsed_index.append(word2id[ind])
            except KeyError:
                parsed_index.append(word2id['<unk>'])
        parsed_index.append(word2id['</s>']) # End token add
        parsed_indices.append(parsed_index)
    return parsed_indices

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res