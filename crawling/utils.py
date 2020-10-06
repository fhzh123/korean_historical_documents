import os
from glob import glob

def NER_dict_setting(NER_korean_dict, NER_chinese_dict, event_id):
    NER_korean_dict[event_id] = dict()
    NER_chinese_dict[event_id] = dict()

    NER_korean_dict[event_id]['idx_place'] = list()
    NER_korean_dict[event_id]['idx_book'] = list()
    NER_korean_dict[event_id]['idx_person'] = list()
    NER_korean_dict[event_id]['idx_era'] = list()

    NER_chinese_dict[event_id]['idx_place'] = list()
    NER_chinese_dict[event_id]['idx_book'] = list()
    NER_chinese_dict[event_id]['idx_person'] = list()
    NER_chinese_dict[event_id]['idx_era'] = list()
    return NER_korean_dict, NER_chinese_dict

def NER_append(NER_dict, event_id, attr_class_list):
    if len(attr_class_list) >= 2:
        if attr_class_list[1] == 'idx_place':
            NER_dict[event_id]['idx_place'].append(idx.text)
        if attr_class_list[1] == 'idx_person':
            NER_dict[event_id]['idx_person'].append(idx.text)
        if attr_class_list[1] == 'idx_book':
            NER_dict[event_id]['idx_book'].append(idx.text)
        if attr_class_list[1] == 'idx_era':
            NER_dict[event_id]['idx_era'].append(idx.text)
    return NER_dict

def changeName(path):
    path_list = glob(path)
    for path in path_list:
        new_path = path.replace('╜┬┴ñ┐°└╧▒Γ', '승정원일기')
        new_path = new_path.replace('├Ñ', '책')
        new_path = new_path.replace('┼╗├╩║╗', '탈초본')
        new_path = new_path.replace('│Γ', '년')
        new_path = new_path.replace('┐∙', '월')
        new_path = new_path.replace('┴∩└º', '즉위')
        new_path = new_path.replace('└▒', '윤')
        # King name
        new_path = new_path.replace('╚┐┴╛', '효종')
        new_path = new_path.replace('╟÷┴╛', '현종')
        new_path = new_path.replace('╝≈┴╛', '숙종')
        new_path = new_path.replace('░µ┴╛', '경종')
        new_path = new_path.replace('┐╡┴╢', '영조')
        new_path = new_path.replace('┴ñ┴╢', '정조')
        new_path = new_path.replace('╝°┴╢', '순조')
        new_path = new_path.replace('╟σ┴╛', '헌종')
        new_path = new_path.replace('├╢┴╛', '철종')
        # Replace file name
        os.rename(path, new_path)