# Import Modules
import pandas as pd

class CustomError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

def dataframe_make(king_list, commnet_list, king_index_list):
    dataset_ = pd.DataFrame({
        'session': king_list,
        'text': commnet_list,
        'time': king_index_list
    })
    dataset_['bow'] = dataset_['text'].apply(lambda x: [i for i in x])
    return dataset_