import os.path

import ir_datasets
import pandas as pd


class DataSet:
    def __init__(self, name):
        print(f'loading data set {name}..')
        self.name = name
        self.file_name = name.split('/')[0]
        self.data = ir_datasets.load(name)
        self.data_frame = self.get_data_frame()
        print(f'data set loaded successfully')

    def get_data_frame(self):
        path = f'data/{self.file_name}.csv'
        if os.path.exists(path):
            data_frame = pd.read_csv(path)
        else:
            data_frame = pd.DataFrame(self.data.docs_iter(), columns=['id', 'doc'])
            data_frame.to_csv(path)
        return data_frame
