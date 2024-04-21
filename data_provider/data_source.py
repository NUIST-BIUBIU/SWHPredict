import pandas as pd
import os
from torch.utils.data import Dataset
from utils.timefeatures import time_features


class DatasetWaveHeight(Dataset):
    def __init__(self, model, station_id, size, features, years, flag, data_rootpath):
        self.model = model
        self.station_id = station_id
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.features = features.split(' ')
        self.years = years
        self.flag = flag
        self.data_rootpath = data_rootpath

        self.__read_data__()

    def __read_data__(self):
        root_path = os.path.join(self.data_rootpath, str(self.station_id))
        year_str = self.years.split(' ')
        start_year = int(year_str[0])
        end_year = start_year if len(year_str) == 1 else int(year_str[1])
        columns = ['Timestamp'] + self.features

        raw_data = []
        for year in range(start_year, end_year + 1):
            data_path = '{}h{}.csv'.format(self.station_id,year)
            new_data = pd.read_csv(os.path.join(root_path, data_path), usecols=columns)
            raw_data.append(new_data)
        raw_data = pd.concat(raw_data, axis=0, ignore_index=True)

        cols_data = raw_data.columns[1:]
        feature_data = raw_data[cols_data].values
        swh_data = raw_data['WaveHeight'].values
        swh_data = swh_data.reshape(-1,1)
        time_data = raw_data['Timestamp'].values

        data_stamp = time_features(time_data)

        dataset_len = feature_data.shape[0]
        border1 = 0 if self.flag == 'train' else int(dataset_len*0.8)
        border2 = int(dataset_len*0.8) if self.flag == 'train' else dataset_len

        self.data_x = feature_data[border1:border2]
        self.data_y = swh_data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]

        if self.model in ['Transformer']:
            r_begin = s_end - self.label_len
            seq_y = self.data_y[r_begin:r_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            return seq_x, seq_y, seq_x_mark, seq_y_mark

        if self.model in ['TCN','TCNAttention']:
            seq_y = self.data_y[r_end-1]
            return seq_x.T, seq_y

        if self.model in ['LSTM']:
            seq_y = self.data_y[r_end-1]
            return seq_x, seq_y


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1