import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings(action='ignore')


class mldldataset(Dataset):
    def __init__(self, path, do_preprocess, flag, sequence_length=10):
        self.path = path
        self.length = 17271
        self.sl = sequence_length
        
        if do_preprocess:
            train = pd.read_parquet(self.path)
            event = pd.read_csv('./train_events.csv')


            for i in tqdm(range(len(event['night']))):
                if event['event'][i] == 'onset':
                    id = event['series_id'][i]
                    onset_step = event['step'][i]
                    wakeup_step = event['step'][i+1]
                    night = event['night'][i]

                    if self._check_existence(onset_step, wakeup_step):
                        one_day = train.iloc[(night-1)*17280:night*17280]
                        if self._check_consistency(one_day):
                            one_day["label"] = 0
                            one_day["label"].iloc[int(onset_step):int(wakeup_step)] = 1
                            table = pa.Table.from_pandas(one_day)
                            if i == 0:
                                pqwriter = pq.ParquetWriter('./train.parquet', table.schema)
                            pqwriter.write_table(table)
                            # pq.write_to_dataset(table, root_path='./train.parquet', )

                            # one_day.to_parquet('./train.parquet', index=False, append=True)
                        else:
                            # print('Inconsistent data')
                            continue
                    else:
                        # print('No onset or wakeup step')
                        continue
            del train, event, one_day
            pqwriter.close()
        
        self.dataset = pd.read_parquet('./train.parquet')
        if flag == "train":
            self.dataset = self.dataset.iloc[:int(len(self.dataset)*0.8)]
        elif flag == "val":
            self.dataset = self.dataset.iloc[int(len(self.dataset)*0.8):]
            
    def _check_existence(self, onset_step, wakeup_step):
        return not np.isnan(onset_step) and not np.isnan(wakeup_step)
    
    def _check_consistency(self, one_day):
        return len(one_day["series_id"].unique()) == 1

    def __len__(self):
        return len(self.dataset) - self.sl + 1
    
    def __getitem__(self, idx):
        ## separated files
        # file_idx = idx // self.length
        # data_idx = idx % self.length

        # file = pd.read_csv('./dataset/{}'.format(self.dataset[file_idx]))
        # data = file.iloc[data_idx:data_idx+self.sl]
        # labels = torch.tensor(data["label"].iloc[-1].astype(np.int64))
        # an = torch.from_numpy(data["anglez"].values.astype(np.float32))
        # en = torch.from_numpy(data["enmo"].values.astype(np.float32))
        # inputs = torch.cat([an.unsqueeze(1), en.unsqueeze(1)], dim=1)
        # return labels, inputs

        ## one file
        labels = torch.tensor(self.dataset["label"].iloc[idx+self.sl-1].astype(np.int64))
        an = torch.from_numpy(self.dataset["anglez"].iloc[idx:idx+self.sl].values.astype(np.float32))
        en = torch.from_numpy(self.dataset["enmo"].iloc[idx:idx+self.sl].values.astype(np.float32))
        inputs = torch.cat([an.unsqueeze(1), en.unsqueeze(1)], dim=1)
        return labels, inputs
                              
        pass
        
if __name__=='__main__':
    path = './train_series.parquet'
    train_dataset = mldldataset(path, do_preprocess=False, flag="train")
    train_dataset.__getitem__(0)
    import pdb; pdb.set_trace()