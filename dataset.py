import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

class mldldataset(Dataset):
    def __init__(self, path, preprocess, flag):
        self.path = path
        
        if not preprocess:
            train = pd.read_parquet(self.path)
            event = pd.read_csv('/home/minhwan/workspace/mldl/train_events.csv')
            
            arr = [-1]
            for i in range(int(len(train["series_id"])/20)-1):
                if train["series_id"][i*20+19] != train["series_id"][(i+1)*20]:
                    arr.append(i)
            arr.append(i+1)
            
            ex = np.array([]).astype(int)
            for idx in range(1, len(arr)):
                ex = np.append(ex, 
                            np.arange((arr[idx]-arr[idx-1])*20//17280*17280+(arr[idx-1]+1)*20,
                                        (arr[idx]+1)*20)
                            )
            
            train = train.drop(ex)
            train = train.reset_index(drop=True)
            
            del arr
            del ex

            event = event.dropna(axis=0, how="any")
            event = event.reset_index(drop=True)
            
            event_drop_arr = []

            for idx, data in enumerate(event["event"]):
                if idx == 0:
                    continue
                else:
                    if event["event"][idx-1] == event["event"][idx]:
                        if event["event"][idx-1] == "onset":
                            event_drop_arr.append(idx-1)
                        else:
                            event_drop_arr.append(idx)
                            
            event = event.drop(event_drop_arr)
            event = event.reset_index(drop=True)

            del event_drop_arr


            event2 = event.iloc[1::2]
            train2 = train.iloc[17279::17280]
            event2['t'] = pd.to_datetime(event2["timestamp"], utc=True).dt.strftime("%Y%m%d")
            train2["t"] = pd.to_datetime(train2["timestamp"], utc=True).dt.strftime("%Y%m%d")
            
            i = 0
            j = 0
            csv_list = []
            arr_list = []
            while i < len(np.array(train2["t"])):
                if np.array(train2["series_id"])[i] != np.array(event2["series_id"])[j] and np.array(train2["series_id"])[i] != np.array(train2["series_id"])[i-1]:
                    csv_list.append(j)
                    j+=1
                if np.array(train2["t"])[i] == np.array(event2["t"])[j]:
                    i+=1
                    j+=1
                else:
                    arr_list.append(i)
                    i+=1
            
            del train2
            del event2

            farr_list = np.array([]).astype(int)
            for i in arr_list:
                farr_list = np.append(farr_list,np.arange(i*17280,(i+1)*17280))  
                
            fcsv_list = np.array([]).astype(int)
            for i in csv_list:
                fcsv_list = np.append(fcsv_list, np.arange(i*2, (i+1)*2))
                
            self.train = train.drop(farr_list)
            self.event = event.drop(fcsv_list)
            
            self.train = self.train.reset_index(drop=True)
            self.event = self.event.reset_index(drop=True)
            self.event["step"] = self.event["step"].astype(int)

            del arr_list
            del csv_list
            del farr_list
            del fcsv_list

            label_list = np.array([]).astype(int)

            for i in tqdm(range(0,len(self.event["step"]),2)):
                label_list = np.append(label_list, np.zeros((self.event["step"][i])%17280))
                if i >1:
                    if self.event["step"][i-2]//17280 != self.event["step"][i-1]//17280:
                        label_list = label_list[:len(label_list)-17280]
                label_list = np.append(label_list, np.ones(self.event["step"][i+1]-self.event["step"][i]))
                label_list = np.append(label_list, np.zeros(17280-(self.event["step"][i+1])%17280))
            self.train["label"] = label_list
            del label_list
            self.train.to_csv("train.csv")

        else:
            self.train = pd.read_csv(self.path)
            # self.train = self.train[:1000]  
            if flag =="train":
                self.train = self.train[:int(len(self.train["label"])*0.8)]
            else:
                self.train = self.train[int(len(self.train["label"])*0.8):]
                self.train = self.train.reset_index(drop=True)
            

    def __len__(self):
        return len(self.train["timestamp"]) - 10
    
    def __getitem__(self, idx):
        labels = torch.tensor(self.train["label"][idx+10], dtype = torch.int64)
        an = torch.from_numpy(self.train["anglez"][idx:idx + 10].values.astype(np.float32))
        en = torch.from_numpy(self.train["enmo"][idx:idx + 10].values.astype(np.float32))
        # time = torch.from_numpy(int(round(self.train["timestamp"][idx:idx + 10])).values.astype(np.int64))
        inputs = torch.cat([an, en])
        inputs = inputs.reshape([10,2])
        return labels, inputs
        
if __name__=='__main__':
    path = '/home/minhwan/workspace/mldl/train.csv'
    train_dataset = mldldataset(path, preprocess=True, flag="train")
    print(train_dataset.train)
    print(train_dataset.train[5000])
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False)