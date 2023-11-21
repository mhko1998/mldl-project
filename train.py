import torch
import torch.nn as nn
from dataset import *

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers,
                            dropout = 0.1,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias = True) 
        
    def reset_hidden_state(self): 
        self.hidden = (
                torch.zeros(self.layers, self.seq_len, self.hidden_dim),
                torch.zeros(self.layers, self.seq_len, self.hidden_dim))
    
    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        return x

def train_model(model, train_df, val_df, num_epochs = None, lr = None, verbose = 10, patience = 10):
     
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    nb_epochs = num_epochs
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[70,140,210], gamma=0.1)
    train_hist = np.zeros(nb_epochs)

    for epoch in tqdm(range(nb_epochs)):
        model.train()
        avg_cost = 0
        total_batch = len(train_df)
        
        for batch_idx, samples in tqdm(enumerate(train_df)):

            label, x_train  = samples
            
            x_train = x_train.to(device)
            # seq별 hidden state reset
            y_train = label.to(device)
            # print(x_train.shape)
            model.reset_hidden_state()
            
            # H(x) 계산
            outputs = model(x_train)
            # print(outputs.shape)
            # cost 계산
            loss = criterion(outputs, y_train)                    
            
            # cost로 H(x) 개선
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            avg_cost += loss/total_batch
               
        train_hist[epoch] = avg_cost        
        
        # if epoch % verbose == 0:
        print('Epoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))
            
        # patience번째 마다 early stopping 여부 확인
        if (epoch % patience == 0) & (epoch != 0):
            
            # loss가 커졌다면 early stop
            if train_hist[epoch-patience] < train_hist[epoch]:
                print('\n Early Stopping')
                
                break
        
        torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': loss,
          }, "ceckpe/"+str(epoch)+".pth")
        evaluate(model, val_df)   
    return model.eval(), train_hist

def evaluate(model, val_df) :
    model.eval()
    corrects, total, total_loss = 0, 0, 0
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_idx, samples in tqdm(enumerate(val_df)):
        label, x_train  = samples
            
        x_train = x_train.to(device)
        y_train = label.to(device)
        outputs = model(x_train)
        loss = criterion(outputs, y_train)        
        total += y_train.size(0)
        total_loss += loss.item()
        corrects += (outputs.max(1)[1].view(y_train.size()).data == y_train.data).sum()
        
    avg_loss = total_loss / len(val_df.dataset)
    avg_accuracy = corrects / total
    print(avg_loss, avg_accuracy.item())

if __name__=='__main__':
    
  path = '/home/minhwan/workspace/mldl/train.csv'
  save_path = '/home/minhwan/workspace/mldl/ckp'
  
  device = 'cuda:0'
  
  input_size, output_size, hidden_dim, sequence_length, num_layer = 2, 2, 64, 10, 2
  
  learning_rate = 0.1
  epochs = 1000
  batch_size = 128
  
  train_dataset = mldldataset(path, preprocess=True, flag = "train")
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=24, shuffle=False)

  val_dataset = mldldataset(path, preprocess=True, flag = "val")
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=24, shuffle=False)
  
  net = Net(input_dim = input_size, hidden_dim= hidden_dim, seq_len=sequence_length, output_dim = output_size, layers = num_layer).to(device)  
  
  
  model, train_hist = train_model(net, train_dataloader, val_df=val_dataloader, num_epochs = epochs, lr = learning_rate, verbose = 20, patience = 10)
  

  ## for inference
#   net.load_state_dict(torch.load("/home/minhwan/workspace/mldl/ceckp/0.pth")["model_state_dict"])
  
#   evaluate(net, val_dataloader)   