import csv
import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import *
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import math
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import ticker
from models import  CNN, RNN, LSTM, GRU, Segrnn,GCN
from tools import visual, metric,EarlyStopping
import os

# gru_model = GRU(input_size=144, hidden_size=144, num_layers=1).cuda()
# cnn_model = CNN(input_channels=1, output_size=144)
# ts_model = Transformer(feature_size=144, num_layers=4, output_size=144)
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return 0.5 * F.mse_loss(x, y) + 0.5 * F.l1_loss(x, y)
def col_norm(data, col_max=None):
        if col_max is None:
            col_max = np.max(data, axis=0)
            # col_min = np.min(data,axis=0)
        data = data / col_max
        data[np.isnan(data)] = 0.0
        data[np.isinf(data)] = 0.0
        return data
class PridictTM():
    def __init__(self,  Seq, Epoch, LR, model_name):
        self.Seq = Seq
        self.epoch = Epoch
        # self.hidden_size = hidden_size
        self.LR = LR
        # self.input_size = input_size
        self.model = Model.cuda()
        self.model_name = model_name
        self.shuff_flag=True
        # print(self.model)



    def vali(self,  vali_loader):
        total_loss = []
        criterion = CustomLoss() 
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                outputs = self.model(batch_x)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss



    def read_data(self, file_name):
        df = pd.read_csv(file_name)
        del df["time"]
        # data_list = np.array(df["data"])
        data_list = df.values
        max_data = np.max(data_list, axis=0)
        data_list = col_norm(data_list,max_data)



        # max_list = np.max(data_list, axis=0)
        # min_list = np.min(data_list, axis=0)

        # data_list = (data_list - min_list) / (max_list - min_list)
        # data_list[np.isnan(data_list)] = 0
        return data_list
        # return data_list, min_list, max_list

    def generate_series(self, data, Seq,pred ):
        x_data = []
        y_data = []
        length = data.shape[0]
        # length = len(data)#od
        # self.stand_scaler = MinMaxScaler()
        # data = self.stand_scaler.fit_transform(data.reshape(-1,1))
        
        print(data.shape)
        for i in range(length - Seq-pred):
            x = data[i:i+Seq]
            y = data[i+Seq:i+Seq+pred]
            x_data.append(x)
            y_data.append(y)
        # self.scaler_x = StandardScaler()
        # self.scaler_x.fit(x_data)
        # x_data = self.scaler_x.transform(x_data)
        # self.scaler_y = StandardScaler()
        # self.scaler_y.fit(y_data)
        # y_data = self.scaler_y.transform(y_data)
       
        
        x_data = torch.from_numpy(np.array(x_data)).float()
        y_data = torch.from_numpy(np.array(y_data)).float()
    
        print(x_data.shape)#torch.Size([19312, 480])
        print(y_data.shape)#torch.Size([19312, 96])

        
        # x_data=x_data.unsqueeze(2)
        # y_data=y_data.unsqueeze(2)#segrnn


        # print(x_data.shape)torch.Size([19312, 480, 1])
        # print(y_data.shape)torch.Size([19312, 96, 1])
        # view(-1, 1)

        return x_data, y_data
    
    def inverse_standardization(self, scaled_data, scaler):
        inverse_data = scaler.inverse_transform(scaled_data)
        return inverse_data

    def generate_batch_loader(self, x_data, y_data):
        torch_dataset = Data.TensorDataset(x_data, y_data)
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=BATCH_SIZE,
            shuffle=self.shuff_flag,
            num_workers=2,
            drop_last=True
        )
        return loader

    def inverse_normalization(self, prediction, y, max_list, min_list):
        inverse_prediction = prediction * (max_list - min_list) + min_list
        inverse_y = y * (max_list - min_list) + min_list

        return inverse_prediction, inverse_y

    def save_TM(self, TM, file_name):
        f = open(file_name, 'w')
        row, column = TM.shape
        for i in range(row):
            for j in range(column):
                if not TM[i][j] == 0.0:
                    temp = str(i + 1) + ' ' + str(j + 1) + ' ' + str(TM[i][j]) + "\n"
                    f.write(temp)
        f.close()

    def train(self, x_data, y_data, train_len,val_len): # Set model to training mode

        self.shuff_flag=True
        print(self.shuff_flag)
        train_data_loader = self.generate_batch_loader(x_data[:train_len], y_data[:train_len])
        val_data_loader=self.generate_batch_loader(x_data[train_len:train_len+val_len],y_data[train_len:train_len+val_len])
        optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.LR)
        loss_func = nn.MSELoss()
        model_name = f"{self.model_name}-{self.Seq}.pkl"
        # Loss=[]
        star_time = time.perf_counter()
        epoch_time = time.time()
        for epoch in range(self.epoch):
            #print(epoch)
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            #print(len(train_data_loader))
            for steps, (batch_x, batch_y) in enumerate(train_data_loader):
                
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                # print('-----------')
                # print(type(batch_x))
                #print('input-shape',batch_x.shape)#([256, 288, 1])
                # print('y',batch_y.shape)([256, 48, 1])
                prediction = self.model(batch_x).cuda()
                #print('pre-shape',prediction.shape)
                # result.append(prediction[-1].cpu().data.numpy())

                loss = loss_func(prediction, batch_y)
                train_loss.append(loss.item())
                # Loss.append(loss.cpu().detach().numpy())
                # Loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #     if (step + 1) % 100 == 0:
            #         print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(step + 1, epoch + 1, loss.item()))
            #         speed = (time.time() - time_now) / iter_count
            #         left_time = speed * ((Epoch - epoch) * Epoch - step)
            #         print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            #         iter_count = 0
            #         time_now = time.time()


            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # train_loss = np.average(train_loss)
            vali_loss = self.vali(val_data_loader)
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
            #     epoch + 1, Epoch, train_loss, vali_loss))

            
            early_stopping=EarlyStopping(vali_loss,self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # print('end')
        end_time = time.perf_counter()
        train_cost=end_time-star_time
        print('total cost time:',train_cost)

        torch.save(self.model.state_dict(), model_name)
        return train_cost
 
        
        # 创建线图
        # plt.plot(Loss)
        #
        # # 设置图表标题和轴标签
        # plt.title('loss')
        # plt.xlabel('epoch')
        # plt.ylabel('value')
        #
        # # # 显示图表
        # plt.show()

    def test(self, x_data, y_data, train_len,test_len,train_cost):
        self.shuff_flag=False
        print(self.shuff_flag)
        # y_tru = []
        # y_pre = []
        # # print(len(x_data), train_len)
        # for i in range(train_len, len(x_data)):
        #     y_tru.append([])
        #     y_pre.append([])
        #   # Set model to evaluation mode
        self.model.load_state_dict(torch.load(f"{self.model_name}-{self.Seq}.pkl"))
        self.model.eval()
        preds = []
        trues = []
        inputx = []
        count = 0
        star_time = time.perf_counter()
        folder_path = './results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        test_data_loader = self.generate_batch_loader(x_data[train_len+test_len:train_len+test_len*2], y_data[train_len+test_len:train_len+test_len*2])
        with torch.no_grad():  # Disable gradient computation for efficiency
            for step, (test_x, test_y) in enumerate(test_data_loader):
                # print('-----------')
                test_x = test_x.cuda()
                test_y = test_y.cuda()
                prediction = self.model(test_x).cuda()
                # print('pred',prediction.shape)


                # print('input-shape',prediction.shape)([256, 96, 1])

                outputs = prediction.detach().cpu().numpy()
                batch_y = test_y.detach().cpu().numpy()

                pred = outputs.squeeze()  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y.squeeze()  # batch_y.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)
                trues.append(true)

        # print(trues[0])
        end_time = time.perf_counter()


        preds = np.array(preds)
        trues = np.array(trues)
        # print(preds.shape, trues.shape)#(11, 256, 96, 1)
        predsplt=preds.reshape(-1)
        trueplt=trues.reshape(-1)
        # plt.figure(figsize=(10, 6))
        # plt.plot(predsplt[:200], label='pred', alpha=0.7)
        # plt.plot(trueplt[:200], label='true', alpha=0.7)
        # # 添加标题和标签
        # plt.title('pred&true')
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # 添加图例
        # plt.legend()
        # plt.show()

        # 显示图形
        #plt.savefig('epoch=100&lr=0.01.png')



        #print(preds.shape)#(11, 256, 96)
        # inputx = np.array(inputx)
        #print(preds.shape[-1], preds.shape[-2])

        # preds=self.stand_scaler.inverse_transform(preds.reshape(-1,1))
        # trues=self.stand_scaler.inverse_transform(trues.reshape(-1,1))

        preds = preds.reshape(-1, preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-1])


        

        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{},detail:{}'.format(mse, mae, rse,self.model_name))
        f = open("results_tm_segrnn_abilene.txt", 'a')
        f.write('detail:{},seq:{},pred:{},mape:{},mse:{},mae:{},rse:{},rmse:{},mspe:{},corr:{},train_time:{},test_time:{}'.format(self.model_name,Seq,Pred,mape,mse, mae, rse,rmse,mspe,corr,train_cost,end_time-star_time))
        f.write('\n')
        f.write('\n')
        f.close()
        print(end_time - star_time)

        return



if __name__ == "__main__":
    # file_name = "./OD_pair/Abilene-OD_pair_2004-08-01.csv"
    # file_name = "D:\pythongc\Traffic-Matrix-Prediction-main\OD_pair\Abilene-OD_pair_2004-08-01.csv"
    # file_name='/home/server/yixin/TMP/OD_pair/ec_data.csv'

    # seq_values = [288]
    # pred_values = [48, 96, 144,192,240,288]
    # for Seq in seq_values:
    #     for Pred in pred_values:
    #         # Rest of your code remains unchanged
    #         # file_name = '/home/server/yixin/TMP/OD_pair/traffic_one_cell.csv'
    #         file_name='/home/server/yixin/TMP/OD_pair/ec_data.csv'
    #         BATCH_SIZE = 256
    #         Epoch = 20
    #         patch = 24

    #         segrnn = Segrnn(seq_len=Seq, pred_len=Pred, enc_in=1, patch_len=patch, d_model=512).cuda()
    #         # ... (rest of your model initialization code)

    #         LR = 0.01
    #         Model = segrnn
    #         train_cost = 0
    #         model_n = 'segrnn'
    #         dataset = 'ukdata'

    #         model_name = f'{model_n}-seq{Seq}-pred{Pred}-patch{patch}-epoch{Epoch}-lr{LR}_{dataset}'

    #         predict_tm_model = PridictTM(Seq, Epoch, LR, model_name)
    #         data = predict_tm_model.read_data(file_name)
    #         x_data, y_data = predict_tm_model.generate_series(data, Seq, Pred)

    #         train_len = int(len(x_data) * 0.7)
    #         test_len = int(len(x_data) * 0.15)
    #         val_len = int(len(x_data) * 0.15)
    #         print(f"Training length for Seq={Seq}, Pred={Pred}: {train_len}")

    #         train_cost = predict_tm_model.train(x_data, y_data, train_len, val_len)
    #         predict_tm_model.test(x_data, y_data, train_len, test_len, train_cost)
    
    
    # file_name='/home/server/yixin/TMP/OD_pair/ec_data.csv'
    file_name='/home/server/yixin/TMP/OD_pair/Abilene-OD_pair.csv'
    Seq = 12
    BATCH_SIZE = 64
    Pred=4
    Epoch = 20
    patch=2


    segrnn = Segrnn(seq_len=Seq,pred_len=Pred,enc_in=144,patch_len=patch,d_model=16).cuda()
    gru = GRU(input_size=1, hidden_size=Pred,output_size=Pred, num_layers=2).cuda()
    rnn= RNN(input_size=1, hidden_size=Pred, output_size=Pred,num_layers=2).cuda()
    lstm= LSTM(input_size=1, hidden_size=Pred, output_size=Pred,num_layers=2).cuda()
    
    #mlp= MLP(input_size=144, layer1_size=64, layer2_size=16, output_size=144, sequence_length=10).cuda()

    LR=0.01
    Model = segrnn
    train_cost=0
    model_n='segrnn'
    dataset = 'abilene'

    model_name = f'{model_n}-seq{Seq}-pred{Pred}-patch{patch}-epoch{Epoch}-lr{LR}_{dataset}'


    # model_name = 'lstm-seq48-pred48-patch48-epoch20-lr0.01_ukdata'
    predict_tm_model = PridictTM( Seq, Epoch, LR, model_name)
    # data, min_list, max_list = predict_tm_model.read_data(file_name)
    data = predict_tm_model.read_data(file_name)
    x_data, y_data = predict_tm_model.generate_series(data, Seq,Pred)

    # print(x_data.shape)
    # print(y_data.shape)
    #train_len = int(int(len(x_data) * 0.8) / 50) * 50
    train_len = int(len(x_data) * 0.7)
    test_len=int(len(x_data)*0.15)
    val_len=int(len(x_data)*0.15)
    print("Training length: ", train_len)

    train_cost=predict_tm_model.train(x_data, y_data, train_len,val_len)
    predict_tm_model.test(x_data, y_data, train_len,test_len,train_cost)
