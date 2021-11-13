"""
Implementation of Time-Series (multivariate) DeepAID Interpreter
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import prettytable as pt
import sys
import numpy as np
sys.path.append("../deepaid/")
from interpreter import Interpreter
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MultiTimeseriesAID(Interpreter):
    def __init__(self, model, thres, 
                    input_size,            # length of input tabular feature
                    feature_desc=None,     # description of each feature dimension 
                    auto_params=True,      # whether automatic calibration of hyperparamters
                    steps=100,             # epoches of optimizing reference (*within auto_params)
                    lr=1.,                 # learning rate of optimizing reference (*within auto_params)
                    k=5,                   # dimensions in interpretation result (*within auto_params)
                    eps=0.01,              # bound of fidelity loss (*within auto_params)
                    lbd=0.001,             # weight_coefficient of stability term (*within auto_params)
                    verbose=False):        # verbose output during training
    
        super(MultiTimeseriesAID,self).__init__(model)
        self.thres = thres
        self.input_size = input_size 
        if feature_desc is None:
            self.feature_desc = ['dim'+str(i) for i in range(self.input_size)]
        else:
            self.feature_desc = feature_desc
        self.steps = steps
        self.lr = lr
        self.eps = eps
        self.k = k
        self.lbd = lbd
        if auto_params:
            self.auto_calibration()
        print('Successfully Initialize <Multivariate Timeseries Interptreter> for Model <{}>'.format(self.model_name))
        self.verbose = verbose


    def forward(self, anomaly_seq, MIN=-np.Inf, MAX=np.Inf):
        begin_time = time.time()
        self.model.train()

        anomaly_seq = torch.from_numpy(anomaly_seq).type(torch.float).to(self.device)
        w = anomaly_seq[0,-1,:]
        w = w.unsqueeze(0).unsqueeze(0)
        w.requires_grad = True

        seq_feat = anomaly_seq[:,:-1,:]
        feat = anomaly_seq[0,1,:].clone()

        optimizer = optim.SGD([w], lr=self.lr)
        Dist = nn.MSELoss()
        Bound = nn.ReLU()

        Last_Loss = np.Inf
        for epoch in range(self.steps):
            MIN = torch.ones_like(w[0,0]) * MIN
            MAX = torch.ones_like(w[0,0]) * MAX
            bounded_x = torch.min(torch.max(w[0,0],MIN), MAX)
            x = torch.cat([seq_feat, w], dim=1)
            dist = Dist(self.model(x),bounded_x)
            loss1 = Bound(dist-(self.thres**2-self.eps))
            loss2 = torch.norm(bounded_x-feat, p=2)
            L0 = torch.norm(bounded_x-feat, p=0)
            Loss =  loss1 + self.lbd * loss2
            
            reserved_x = w.clone().detach()[0,0]
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            # L0-norm clipping 
            importance = w.grad[0,0] * w.data[0,0]
            clip_index = torch.argsort(importance)
            clip_index = clip_index[:-self.k]
            w.data[0,0][clip_index] = reserved_x[clip_index]

            # if epoch % 10 == 0:
            #     # print('\r epoch:{}'.format(epoch), '|Loss:%.4f'%Loss.item(), 
            #     # '|loss1:%.4f'%loss1.item(),'(dist:{%.4f}'%dist, 'bound:{%.4f})'%(thres**2-eps),
            #     # '|loss2:%.4f'%loss2.item(),'|L0:%.4f'%L0.item(),end='')

            #     print('epoch:{}'.format(epoch), '|Loss:%.4f'%Loss.item(), 
            #     '|loss1:%.4f'%loss1.item(),'(dist:{%.4f}'%dist, 'bound:{%.4f})'%(thres**2-eps),
            #     '|loss2:%.4f'%loss2.item(),'|L0:%.4f'%L0.item())
            
            if Last_Loss - Loss < 0.00001:
                # print('')
                # print("Early stop! epoch is:",epoch)
                break
            else:
                Last_Loss = Loss

        if self.verbose:
            print('STOP! epoch:{}'.format(epoch), '|Loss:%.4f'%Loss.item(), 
                    '|loss1:%.4f'%loss1.item(),'(dist:{%.4f}'%dist, 'bound:{%.4f})'%(self.thres**2-self.eps),
                    '|loss2:%.4f'%loss2.item(),'|L0:%.4f'%L0.item())

        print('Finish Interpretation after {} steps'.format(epoch), '(Final loss: %.2f,'%Loss.item(), "Time elasped: %.2fs)"%(time.time()-begin_time)) 

        feat = feat.cpu()
        refer_feat = w.data.cpu()[0,0] 
        importance = (refer_feat - feat) ** 2
        clip_index = torch.argsort(importance)[-self.k:]
        # print(clip_index)
        
        return {'index_inc':clip_index.numpy(), 'value_inc':refer_feat[clip_index].numpy()} # increase order of importance


    def auto_calibration(self, ): # auto calibration of hyperparamters
        pass
    
    def show_plot(self, anomaly, interpretation, normer):
        print('\nVisualize Interpretation (Plot View)')
        plt.figure()
        x = [i+1 for i in range(self.k)]
        index_inc = interpretation['index_inc']
        value_inc = interpretation['value_inc']
        pseudo_value = np.asarray([0]*len(anomaly))
        pseudo_value[index_inc] = value_inc
        value_inc = normer.restore(pseudo_value)[index_inc]
        anomaly = normer.restore(anomaly)[index_inc]
        reference = value_inc
        plt.plot(x,anomaly,'ro--',label='anomaly')
        plt.plot(x,reference,'g^--',label='reference')
        for xx, y1, y2 in zip(x, anomaly,reference):
            plt.text(xx, y1*0.75, '%.0f' % y1, ha='center', va= 'bottom',fontsize=11, c='r')
            plt.text(xx, y2*1.25, '%.0f' % y2, ha='center', va= 'bottom',fontsize=11, c='g')
        
        feature_desc = np.asarray(self.feature_desc)[index_inc].tolist()
        plt.xticks(x, feature_desc, rotation='80')
        plt.legend()
        plt.subplots_adjust(bottom=0.5)
        plt.show()
    
    def show_heatmap(self, anomaly, interpretation, normer):
        print('\nVisualize Interpretation (HeatMap View)')
        import random

        index_inc = interpretation['index_inc']
        value_inc = interpretation['value_inc']
        pseudo_value = np.asarray([0]*len(anomaly))
        pseudo_value[index_inc] = value_inc
        reference = normer.restore(pseudo_value)[index_inc]
        anomaly_norm = normer.restore(anomaly)[index_inc]

        anomaly = anomaly[index_inc]        
        while np.max(anomaly)>1. or np.max(value_inc)>1.:
            anomaly /= 2.
            value_inc /= 2.
    
        fig1 = plt.figure(figsize=(20,5))
        ax1 = fig1.add_subplot(111, aspect='equal')
        cm = np.linspace(len(index_inc)-1, 0, len(index_inc))/(len(index_inc)*1.25)
        for i in range(self.input_size):
            if i in index_inc:
                ax1.add_patch(
                    patches.Rectangle(
                        (0.1*i, 0),   # (x,y)
                        0.1,          # width
                        0.2,          # height
                        color=cm.astype(str)[np.where(index_inc==i)][0]
                    )
                )
                ax1.add_patch(
                    patches.Rectangle(
                        (0.1*i, 0.2),   # (x,y)
                        0.05,          # width
                        anomaly[np.where(index_inc==i)],          # height
                        color='r'
                    ),
                )
                plt.text(0.1*i, 0.2, '%.0f' % anomaly_norm[np.where(index_inc==i)], ha='center', va= 'bottom', rotation=90, fontsize=11, c='black')
                ax1.add_patch(
                    patches.Rectangle(
                        (0.1*i+0.05, 0.2),   # (x,y)
                        0.05,          # width
                        value_inc[np.where(index_inc==i)],          # height
                        color='g'
                    )
                )
                plt.text(0.1*i+0.1, 0.2, '%.0f' % reference[np.where(index_inc==i)], ha='center', va= 'bottom',rotation=90,fontsize=11, c='black')
                arr_h = random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
                ax1.arrow(0.1*i+0.05, 0., 0.05, -arr_h, head_width=0.1, head_length=0.05, fc='k', ec='k')
                ax1.text(0.1*i+0.1, -arr_h-0.1, self.feature_desc[i], fontsize=12)
            else:
                ax1.add_patch(
                    patches.Rectangle(
                        (0.1*i, 0),   # (x,y)
                        0.1,          # width
                        0.2,          # height
                        color=str(0.9)
                    )
                )
        plt.axis('off')
        plt.ylim(-1,1)
        plt.xlim(0,10)
        plt.bar([-1],[0],color='r',label='anomaly')
        plt.bar([-1],[0],color='g',label='reference')
        plt.bar([-1],[0],color='0.25',label='importance')
        plt.legend()
        plt.show()

    def show_table(self, anomaly, interpretation, normer):
        print('\nVisualize Interpretation (Table View)')

        index_inc = interpretation['index_inc'][::-1]
        value_inc = interpretation['value_inc'][::-1]
        pseudo_value = np.asarray([0]*len(anomaly))
        pseudo_value[index_inc] = value_inc
        value_inc = normer.restore(pseudo_value)[index_inc]
        anomaly = normer.restore(anomaly)[index_inc]
        reference = value_inc
        
        tb = pt.PrettyTable()
        tb.field_names = ["Feature Description", "Value in Anomaly", "comp.", "Value in Reference"]
        for i in range(len(reference)):
            row = [self.feature_desc[index_inc[i]], round(anomaly[i],3), '?', round(reference[i],3)]
            if anomaly[i] > reference[i]: 
                if anomaly[i] > 10*reference[i] and anomaly[i]*reference[i]>0:
                    row[2] = '>>'
                row[2] = '>'
            else:
                if anomaly[i] < 10*reference[i] and anomaly[i]*reference[i]>0:
                    row[2] = '<<'
                row[2] = '<'
            tb.add_row(row)

        print(tb)