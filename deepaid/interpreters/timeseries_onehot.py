"""
Implementation of Time-Series (univariate) DeepAID Interpreter
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import prettytable as pt
import sys
sys.path.append("../deepaid/")
from interpreter import Interpreter

class UniTimeseriesAID(Interpreter):
    
    def __init__(self, model, 
                 feature_desc=None,     # description of each dimension of one-hot vectors
                 auto_params=True,      # whether automatic calibration of hyperparamters
                 class_num = None,      # dimension of one-hot vector
                 steps=20,              # epoches of optimizing reference (*within auto_params)
                 lr=0.5,                # learning rate of optimizing reference (*within auto_params)
                 k=1,                   # dimensions in interpretation result (*within auto_params)
                 bound_thres=None,      # bound of fidelity loss (*within auto_params)
                 lbd=0.001,             # weight_coefficient of stability term (*within auto_params)
                 pos_thres = 0.4,       # threshold for saliency test
                 grad_thres = 0.01,     # threshold for saliency test 
                 num_candidates=9,      # number of candidate in detection model
                 verbose=False):        # verbose output during training
        super(UniTimeseriesAID,self).__init__(model)
        if feature_desc is None:
            self.feature_desc = ['dim'+str(i) for i in range(class_num)]
        else:
            self.feature_desc = feature_desc
        self.steps = steps
        self.lr = lr
        self.bound_thres = bound_thres
        self.pos_thres = pos_thres
        self.grad_thres = grad_thres
        self.k = k
        self.lbd = lbd
        if auto_params:
            self.auto_calibration(num_candidates=num_candidates)
        
        print('Successfully Initialize <Univariate Timeseries Interptreter> for Model <{}>'.format(self.model_name))
        self.verbose = verbose
        

    def forward(self, anomaly_seq, label, window_size=10):
        self.model.train()
        max_dim = self.k 
        begin_time = time.time()

        w = anomaly_seq.unsqueeze(0).clone().detach()
        w.requires_grad = True

        optimizer = optim.SGD([w], lr=self.lr)
        Bound = nn.ReLU()
        Logit = nn.Softmax()

        out = self.model(w)[0]
        loss_determinism = torch.norm(w-anomaly_seq, p='fro')
        loss_accuracy = Bound(self.bound_thres-Logit(out)[label[0]])
        Loss = loss_accuracy + self.lbd * loss_determinism
        optimizer.zero_grad()
        Loss.backward()

        seq_id = torch.argmax(anomaly_seq,dim=1).cpu().numpy() # reverse one-hot to keyno sequence

        ## interp res list 
        IDX1 = []
        IDX2 = []
        REF = []

        ## IF interp POS at Xt
        if torch.max(Logit(out)).cpu().data > self.pos_thres:
            if torch.max(w.grad).cpu().numpy()<self.grad_thres:
                IDX1.append(window_size)
                IDX2.append(torch.argmax(Logit(out)).cpu().data.numpy().tolist())
                REF.append(1.)

                IDX1.append(window_size)
                IDX2.append(label[0].cpu().data.numpy().tolist())
                REF.append(0.)

                for i in range(max_dim-1):
                    IDX1.append(window_size)
                    IDX2.append(torch.argmax(Logit(out)).cpu().data.numpy().tolist())
                    REF.append(1.)

                    IDX1.append(window_size)
                    IDX2.append(label[0].cpu().data.numpy().tolist())
                    REF.append(0.)

                return [IDX1, IDX2], REF

        max_grad = []
        max_idx = []
        print('w.grad',w.grad.shape)
        for i in range(window_size):
            max_grad.append(torch.max(w.grad[0,i]))
            max_idx.append(torch.argmax(w.grad[0,i]))
        
        idx_idx = torch.argsort(-torch.tensor(max_grad))

        for cnt, i in enumerate(idx_idx):
            IDX1.append(i.numpy().tolist())
            IDX2.append(seq_id[i])
            REF.append(0.)

            IDX1.append(i.numpy().tolist())
            IDX2.append(max_idx[i].cpu().data.numpy().tolist())
            REF.append(1.)

            if cnt >= max_dim-1:
                break

        end_time = time.time()
        print('Finish Interpretation after {} steps'.format(self.k), '(Final loss: %.2f,'%Loss.item(), "Time elasped: %.2fs)"%(end_time-begin_time)) 
        return [IDX1, IDX2], REF


    def auto_calibration(self, num_candidates): # auto calibration of hyperparamters
        self.bound_thres = 1./(num_candidates+1)

    
    def show_table(self, anomaly, interpretation):
        if self.k != 1:
            raise NotImplementedError

        print('\nVisualize Interpretation (Table View)')
        index = interpretation[0]
        value = interpretation[1]
        
        reference = anomaly.copy()
        if index[0][0] != index[0][1]:
            print("error at index[0]!")
            exit(-1)
        else:
            if value[0] == 1.:
                if reference[index[0][0]] != index[1][1]:
                    print("fatal error in interpretation!")
                    exit(-1)
                else:
                    reference[index[0][0]] = index[1][0]
            elif value[1] == 1.:
                if reference[index[0][0]] != index[1][0]:
                    print("fatal error in interpretation!")
                    exit(-1)
                else:
                    reference[index[0][0]] = index[1][1]
            else:
                print("error: abnormal value of value!")
                exit(-1)
        tb = pt.PrettyTable()
        tb.field_names = [ "Ano.","Meaning","Diff.", "Ref.", "Meaning*"]
        for i in range(len(reference)):
            if anomaly[i] == reference[i]:
                row = [anomaly[i], self.feature_desc[anomaly[i]], '', reference[i], self.feature_desc[reference[i]]]
            else:
                row = [anomaly[i], self.feature_desc[anomaly[i]], '!=', reference[i], self.feature_desc[reference[i]]]
            tb.add_row(row)

        print(tb)
