import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
import torch

def validate_by_rmse(rmse_vec,thres,label):
    pred = np.asarray([0] * len(rmse_vec))
    idx = np.where(rmse_vec>thres)
    pred[idx] = 1
    cnf_matrix = confusion_matrix(label, pred)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    TPR = (TP/(TP+FN))[1]
    FPR = (FP/(FP+TN))[1]
    print("TPR:",TPR,"|FPR:",FPR)

    return pred

class Normalizer:
    def __init__(self, 
            dim, 
            normer="minmax",
            online_minmax=False): # whether fit_transform online (see Kitsune), *available only for normer="minmax"

        self.dim = dim # feature dimensionality
        self.normer = normer
        if self.normer == 'minmax':
            self.online_minmax = online_minmax
            self.norm_max = [-np.Inf] * self.dim
            self.norm_min = [np.Inf] * self.dim
        else:
            raise NotImplementedError # Implement other Normalizer here
        
    def fit_transform(self,train_feat):
        if self.normer == 'minmax':
            return self._minmax_fit_transform(train_feat)
        else:
            raise NotImplementedError # Implement other Normalizer here

    def transform(self,feat):
        if self.normer == 'minmax':
            return self._minmax_transform(feat)
        else:
            raise NotImplementedError # Implement other Normalizer here

    def restore(self,feat):
        if self.normer == 'minmax':
            return self._minmax_restore(feat)
        else:
            raise NotImplementedError # Implement other Normalizer here
        
    def _minmax_fit_transform(self,train_feat):
        if not self.online_minmax:
            self.norm_min = np.min(train_feat,axis=0)
            self.norm_max = np.max(train_feat,axis=0)
            norm_feat = (train_feat - self.norm_min) / (self.norm_max-self.norm_min+1e-10)
            return norm_feat
        else:
            norm_feat = []
            self.norm_max, self.norm_min = np.asarray(self.norm_max), np.asarray(self.norm_min)
            for i in range(len(train_feat)):
                x = train_feat[i]
                self.norm_max[x>self.norm_max] = x[x>self.norm_max]
                self.norm_min[x<self.norm_min] = x[x<self.norm_min]
                norm_feat.append((x - self.norm_min) / (self.norm_max-self.norm_min+1e-10))
            return np.asarray(norm_feat)

    def _minmax_transform(self, feat):
        norm_feat = (feat - self.norm_min) / (self.norm_max-self.norm_min+1e-10)
        return norm_feat

    def _minmax_restore(self, feat):
        denorm_feat = feat * (self.norm_max-self.norm_min+1e-10) + self.norm_min
        return denorm_feat
    
    # def _olminmax_fit_transform(self, train_feat):
    #     norm_feat = []
    #     for i in range(len(train_feat)):
    #         x = train_feat[i]
    #         self.norm_max[x>self.norm_max] = x[x>self.norm_max]
    #         self.norm_min[x<self.norm_min] = x[x<self.norm_min]
    #         norm_feat.append(x - self.norm_min) / (self.norm_max-self.norm_min+1e-10)
    #     return np.asarray(norm_feat)
    
    # def _olminmax_transform(self, feat):
    #     norm_feat = (feat - self.norm_min) / (self.norm_max-self.norm_min+1e-10)
    #     return norm_feat


""" Deeplog tools """
def deeplogtools_seqformat(model, abnormal_data, num_candidates, index=0):
    import keras.utils.np_utils as np_utils
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = abnormal_data.copy()
    y, X = X[:,-1], np_utils.to_categorical(X[:,:-1])
    Output = model(torch.from_numpy(X).type(torch.float).to(device))
    TP_idx = []
    for i in range(len(Output)):
        output = Output[i]
        label = y[i]
        predicted = torch.argsort(output)[-num_candidates:]
        if label not in predicted:
            TP_idx.append(i)
    seq_feat = np_utils.to_categorical(abnormal_data[TP_idx])
    feat = seq_feat[index]
    seq = torch.from_numpy(feat[:-1,:]).to(device)
    label = torch.tensor(np.argmax(feat[-1])).unsqueeze(0).to(device)
    return seq,label, abnormal_data[TP_idx][index]

""" Multi LSTM tools """
def multiLSTM_seqformat(test_feat, seq_len = 5, index=0):
    import more_itertools

    X_test = more_itertools.windowed(test_feat[:,:],n=seq_len,step=1)
    X_test = np.asarray(list(X_test))
    y_test = np.asarray(test_feat[seq_len-1:])

    # print("X_test:",X_test.shape,"y_test:",y_test.shape)
    i = index
    interp_feat = y_test[i]
    seq_feat = np.asarray([X_test[i]]) 
    # print("seq_feat:",seq_feat.shape,"interp_feat:",interp_feat.shape)

    return seq_feat, interp_feat