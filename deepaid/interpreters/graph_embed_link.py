import numpy as np
import _pickle as pkl
import heapq
import sys
sys.path.append("../deepaid/")
from interpreter import Interpreter

PRED = 0
LEVEL = 1
DIS = 2
A = 3 
B = 4 

def get_adjacency_list(edgelist_file,node_num,directed=False):
    adjacency = dict()
    with open(edgelist_file,'r') as f:
        for line in f.readlines():
            pair =  line.strip().split()
            # if int(pair[0]) >= node_num or int(pair[1]) >= node_num:
            #     print(pair)
            #     continue
            if not pair[0] in adjacency.keys():
                adjacency[pair[0]] = [pair[1]]
            else:
                adjacency[pair[0]].append(pair[1])
            if not directed:
                if not pair[1] in adjacency.keys():
                    adjacency[pair[1]] = [pair[0]]
                else:
                    adjacency[pair[1]].append(pair[0])
    
    pkl.dump(adjacency,open('save/adjacency.pkl','wb'))
    for i in range(node_num):
        if not str(i) in adjacency.keys():
            adjacency[str(i)] = []
        else:
            adjacency[str(i)] = list(set(adjacency[str(i)]))
    return adjacency

# def get_adjacency_list(edgelist_file,node_num,directed=False):
#     adjacency = dict()
#     for i in range(node_num):
#         for j in range(node_num//100):
#             if j == 0:
#                 adjacency[str(i)] = [j]
#             else:
#                 adjacency[str(i)].append(j)
#         # if i%10000 == 0:
#         #     print(i)
#     return adjacency

class GraphEmbedLink(Interpreter):
    
    def __init__(self, model, 
                 input_size,            # length of input tabular feature
                 feature_desc=None,     # description of each feature dimension 
                 max_depth = 5,         # max_depth of traversal
                 edge_list_file = None,      
                 adjacency_list = None, 
                 node_name_map = None,
                 embeddings = None,
                 verbose=False):        # verbose output during training
        super(GraphEmbedLink,self).__init__(model)
        self.input_size = input_size
        self.max_depth = max_depth
        self.edge_list = node_name_map
        self.embeddings = embeddings

        self.edge_list_file = edge_list_file
        self.adjacency_list = adjacency_list
        if self.edge_list_file is not None and self.adjacency_list is None:
            self.adjacency_list = get_adjacency_list(edge_list_file,len(self.embeddings))
        
        print('Successfully Initialize <Graph_Embed_Link Interptreter> for Model <{}>'.format(self.model_name))
        self.verbose = verbose
        
    def forward(self, anomaly_embedding, anomaly_node_pair):
        # print('anomaly_node_pair',anomaly_node_pair)
        PQ = []
        RES = []
        visit_node = [0]*len(self.embeddings)
        level = 1
        for node in self.adjacency_list[str(anomaly_node_pair[0])]:
            node = int(node)
            if visit_node[node] == 0:
                link_embedings = self.embeddings[anomaly_node_pair[0]]*self.embeddings[node]
                pred = self.model.predict([link_embedings])[0]
                dis = np.linalg.norm(anomaly_embedding-link_embedings)
                heapq.heappush(PQ,(pred, level, dis, anomaly_node_pair[0], node))
                # visit_link.add((anomaly_node_pair[0], node)
        visit_node[anomaly_node_pair[0]] = 1
        
        for node in self.adjacency_list[str(anomaly_node_pair[1])]:
            node = int(node)
            if visit_node[node] == 0:
                link_embedings = self.embeddings[anomaly_node_pair[1]]*self.embeddings[node]
                pred = self.model.predict([link_embedings])[0]
                dis = np.linalg.norm(anomaly_embedding-link_embedings)
                heapq.heappush(PQ,(pred, level, dis, anomaly_node_pair[1], node))
                # visit_link.add((anomaly_node_pair[1], node))
        visit_node[anomaly_node_pair[1]] = 1

        while PQ:
            item = heapq.heappop(PQ)
            # print('item,level', item,level)
            heapq.heappush(RES, item)
            level = item[LEVEL] + 1
            if level > self.max_depth:
                continue
            if visit_node[item[A]] == 0:
                for node in self.adjacency_list[str(item[A])]:
                    node = int(node)
                    if visit_node[node] == 0:
                        link_embedings = self.embeddings[item[A]]*self.embeddings[node]
                        pred = self.model.predict([link_embedings])[0]
                        dis = np.linalg.norm(anomaly_embedding-link_embedings)
                        heapq.heappush(PQ,(pred, level, dis, item[A], node))
                visit_node[item[A]] = 1

            elif visit_node[item[B]] == 0:
                for node in self.adjacency_list[str(item[B])]:
                    node = int(node)
                    if visit_node[node] == 0:
                        link_embedings = self.embeddings[item[B]]*self.embeddings[node]
                        pred = self.model.predict([link_embedings])[0]
                        dis = np.linalg.norm(anomaly_embedding-link_embedings)
                        heapq.heappush(PQ,(pred, level, dis, item[B], node))
                visit_node[item[B]] = 1
        
        # print('**RES**', heapq.heappop(RES))
        return heapq.heappop(RES)

# if __name__ == "__main__":
    


