"Implementation of Neural Graph Filtering (NGCF) using pytorch"

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


def _convert_sp_mat_to_sp_tensor(self, X):
        """
        Convert scipy sparse matrix to PyTorch sparse matrix

        Arguments:
        ----------
        X = Adjacency matrix, scipy sparse matrix
        """
        coo = X.tocoo().astype(np.float32)
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        res = torch.sparse.FloatTensor(i, v, coo.shape).to(device)
        return res


"Weights initialization"


def _init_weights(self):
        print("Initializing weights...")
        weight_dict = nn.ParameterDict()

        initializer = torch.nn.init.xavier_uniform_
        
        weight_dict['user_embedding'] = nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim).to(device)))
        weight_dict['item_embedding'] = nn.Parameter(initializer(torch.empty(self.n_items, self.emb_dim).to(device)))

        weight_size_list = [self.emb_dim] + self.layers

        for k in range(self.n_layers):
            weight_dict['W_gc_%d' %k] = nn.Parameter(initializer(torch.empty(weight_size_list[k], weight_size_list[k+1]).to(device)))
            weight_dict['b_gc_%d' %k] = nn.Parameter(initializer(torch.empty(1, weight_size_list[k+1]).to(device)))
            
            weight_dict['W_bi_%d' %k] = nn.Parameter(initializer(torch.empty(weight_size_list[k], weight_size_list[k+1]).to(device)))
            weight_dict['b_bi_%d' %k] = nn.Parameter(initializer(torch.empty(1, weight_size_list[k+1]).to(device)))
            
        return weight_dict


"Training"


def train(model, data_generator, optimizer):
	"""
    Train the model PyTorch style

    Arguments:
    ---------
    model: PyTorch model
    data_generator: Data object
    optimizer: PyTorch optimizer
    """
    model.train()
    n_batch = data_generator.n_train // data_generator.batch_size + 1
    running_loss=0
    for _ in range(n_batch):
        u, i, j = data_generator.sample()
        optimizer.zero_grad()
        loss = model(u,i,j)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss



def forward(self, u, i, j):
    """
    Computes the forward pass
    
    Arguments:
    ---------
    u = user
    i = positive item (user interacted with item)
    j = negative item (user did not interact with item)
    """
    # apply drop-out mask
    if self.node_dropout > 0.:
        self.A_hat = self._droupout_sparse(self.A)  
        
    ego_embeddings = torch.cat([self.weight_dict['user_embedding'], self.weight_dict['item_embedding']], 0)

    all_embeddings = [ego_embeddings]

    # forward pass for 'n' propagation layers
    for k in range(self.n_layers):

        # weighted sum messages of neighbours
        if self.node_dropout > 0.:
            side_embeddings = torch.sparse.mm(self.A_hat, ego_embeddings)
        else:      
            side_embeddings = torch.sparse.mm(self.A, ego_embeddings)

        # transformed sum weighted sum messages of neighbours
        sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k] + self.weight_dict['b_gc_%d' % k])
        sum_embeddings = F.leaky_relu(sum_embeddings)

        # bi messages of neighbours
        bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
        # transformed bi messages of neighbours
        bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k] + self.weight_dict['b_bi_%d' % k])
        bi_embeddings = F.leaky_relu(bi_embeddings)

        # non-linear activation 
        ego_embeddings = sum_embeddings + bi_embeddings
        # + message dropout
        mess_dropout_mask = nn.Dropout(self.mess_dropout)
        ego_embeddings = mess_dropout_mask(ego_embeddings)

        # normalize activation
        norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

        all_embeddings.append(norm_embeddings)

    all_embeddings = torch.cat(all_embeddings, 1)
    
    # back to user/item dimension
    u_g_embeddings, i_g_embeddings = all_embeddings.split([self.n_users, self.n_items], 0)

    self.u_g_embeddings = nn.Parameter(u_g_embeddings)
    self.i_g_embeddings = nn.Parameter(i_g_embeddings)
    
    u_emb = u_g_embeddings[u] # user embeddings
    p_emb = i_g_embeddings[i] # positive item embeddings
    n_emb = i_g_embeddings[j] # negative item embeddings

    y_ui = torch.mul(u_emb, p_emb).sum(dim=1)
    y_uj = torch.mul(u_emb, n_emb).sum(dim=1)
    log_prob = (torch.log(torch.sigmoid(y_ui-y_uj))).mean()

    # compute bpr-loss
    bpr_loss = -log_prob
    if self.reg > 0.:
        l2norm = (torch.sum(u_emb**2)/2. + torch.sum(p_emb**2)/2. + torch.sum(n_emb**2)/2.) / u_emb.shape[0]
        l2reg  = self.reg*l2norm
        bpr_loss =  -log_prob + l2reg

    return bpr_loss



"Model evaluation"



def eval_model(u_emb, i_emb, Rtr, Rte, k):
    """
    Evaluate the model
    
    Arguments:
    ---------
    u_emb: User embeddings
    i_emb: Item embeddings
    Rtr: Sparse matrix with the training interactions
    Rte: Sparse matrix with the testing interactions
    k : kth-order for metrics
    
    Returns:
    --------
    result: Dictionary with lists correponding to the metrics at order k for k in Ks
    """
    # split matrices
    ue_splits = split_matrix(u_emb)
    tr_splits = split_matrix(Rtr)
    te_splits = split_matrix(Rte)

    recall_k, ndcg_k= [], []
    # compute results for split matrices
    for ue_f, tr_f, te_f in zip(ue_splits, tr_splits, te_splits):

        scores = torch.mm(ue_f, i_emb.t())

        test_items = torch.from_numpy(te_f.todense()).float().cuda()
        non_train_items = torch.from_numpy(1-(tr_f.todense())).float().cuda()
        scores = scores * non_train_items

        _, test_indices = torch.topk(scores, dim=1, k=k)
        pred_items = torch.zeros_like(scores).float()
        pred_items.scatter_(dim=1,index=test_indices,src=torch.tensor(1.0).cuda())

        topk_preds = torch.zeros_like(scores).float()
        topk_preds.scatter_(dim=1,index=test_indices[:, :k],src=torch.tensor(1.0))

        TP = (test_items * topk_preds).sum(1)
        rec = TP/test_items.sum(1)
        ndcg = compute_ndcg_k(pred_items, test_items, test_indices, k)

        recall_k.append(rec)
        ndcg_k.append(ndcg)

    return torch.cat(recall_k).mean(), torch.cat(ndcg_k).mean()


"Early stopping"


def early_stopping(log_value, best_value, stopping_step, flag_step, expected_order='asc'):
    """
    Check if early_stopping is needed
    Function copied from original code
    """
    assert expected_order in ['asc', 'des']
    if (expected_order == 'asc' and log_value >= best_value) or (expected_order == 'des' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False

    return best_value, stopping_step, should_stop


'''

DATASET USED - KASANDR is a novel, 
publicly available collection for recommendation systems that records 
the behavior of customers of the European leader in e-Commerce advertising, Kelkoo.


HYPERPARAMETRS-

batch_size: 128, 256, 512, 1024
node_dropout: 0.0, 0.1
message_dropout: 0.0, 0.1
learning_rate: 0.0001, 0.0005, 0.001, 0.005

'''

