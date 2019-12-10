import torch
import torch.nn.functional as F
#from tensorboard_logger import log_value
import numpy as np
from torch.autograd import Variable
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def MILL(element_logits, seq_len, batch_size, labels, device):
    ''' element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over, 
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) '''

    k = np.ceil(seq_len/8).astype('int32')
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    for i in range(batch_size):
        tmp, _ = torch.topk(element_logits[i][:seq_len[i]], k=int(k[i]), dim=0) # T''x20
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(Variable(labels) * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CASL(x, element_logits, seq_len, n_similar, labels, device):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature), 
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class) 
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

    # First 2*n_similar elements out of B is.. "similar pairs"
    # x: BxT''x2048
    # element_logits: BxT''x20
    # seq_len: (B,)
    # labels: Bx20
    margin = 0.5
    sim_loss = 0.
    n_tmp = 0.
    for i in range(0, n_similar*2, 2):
        atn1 = F.softmax(element_logits[i][:seq_len[i]], dim=0) # T1x20: each (T1,) vector is a probability distribution
        atn2 = F.softmax(element_logits[i+1][:seq_len[i+1]], dim=0) # T2x20

        n1 = torch.FloatTensor([np.maximum(seq_len[i]-1, 1)]).to(device)
        n2 = torch.FloatTensor([np.maximum(seq_len[i+1]-1, 1)]).to(device)
        Hf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn1) # 2048xT1, T1x20-> 2048x20
        Hf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), atn2) # 2048xT2, T2x20 -> 2048x20
        Lf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), (1 - atn1)/n1)
        Lf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), (1 - atn2)/n2)

        d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0)) # (20,)
        d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))

        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+margin, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+margin, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        n_tmp = n_tmp + torch.sum(Variable(labels[i,:])*Variable(labels[i+1,:]))
    sim_loss = sim_loss / n_tmp
    return sim_loss


def train(itr, dataset, args, model, optimizer, logger, device):

    features, labels = dataset.load_data(n_similar=args.num_similar)
    s1 = features.shape # BxT'x2048
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1) # BxT' -> (B,)
    features = features[:,:np.max(seq_len),:] # T' -> np.max(seq_len)=T''
    s2 = features.shape # BxT''x2048

    features = torch.from_numpy(features).float().to(device) # BxT''x2048
    labels = torch.from_numpy(labels).float().to(device) # BxC

    #print ('train: ', s1, s2, features.shape, labels.shape, seq_len)
    final_features, element_logits = model(Variable(features))

    milloss = MILL(element_logits, seq_len, args.batch_size, labels, device)
    casloss = CASL(final_features, element_logits, seq_len, args.num_similar, labels, device)

    #milloss *= 2
    #casloss = 0

    total_loss = args.Lambda * milloss + (1-args.Lambda) * casloss

    #logger.scalar_summary('train/milloss', milloss, step=itr)
    #logger.scalar_summary('train/casloss', casloss, step=itr)
    #logger.scalar_summary('train/totloss', total_loss, step=itr)
    logger.add_scalar('train/milloss', milloss, global_step=itr)
    logger.add_scalar('train/casloss', casloss, global_step=itr)
    logger.add_scalar('train/totloss', total_loss, global_step=itr)

    print('Iteration: %d, Loss: %.3f' %(itr, total_loss.data.cpu().numpy()), flush=True)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

