import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from classificationMAP import getClassificationMAP as cmAP
from detectionMAP3 import getDetectionMAP as dmAP
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def test(itr, dataset, args, model, logger, device, exp_name):
    done = False
    instance_logits_stack = []
    element_logits_stack = []
    labels_stack = []
    while not done:
        if dataset.currenttestidx % 100 ==0:
            print('Testing test data point %d of %d' %(dataset.currenttestidx, len(dataset.testidx)))

        features, labels, done = dataset.load_data(is_training=False)
        features = torch.from_numpy(features).float().to(device)

        #print ('test: ', features.shape, labels.shape)

        with torch.no_grad():
           _, element_logits = model(Variable(features), is_training=False)
        tmp = F.softmax(torch.mean(torch.topk(element_logits, k=int(np.ceil(len(features)/8)), dim=0)[0], dim=0), dim=0).cpu().data.numpy()
        element_logits = element_logits.cpu().data.numpy()

        instance_logits_stack.append(tmp)
        element_logits_stack.append(element_logits)
        labels_stack.append(labels)

    np.save('./ckpt/' + exp_name + '/%s_%06d_cas.npy' % (exp_name, itr), np.array(element_logits_stack), allow_pickle=True)
    np.save('./ckpt/' + exp_name + '/%s_%06d_pmf.npy' % (exp_name, itr), np.array(instance_logits_stack), allow_pickle=True)
    np.save('./ckpt/' + exp_name + '/%s_%06d_lab.npy' % (exp_name, itr), np.array(labels_stack), allow_pickle=True)

    instance_logits_stack = np.array(instance_logits_stack)
    labels_stack = np.array(labels_stack)

    dmap, iou = dmAP(element_logits_stack, dataset.path_to_annotations, args)

    cmap = cmAP(instance_logits_stack, labels_stack)
    print('Classification map %f' %cmap)
    print('Detection map @ %f = %f' %(iou[0], dmap[0]))
    print('Detection map @ %f = %f' %(iou[1], dmap[1]))
    print('Detection map @ %f = %f' %(iou[2], dmap[2]))
    print('Detection map @ %f = %f' %(iou[3], dmap[3]))
    print('Detection map @ %f = %f' %(iou[4], dmap[4]))
    print('Detection map @ %f = %f' %(iou[5], dmap[5]))
    print('Detection map @ %f = %f' %(iou[6], dmap[6]), flush=True)

    #logger.scalar_summary('test/c_mAP', cmap, step=itr)
    logger.add_scalar('test/c_mAP', cmap, global_step=itr)
    for item in list(zip(dmap,iou)):
        #logger.scalar_summary('test/d_mAP_at_%s'%str(item[1]), item[0], step=itr)
        logger.add_scalar('test/d_mAP_at_%s'%str(item[1]), item[0], global_step=itr)

