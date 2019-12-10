import numpy as np

def str2ind(categoryname,classlist):
    return [i for i in range(len(classlist)) if categoryname==classlist[i]][0]

def filter_segments(segment_predict, videonames, ambilist, factor):
    ind = np.zeros(segment_predict.shape[0])
    for i in range(segment_predict.shape[0]):
        vn = videonames[int(segment_predict[i,0])]
        for a in ambilist:
            if a[0]==vn:
                gt = range(round(float(a[2])*factor), round(float(a[3])*factor))
                pd = range(int(segment_predict[i][1]),int(segment_predict[i][2]))
                if len(set(gt).intersection(set(pd))) > 0:
                    ind[i] = 1
    s = [segment_predict[i,:] for i in range(segment_predict.shape[0]) if ind[i]==0]
    return np.array(s)

# Inspired by Pascal VOC evaluation tool.
def _ap_from_pr(prec, rec):
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])

    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])

    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])

    return ap

def getLocMAP(predictions, th, annotation_path, args):
    gtsegments = np.load(annotation_path + '/segments.npy', allow_pickle=True)
    gtlabels = np.load(annotation_path + '/labels.npy', allow_pickle=True)
    videoname = np.load(annotation_path + '/videoname.npy', allow_pickle=True)
    videoname = np.array([v.decode('utf-8') for v in videoname])
    subset = np.load(annotation_path + '/subset.npy', allow_pickle=True)
    subset = np.array([s.decode('utf-8') for s in subset])
    classlist = np.load(annotation_path + '/classlist.npy', allow_pickle=True)
    classlist = np.array([c.decode('utf-8') for c in classlist])
    factor = 25.0/16.0

    ambilist = list(open(annotation_path + '/Ambiguous_test.txt', 'r'))
    ambilist = [a.strip('\n').split(' ') for a in ambilist]

    # Keep only the test subset annotations
    j = 0
    gts, gtl, vn, pred = [], [], [], []
    for i, s in enumerate(subset):
        if subset[i]=='test' and len(gtsegments[i]):
            gts.append(gtsegments[i])
            gtl.append(gtlabels[i])
            vn.append(videoname[i])
            pred.append(predictions[j])
            j += 1
    gtsegments = gts
    gtlabels = gtl
    videoname = vn
    predictions = pred

    # which categories have temporal labels ?
    templabelcategories = sorted(list(set([l for gtl in gtlabels for l in gtl])))

    # the number index for those categories.
    templabelidx = []
    for t in templabelcategories:
        templabelidx.append(str2ind(t,classlist))

    # process the predictions such that classes having greater than a certain threshold are detected only
    predictions_mod = []
    c_score = []
    for p in predictions:
        pp = -p
        for i in range(pp.shape[1]):
            pp[:,i].sort()
        pp = -pp
        c_s = np.mean(pp[:int(np.ceil(pp.shape[0]/8)),:], axis=0)
        #ind = c_s > 0.0
        ind = (c_s > np.max(c_s)/2) * (c_s > 0)
        c_score.append(c_s)
        predictions_mod.append(p*ind)
    predictions = predictions_mod

    ap = []
    for c in templabelidx:
        segment_predict = []
        # Get list of all predictions for class c
        for i in range(len(predictions)):
            tmp = predictions[i][:,c]
            threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp))*0.5
            vid_pred = np.concatenate([np.zeros(1),(tmp>threshold).astype('float32'),np.zeros(1)], axis=0)
            vid_pred_diff = [vid_pred[idt]-vid_pred[idt-1] for idt in range(1,len(vid_pred))]
            s = [idk for idk,item in enumerate(vid_pred_diff) if item==1]
            e = [idk for idk,item in enumerate(vid_pred_diff) if item==-1]
            for j in range(len(s)):
                aggr_score = np.max(tmp[s[j]:e[j]]) + c_score[i][c]
                if e[j]-s[j]>=2:
                    segment_predict.append([i, s[j], e[j], aggr_score])
        segment_predict = np.array(segment_predict)
        #segment_predict = filter_segments(segment_predict, videoname, ambilist, factor)

        # Sort the list of predictions for class c based on score
        if len(segment_predict) == 0:
            return 0
        segment_predict = segment_predict[np.argsort(-segment_predict[:,3])]

        segment_predict = filter_segments(segment_predict, videoname, ambilist, factor)

        # Create gt list 
        segment_gt = [[i, gtsegments[i][j][0], gtsegments[i][j][1]] for i in range(len(gtsegments)) for j in range(len(gtsegments[i])) if str2ind(gtlabels[i][j],classlist)==c]
        gtpos = len(segment_gt)

        # Compare predictions and gt
        tp, fp = [], []
        for i in range(len(segment_predict)):
            flag = 0.
            best_iou = 0
            for j in range(len(segment_gt)):
                if segment_predict[i][0]==segment_gt[j][0]:
                    gt = range(round(segment_gt[j][1]*factor), round(segment_gt[j][2]*factor))
                    p = range(int(segment_predict[i][1]),int(segment_predict[i][2]))
                    IoU = float(len(set(gt).intersection(set(p))))/float(len(set(gt).union(set(p))))
                    if IoU >= th:
                        flag = 1.
                        if IoU > best_iou:
                            best_iou = IoU
                            best_j = j
            if flag > 0:
                del segment_gt[best_j]
            tp.append(flag)
            fp.append(1.-flag)
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        if sum(tp)==0:
            prc = 0.
        else:
            cur_prec = tp_c / (fp_c+tp_c)
            cur_rec = tp_c / gtpos
            prc = _ap_from_pr(cur_prec, cur_rec)
        ap.append(prc)

    return 100*np.mean(ap)

def getDetectionMAP(predictions, annotation_path, args):
   iou_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
   dmap_list = []
   for iou in iou_list:
      print('Testing for IoU %f' %iou)
      dmap_list.append(getLocMAP(predictions, iou, annotation_path, args))

   return dmap_list, iou_list
