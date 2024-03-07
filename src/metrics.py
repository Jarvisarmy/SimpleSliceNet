import numpy as np
from sklearn import metrics
import cv2
from skimage import measure

def auroc_score(gt,scores):
    if isinstance(scores, list):
        scores = np.stack(scores)
    if isinstance(gt, list):
        gt = np.stack(gt)
    
    flat_scores = scores.ravel()
    flat_gt = gt.ravel()
    try:
        auroc = metrics.roc_auc_score(
            flat_gt.astype(int), flat_scores
        )
        return auroc
    except Exception as e:
        print(e)
        l = len(flat_scores)
        mid = l//2
        flat_scores_1, flat_gt_1 = flat_scores[:mid], flat_gt[:mid]
        flat_scores_2, flat_gt_2 = flat_scores[mid:], flat_gt[mid:]
        auroc_1 = metrics.roc_auc_score(
            flat_gt_1.astype(int), flat_scores_1
        )
        auroc_2 = metrics.roc_auc_score(
            flat_gt_2.astype(int), flat_scores_2
        )
        return (auroc_1+auroc_2)/2

def auprc_score(gt,scores):
    if isinstance(scores, list):
        scores = np.stack(scores)
    if isinstance(gt, list):
        gt = np.stack(gt)
    
    flat_scores = scores.ravel()
    flat_gt = gt.ravel()
    precision, recall,_ = metrics.precision_recall_curve(
        flat_gt.astype(int), flat_scores
    )
    auprc = metrics.auc(recall, precision)
    return auprc


def pro_score(masks, amaps, num_th=100):
    df_pro = []
    df_fpr = []
    df_threshold = []
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # get a kernel
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k) # add dialation
            for region in measure.regionprops(measure.label(mask)): # get all the ground truth anomaly region
                # measure.label(mask) label connected regions of an integer array
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum() # get the predict anomaly pixel
                pros.append(tp_pixels / region.area) # tp/(tp+fn) = predicted abnomal pixel/groud truth abnormal 

        inverse_masks = 1 - masks # the inverse mask, 1 for normal pixels
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum() #fp/(fp+tn) = predicted normal pixel/ground truth normal
        df_pro.append(np.mean(pros))
        df_fpr.append(fpr)
        df_threshold.append(th)     
    df_pro = np.array(df_pro)
    df_fpr = np.array(df_fpr)
    df_threshold = np.array(df_threshold)
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3

    idx = df_fpr < 0.3
    df_pro = df_pro[idx]
    df_fpr = df_fpr[idx]
    df_threshold = df_threshold[idx]
    df_fpr = df_fpr / df_fpr.max()

    pro_auc = metrics.auc(df_fpr, df_pro)
    return pro_auc

'''
def pro_score(ground_truth, scores, num_th=200):
    pros = []
    fprs = []
    threds = []
    min_th = scores.min()
    max_th = scores.max()
    delta = (max_th-min_th)/num_th
    binary_score_maps = np.zeros_like(scores, dtype=bool)
    for th in np.arange(min_th, max_th, delta):
        binary_score_maps[scores <= th] = 0
        binary_score_maps[scores < th] = 1
        pro = []
        for i in range(len(binary_score_maps)):
            props = measure.regionprops(measure.label(ground_truth[i], connectivity=2), binary_score_maps[i])
            for prop in props:
                pro.append(prop.intensity_image.sum()/prop.area)
            
        #fpr
        masks_neg = 1-ground_truth
        fpr = np.logical_and(masks_neg, binary_score_maps).sum()/masks_neg.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        threds.append(th)
    pros = np.array(pros)
    fprs = np.array(fprs)
    threds = np.array(threds)
    idx = fprs <=0.3
    fprs = fprs[idx]
    pros = pros[idx]
    fprs = fprs/fprs.max()
    pro_auc = metrics.auc(fprs, pros)
    return pro_auc
'''
def return_best_thr(y_true, y_score):
    if isinstance(y_score, list):
        y_score = np.stack(y_score)
    if isinstance(y_true, list):
        y_true = np.stack(y_true)
    
    y_score = y_score.ravel()
    y_true = y_true.ravel()
    precs, recs, thrs = metrics.precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N

def accuracy_score(y_true, y_score):
    return metrics.accuracy_score(y_true,y_score)
def f1_score(y_true, y_score):
    return metrics.f1_score(y_true, y_score)
def recall_score(y_true, y_score):
    return metrics.recall_score(y_true, y_score)

if __name__ == '__main__':
    y_true = np.ones(6)
    y_prob = np.array([0.7, 0.8, 0.4, 0.5, 0.8,0.9])
    thres = return_best_thr(y_true, y_prob)
    acc = accuracy_score(y_true, y_prob >= thres)
    f1 = f1_score(y_true, y_prob >= thres)
    recall = recall_score(y_true, y_prob >= thres)
    specificity = specificity_score(y_true, y_prob>=thres)

    auroc = auroc_score(y_true, y_prob)
    auprc = auprc_score(y_true, y_prob)
    print('acc: {}, f1: {}, recall: {}, specificity: {}, auroc: {}, auprc: {}'.format(acc, f1, recall, specificity, auroc, auprc))