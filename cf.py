import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,

    auc, 
    precision_recall_curve, 
    roc_curve,
    confusion_matrix
)



def metric_scores(y_true, y_pred): #return dic
    ### double check the input y_pred
    #is already    [1,0,0,1]
    #or still prob [0.6, 0.4, 0.3, 0.5]

    y_pred_class = np.around(y_pred)
    #y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    #'auROC_v2': roc_auc_score(va_lali, y_pred_prob),
    #'auPRC_v2': average_precision_score(va_lali, y_pred_prob)
    
    return {'accuracy': round(accuracy_score(y_true, y_pred_class),3),
            'specificity': round(recall_score(y_true, y_pred_class, pos_label=0),3),
            'precision': round(precision_score(y_true, y_pred_class, zero_division=0),3),
            'recall': round(recall_score(y_true, y_pred_class),3),
            'F1': round(f1_score(y_true, y_pred_class, zero_division=0),3),
            'MCC': round(matthews_corrcoef(y_true, y_pred_class),3),
            'auROC': round(auc(fpr, tpr),3),
            'auPRC': round(auc(recall, precision),3)         
            }



def show_table(values, headers=None, v_headers=None, title=None, float_fmt='%.3f'):

    values = [list(_) for _ in values]

    if headers is not None:
        headers = list(headers)
        item_widths = [len(_) for _ in headers]
    else:
        item_widths = [0 for _ in range(len(values[0]))]

    if v_headers is not None:
        if headers is not None:
            headers.insert(0, '')
        for i, row in enumerate(values):
            row.insert(0, v_headers[i])
        item_widths.insert(0, 0)
    
    for row in values:
        for i, v in enumerate(row):
            row[i] = float_fmt % v if isinstance(v, float) else str(v)
            item_widths[i] = max(item_widths[i], len(row[i]))

    sep_line = '+%s+' % '+'.join('-' * (w + 2) for w in item_widths)

    if title is not None:
        print('+%s+' % ('-' * (len(sep_line) - 2)))
        print('| %%-%ds |' % (len(sep_line) - 4) % title)
    
    print(sep_line)

    if headers is not None:
        for i, h in enumerate(headers):
            print('| %%%ds ' % item_widths[i] % h, end='')
        print('|')
        print(sep_line)

    for row in values:
        for i, v in enumerate(row):
            print('| %%%ds ' % item_widths[i] % v, end='')
        print('|')
    print(sep_line)



def main_cf(txt1_d, t1_title):
    each_li, md_name_li, label_li =[],[],[]
    prob_d={}
    n=0
    for txt_name, txt_path in txt1_d.items():
        #print(txt_name, txt_path)
        n+=1
        prob_d[ txt_name ]=[]
        with open(txt_path, 'r') as f:
            for l in f:
                #0_0.2805816603359451
                l=l.strip().split('_')             
                prob_d[ txt_name ].append(float(l[1]))
                if n==1:
                    label_li.append(int(l[0]))
        prob_d[ txt_name ]=np.array(prob_d[ txt_name ])
        #-----------------
        sub_mtx = metric_scores(label_li, prob_d[ txt_name ])
        md_name_li.append(txt_name)
        each_li.append(sub_mtx)

    #label_li=np.array(label_li)
    show_table([_.values() for _ in each_li],
                headers=each_li[0].keys(),
                v_headers=md_name_li,
                title=t1_title, float_fmt='%.3f')

    



#create DL model name, prediction result (label and probability) in a dictionary
txt2_d={
    'CNN + PC6':'cnn.txt', 
    'MLP + PepBERT':'mlp.txt'} 


main_cf(txt2_d, 'test set prediction')


#~$ python cf.py

#output example
'''
+---------------------------------------------------------------------------------------------+
| title                                                                                       |
+---------------+----------+-------------+-----------+--------+-------+-------+-------+-------+
|               | accuracy | specificity | precision | recall |    F1 |   MCC | auROC | auPRC |
+---------------+----------+-------------+-----------+--------+-------+-------+-------+-------+
|     CNN + PC6 |    0.825 |       0.899 |     0.600 |  0.556 | 0.577 | 0.468 | 0.854 | 0.580 |
| MLP + PepBERT |    0.865 |       0.960 |     0.778 |  0.519 | 0.622 | 0.561 | 0.838 | 0.665 |
+---------------+----------+-------------+-----------+--------+-------+-------+-------+-------+
'''
