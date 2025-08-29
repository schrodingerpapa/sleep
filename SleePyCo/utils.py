import os
import sys
import math
import time
import torch
import random
import numpy as np
import sklearn.metrics as skmet
from terminaltables import SingleTable
from termcolor import colored
import shutil

term_width = shutil.get_terminal_size().columns

TOTAL_BAR_LENGTH = 25.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, ckpt_path='./checkpoints', ckpt_name='checkpoint.pth', mode='min'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        if mode == 'max':
            self.init_metric = 0
        elif mode == 'min':
            self.init_metric = -np.inf
        else:
            raise NotImplementedError
            
        self.delta = delta
        self.ckpt_path = ckpt_path
        self.ckpt_name = ckpt_name if '.pth' in ckpt_name else ckpt_name + '.pth'

        os.makedirs(self.ckpt_path, exist_ok=True)


    def __call__(self, val_acc, val_loss, model):
        
        if self.mode == 'max':
            score = val_acc
            val_metric = val_acc
        elif self.mode == 'min':
            score = -val_loss
            val_metric = val_loss
        else:
            raise NotImplementedError

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.mode == 'max':
                print(f'[INFO] Validation accuracy increased ({self.init_metric:.6f} --> {val_metric:.6f}).  Saving model ...\n')
            elif self.mode == 'min':
                print(f'[INFO] Validation loss decreased ({self.init_metric:.6f} --> {val_metric:.6f}).  Saving model ...\n')
            else:
                raise NotImplementedError

        torch.save(model.state_dict(), os.path.join(self.ckpt_path, self.ckpt_name))
        self.init_metric = val_metric

def summarize_result(config, fold, y_true, y_pred, save=True):
    os.makedirs('results', exist_ok=True)
    y_pred_argmax = np.argmax(y_pred, 1)  # 取每行的最大值的索引作为预测类别
    
    # 使用 sklearn 的 classification_report 并指定 labels 和 zero_division 参数
    from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
    
    # 确保指定了所有可能的标签
    num_classes = config['classifier']['num_classes']
    labels = list(range(num_classes))
    
    # 生成分类报告，处理零除情况
    result_dict = classification_report(y_true, y_pred_argmax, labels=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred_argmax, labels=labels)
    
    # 获取整体指标
    accuracy = round(result_dict['accuracy']*100, 1) if 'accuracy' in result_dict else 0.0
    macro_f1 = round(result_dict['macro avg']['f1-score']*100, 1) if 'macro avg' in result_dict and 'f1-score' in result_dict['macro avg'] else 0.0
    kappa = round(cohen_kappa_score(y_true, y_pred_argmax), 3)
    
    # 初始化各类别指标
    class_metrics = {}
    for i in range(num_classes):
        class_key = str(i)  # classification_report 使用字符串作为键
        if class_key in result_dict:
            class_metrics[i] = {
                'precision': round(result_dict[class_key]['precision'] * 100, 1),
                'recall': round(result_dict[class_key]['recall'] * 100, 1),
                'f1': round(result_dict[class_key]['f1-score'] * 100, 1)
            }
        else:
            class_metrics[i] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
    
    # 提取各类别指标
    wpr, wre, wf1 = class_metrics[0]['precision'], class_metrics[0]['recall'], class_metrics[0]['f1']
    n1pr, n1re, n1f1 = class_metrics[1]['precision'], class_metrics[1]['recall'], class_metrics[1]['f1']
    n2pr, n2re, n2f1 = class_metrics[2]['precision'], class_metrics[2]['recall'], class_metrics[2]['f1']
    n3pr, n3re, n3f1 = class_metrics[3]['precision'], class_metrics[3]['recall'], class_metrics[3]['f1']
    rpr, rre, rf1 = class_metrics[4]['precision'], class_metrics[4]['recall'], class_metrics[4]['f1']
    
    # 确保混淆矩阵大小正确
    # 填充或裁剪混淆矩阵为5x5
    cm_formatted = np.zeros((5, 5))
    cm_rows, cm_cols = cm.shape if len(cm.shape) == 2 else (0, 0)
    if cm_rows > 0 and cm_cols > 0:
        min_rows = min(cm_rows, 5)
        min_cols = min(cm_cols, 5)
        cm_formatted[:min_rows, :min_cols] = cm[:min_rows, :min_cols]
    
    overall_data = [
        ['ACC', 'MF1', '\u03BA'],
        [accuracy, macro_f1, kappa],
    ]
    
    perclass_data = [
        [colored('A', 'cyan') + '\\' + colored('P', 'green'), 'W', 'N1', 'N2', 'N3', 'R', 'PR', 'RE', 'F1'],
        ['W', cm_formatted[0][0], cm_formatted[0][1], cm_formatted[0][2], cm_formatted[0][3], cm_formatted[0][4], wpr, wre, wf1],
        ['N1', cm_formatted[1][0], cm_formatted[1][1], cm_formatted[1][2], cm_formatted[1][3], cm_formatted[1][4], n1pr, n1re, n1f1],
        ['N2', cm_formatted[2][0], cm_formatted[2][1], cm_formatted[2][2], cm_formatted[2][3], cm_formatted[2][4], n2pr, n2re, n2f1],
        ['N3', cm_formatted[3][0], cm_formatted[3][1], cm_formatted[3][2], cm_formatted[3][3], cm_formatted[3][4], n3pr, n3re, n3f1],
        ['R', cm_formatted[4][0], cm_formatted[4][1], cm_formatted[4][2], cm_formatted[4][3], cm_formatted[4][4], rpr, rre, rf1],
    ]
    
    overall_dt = SingleTable(overall_data, colored('OVERALL RESULT', 'red'))
    perclass_dt = SingleTable(perclass_data, colored('PER-CLASS RESULT', 'red'))
    
    print('\n[INFO] Evaluation result from fold 1 to {}'.format(fold))
    print('\n' + overall_dt.table)
    print('\n' + perclass_dt.table)
    print(colored(' A', 'cyan') + ': Actual Class, ' + colored('P', 'green') + ': Predicted Class' + '\n\n')
    
    if save:
        with open(os.path.join('results', config['name'] + '.txt'), 'a') as f:
            f.write(
                str(fold) + ' ' +
                str(accuracy) + ' ' + 
                str(macro_f1) + ' ' + 
                str(kappa) + ' ' +
                str(wf1) + ' ' +
                str(n1f1) + ' ' +
                str(n2f1) + ' ' +
                str(n3f1) + ' ' +
                str(rf1) + '\n'  # 添加换行符
            )


# def summarize_result(config, fold, y_true, y_pred, save=True):
#     os.makedirs('results', exist_ok=True)
#     y_pred_argmax = np.argmax(y_pred, 1) # 取每行的最大值的索引作为预测类别
#     result_dict = skmet.classification_report(y_true, y_pred_argmax, digits=3, output_dict=True)
#     cm = skmet.confusion_matrix(y_true, y_pred_argmax)
    
#     accuracy = round(result_dict['accuracy']*100, 1)
#     macro_f1 = round(result_dict['macro avg']['f1-score']*100, 1)
#     kappa = round(skmet.cohen_kappa_score(y_true, y_pred_argmax), 3)
    
#     wpr = round(result_dict['0.0']['precision']*100, 1)
#     wre = round(result_dict['0.0']['recall']*100, 1)
#     wf1 = round(result_dict['0.0']['f1-score']*100, 1)
    
#     n1pr = round(result_dict['1.0']['precision']*100, 1)
#     n1re = round(result_dict['1.0']['recall']*100, 1)
#     n1f1 = round(result_dict['1.0']['f1-score']*100, 1)

#     n2pr = round(result_dict['2.0']['precision']*100, 1)
#     n2re = round(result_dict['2.0']['recall']*100, 1)
#     n2f1 = round(result_dict['2.0']['f1-score']*100, 1)
    
#     n3pr = round(result_dict['3.0']['precision']*100, 1)
#     n3re = round(result_dict['3.0']['recall']*100, 1)
#     n3f1 = round(result_dict['3.0']['f1-score']*100, 1)
    
#     rpr = round(result_dict['4.0']['precision']*100, 1)
#     rre = round(result_dict['4.0']['recall']*100, 1)
#     rf1 = round(result_dict['4.0']['f1-score']*100, 1)
    
#     overall_data = [
#         ['ACC', 'MF1', '\u03BA'],
#         [accuracy, macro_f1, kappa],
#     ]
    
#     perclass_data = [
#         [colored('A', 'cyan') + '\\' + colored('P', 'green'), 'W', 'N1', 'N2', 'N3', 'R', 'PR', 'RE', 'F1'],
#         ['W', cm[0][0], cm[0][1], cm[0][2], cm[0][3], cm[0][4], wpr, wre, wf1],
#         ['N1', cm[1][0], cm[1][1], cm[1][2], cm[1][3], cm[1][4], n1pr, n1re, n1f1],
#         ['N2', cm[2][0], cm[2][1], cm[2][2], cm[2][3], cm[2][4], n2pr, n2re, n2f1],
#         ['N3', cm[3][0], cm[3][1], cm[3][2], cm[3][3], cm[3][4], n3pr, n3re, n3f1],
#         ['R', cm[4][0], cm[4][1], cm[4][2], cm[4][3], cm[4][4], rpr, rre, rf1],
#     ]
    
#     overall_dt = SingleTable(overall_data, colored('OVERALL RESULT', 'red'))
#     perclass_dt = SingleTable(perclass_data, colored('PER-CLASS RESULT', 'red'))
    
#     print('\n[INFO] Evaluation result from fold 1 to {}'.format(fold))
#     print('\n' + overall_dt.table)
#     print('\n' + perclass_dt.table)
#     print(colored(' A', 'cyan') + ': Actual Class, ' + colored('P', 'green') + ': Predicted Class' + '\n\n')
    
#     if save:
#         with open(os.path.join('results', config['name'] + '.txt'), 'w') as f:
#             f.write(
#                 str(fold) + ' ' +
#                 str(round(result_dict['accuracy']*100, 1)) + ' ' + 
#                 str(round(result_dict['macro avg']['f1-score']*100, 1)) + ' ' + 
#                 str(round(kappa, 3)) + ' ' +
#                 str(round(result_dict['0.0']['f1-score']*100, 1)) + ' ' +
#                 str(round(result_dict['1.0']['f1-score']*100, 1)) + ' ' +
#                 str(round(result_dict['2.0']['f1-score']*100, 1)) + ' ' +
#                 str(round(result_dict['3.0']['f1-score']*100, 1)) + ' ' +
#                 str(round(result_dict['4.0']['f1-score']*100, 1)) + ' '
#             )


def set_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value) # Python hash buildin
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
