"""
Author: Dr. Jin Zhang
E-mail: j.zhang.vision@gmail.com
Codes for "Flotation process monitoring via momentum memory network based on froth videos"
Created on 2022.02.10
"""

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, Subset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import time
import argparse
import numpy as np

from dataset_clustering import OnStreamingTailingSet
from CL_StableSGD import LinearModel as StableSGD_Lerner
from CL_ERReservior import EpisodicMemoryLearner as ERReservoir_Lerner
from CL_OCS import EpisodicMemoryLearner as OCS_Lerner
from CL_MEMO import EpisodicMemoryLearner as MEMO_Lerner
from CL_DualBuffer import EpisodicMemoryLearner as DualBuffer_Lerner
from CL_DualNet import EpisodicMemoryLearner as DualNet_Lerner
from util import AverageMeter

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score

import seaborn as sns
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=40, help='batch_size')  # -->GPU支持到60
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers=4*num_GPU')
    parser.add_argument('--epoch', type=int, default=0, help='number of training epochs')
    parser.add_argument('--load_epoch', type=int, default=400, help='number of training epochs')
    parser.add_argument('--epochs_per_task', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.4, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model dataset
    parser.add_argument('--model_name', type=str, default='ERReservoir')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # temperature
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')

    opt = parser.parse_args()

    opt.save_folder = os.path.join('./save', opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt


def set_loader(opt, full_data):
    train_size = int(0.6 * len(full_data))
    val_size = int(0.2 * len(full_data))
    test_size = len(full_data) - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(full_data, [train_size, val_size, test_size],
                                                                    generator=torch.Generator().manual_seed(32))
    train_loader = DataLoader(train_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    val_loader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    return train_loader, val_loader, test_loader


def set_model():
    StableSGD = StableSGD_Lerner()
    ERReservoir = ERReservoir_Lerner(memory_capacity=3000, memory_sample_sz=10)
    OCR = OCS_Lerner(memory_capacity=3000, memory_sample_sz=10)
    MEMO = MEMO_Lerner(memory_capacity=3000, memory_sample_sz=10)
    DualBuffer = DualBuffer_Lerner(memory_capacity=3000, memory_sample_sz=10)
    DualNet = DualNet_Lerner(memory_capacity=3000, memory_sample_sz=10)

    if torch.cuda.is_available():
        StableSGD = StableSGD.cuda()
        ERReservoir = ERReservoir.cuda()
        OCR = OCR.cuda()
        MEMO = MEMO.cuda()
        DualBuffer = DualBuffer.cuda()
        DualNet = DualNet.cuda()
        cudnn.benchmark = True

    return StableSGD, ERReservoir, OCR, MEMO, DualBuffer, DualNet


def cal_metrics(y_pred, y_true):
    # r2_score: R2
    R2_0 = r2_score(y_true[:, 0], y_pred[:, 0])
    R2_1 = r2_score(y_true[:, 1], y_pred[:, 1])
    R2_2 = r2_score(y_true[:, 2], y_pred[:, 2])
    # explained_variance_score: EVS
    EVS_0 = explained_variance_score(y_true[:, 0], y_pred[:, 0])
    EVS_1 = explained_variance_score(y_true[:, 1], y_pred[:, 1])
    EVS_2 = explained_variance_score(y_true[:, 2], y_pred[:, 2])
    # mean_squared_error (If True returns MSE value, if False returns RMSE value.): RMSE
    RMSE_0 = mean_squared_error(y_true[:, 0], y_pred[:, 0], squared=False)
    RMSE_1 = mean_squared_error(y_true[:, 1], y_pred[:, 1], squared=False)
    RMSE_2 = mean_squared_error(y_true[:, 2], y_pred[:, 2], squared=False)
    # mean_absolute_error: MAE
    MAE_0 = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    MAE_1 = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    MAE_2 = mean_absolute_error(y_true[:, 2], y_pred[:, 2])
    return R2_0, R2_1, R2_2, EVS_0, EVS_1, EVS_2, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2


def CL_metrics(acc_matrix):
    """
    acc_matrix: A matrix of shape (T, T), where T is the number of tasks.
    row direction: finish the i-th task
    column direction: test on the j-th task
    """
    num_tasks = acc_matrix.shape[0]
    average_acc = []
    forgetting = []
    for i in range(num_tasks): # finish the i-th task
        # average accuracy after incremental training
        test_acc_seq = acc_matrix[i, :i+1]
        average_acc.append(np.mean(test_acc_seq))
    for j in range(num_tasks-1):
        # average forgetting after finish all tasks
        acc_diff = acc_matrix[j:-1, j] - acc_matrix[-1, j]
        max_diff = np.max(acc_diff)
        forgetting.append(max(0, max_diff))
    average_forgetting = np.mean(forgetting)
    return average_acc, average_forgetting


def main():
    args = parse_option()

    sns.set_theme(style='white', font='Times New Roman', font_scale=1.6)

    acc_matrix_SGD_acc0 = np.zeros((4, 4))
    acc_matrix_SGD_acc1 = np.zeros((4, 4))
    acc_matrix_SGD_acc2 = np.zeros((4, 4))
    acc_matrix_ERReservoir_acc0 = np.zeros((4, 4))
    acc_matrix_ERReservoir_acc1 = np.zeros((4, 4))
    acc_matrix_ERReservoir_acc2 = np.zeros((4, 4))
    acc_matrix_OCR_acc0 = np.zeros((4, 4))
    acc_matrix_OCR_acc1 = np.zeros((4, 4))
    acc_matrix_OCR_acc2 = np.zeros((4, 4))
    acc_matrix_MEMO_acc0 = np.zeros((4, 4))
    acc_matrix_MEMO_acc1 = np.zeros((4, 4))
    acc_matrix_MEMO_acc2 = np.zeros((4, 4))
    acc_matrix_DualBuffer_acc0 = np.zeros((4, 4))
    acc_matrix_DualBuffer_acc1 = np.zeros((4, 4))
    acc_matrix_DualBuffer_acc2 = np.zeros((4, 4))
    acc_matrix_DualNet_acc0 = np.zeros((4, 4))
    acc_matrix_DualNet_acc1 = np.zeros((4, 4))
    acc_matrix_DualNet_acc2 = np.zeros((4, 4))

    acc_table0 = []
    acc_table1 = []
    acc_table2 = []
    fgt_table0 = []
    fgt_table1 = []
    fgt_table2 = []

    StableSGD, ERReservoir, OCR, MEMO, DualBuffer, DualNet = set_model()
    full_data_1, full_data_2, full_data_3, full_data_4 = OnStreamingTailingSet(train_mode="train", clip_mode='seq')

    for task_id in range(1, 5):
        enc_pth_file = './save/StableSGD/checkpoints_{task_id}_400.pth'.format(task_id=task_id)
        StableSGD.load_state_dict(torch.load(enc_pth_file))
        enc_pth_file = './save/ERReservoir/checkpoints_{task_id}_400.pth'.format(task_id=task_id)
        ERReservoir.load_state_dict(torch.load(enc_pth_file))
        enc_pth_file = './save/OCS/checkpoints_{task_id}_400.pth'.format(task_id=task_id)
        OCR.load_state_dict(torch.load(enc_pth_file))
        enc_pth_file = './save/MEMO/checkpoints_{task_id}_400.pth'.format(task_id=task_id)
        MEMO.load_state_dict(torch.load(enc_pth_file))
        enc_pth_file = './save/DualBuffer/checkpoints_{task_id}_400.pth'.format(task_id=task_id)
        DualBuffer.load_state_dict(torch.load(enc_pth_file))
        enc_pth_file = './save/DualNet/checkpoints_{task_id}_400.pth'.format(task_id=task_id)
        DualNet.load_state_dict(torch.load(enc_pth_file))
        StableSGD.eval()
        ERReservoir.eval()
        OCR.eval()
        MEMO.eval()
        DualBuffer.eval()
        DualNet.eval()
        for prev_task_id in range(1, task_id + 1):
            if prev_task_id == 1:
                full_data = full_data_1
            elif prev_task_id == 2:
                full_data = full_data_2
            elif prev_task_id == 3:
                full_data = full_data_3
            else:
                full_data = full_data_4
            _, _, test_loader = set_loader(args, full_data)


            with torch.no_grad():
                for idx, (reagents, images, targets, _) in enumerate(test_loader):
                    reagents = reagents.cuda(non_blocking=True)
                    images = images.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True)
                    # forward
                    pred_StableSGD = StableSGD(reagents.float(), images)
                    pred_ERReservoir = ERReservoir(reagents.float(), images)
                    pred_OCR = OCR(reagents.float(), images)
                    pred_MEMO, _ = MEMO(reagents.float(), images)
                    pred_DualBuffer = DualBuffer(reagents.float(), images)
                    pred_DualNet = DualNet(reagents.float(), images)
                    # update metric
                    if idx:
                        predict_StableSGD_set = np.append(predict_StableSGD_set, pred_StableSGD.detach().cpu().numpy(), axis=0)
                        predict_ERReservoir_set = np.append(predict_ERReservoir_set, pred_ERReservoir.detach().cpu().numpy(), axis=0)
                        predict_OCR_set = np.append(predict_OCR_set, pred_OCR.detach().cpu().numpy(), axis=0)
                        predict_MEMO_set = np.append(predict_MEMO_set, pred_MEMO.detach().cpu().numpy(), axis=0)
                        predict_DualBuffer_set = np.append(predict_DualBuffer_set, pred_DualBuffer.detach().cpu().numpy(), axis=0)
                        predict_DualNet_set = np.append(predict_DualNet_set, pred_DualNet.detach().cpu().numpy(), axis=0)
                        target_set = np.append(target_set, targets.cpu().numpy(), axis=0)
                    else:
                        predict_StableSGD_set = pred_StableSGD.detach().cpu().numpy()
                        predict_ERReservoir_set = pred_ERReservoir.detach().cpu().numpy()
                        predict_OCR_set = pred_OCR.detach().cpu().numpy()
                        predict_MEMO_set = pred_MEMO.detach().cpu().numpy()
                        predict_DualBuffer_set = pred_DualBuffer.detach().cpu().numpy()
                        predict_DualNet_set = pred_DualNet.detach().cpu().numpy()
                        target_set = targets.cpu().numpy()

            acc0, acc1, acc2, _, _, _, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2 = cal_metrics(predict_StableSGD_set, target_set)
            acc_matrix_SGD_acc0[task_id - 1, prev_task_id - 1] = acc0
            acc_matrix_SGD_acc1[task_id - 1, prev_task_id - 1] = acc1
            acc_matrix_SGD_acc2[task_id - 1, prev_task_id - 1] = acc2
            acc0, acc1, acc2, _, _, _, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2 = cal_metrics(predict_ERReservoir_set, target_set)
            acc_matrix_ERReservoir_acc0[task_id - 1, prev_task_id - 1] = acc0
            acc_matrix_ERReservoir_acc1[task_id - 1, prev_task_id - 1] = acc1
            acc_matrix_ERReservoir_acc2[task_id - 1, prev_task_id - 1] = acc2
            acc0, acc1, acc2, _, _, _, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2 = cal_metrics(predict_OCR_set, target_set)
            acc_matrix_OCR_acc0[task_id - 1, prev_task_id - 1] = acc0
            acc_matrix_OCR_acc1[task_id - 1, prev_task_id - 1] = acc1
            acc_matrix_OCR_acc2[task_id - 1, prev_task_id - 1] = acc2
            acc0, acc1, acc2, _, _, _, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2 = cal_metrics(predict_MEMO_set, target_set)
            acc_matrix_MEMO_acc0[task_id - 1, prev_task_id - 1] = acc0
            acc_matrix_MEMO_acc1[task_id - 1, prev_task_id - 1] = acc1
            acc_matrix_MEMO_acc2[task_id - 1, prev_task_id - 1] = acc2
            acc0, acc1, acc2, _, _, _, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2 = cal_metrics(predict_DualBuffer_set, target_set)
            acc_matrix_DualBuffer_acc0[task_id - 1, prev_task_id - 1] = acc0
            acc_matrix_DualBuffer_acc1[task_id - 1, prev_task_id - 1] = acc1
            acc_matrix_DualBuffer_acc2[task_id - 1, prev_task_id - 1] = acc2
            acc0, acc1, acc2, _, _, _, RMSE_0, RMSE_1, RMSE_2, MAE_0, MAE_1, MAE_2 = cal_metrics(predict_DualNet_set, target_set)
            acc_matrix_DualNet_acc0[task_id - 1, prev_task_id - 1] = acc0
            acc_matrix_DualNet_acc1[task_id - 1, prev_task_id - 1] = acc1
            acc_matrix_DualNet_acc2[task_id - 1, prev_task_id - 1] = acc2
    #########    Zn   #########
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_SGD_acc0, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('StableSGD: Zn')
    plt.savefig('save/acc_matrix_SGD_acc0.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_ERReservoir_acc0, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('ERReservoir: Zn')
    plt.savefig('save/acc_matrix_ERReservoir_acc0.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_OCR_acc0, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('OCS: Zn')
    plt.savefig('save/acc_matrix_OCS_acc0.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_MEMO_acc0, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('MEMO: Zn')
    plt.savefig('save/acc_matrix_MEMO_acc0.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_DualBuffer_acc0, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('DualBuffer: Zn')
    plt.savefig('save/acc_matrix_DualBuffer_acc0.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_DualNet_acc0, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('DualNet: Zn')
    plt.savefig('save/acc_matrix_DualNet_acc0.png')
    plt.close()
    #########    Pb   #########
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_SGD_acc1, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('StableSGD: Pb')
    plt.savefig('save/acc_matrix_SGD_acc1.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_ERReservoir_acc1, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('ERReservoir: Pb')
    plt.savefig('save/acc_matrix_ERReservoir_acc1.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_OCR_acc1, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('OCS: Pb')
    plt.savefig('save/acc_matrix_OCS_acc1.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_MEMO_acc1, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('MEMO: Pb')
    plt.savefig('save/acc_matrix_MEMO_acc1.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_DualBuffer_acc1, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('DualBuffer: Pb')
    plt.savefig('save/acc_matrix_DualBuffer_acc1.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_DualNet_acc1, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('DualNet: Pb')
    plt.savefig('save/acc_matrix_DualNet_acc1.png')
    plt.close()
    #########    Fe   #########
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_SGD_acc2, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('StableSGD: Fe')
    plt.savefig('save/acc_matrix_SGD_acc2.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_ERReservoir_acc2, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('ERReservoir: Fe')
    plt.savefig('save/acc_matrix_ERReservoir_acc2.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_OCR_acc2, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('OCS: Fe')
    plt.savefig('save/acc_matrix_OCS_acc2.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_MEMO_acc2, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('MEMO: Fe')
    plt.savefig('save/acc_matrix_MEMO_acc2.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_DualBuffer_acc2, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('DualBuffer: Fe')
    plt.savefig('save/acc_matrix_DualBuffer_acc2.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(acc_matrix_DualNet_acc2, annot=True, fmt=".2f", cmap='Blues', cbar=False)
    ax.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    ax.set_yticklabels(['After T1', 'After T2', 'After T3', 'After T4'])
    plt.title('DualNet: Fe')
    plt.savefig('save/acc_matrix_DualNet_acc2.png')
    plt.close()


    ave_acc, ave_fgt = CL_metrics(acc_matrix_SGD_acc0)
    acc_table0.append(ave_acc)
    fgt_table0.append(ave_fgt)
    print(f"StableSGD: Acc matrix: {acc_matrix_SGD_acc0}    Zn Average Acc: {ave_acc}    Zn Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_SGD_acc1)
    acc_table1.append(ave_acc)
    fgt_table1.append(ave_fgt)
    print(f"StableSGD: Acc matrix: {acc_matrix_SGD_acc1}    Pb Average Acc: {ave_acc}    Pb Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_SGD_acc2)
    acc_table2.append(ave_acc)
    fgt_table2.append(ave_fgt)
    print(f"StableSGD: Acc matrix: {acc_matrix_SGD_acc2}    Fe Average Acc: {ave_acc}    Fe Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_ERReservoir_acc0)
    acc_table0.append(ave_acc)
    fgt_table0.append(ave_fgt)
    print(f"ERReservoir: Acc matrix: {acc_matrix_ERReservoir_acc0}    Zn Average Acc: {ave_acc}    Zn Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_ERReservoir_acc1)
    acc_table1.append(ave_acc)
    fgt_table1.append(ave_fgt)
    print(f"ERReservoir: Acc matrix: {acc_matrix_ERReservoir_acc1}    Pb Average Acc: {ave_acc}    Pb Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_ERReservoir_acc2)
    acc_table2.append(ave_acc)
    fgt_table2.append(ave_fgt)
    print(f"ERReservior: Acc matrix: {acc_matrix_ERReservoir_acc2}    Fe Average Acc: {ave_acc}    Fe Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_OCR_acc0)
    acc_table0.append(ave_acc)
    fgt_table0.append(ave_fgt)
    print(f"OCS: Acc matrix: {acc_matrix_OCR_acc0}    Zn Average Acc: {ave_acc}    Zn Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_OCR_acc1)
    acc_table1.append(ave_acc)
    fgt_table1.append(ave_fgt)
    print(f"OCS: Acc matrix: {acc_matrix_OCR_acc1}    Pb Average Acc: {ave_acc}    Pb Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_OCR_acc2)
    acc_table2.append(ave_acc)
    fgt_table2.append(ave_fgt)
    print(f"OCS: Acc matrix: {acc_matrix_OCR_acc2}    Fe Average Acc: {ave_acc}    Fe Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_MEMO_acc0)
    acc_table0.append(ave_acc)
    fgt_table0.append(ave_fgt)
    print(f"MEMO: Acc matrix: {acc_matrix_MEMO_acc0}    Zn Average Acc: {ave_acc}    Zn Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_MEMO_acc1)
    acc_table1.append(ave_acc)
    fgt_table1.append(ave_fgt)
    print(f"MEMO: Acc matrix: {acc_matrix_MEMO_acc1}    Pb Average Acc: {ave_acc}    Pb Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_MEMO_acc2)
    acc_table2.append(ave_acc)
    fgt_table2.append(ave_fgt)
    print(f"MEMO: Acc matrix: {acc_matrix_MEMO_acc2}    Fe Average Acc: {ave_acc}    Fe Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_DualBuffer_acc0)
    acc_table0.append(ave_acc)
    fgt_table0.append(ave_fgt)
    print(f"DualBuffer: Acc matrix: {acc_matrix_DualBuffer_acc0}    Zn Average Acc: {ave_acc}    Zn Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_DualBuffer_acc1)
    acc_table1.append(ave_acc)
    fgt_table1.append(ave_fgt)
    print(f"DualBuffer: Acc matrix: {acc_matrix_DualBuffer_acc1}    Pb Average Acc: {ave_acc}    Pb Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_DualBuffer_acc2)
    acc_table2.append(ave_acc)
    fgt_table2.append(ave_fgt)
    print(f"DualBuffer: Acc matrix: {acc_matrix_DualBuffer_acc2}    Fe Average Acc: {ave_acc}    Fe Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_DualNet_acc0)
    acc_table0.append(ave_acc)
    fgt_table0.append(ave_fgt)
    print(f"DualNet: Acc matrix: {acc_matrix_DualNet_acc0}    Zn Average Acc: {ave_acc}    Zn Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_DualNet_acc1)
    acc_table1.append(ave_acc)
    fgt_table1.append(ave_fgt)
    print(f"DualNet: Acc matrix: {acc_matrix_DualNet_acc1}    Pb Average Acc: {ave_acc}    Pb Average Forgetting: {ave_fgt}")
    ave_acc, ave_fgt = CL_metrics(acc_matrix_DualNet_acc2)
    acc_table2.append(ave_acc)
    fgt_table2.append(ave_fgt)
    print(f"DualNet: Acc matrix: {acc_matrix_DualNet_acc2}    Fe Average Acc: {ave_acc}    Fe Average Forgetting: {ave_fgt}")
    #save acc_tables as csv files
    np.savetxt("save/acc_table_Zn4.csv", acc_table0, delimiter=",")
    np.savetxt("save/acc_table_Pb4.csv", acc_table1, delimiter=",")
    np.savetxt("save/acc_table_Fe4.csv", acc_table2, delimiter=",")
    np.savetxt("save/fgt_table_Zn4.csv", fgt_table0, delimiter=",")
    np.savetxt("save/fgt_table_Pb4.csv", fgt_table1, delimiter=",")
    np.savetxt("save/fgt_table_Fe4.csv", fgt_table2, delimiter=",")
if __name__ == "__main__":
    main()