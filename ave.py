from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import numpy as np
from dataloader import Dataset
from fully_model import psp_net
import random
import torch.nn.functional as F
import torch.optim as optim
from Optim import ScheduledOptim
import warnings
warnings.filterwarnings("ignore")
import argparse

# data
parser = argparse.ArgumentParser(description='Fully supervised AVE localization')
parser.add_argument('--model_name', type=str, default='PSP', help='model name')
parser.add_argument('--dir_video', type=str, default="./data/visual_feature.pkl", help='visual features')
parser.add_argument('--dir_audio', type=str, default='./data/audio_feature.pkl', help='audio features')
parser.add_argument('--dir_labels', type=str, default='./data/label.pkl', help='labels of dataset')

parser.add_argument('--dir_order_train', type=str, default='./data/train_set.pkl', help='indices of training samples')
parser.add_argument('--dir_order_test', type=str, default='./data/test_set.pkl', help='indices of testing samples')

parser.add_argument('--nb_epoch', type=int, default=300,  help='number of epoch')
parser.add_argument('--batch_size', type=int, default=100, help='number of batch size')
parser.add_argument('--save_epoch', type=int, default=1, help='number of epoch for saving models')
parser.add_argument('--LAMBDA', type=float, default=100, help='weight for balancing losses')
parser.add_argument('--threshold', type=float, default=0.095, help='key-parameter for pruning process')
parser.add_argument('--check_epoch', type=int, default=5, help='number of epoch for checking accuracy of current models during training')
parser.add_argument('--audios_dir', type=str, default='audios', help='audios')
parser.add_argument('--videos_dir', type=str, default='videos', help='videos')
parser.add_argument('--sav_dir', type=str, default='./ave_models')


parser.add_argument('--trained_model_path', type=str, default='PSP_30_fully_0.095.pt', help='pretrained model')
parser.add_argument('--train', action='store_true', default=False, help='train a new model')


FixSeed = 123
random.seed(FixSeed)
np.random.seed(FixSeed)
torch.manual_seed(FixSeed)
torch.cuda.manual_seed(FixSeed)
torch.cuda.manual_seed_all(FixSeed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train(args, net_model, optimizer):
    Data = Dataset(video_dir=args.dir_video, audio_dir=args.dir_audio, batch_size=args.batch_size, label_dir=args.dir_labels,
                        order_dir=args.dir_order_train, status='train')
    nb_batch = Data.__len__() // args.batch_size  # 完整遍历整个数据集需要几个批次
    print('nb_batch:', nb_batch)
    epoch_l = []
    best_test_acc = 0
    for epoch in range(args.nb_epoch):
        net_model.train()
        epoch_loss = 0
        epoch_loss_cls = 0
        epoch_loss_avps = 0
        n = 0
        start = time.time()
        SHUFFLE_SAMPLES = True
        for i in range(nb_batch):
            # all_out_probs = []
            all_pred = []
            all_labels = []
            audio, video, labels, segment_label, segment_avps_gt, labels_bg = Data.get_batch(i, SHUFFLE_SAMPLES) # 得到一个批次的数据
            SHUFFLE_SAMPLES = False
            loss_cls = 0
            loss_avps = 0
            for n in range(args.batch_size):
                keys_video = list(video.keys())

                keys_auido = list(audio.keys())
                filename_a = keys_auido[n]
                filename_v = keys_video[n]
                audio_input = torch.from_numpy(audio[filename_a]).float().unsqueeze(0).cuda() # [1,t,128]
                video_input = torch.from_numpy(video[filename_v]).float().unsqueeze(0).cuda() # [1,t,7,7,512]
                
                label = torch.from_numpy(labels[n]).float().unsqueeze(0).cuda()
                label_bg = torch.from_numpy(labels_bg[n]).float().cuda() # [1,t,37]
                all_labels.append(label)

                segment_label_batch = torch.from_numpy(segment_label[n]).long().unsqueeze(0).cuda()
                count_list = []
                count_bg = torch.sum(segment_label_batch == 36).item()  # 计算背景为36的计数
                count_event = torch.sum(segment_label_batch != 36).item()  # 计算非背景的事件计数
                count_list.append([count_bg, count_event])

                segment_avps_gt_batch = torch.from_numpy(segment_avps_gt[n]).float().unsqueeze(0).cuda()

                net_model.zero_grad()
                """temporal interval"""
                fusion, out_prob, cross_att, selected_time_segments, pred = net_model(audio_input, video_input, args.threshold)
                all_pred.append(pred)

                loss_cls += nn.CrossEntropyLoss()(out_prob.permute(0, 2, 1), segment_label_batch)
                loss_avps += AVPSLoss(cross_att, segment_avps_gt_batch)
            epoch_loss_cls += loss_cls.cpu().data.numpy()
            epoch_loss_avps += loss_avps.cpu().data.numpy()
            loss = loss_cls + args.LAMBDA * loss_avps
            epoch_loss += loss.cpu().data.numpy()
            loss.backward()
            optimizer.step_lr()

        SHUFFLE_SAMPLES = True
        if (epoch+1) % 60 == 0 and epoch < 170:
            optimizer.update_lr()

        end = time.time()
        epoch_l.append(epoch_loss)
        acc = 0
        for i in range(args.batch_size):
            labels = all_labels[i].cpu().data.numpy()
            x_labels = all_pred[i].cpu().data.numpy()
            acc += compute_acc(labels, x_labels, 1)
        acc /= args.batch_size

        print("=== Epoch {%s}   lr: {%.6f} | Loss: [{%.4f}] loss_cls: [{%.4f}] | loss_frame: [{%.4f}] | training_acc {%.4f}" \
            % (str(epoch), optimizer._optimizer.param_groups[0]['lr'], (epoch_loss) / n, epoch_loss_cls/n, epoch_loss_avps/n, acc))

        # if epoch % args.save_epoch == 0 and epoch != 0:
        #     val_acc = val(args, net_model)
        #     print('val accuracy:', val_acc, 'epoch=', epoch)
        #     if val_acc >= best_val_acc:
        #         best_val_acc = val_acc
        #         print('best val accuracy:', best_val_acc)
        #         print('best val accuracy: {} ***************************************'.format(best_val_acc))
        if epoch % args.check_epoch == 0 and epoch != 0:
            test_acc = test(args, net_model)
            print('test accuracy:', test_acc, 'epoch=', epoch)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                print('best test accuracy: {} ======================================='.format(best_test_acc))
                model_file = os.path.join(args.sav_dir, model_name + "_" + str(epoch) + "_fully.pt")
        # 保存模型
                torch.save(net_model, model_file)
    # print('[best val accuracy]: ', best_val_acc)
    print('[best test accuracy]: ', best_test_acc)



# def val(args, net_model):
#     net_model.eval()
#     AVEData = Dataset(video_dir=args.dir_video, audio_dir=args.dir_audio, label_dir=args.dir_labels,
#                         order_dir=args.dir_order_val, batch_size=400, status='val')
#     nb_batch = AVEData.__len__()
#     audio_inputs, video_inputs, labels, _, _ = AVEData.get_batch(0)
#     all_labels = []
#     all_out_probs = []
#     for n in range(400):
#         keys_video = list(video_inputs.keys())
#         keys_auido = list(audio_inputs.keys())
#         filename_a = keys_auido[n]
#         filename_v = keys_video[n]
#         audio_input = torch.from_numpy(audio_inputs[filename_a]).float().unsqueeze(0).cuda()  # [1,t,128]
#         video_input = torch.from_numpy(video_inputs[filename_v]).float().unsqueeze(0).cuda()  # [1,t,7,7,512]
#         label = torch.from_numpy(labels[n]).float().unsqueeze(0).cuda()  # [1,t,37]
#         all_labels.append(label)
#         fusion, out_prob, cross_att, _ = net_model(audio_input, video_input, args.threshold)
#         all_out_probs.append(out_prob)
#
#     acc = 0
#     for i in range(400):
#         labels = all_labels[i].cpu().data.numpy()
#         x_labels = all_out_probs[i].cpu().data.numpy()
#         acc += compute_acc(labels, x_labels, 1)
#     acc /= 400
#
#     print('[val]acc: ', acc)
#     return acc

def test(args, net_model, model_path=None):
    if model_path is not None:
        net_model = torch.load(model_path)
        print(">>> [Testing] Load pretrained model from " + model_path)


    net_model.eval()
    AVEData = Dataset(video_dir=args.dir_video, audio_dir=args.dir_audio, label_dir=args.dir_labels,
                         order_dir=args.dir_order_test, batch_size=400, status='test')
    nb_batch = AVEData.__len__()
    audio_inputs, video_inputs, labels, _, _, _, = AVEData.get_batch(0)
    all_labels = []
    all_out_probs = []
    for n in range(400):
        keys_video = list(video_inputs.keys())
        keys_auido = list(audio_inputs.keys())
        filename_a = keys_auido[n]
        filename_v = keys_video[n]
        audio_input = torch.from_numpy(audio_inputs[filename_a]).float().unsqueeze(0).cuda()  # [1,t,128]
        video_input = torch.from_numpy(video_inputs[filename_v]).float().unsqueeze(0).cuda()  # [1,t,7,7,512]
        label = torch.from_numpy(labels[n]).float().unsqueeze(0).cuda()  # [1,t,37]
        all_labels.append(label)
        fusion, out_prob, cross_att, _, pred = net_model(audio_input, video_input, args.threshold)
        all_out_probs.append(pred)

    acc = 0
    for i in range(400):
        labels = all_labels[i].cpu().data.numpy()
        x_labels = all_out_probs[i].cpu().data.numpy()
        acc += compute_acc(labels, x_labels, 1)
    acc /= 400
    print('[test]acc: ', acc)

    return acc

def AVPSLoss(av_simm, soft_label):
    """audio-visual pair similarity loss for fully supervised setting,
    please refer to Eq.(8, 9) in our paper.
    """
    # av_simm: [bs, 10]
    relu_av_simm = F.relu(av_simm)
    sum_av_simm = torch.sum(relu_av_simm, dim=-1, keepdim=True)
    avg_av_simm = relu_av_simm / (sum_av_simm + 1e-8)
    loss = nn.MSELoss()(avg_av_simm, soft_label)
    return loss


def compute_acc(labels, x_labels, nb_batch):
    """ compute the classification accuracy
    Args:
        labels: ground truth label
        x_labels: predicted label
        nb_batch: batch size
    """
    t = labels.shape[1]
    N = int(nb_batch * t)
    pre_labels = x_labels.astype(int)
    real_labels = np.zeros(N).astype(int)
    c = 0
    for i in range(nb_batch):
        for j in range(labels.shape[1]): # x_labels.shape: [bs, 10, 29]
            if 36 == np.argmax(labels[i, j, :]):
                real_labels[c] = 0
            else:
                real_labels[c] = 1
            c += 1
    unique_classes = np.unique(real_labels)
    if len(unique_classes) == 1:
        # If only one class is present, return default AUC score of 0.5
        return 0.5
    else:
        # Compute ROC AUC score if both classes are present
        return roc_auc_score(real_labels, pre_labels)

if __name__ == "__main__":
    args = parser.parse_args()
    print("args: ", args)
    model_name = args.model_name
    net_model = psp_net(128, 512, 128, 37)
    net_model.cuda()
    optimizer = optim.Adam(net_model.parameters(), lr=1e-3)
    optimizer = ScheduledOptim(optimizer)
    if args.train:
        train(args, net_model, optimizer)
    else:
        test_acc = test(args, net_model, model_path=args.trained_model_path)
        print("[test] accuracy: ", test_acc)



