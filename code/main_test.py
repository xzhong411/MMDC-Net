import glob

import torch
import torch.nn as nn

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tensorboard.program import TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint

from core.models import UNet
import os
import argparse

import torch.nn.functional as F
from core.unet_parts import RecallCrossEntropy
from core.utils import calculate_Accuracy, get_model, get_data
from pylab import *
import random

from split_data import get_kflod_data

plt.switch_backend('agg')

# --------------------------------------------------------------------------------
model_name= 'MNet'
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')

parser.add_argument('--epochs', type=int, default=100,
                    help='the epochs of this run')
parser.add_argument('--n_class', type=int, default=2,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')
parser.add_argument('--lr', type=float, default=0.00015,
                    help='initial learning rate')
parser.add_argument('--GroupNorm', type=bool, default=True,
                    help='decide to use the GroupNorm')
parser.add_argument('--BatchNorm', type=bool, default=False,
                    help='decide to use the BatchNorm')
# ---------------------------
# model
# ---------------------------
#parser.add_argument('--data_path', type=str, default='../data/CHASEDB1_1',help='dir of the all img')
#parser.add_argument('--model_save', type=str, default='../models/chase_best_model.pth',help='dir of the model.pth')

# parser.add_argument('--data_path', type=str, default='../data/CVC-612',help='dir of the all img')
# parser.add_argument('--model_save', type=str, default='../models/cvc_best_model.pth',help='dir of the model.pth')

# parser.add_argument('--data_path', type=str, default='../data/CVC-ClinicDB',help='dir of the all img')
# parser.add_argument('--model_save', type=str, default='../models/clin_best_model.pth',help='dir of the model.pth')

# parser.add_argument('--data_path', type=str, default='../data/DRHAGIS',help='dir of the all img')
# parser.add_argument('--model_save', type=str, default='../models/drha_best_model.pth',help='dir of the model.pth')

parser.add_argument('--data_path', type=str, default='../data/HRF',help='dir of the all img')
parser.add_argument('--model_save', type=str, default='../models/HRF_best_model.pth',help='dir of the model.pth')

# parser.add_argument('--data_path', type=str, default='../data/Kvasir-SEG',help='dir of the all img')
# parser.add_argument('--model_save', type=str, default='../models/SEG_best_model.pth',help='dir of the model.pth')

#parser.add_argument('--data_path', type=str, default='../data/DRIVE_1',help='dir of the all img')
#parser.add_argument('--model_save', type=str, default='../models/dri_best_model.pth',help='dir of the model.pth')

# parser.add_argument('--data_path', type=str, default='../data/STARE_1',help='dir of the all img')
# parser.add_argument('--model_save', type=str, default='../models/stare_best_model.pth',help='dir of the model.pth')
parser.add_argument('--my_description', type=str, default='',
                    help='some description define your training')
parser.add_argument('--batch_size', type=int, default=2,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=512,
                    help='the training img size')
# ---------------------------
# GPU
# ---------------------------
parser.add_argument('--use_gpu', type=bool, default=True,
                    help='dir of the all ori img')
parser.add_argument('--gpu_avaiable', type=str, default='0,1,2,3',
                    help='the gpu used')

args = parser.parse_args()
print(args)
# --------------------------------------------------------------------------------


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable

model = UNet(n_channels=3,n_classes=args.n_class)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model).cuda()
if args.use_gpu:
    model.cuda()
    print('GPUs used: (%s)' % args.gpu_avaiable)
    print('------- success use GPU --------')
print("  ''''''''")
EPS = 1e-12
# define path
data_path = args.data_path



# 随机将数据划分为5分。为了确保每次划分的结果都一样，加入随机数种子。
# img_list = get_img_list(args.data_path, flag='training')
# test_img_list = get_img_list(args.data_path, flag='test')
train_img_list,test_img_list = get_kflod_data(args.data_path,Kflod=2)
# print(sorted(train_img_list)==sorted(d)) # 判斷每次身成的數據是否相同

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
# criterion = RecallCrossEntropy()
criterion = nn.NLLLoss2d()
softmax_2d = nn.Softmax2d()

IOU_best = 0

print('This model is %s_%s_%s_%s' % (model_name, args.n_class, args.img_size,args.my_description))
if not os.path.exists(r'../models/%s_%s' % (model_name, args.my_description)):
    os.mkdir(r'../models/%s_%s' % (model_name, args.my_description))

with open(r'../logs/%s_%s.txt' % (model_name, args.my_description), 'w+') as f:
    f.write('This model is %s_%s: ' % (model_name, args.my_description)+'\n')
    f.write('args: '+str(args)+'\n')
    f.write('training lens: '+str(len(train_img_list))+' | test lens: '+str(len(test_img_list)))
    f.write('\n\n---------------------------------------------\n\n')

train_losses = []
train_acces = []
eval_losses = []
eval_acces = []
for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    train_acc = 0
    begin_time = time.time()
    print ('This model is %s_%s_%s_%s' % (
        model_name, args.n_class, args.img_size, args.my_description))
    random.shuffle(train_img_list)

    if 'arg' in args.data_path:
        if (epoch % 10 ==  0) and epoch != 0 and epoch < 400:
            args.lr /= 10
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            # optimizer = torch.optim.Adam(model.parameters(),lr=adjust_learning_rate(optimizer, epoch, args.lr, args.epochs//8, args.epochs, 0.9))
    for i, (start, end) in enumerate(zip(range(0, len(train_img_list), args.batch_size),
                                         range(args.batch_size, len(train_img_list) + args.batch_size,
                                               args.batch_size))):
        path = train_img_list[start:end]
        img, gt, tmp_gt, img_shape,label_ori = get_data(args.data_path, path, img_size=args.img_size, gpu=args.use_gpu)
        optimizer.zero_grad()
        out = model(img)
        out = torch.log(softmax_2d(out) + EPS)
        loss = criterion(out, gt)

        out = torch.log(softmax_2d(out) + EPS)
        loss.backward()
        optimizer.step()

        ppi = np.argmax(out.cpu().data.numpy(), 1)
        tmp_out = ppi.reshape([-1])
        tmp_gt = tmp_gt.reshape([-1])

        my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
        IU,Dice,Acc,Se,Sp,f1= calculate_Accuracy(my_confusion)
        print(str('model: {:s}_{:s} | epoch_batch: {:d}_{:d} | loss: {:f}  | Acc: {:.3f} | Se: {:.3f} | Sp: {:.3f}| f1: {:.3f} | IU: {:.3f}| Dice: {:.3f}'
                  ).format(model_name, args.my_description,epoch, i, loss.item(), Acc,Se,Sp,
                                                                                  f1,IU,Dice))
        train_loss += loss.item()
        train_acc += Acc

    print('training finish, time: %.1f s' % (time.time() - begin_time))
    train_losses.append((train_loss/ len(train_img_list))*args.batch_size)
    train_acces.append((train_acc / len(train_img_list))*args.batch_size)
    if epoch % 10 == 0 and epoch != 0:
        torch.save(model.state_dict(), args.model_save)
        print('success save Nucleus_best model')
    torch.save(model.state_dict(), args.model_save)
    eval_loss = 0
    eval_acc = 0
    ACC = []
    SE = []
    SP = []
    AUC = []
    F1 = []
    meanDice = []
    meanIU = []
    alltime = []
    model.eval()
    for i, img_path in enumerate(test_img_list):
        with torch.no_grad():
            test_img, test_gt, tmp_gt, img_shape,label_ori = get_data(args.data_path, [img_path], img_size=args.img_size, gpu=args.use_gpu)
            out= model(test_img)
            out = torch.log(softmax_2d(out) + EPS)
            loss_1 = criterion(out,test_gt)

            eval_loss +=loss_1.item()
            out = torch.log(softmax_2d(out) + EPS)

            out = F.upsample(out, size=(img_shape[0][0], img_shape[0][1]), mode='bilinear')
            out = out.cpu().data.numpy()

            y_pred = out[:, 1, :, :]
            y_pred = y_pred.reshape([-1])
            ppi = np.argmax(out, 1)

            tmp_out = ppi.reshape([-1])
            tmp_gt = label_ori.reshape([-1])

            my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
            IU,Dice,Acc,Se,Sp,f1= calculate_Accuracy(my_confusion)
            eval_acc += Acc
            Auc = roc_auc_score(tmp_gt, y_pred)
            AUC.append(Auc)
            ACC.append(Acc)
            SE.append(Se)
            SP.append(Sp)
            F1.append(f1)
            meanIU.append(IU)
            meanDice.append(Dice)



            fpr, tpr, thresh = metrics.roc_curve(tmp_gt, y_pred)
            # print(fpr)
            auc = metrics.roc_auc_score(tmp_gt, y_pred)

    eval_losses.append(eval_loss / len(test_img_list))
    eval_acces.append(eval_acc / len(test_img_list))
    print('Acc: %s  |  Se: %s |  Sp: %s |  Auc: %s |  f1: %s  |  IU: %s  |  Dice: %s |  time:%s' % (
        str(np.mean(np.stack(ACC))), str(np.mean(np.stack(SE))), str(np.mean(np.stack(SP))), str(np.mean(np.stack(AUC))),
        str(np.mean(np.stack(F1))),str(np.mean(np.stack(meanIU))),str(np.mean(np.stack(meanDice))),str(np.mean(alltime))))
plt.figure(dpi=600)
x1 = range(0,args.epochs)
x2 = range(0,args.epochs)
y1 = eval_acces
y11 = train_acces
y2=  eval_losses
y22 = train_losses
plt.figure(dpi=600)

plt.plot(x1,y1,label="validate")
plt.plot(x1,y11,label="train")

plt.ylabel('accuracy')
plt.xlabel("epochs")
plt.legend()
plt.savefig('./s1.png')
plt.show()
plt.figure(dpi=600)
plt.plot(x2,y2,label="validate")
plt.plot(x2,y22,label="train")
plt.xlabel("epochs")
plt.ylabel('loss')

plt.legend()
plt.savefig('./s2.png')
plt.show()

