import glob

import torch
import torch.nn as nn

import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from tensorboard.program import TensorBoard


from core.models import MMDC
import os
import argparse

import torch.nn.functional as F
from core.unet_parts import RecallCrossEntropy
from core.utils import calculate_Accuracy, get_model, get_data
from pylab import *


plt.switch_backend('agg')

# --------------------------------------------------------------------------------
model_name= 'MMDC-Net'
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')

parser.add_argument('--epochs', type=int, default=100,
                    help='the epochs of this run')
parser.add_argument('--n_class', type=int, default=2,    #number of class,set to 2
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')
parser.add_argument('--lr', type=float, default=0.00015,     #learning rate
                    help='initial learning rate')
parser.add_argument('--GroupNorm', type=bool, default=True,
                    help='decide to use the GroupNorm')
parser.add_argument('--BatchNorm', type=bool, default=False,
                    help='decide to use the BatchNorm')
# ---------------------------
# model
# ---------------------------

parser.add_argument('--data_path', type=str, default='../data/STARE',help='dir of the all img')
parser.add_argument('--model_save', type=str, default='../models/stare_best_model.pth',help='dir of the model.pth')
parser.add_argument('--my_description', type=str, default='',
                    help='some description define your training')
parser.add_argument('--batch_size', type=int, default=6,
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

model = MMDC(n_channels=3,n_classes=args.n_class)
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
train_img_list=glob.glob(os.path.join(data_path, 'train/image/*.png'))   #get the training images
test_img_list=glob.glob(os.path.join(data_path, 'test/image/*.png'))    #get the testing images


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

criterion = RecallCrossEntropy()   #recall loss
softmax_2d = nn.Softmax2d()

IOU_best = 0

print('This model is %s_%s_%s_%s' % (model_name, args.n_class, args.img_size,args.my_description))
if not os.path.exists(r'../models/%s_%s' % (model_name, args.my_description)):
    os.mkdir(r'../models/%s_%s' % (model_name, args.my_description))   #Model parameter description

#Model run logging
with open(r'../logs/%s_%s.txt' % (model_name, args.my_description), 'w+') as f:
    f.write('This model is %s_%s: ' % (model_name, args.my_description)+'\n')
    f.write('args: '+str(args)+'\n')
    f.write('training lens: '+str(len(train_img_list))+' | test lens: '+str(len(test_img_list)))
    f.write('\n\n---------------------------------------------\n\n')
#Model training
for epoch in range(args.epochs):
    model.train()

    begin_time = time.time()
    print ('This model is %s_%s_%s_%s' % (
        model_name, args.n_class, args.img_size, args.my_description))
    random.shuffle(train_img_list)

    if 'arg' in args.data_path:
        if (epoch % 10 ==  0) and epoch != 0 and epoch < 400:  #Automatically adjust the learning rate
            args.lr /= 10
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            # optimizer = torch.optim.Adam(model.parameters(),lr=adjust_learning_rate(optimizer, epoch, args.lr, args.epochs//8, args.epochs, 0.9))
    for i, (start, end) in enumerate(zip(range(0, len(train_img_list), args.batch_size),
                                         range(args.batch_size, len(train_img_list) + args.batch_size,
                                               args.batch_size))):
        path = train_img_list[start:end]
        img, gt, tmp_gt, img_shape,label_ori = get_data(args.data_path, path, img_size=args.img_size, gpu=args.use_gpu)
        optimizer.zero_grad()
        out = model(img)      #Get the final output of the image
        # out = torch.log(softmax_2d(out) + EPS)
        loss = criterion(out, gt)

        # out = torch.log(softmax_2d(out) + EPS)
        loss.backward()
        optimizer.step()

        ppi = np.argmax(out.cpu().data.numpy(), 1)
        tmp_out = ppi.reshape([-1])
        tmp_gt = tmp_gt.reshape([-1])

        my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)   #Get the confusion
        IU,Dice,Acc,Se,Sp,f1= calculate_Accuracy(my_confusion)

	#Print model real-time metrics
        print(str('model: {:s}_{:s} | epoch_batch: {:d}_{:d} | loss: {:f}  | Acc: {:.3f} | Se: {:.3f} | Sp: {:.3f}| f1: {:.3f} | IU: {:.3f}| Dice: {:.3f}'
                  ).format(model_name, args.my_description,epoch, i, loss.item(), Acc,Se,Sp,
                                                                                  f1,IU,Dice))

    print('training finish, time: %.1f s' % (time.time() - begin_time))
    if epoch % 10 == 0 and epoch != 0:
        torch.save(model.state_dict(), args.model_save)
        print('success save Nucleus_best model')

