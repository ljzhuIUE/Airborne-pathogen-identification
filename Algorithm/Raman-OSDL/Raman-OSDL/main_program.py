
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import argparse
import sys

#from models import *
sys.path.append("..")
#import backbones.cifar as models
from Utils import adjust_learning_rate, progress_bar, Logger, mkdir_p, Evaluation
from openmax import compute_train_score_and_mavs_and_dists, fit_weibull, openmax
from Modelbuilder import Network


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

#When using a trained model for test another dataset
parser.add_argument('--test_path', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--other_path', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--arch', default='ANN')

parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--es', default=100, type=int, help='epoch size')

#Classes number in training set and test set
parser.add_argument('--train_class_num', default=6, type=int, help='Classes used in training')
parser.add_argument('--test_class_num', default=7, type=int, help='Classes used in testing')
parser.add_argument('--includes_all_train_class', default=True,  action='store_true',
                    help='If required all known classes included in testing')
parser.add_argument('--evaluate', action='store_true',
                    help='Evaluate without training')

#Parameters for weibull distribution fitting.
parser.add_argument('--weibull_tail', default=20, type=int, help='Classes used in testing')
parser.add_argument('--weibull_alpha', default=3, type=int, help='Classes used in testing')
parser.add_argument('--weibull_threshold', default=0.98, type=float, help='Classes used in testing')

args = parser.parse_args()

global close_acc_openmax
global acc_openmax

global close_acc_softmax
global acc_softmax

close_acc_openmax = 0
close_acc_softmax = 0
acc_openmax=0
acc_softmax=0
def main():
   
    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    print(device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # checkpoint
    args.checkpoint = './checkpoints/models/' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing data..')

    
    
    from dataset import MyCustomDataset
    from dataset_other import MyCustomDataset_other
   

    # Model
    print('==> Building model..')
    net = Network(backbone='ANN', num_classes=args.train_class_num)
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    
    global resume_path 
    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):
            resume_path=args.resume.split("/")[-2]
            
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            # best_acc = checkpoint['acc']
            # print("BEST_ACCURACY: "+str(best_acc))
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss','Train Acc.', 'Test Loss', 'Test Acc.'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    epoch=0
    
    
    '''
            ------------------------------------Reading Dataset--------------------------------------
    '''
    #Data for training and test, including five pathogen classes and 'unknown for train' class
    global set_data 
    set_data = "/home/zhulongji/openmax_v2/Openmax/OpenMax/data/new_data/train_test"
    #Data for test only ('unknown for test' class, which has never been seen in the training set)
    global other_data
    other_data = "/home/zhulongji/openmax_v2/Openmax/OpenMax/data/new_data/other"

    if not args.evaluate:
      #Cross validation
      for k in range(0,10):   
            data_train = MyCustomDataset(k=k,root= set_data,transforms=None,mode="train")
            data_test = MyCustomDataset(k=k,root= set_data,transforms=None,mode="test")
            data_other = MyCustomDataset_other(root= other_data,transforms=None,label=args.train_class_num)
            
            trainloader = DataLoader(data_train, batch_size=64, shuffle=True)
            testloader = DataLoader(data_test, batch_size=50, shuffle=False)
            otherloader = DataLoader(data_other, batch_size=50, shuffle=False)

            save_path = os.path.join(args.checkpoint,"{}折".format(str(k)))
            if not os.path.isdir(save_path):
              mkdir_p(save_path)
            for epoch in range(start_epoch, args.es):
                print('\nEpoch: %d   Learning rate: %f' % (epoch+1, optimizer.param_groups[0]['lr']))
                adjust_learning_rate(optimizer, epoch, args.lr)
                train_loss, train_acc = train(net,trainloader,optimizer,criterion,device,k)
                save_model(net, None, epoch, os.path.join(save_path,'last_model.pth'))
                test_loss, test_acc = 0, 0
                #
                logger.append([epoch+1, optimizer.param_groups[0]['lr'], train_loss, train_acc, test_loss, test_acc])

                # compute_train_score_and_mavs_and_dists
                if epoch % 49 == 0 and epoch!=0:
                    test(epoch, net, trainloader, testloader, otherloader,criterion, device, k , False)
            test(epoch, net, trainloader, testloader, otherloader,criterion, device, k, True)
      logger.close()

    else:
          data_train = MyCustomDataset(k=int(resume_path[0]),root= set_data,transforms=None,mode="train")
          trainloader = DataLoader(data_train, batch_size=64, shuffle=True)

          data_last = MyCustomDataset(k=0, root= args.test_path,transforms=None,mode="last")
          lastloader = DataLoader(data_last, batch_size=50, shuffle=False)
          
          data_other = MyCustomDataset_other(root= args.other_path,transforms=None,label=args.train_class_num)
          otherloader = DataLoader(data_other, batch_size=50, shuffle=False)

          test(epoch, net, trainloader, lastloader, otherloader,criterion, device, int(resume_path[0]), True)


def train(net,trainloader,optimizer,criterion,device,k):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        _, outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | %d折'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total,k))
    return train_loss/(batch_idx+1), correct/total


def test(epoch, net,trainloader,  testloader, otherloader, criterion, device ,k ,is_save):
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    scores, labels = [], []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = net(inputs)
            scores.append(outputs)
            labels.append(targets)
            progress_bar(batch_idx, len(testloader))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(otherloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = net(inputs)
            scores.append(outputs)
            labels.append(targets)
            progress_bar(batch_idx, len(otherloader))
    
    # Get the prdict results.
    scores = torch.cat(scores,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    scores = np.array(scores)[:, np.newaxis, :]
    labels = np.array(labels)

    # Fit the weibull distribution from training data.
    print("Fittting Weibull distribution...")
    _, mavs, dists = compute_train_score_and_mavs_and_dists(args.train_class_num, trainloader, device, net)
    categories = list(range(0, args.train_class_num))
    weibull_model = fit_weibull(mavs, dists, categories, args.weibull_tail, "euclidean")

    pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
    score_softmax, score_openmax = [], []

    for score in scores:
        so, ss = openmax(weibull_model, categories, score,
                         0.5, args.weibull_alpha, "euclidean")  # openmax_prob, softmax_prob
        pred_softmax.append(np.argmax(ss))
        pred_softmax_threshold.append(np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)

        pred_openmax.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
        score_softmax.append(ss)
        score_openmax.append(so)
   

    print("Evaluation...")

    print(f"_________________________________________")
    print("Current is {} fold".format(k))

    eval_softmax = Evaluation(pred_softmax, labels, args.train_class_num ,score_softmax)
    torch.save(eval_softmax, os.path.join(args.checkpoint, 'eval_softmax.pkl'))

    print(f"Softmax close_acc is %.3f" % (eval_softmax.acc_close))
    print(f"Softmax accuracy is %.3f" % (eval_softmax.accuracy))
    print(f"Softmax F1 is %.3f" % (eval_softmax.f1_measure))
    print(f"Softmax f1_macro is %.3f" % (eval_softmax.f1_macro))
    print(f"Softmax f1_macro_weighted is %.3f" % (eval_softmax.f1_macro_weighted))
    print(f"Softmax area_under_roc is %.3f" % (eval_softmax.area_under_roc))

    
    print(f"_________________________________________")
    print("Current test {} fold".format(k))
    eval_softmax_threshold = Evaluation(pred_softmax_threshold, labels, args.train_class_num ,score_softmax)
    torch.save(eval_softmax_threshold, os.path.join(args.checkpoint, 'eval_softmax_threshold.pkl'))
    print(f"SoftmaxThreshold close_acc is %.3f" % (eval_softmax_threshold.acc_close))
    print(f"SoftmaxThreshold accuracy is %.3f" % (eval_softmax_threshold.accuracy))
    print(f"SoftmaxThreshold F1 is %.3f" % (eval_softmax_threshold.f1_measure))
    print(f"SoftmaxThreshold f1_macro is %.3f" % (eval_softmax_threshold.f1_macro))
    print(f"SoftmaxThreshold f1_macro_weighted is %.3f" % (eval_softmax_threshold.f1_macro_weighted))
    print(f"SoftmaxThreshold area_under_roc is %.3f" % (eval_softmax_threshold.area_under_roc))
    print(f"_________________________________________")
    print("Current test {} fold".format(k))
    eval_openmax = Evaluation(pred_openmax, labels, args.train_class_num,score_openmax)
    torch.save(eval_openmax, os.path.join(args.checkpoint, 'eval_openmax.pkl'))
    print(f"OpenMax close_acc is %.3f" % (eval_openmax.acc_close))
    print(f"OpenMax accuracy is %.3f" % (eval_openmax.accuracy))
    print(f"OpenMax F1 is %.3f" % (eval_openmax.f1_measure))
    print(f"OpenMax f1_macro is %.3f" % (eval_openmax.f1_macro))
    print(f"OpenMax f1_macro_weighted is %.3f" % (eval_openmax.f1_macro_weighted))
    print(f"OpenMax area_under_roc is %.3f" % (eval_openmax.area_under_roc))
    print(f"_________________________________________")
    if is_save :
        global close_acc_openmax
        global acc_openmax

        global close_acc_softmax
        global acc_softmax
        
        close_acc_softmax+=eval_softmax_threshold.acc_close
        acc_softmax+=eval_softmax_threshold.accuracy
        

        close_acc_openmax+=eval_openmax.acc_close
        acc_openmax+=eval_openmax.accuracy

    if k==9 and is_save:
        print("Average score of 10 folds SoftmaxThreshold close_acc is %.3f" % (close_acc_softmax/10))
        print("Average score of 10 folds SoftmaxThreshold accuracy is %.3f" % (acc_softmax/10))

        print("Average score of 10 folds OpenMax close_acc is %.3f" % (close_acc_openmax/10))
        print("Average score of 10 folds OpenMax accuracy is %.3f" % (acc_openmax/10))
        

def save_model(net, acc, epoch, path):
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'testacc': acc,
        'epoch': epoch,
    }
    torch.save(state, path)

if __name__ == '__main__':
    main()

