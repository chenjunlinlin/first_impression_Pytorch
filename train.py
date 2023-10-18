from model import model_LSTM
from model import init_model
from data import dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as dataloader
import tqdm
import os
from tensorboardX import SummaryWriter
import options
from collections import OrderedDict
from utils.extract_frames import mkdir_p
import json
from datetime import datetime
from model.network import set_parameter_requires_grad
import math

import warnings
warnings.filterwarnings("ignore")

args = options.get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
devices = [0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(epoch):
    model.train()
    train_loss = 0
    train_Blloss = 0
    MACCS = 0
    total_iter = len(trainset) // args.batch_size
    for video_feat, audio_feat, label in tqdm.tqdm(train_loader, desc='epoch:{}'.format(epoch)):
        if iscuda :
            # video_feat, flow_feat, audio_feat, label = video_feat.cuda(), flow_feat.cuda(), audio_feat.cuda(), label.cuda()
            video_feat, audio_feat, label = video_feat.cuda(), audio_feat.cuda(), label.cuda()
        if epoch == 1:
            set_parameter_requires_grad(model=model.module, freezing=False)
        optimizer.zero_grad()
        output = model(video_feat, audio_feat)
        loss = criterion1(output, label)
        bel_loss = criterion2(output, label)
        total_loss = bel_loss + loss
        total_loss.backward()
        optimizer.step()
        train_loss += loss.item()*label.size(0)
        train_Blloss += bel_loss.item()*label.size(0)
        acc = (1 - (output.detach() - label.detach()).abs()).sum(0) / args.batch_size 
        MACCS += acc.sum() / 5
    train_loss = train_loss/len(train_loader.dataset)
    writer_t.add_scalar("loss", train_loss, epoch)
    writer_t.add_scalar("Bl_loss", train_Blloss, epoch)
    writer_t.add_scalar("MACC", MACCS / total_iter, epoch)
    writer_t.add_scalar("Lr", optimizer.param_groups[0]['lr'], epoch)
    print('Epoch: {} \tTraining Loss: {:.6f} \tBell_Loss : {:.6f} \tLr :{:.6f}'.format(epoch, train_loss, train_Blloss, optimizer.param_groups[0]['lr']))
    print('Epoch: {} \tTraining MACC: {:.6f}'.format(epoch, MACCS / total_iter))
    torch.cuda.empty_cache()

    return train_loss, MACCS / total_iter

def val(epoch):
    model.eval()
    val_loss = 0
    MACCS = 0
    total_iter = len(valset) // args.batch_size
    with torch.no_grad():
        for video_feat, audio_feat, label in tqdm.tqdm(val_loader, desc='Epoch: {}'.format(epoch)):
            if iscuda :
                # video_feat, flow_feat, audio_feat, label = video_feat.cuda(), flow_feat.cuda(), audio_feat.cuda(), label.cuda()
                video_feat, audio_feat, label = video_feat.cuda(), audio_feat.cuda(), label.cuda()
            output = model(video_feat, audio_feat)
            loss = criterion1(output, label) 
            val_loss += loss.item()*label.size(0)
            acc = (1 - (output - label).abs()).sum(0) / args.batch_size 
            MACCS += acc.sum() / 5
        val_loss = val_loss/len(val_loader.dataset)
        writer_v.add_scalar("loss", val_loss, epoch)
        writer_v.add_scalar("MACC", MACCS / total_iter, epoch)
        print('Epoch: {} \tvaling Loss: {:.6f} '.format(epoch, val_loss))
        print('Epoch: {} \tvaling MACC: {:.6f}'.format(epoch, MACCS / total_iter))
        torch.cuda.empty_cache()

    return val_loss, MACCS / total_iter

if __name__=='__main__' :

    iscuda = args.iscuda
    if torch.cuda.is_available and iscuda :
        iscuda = True
    else : iscuda = False

    trainset = dataset.MY_DATASET(video_dir=args.train_video_dir, flow_dir=args.train_flow_dir, audio_dir=args.train_audio_dir, csv_file=args.train_csv_path, num_flow=args.num_flow, n=args.N)
    valset = dataset.MY_DATASET(video_dir=args.val_video_dir, flow_dir=args.val_flow_dir, audio_dir=args.val_audio_dir, csv_file=args.val_csv_path, num_flow=args.num_flow, n=args.N)
    

    train_loader = dataloader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = dataloader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True, pin_memory=True)

    writer_t = SummaryWriter("./logs/{}/train".format(args.name))
    writer_v = SummaryWriter("./logs/{}/val".format(args.name))
    model = model_LSTM.BIO_MODEL_LSTM(arg=args)
    # writer_t.add_graph(model=model, input_to_model = (torch.rand(6, 6, 3, 224, 224), torch.rand(6, 6, 68)))
    # model = vol_model.VOL_MODEL()
    print('--------模型初始化---------')
    if args.pretrain and os.path.exists(os.path.join(args.best_model_save_dir, "best_{}.pth".format(args.name))):
        state_dict = torch.load(os.path.join(args.best_model_save_dir, "best_{}.pth".format(args.name)))
        new_state_dict = OrderedDict()
        for k in state_dict:
            new_k = k.replace('module.', '')
            new_state_dict[new_k] = state_dict[k]
        model.load_state_dict(state_dict=new_state_dict)
        print("预训练模型加载成功：{}".format(os.path.join(args.best_model_save_dir, "best_{}.pth".format(args.name))))
    else:
        # init_model.initialize_weights(model)
        print("无预训练模型")
    if iscuda :
        print("load model to cuda")
        model = nn.DataParallel(model.cuda(), device_ids=devices)
    criterion1 = nn.MSELoss().cuda()
    criterion2 = model_LSTM.Bell_loss(gama=args.gama, sita=args.sita)
    # criterion2 = nn.L1Loss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay = args.weight_decay, lr=args.lr, momentum=args.momentum)
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay = args.weight_decay, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=5, T_mult=2, eta_min=0.001)

    best_model = args.__dict__
    best_model_path = os.path.join(args.logs, args.name + ".json")
    
    for epoch in range(args.epochs):
        if os.path.exists(best_model_path):
            with open(best_model_path, 'r') as f:
                logs = json.load(f)
                best_model["loss"] = logs["loss"]
        loss, MACC = train(epoch)
        if 'loss' not in best_model or loss < best_model['loss']:
            best_model['loss'] = loss
            best_model['MACC'] = MACC.item()
            best_model['path'] = os.path.join(args.best_model_save_dir , "best_{}.pth".format(args.name))
            best_model['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            torch.save(model.state_dict(), os.path.join(args.best_model_save_dir , "best_{}.pth".format(args.name)))
        with open(best_model_path, 'w') as f:
            json.dump(best_model, f, indent=4, ensure_ascii=False)
        if epoch % 5 == 0 and epoch != 0:
            val_loss, val_MACC = val(epoch)
            if 'val_loss' not in best_model or val_loss < best_model['val_loss']:
                best_model['val_loss'] = val_loss
                best_model['val_MACC'] = val_MACC.item()
                best_model['epoch num'] = epoch
            torch.save(model.state_dict(), os.path.join(args.model_save_dir, args.name) + '{}.pth'.format(epoch))
        scheduler.step()
    writer_t.close()
    writer_v.close()

