from model import model_LSTM
from data import dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as dataloader
import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from model import network
import options
import json
from datetime import datetime
from model.network import set_parameter_requires_grad
from utils import logs, train_utils
from utils.extract_flows import mkdir_p

import warnings
warnings.filterwarnings("ignore")

args = options.get_args()
train_utils.get_cudainfo()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(epoch):
    model.train()
    train_loss = 0
    train_Blloss = 0
    MACCS = 0
    total_iter = len(train_loader.dataset) // args.batch_size
    for video_feat, audio_feat, label in tqdm.tqdm(train_loader, desc='epoch:{}'.format(epoch)):
        if args.iscuda:
            # video_feat, gloaudio_feat, audio_feat, label = video_feat.cuda(
            # ), gloaudio_feat.cuda(), audio_feat.cuda(), label.cuda()
            video_feat, audio_feat, label = video_feat.cuda(
            ), audio_feat.cuda(), label.cuda()
        if epoch == 15 and args.backbone.startswith("resnet"):
            model1 = model.module if args.DP else model
            set_parameter_requires_grad(model=model1, freezing=False)
        optimizer.zero_grad()
        output = model(video_feat, audio_feat)
        loss = criterion1(output, label)
        if args.gama == 0:
            bel_loss = torch.tensor(0)
        else:
            bel_loss = criterion2(output, label)
        total_loss = bel_loss + loss
        total_loss.backward()
        optimizer.step()
        train_loss += loss.item()*label.size(0)
        train_Blloss += bel_loss.item()*label.size(0)
        acc = (1 - (output.detach() - label.detach()).abs()
               ).sum(0) / args.batch_size
        MACCS += acc.sum() / 5
    train_loss = train_loss/len(train_loader.dataset)
    writer_t.add_scalar("loss", train_loss, epoch)
    writer_t.add_scalar("Bl_loss", train_Blloss, epoch)
    writer_t.add_scalar("MACC", MACCS / total_iter, epoch)
    writer_t.add_scalar("Lr", optimizer.param_groups[0]['lr'], epoch)
    print('Epoch: {} \tTraining Loss: {:.6f} \tLr :{:.6f}'.format(
        epoch, train_loss, optimizer.param_groups[0]['lr']))
    print('Epoch: {} \tTraining MACC: {:.6f}'.format(epoch, MACCS / total_iter))
    torch.cuda.empty_cache()

    return train_loss, MACCS / total_iter


def val(epoch):
    model.eval()
    val_loss = 0
    MACCS = 0
    total_iter = len(val_loader.dataset) // args.batch_size
    with torch.no_grad():
        for video_feat, audio_feat, label in tqdm.tqdm(val_loader, desc='Epoch: {}'.format(epoch)):
            if args.iscuda:
                # video_feat, gloaudio_feat, audio_feat, label = video_feat.cuda(
                # ), gloaudio_feat.cuda(), audio_feat.cuda(), label.cuda()
                video_feat, audio_feat, label = video_feat.cuda(
                ), audio_feat.cuda(), label.cuda()
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


if __name__ == '__main__':

    train_utils.set_random_seed(seed=args.seed)

    train_loader = train_utils.get_dataloader(cfg=args, is_train=True)
    val_loader = train_utils.get_dataloader(cfg=args, is_train=False)

    exp_name = logs.get_exp_name(args=args)
    print(f"The log of this experiment is saved in “{exp_name}”")

    writer_t = SummaryWriter("{}/{}/train".format(args.logs, exp_name))
    writer_v = SummaryWriter("{}/{}/val".format(args.logs, exp_name))
    model = model_LSTM.BIO_MODEL_LSTM(arg=args)

    model, start_epoch, opt_para = train_utils.get_model(
        cfg=args, model=model, exp_name=exp_name)

    criterion1 = nn.MSELoss().cuda()
    criterion2 = model_LSTM.Bell_loss(gama=args.gama, sita=args.sita)

    optimizer = train_utils.get_opt(cfg=args, model=model, opt_para=opt_para)
    scheduler = train_utils.get_scheduler(cfg=args,
                                          opt=optimizer,
                                          epoch=start_epoch)

    best_model = args.__dict__
    best_model_path = os.path.join(args.logs, exp_name + ".json")

    checkpoint = {
        'net': None,
        'optimizer': None,
        'epoch': None
    }

    mkdir_p(args.model_save_dir)
    mkdir_p(os.path.join(args.model_save_dir, exp_name))

    if start_epoch is None:
        start_epoch = 0

    for epoch in range(start_epoch + 1, args.epochs):
        if os.path.exists(best_model_path):
            with open(best_model_path, 'r') as f:
                logs = json.load(f)
                best_model["loss"] = logs["loss"]
        network.warmup(optimizer=optimizer, Lr=args.lr,
                       total_epoch=15, cur_epoch=epoch)
        loss, MACC = train(epoch)
        if 'loss' not in best_model or loss < best_model['loss']:
            best_model['loss'] = loss
            best_model['MACC'] = MACC.item()
            best_model['path'] = os.path.join(
                args.best_model_save_dir, "best_{}.pth".format(exp_name))
            best_model['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            checkpoint = train_utils.update_checkpoint(checkpoint=checkpoint,
                                                       net_para=model.state_dict(),
                                                       opt_para=optimizer.state_dict(),
                                                       epoch=epoch
                                                       )
            torch.save(checkpoint, os.path.join(
                args.best_model_save_dir, "best_{}.pth".format(exp_name)))
        if epoch % 3 == 0 and epoch != 0:
            val_loss, val_MACC = val(epoch)
            if 'val_loss' not in best_model or val_loss < best_model['val_loss']:
                best_model['val_loss'] = val_loss
                best_model['val_MACC'] = val_MACC.item()
                best_model['epoch_num'] = epoch
                checkpoint = train_utils.update_checkpoint(
                    checkpoint=checkpoint, net_para=model.state_dict(), opt_para=optimizer.state_dict(), epoch=epoch)
            torch.save(checkpoint, os.path.join(
                args.model_save_dir, exp_name, '{:04d}.pth'.format(epoch)))
        with open(best_model_path, 'w') as f:
            json.dump(best_model, f, indent=4, ensure_ascii=False)
        scheduler.step()
    writer_t.close()
    writer_v.close()
