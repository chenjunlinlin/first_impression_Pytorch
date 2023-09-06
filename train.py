from model import model_LSTM
from model import init_model
import dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as dataloader
import tqdm
import json 
import os
from tensorboardX import SummaryWriter
from options import *
from datetime import datetime

args = get_args()


def train(epoch):
    model.train()
    train_loss = 0
    for video_feat, audio_feat, label in tqdm.tqdm(train_loader, desc='epoch:{}'.format(epoch)):
        if iscuda :
            video_feat, audio_feat, label = video_feat.cuda(), audio_feat.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(video_feat, audio_feat)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*label.size(0)
    train_loss = train_loss/len(train_loader.dataset)
    acc = (1 - (output - label).abs()).sum(0) / args.batch_size 
    writer_t.add_scalar("loss", train_loss, epoch)
    writer_t.add_scalar("MACC", acc.sum() / 5, epoch)
    print('Epoch: {} \tTraining Loss: {:.6f} \tLr :{:.6f}'.format(epoch, train_loss, optimizer.param_groups[0]['lr']))
    print('Epoch: {} \tTraining MACC: {:.6f}'.format(epoch, acc.sum() / 5))

    return loss, acc.sum() / 5

def val(epoch):
    model.eval()
    val_loss = 0
    for video_feat, audio_feat, label in tqdm.tqdm(val_loader, desc='epoch:{}'.format(epoch)):
        if iscuda :
            video_feat, audio_feat, label = video_feat.cuda(), audio_feat.cuda(), label.cuda()
        output = model(video_feat, audio_feat)
        loss = criterion(output, label)
        val_loss += loss.item()*label.size(0)
    val_loss = val_loss/len(val_loader.dataset)
    acc = (1 - (output - label).abs()).sum(0) / args.batch_size
    writer_v.add_scalar("loss", val_loss, epoch)
    writer_v.add_scalar("MACC", acc.sum() / 5, epoch)
    print('Epoch: {} \tvaling Loss: {:.6f} '.format(epoch, val_loss))
    print('Epoch: {} \tvaling MACC: {:.6f}'.format(epoch, acc.sum() / 5))

if __name__=='__main__' :

    iscuda = args.iscuda
    if torch.cuda.is_available and iscuda :
        iscuda = True
    else : iscuda = False

    trainset = dataset.MY_DATASET(video_dir=args.train_video_dir, audio_dir=args.train_audio_dir, csv_file=args.train_csv_path, n=args.N)
    valset = dataset.MY_DATASET(video_dir=args.val_video_dir, audio_dir=args.val_audio_dir, csv_file=args.val_csv_path, n=args.N)
    

    train_loader = dataloader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    val_loader = dataloader(valset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True, pin_memory=True)

    writer_t = SummaryWriter("./logs/{}/train".format(args.name))
    writer_v = SummaryWriter("./logs/{}}/val".format(args.name))
    model = model_LSTM.BIO_MODEL_LSTM()
    # model = vol_model.VOL_MODEL()
    writer_t.add_graph(model=model.eval(), input_to_model = (torch.rand(1,3, 6, 112, 112), torch.rand(1, 6, 68)))
    print('--------模型初始化---------')
    if args.pretrain and os.path.exists(args.best_model_save_dir):
        model.load_state_dict(torch.load(args.best_model_save_dir))
        print("预训练模型加载成功：{}".format(args.best_model_save_dir))
    else:
        init_model.initialize_weights(model)
        print("无预训练模型")
    if iscuda :
        print("load model to cuda")
        model = nn.DataParallel(model.cuda(), device_ids=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay = args.weight_decay, lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

    with open(args.best_model, 'r+') as f:
        best_model = json.load(f)
        for epoch in range(args.epochs):
            loss, MACC = train(epoch)
            if loss < best_model['loss'] or best_model['loss'] == None:
                best_model['name'] = args.name
                best_model['loss'] = loss
                best_model['MACC'] = MACC
                best_model['path'] = os.path.join(args.best_model_save_dir , "best_{}.pth".format(args.name))
                best_model['time'] = datetime.now
                torch.save(model.state_dict(), os.path.join(args.best_model_save_dir , "best_{}.pth".format(args.name)))
            if epoch % 5 == 0:
                torch.save(model.state_dict(), os.path.join(args.model_save_dir, args.name) + '{}.pth'.format(epoch))
            scheduler.step()
            val(epoch)
        f.seek(0)
        json.dump(best_model, f, indent=4, ensure_ascii=False)
        f.close()
    writer_t.close()
    writer_v.close()

