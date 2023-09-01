import model_LSTM
import vol_model
import init_model
import dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as dataloader
import tqdm
import json 
# from datetime import datetime
from torchinfo import summary
import os
from tensorboardX import SummaryWriter

def train(epoch):
    model.train()
    train_loss = 0
    for video_feat, audio_feat, label in tqdm.tqdm(train_loader, desc='epoch:{}'.format(epoch)):
        if iscuda :
            video_feat, audio_feat, label = video_feat.cuda(), audio_feat.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(video_feat, audio_feat)
        # print(torch.isnan(output).all(), torch.isnan(label).all())
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()*label.size(0)
    train_loss = train_loss/len(train_loader.dataset)
    acc = (1 - (output - label).abs()).sum(0) / batch_size
    writer_t.add_scalar("loss", train_loss, epoch)
    writer_t.add_scalar("MACC", acc.sum() / 5, epoch)
    print('Epoch: {} \tTraining Loss: {:.6f} \tLr :{}'.format(epoch, train_loss, optimizer.param_groups[0]['lr']))
    print('Epoch: {} \tTraining MACC: {:.6f}'.format(epoch, acc.sum() / 5))

    return loss

def val(epoch):
    model.eval()
    val_loss = 0
    for video_feat, audio_feat, label in tqdm.tqdm(val_loader, desc='{}'.format(epoch)):
        if iscuda :
            video_feat, audio_feat, label = video_feat.cuda(), audio_feat.cuda(), label.cuda()
        output = model(video_feat, audio_feat)
        loss = criterion(output, label)
        val_loss += loss.item()*label.size(0)
    val_loss = val_loss/len(val_loader.dataset)
    acc = (1 - (output - label).abs()).sum(0) / batch_size
    writer_v.add_scalar("loss", val_loss, epoch)
    writer_v.add_scalar("MACC", acc.sum() / 5, epoch)
    print('Epoch: {} \tvaling Loss: {:.6f} '.format(epoch, val_loss))
    print('Epoch: {} \tvaling MACC: {:.6f}'.format(epoch, acc.sum() / 5))

if __name__=='__main__' :
    with open('./config.json', 'r') as josnfile:
        config = json.load(josnfile)
    train_csv_path = config["path"]["train_csv_path"]
    val_csv_path = config["path"]["val_csv_path"]
    train_audio_dir = config["path"]["train_audio_dir"]
    train_video_dir = config["path"]["train_video_dir"]
    val_audio_dir = config["path"]["val_audio_dir"]
    val_video_dir = config["path"]["val_video_dir"]
    best_model_save_dir = config["path"]["best_model_save_dir"]
    model_save_dir = config["path"]["model_save_dir"]

    N = config["para"]["N"]
    batch_size = config["para"]["batch_size"]
    lr = config["para"]["lr"]
    momentum = config["para"]["momentum"]
    weight_decay = config["para"]["weight_decay"]
    epochs = config["para"]["epochs"]
    pretrain = config["pretrain"]

    iscuda = config["iscuda"]
    if torch.cuda.is_available and iscuda :
        iscuda = True
    else : iscuda = False

    trainset = dataset.MY_DATASET(video_dir=train_video_dir, audio_dir=train_audio_dir, csv_file=train_csv_path, n=N)
    valset = dataset.MY_DATASET(video_dir=val_video_dir, audio_dir=val_audio_dir, csv_file=val_csv_path, n=N)
    

    train_loader = dataloader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = dataloader(valset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    writer_t = SummaryWriter("../logs/runs/train")
    writer_v = SummaryWriter("../logs/runs/val")
    model = model_LSTM.BIO_MODEL_LSTM()
    # model = vol_model.VOL_MODEL()
    writer_t.add_graph(model=model, input_to_model = (torch.rand(1,3, 6, 112, 112), torch.rand(1, 6, 68)))
    print('--------模型初始化---------')
    if pretrain and os.path.exists(best_model_save_dir):
        model.load_state_dict(torch.load(best_model_save_dir))
        print("预训练模型加载成功：{}".format(best_model_save_dir))
    else:
        init_model.initialize_weights(model)
        print("无预训练模型")
    if iscuda :
        print("load model to cuda")
        model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay = weight_decay, lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

    best_loss = 999
    for epoch in range(epochs):
        loss = train(epoch)
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), best_model_save_dir)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), model_save_dir + '{}.pth'.format(epoch))
        val(epoch)
        scheduler.step()
    writer_t.close()
    writer_v.close()

