import torch
import torch.nn as nn
import torch.optim
import os
import argparse
import dataloader
from Ours.BrightsightNet import enhanceNet

import Myloss
import pandas as pd
# Create an empty DataFrame object
df = pd.DataFrame(columns=['Iteration', 'Loss1', 'Loss2'])




def loss(r,enhance,orign,gloab = True,exp=0.6):
    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16, exp)
    L_TV = Myloss.L_TV()
    L_global = Myloss.Global_exp(exp)
    Loss_TV = 10*L_TV(r)
    loss_spa = torch.mean(L_spa(enhance, orign))
    loss_col =  2*torch.mean(L_color(enhance))
    loss_exp = 5*torch.mean(L_exp(enhance))
    loss_global = 5*L_global(enhance)
    # print("Loss_TV",Loss_TV.item())
    # print("loss_spa", loss_spa.item())
    # print("loss_col", loss_col.item())
    # print("loss_exp", loss_exp.item())
    # print("loss_global", loss_global.item())

    if gloab:
        loss = Loss_TV + loss_spa + loss_col  + loss_exp + loss_global
    else:
        loss = Loss_TV + loss_spa + loss_col + loss_exp
    return loss







def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    num_iter = 0
    DCE_net = enhanceNet().cuda()

    # DCE_net.apply(weights_init)
    if config.load_pretrain == True:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir))
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)




    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    DCE_net.train()

    for epoch in range(config.num_epochs):
        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.cuda()
            enhance_image, enhance_image_stage2, r_1, r_2 = DCE_net(img_lowlight)
            loss1 = loss(r=r_1,enhance=enhance_image,orign=img_lowlight,exp=0.4,gloab=False)
            loss2 = loss(r=r_2,enhance=enhance_image_stage2,orign=img_lowlight,exp=0.6, gloab=False)
            total_loss = loss1 + loss2

            df.loc[len(df)] = [num_iter, loss1.item(), loss2.item()]
            num_iter = num_iter+1
            # 每隔一定步数保存 DataFrame 到 CSV 文件
            if num_iter % 50 == 0:
                df.to_csv('loss.csv', index=False)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print(str(epoch) + ":Loss at iteration", iteration + 1, ":", total_loss.item())
                print({"loss1": loss1.item(), "loss2": loss2.item()})
            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch_ours_final" + str(epoch) + '.pth')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default=r"E:\CZC\data\train_red_rail")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/Epoch99.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)
