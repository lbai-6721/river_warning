import os

import torch

from get_miou import calculate_miou
from nets.deeplabv3_training import (CE_Loss, Dice_loss, Focal_Loss,
                                     weights_init)
from tqdm import tqdm

from sklearn.model_selection import KFold
import numpy as np

from Utils.utils import get_lr
from Utils.utils_metrics import f_score


def fit_one_epoch666(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, \
                  fp16, scaler, save_period, save_dir, local_rank=0):
    # 初始化指标容器
    total_loss = 0
    total_f_score = 0
    val_loss = 0
    val_f_score = 0
    miou = 0  # 新增mIoU指标

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # ==================== 训练阶段 ====================
    model_train.train()
    for iteration, batch in enumerate(gen):

        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(imgs)
            # ----------------------#
            #   计算损失
            # ----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model_train(imgs)
                # ----------------------#
                #   计算损失
                # ----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    _f_score = f_score(outputs, labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    conf_matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(imgs)
            # ----------------------#
            #   计算损失
            # ----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)

            # 获取预测结果
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            trues = labels.cpu().numpy()

            # 更新混淆矩阵
            for lt, lp in zip(trues.flatten(), preds.flatten()):
                lt = int(lt)
                lp = int(lp)

                conf_matrix[lt, lp] += 1

            val_loss += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                    'f_score': val_f_score / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    # ==================== 计算mIoU ====================
    '''intersection = np.diag(conf_matrix)
    union = conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - intersection
    iou = intersection / (union + 1e-8)


    miou = np.nanmean(iou)  # 忽略NaN值求平均'''
    result = calculate_miou(
        miou_mode=0,
        num_classes=2,
        name_classes=["background", "riverbed"],
        VOCdevkit_path='VOCdevkit',
        day_night_mode='night',
        #model_arch='unet'    此处为使用unet
        #model_arch='segnet'
        model_arch='pspnet'
    )
    miou = result['mIoU']

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print(f'Epoch:{epoch + 1}/{Epoch}')
        print(
            f'Total Loss: {total_loss / epoch_step:.3f} || Val Loss: {val_loss / epoch_step_val:.3f}||miou:{miou:.3f} ')


        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
            epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

    # 返回三个指标：训练损失均值、验证损失均值、mIoU
    return (total_loss / epoch_step,
            val_loss / epoch_step_val,
            miou)


# 以下是使用fit_one_epoch函数的示例代码，假设你有相应的数据生成器等
if __name__ == "__main__":
    # 这里需要初始化相关的参数和对象，比如model_train, model, loss_history等
    # 为了示例，这里只是简单赋值为None
    model_train = None
    model = None
    loss_history = None
    eval_callback = None
    optimizer = None
    epoch = 0
    epoch_step = 100
    epoch_step_val = 50
    gen = None
    gen_val = None
    Epoch = 10
    cuda = True
    dice_loss = False
    focal_loss = False
    cls_weights = np.ones(10)
    num_classes = 10
    fp16 = False
    scaler = None
    save_period = 5
    save_dir = "./saved_models"

    train_loss, val_loss, miou = fit_one_epoch666(model_train, model, loss_history, eval_callback, optimizer, epoch,
                                               epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss,
                                               focal_loss, cls_weights, num_classes, fp16, scaler, save_period,
                                               save_dir)
    print(f"Train Loss: {train_loss}, Val Loss: {val_loss}, mIoU: {miou}")