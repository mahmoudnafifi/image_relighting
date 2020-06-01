__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torchvision
# import torch.nn as nn
from torch import optim
from tqdm import tqdm
from src.dataset import DataLoading
from torch.utils.data import DataLoader, random_split
from src import Relighting
from src import utils

try:
    from torch.utils.tensorboard import SummaryWriter

    use_tb = True
except ImportError:
    use_tb = False


def train_net(net, device, epochs=60, batch_size=32, lr=0.0001, val_percent=0.1, lrdf=0.5, lrdp=25, chkpointperiod=1,
              patchsz=256, validationFrequency=4, dataset_dir='', task='', save_cp=True):
    if task == 'normalization':
        logging.info(f'Current task is {task}.')
        in_dir_img = os.path.join(dataset_dir, 'input_aug/')
        gt_dir_img = os.path.join(dataset_dir, 'gt_images_aug/')
        norm_net = train_normalization(net, device, epochs=epochs, batch_size=int(batch_size / 2), lr=lr,
                                       val_percent=val_percent, lrdf=lrdf, lrdp=lrdp, chkpointperiod=chkpointperiod,
                                       patchsz=patchsz,
                                       validationFrequency=validationFrequency, in_dir_img=in_dir_img,
                                       gt_dir_img=gt_dir_img, save_cp=save_cp)

        net = Relighting.Relighting(task='relighting_one_to_one', device=device)
        net.to(device=device)
        net.load_normalization_net(norm_net.norm_net)
        net.task = 'relighting_one_to_one'

        logging.info(f'Current task is {net.task}.')
        in_dir_img = os.path.join(dataset_dir, 'train_t1/input_aug/')
        gt_dir_img = os.path.join(dataset_dir, 'train_t1/target_aug/')

        train_relighting_one_to_one(net, device, epochs=epochs, batch_size=int(batch_size / 2), lr=lr,
                                    val_percent=val_percent, lrdf=lrdf, lrdp=lrdp, chkpointperiod=chkpointperiod,
                                    patchsz=patchsz, validationFrequency=validationFrequency,
                                    in_dir_img=in_dir_img, gt_dir_img=gt_dir_img, save_cp=save_cp)

        net = Relighting.Relighting(task='relighting_one_to_any', device=device)
        net.to(device=device)
        net.load_normalization_net(norm_net.norm_net)
        net.task = 'relighting_one_to_any'

        logging.info(f'Current task is {net.task}.')

        in_dir_img = os.path.join(dataset_dir, 'train_t3/')
        gt_dir_img = os.path.join(dataset_dir, 'train_t3/')
        tr_dir_img = os.path.join(dataset_dir, 'train_t3/')

        train_relighting_one_to_any(net, device, epochs=epochs, batch_size=int(batch_size / 4), lr=lr,
                                    val_percent=val_percent,
                                    lrdf=lrdf, lrdp=lrdp, chkpointperiod=chkpointperiod, patchsz=patchsz,
                                    validationFrequency=validationFrequency, in_dir_img=in_dir_img,
                                    gt_dir_img=gt_dir_img,
                                    tr_dir_img=tr_dir_img, save_cp=save_cp)

    elif task == 'relighting_one_to_one':
        logging.info(f'Current task is {task}.')
        in_dir_img = os.path.join(dataset_dir, 'train_t1/input_aug/')
        gt_dir_img = os.path.join(dataset_dir, 'train_t1/target_aug/')

        train_relighting_one_to_one(net, device, epochs=epochs, batch_size=int(batch_size / 2), lr=lr,
                                    val_percent=val_percent, lrdf=lrdf, lrdp=lrdp, chkpointperiod=chkpointperiod,
                                    patchsz=patchsz, validationFrequency=validationFrequency,
                                    in_dir_img=in_dir_img, gt_dir_img=gt_dir_img, save_cp=save_cp)

        norm_net = net.norm_net
        net = Relighting.Relighting(task='relighting_one_to_any', device=device)
        net.to(device=device)
        net.load_normalization_net(norm_net)
        net.task = 'relighting_one_to_any'

        logging.info(f'Current task is {net.task}.')

        in_dir_img = os.path.join(dataset_dir, 'train_t3/')
        gt_dir_img = os.path.join(dataset_dir, 'train_t3/')
        tr_dir_img = os.path.join(dataset_dir, 'train_t3/')

        train_relighting_one_to_any(net, device, epochs=epochs, batch_size=int(batch_size / 4), lr=lr,
                                    val_percent=val_percent,
                                    lrdf=lrdf, lrdp=lrdp, chkpointperiod=chkpointperiod, patchsz=patchsz,
                                    validationFrequency=validationFrequency, in_dir_img=in_dir_img,
                                    gt_dir_img=gt_dir_img,
                                    tr_dir_img=tr_dir_img, save_cp=save_cp)

    elif task == 'relighting_one_to_any':
        logging.info(f'Current task is {task}.')
        in_dir_img = os.path.join(dataset_dir, 'train_t3/')
        gt_dir_img = os.path.join(dataset_dir, 'train_t3/')
        tr_dir_img = os.path.join(dataset_dir, 'train_t3/')

        train_relighting_one_to_any(net, device, epochs=epochs, batch_size=int(batch_size / 4), lr=lr,
                                    val_percent=val_percent, lrdf=lrdf, lrdp=lrdp, chkpointperiod=chkpointperiod,
                                    patchsz=patchsz, validationFrequency=validationFrequency, in_dir_img=in_dir_img,
                                    gt_dir_img=gt_dir_img, tr_dir_img=tr_dir_img, save_cp=save_cp)


def train_normalization(net, device, epochs, batch_size, lr, val_percent, lrdf, lrdp, chkpointperiod, patchsz,
                        validationFrequency, in_dir_img, gt_dir_img, save_cp):
    dir_checkpoint = 'checkpoints_normalization/'
    dataset = DataLoading(in_dir_img, gt_dir=gt_dir_img)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    if use_tb:
        writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training normalization net:
        Epochs:          {epochs} epochs
        Batch size:      {batch_size}
        Patch size:      {patchsz} x {patchsz}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Validation Frq.: {validationFrequency}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        TensorBoard:     {use_tb}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, lrdp, gamma=lrdf, last_epoch=-1)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['input']
                gts = batch['gt']
                assert imgs.shape[1] == 3, \
                    f'Network has been defined with 3 input channels, ' \
                    f'but loaded training images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                assert gts.shape[1] == 3, \
                    f'Network has been defined with 3 input channels, ' \
                    f'but loaded GT images have {gts.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                gts = gts.to(device=device, dtype=torch.float32)

                results = net(imgs)
                loss = utils.compute_loss(results, gts)
                epoch_loss += loss.item()
                if use_tb:
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(np.ceil(imgs.shape[0]))
                global_step += 1

        if (epoch + 1) % validationFrequency == 0:
            val_score = vald_net(net, val_loader, 'normalization', device)
            logging.info('Validation MAE: {}'.format(val_score))
            if use_tb:
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('Loss/test', val_score, global_step)
                writer.add_images('images', imgs, global_step)
                writer.add_images('result', results, global_step)
                writer.add_images('GT', gts, global_step)

        scheduler.step()

        if save_cp and (epoch + 1) % chkpointperiod == 0:
            if not os.path.exists(dir_checkpoint):
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')

            torch.save(net.state_dict(), dir_checkpoint + f'normalization_net_{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved!')

    if not os.path.exists('models'):
        os.mkdir('models')
        logging.info('Created trained models directory')
    torch.save(net.state_dict(), 'models/' + 'normalization_net.pth')
    logging.info('Saved trained model!')
    if use_tb:
        writer.close()
    logging.info('End of training normalization net')
    return net



def train_relighting_one_to_one(net, device, epochs, batch_size, lr, val_percent, lrdf, lrdp, chkpointperiod, patchsz,
                     validationFrequency, in_dir_img, gt_dir_img, save_cp):
    dir_checkpoint = 'checkpoints_relighting/'
    dataset = DataLoading(in_dir_img, gt_dir=gt_dir_img)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    if use_tb:
        writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training relighting net:
        Epochs:          {epochs} epochs
        Batch size:      {batch_size}
        Patch size:      {patchsz} x {patchsz}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Validation Frq.: {validationFrequency}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        TensorBoard:     {use_tb}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, lrdp, gamma=lrdf, last_epoch=-1)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['input']
                gts = batch['gt']
                assert imgs.shape[1] == 3, \
                    f'Network has been defined with 3 input channels, ' \
                    f'but loaded training images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                assert gts.shape[1] == 3, \
                    f'Network has been defined with 3 input channels, ' \
                    f'but loaded GT images have {gts.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                gts = gts.to(device=device, dtype=torch.float32)

                results = net(imgs)
                loss = utils.compute_loss(results, gts)
                epoch_loss += loss.item()
                if use_tb:
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(np.ceil(imgs.shape[0]))
                global_step += 1

        if (epoch + 1) % validationFrequency == 0:
            val_score = vald_net(net, val_loader, 'relighting_one_to_one', device)
            logging.info('Validation MAE: {}'.format(val_score))
            if use_tb:
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('Loss/test', val_score, global_step)
                writer.add_images('images', imgs, global_step)
                writer.add_images('result', results, global_step)
                writer.add_images('GT', gts, global_step)

        scheduler.step()

        if save_cp and (epoch + 1) % chkpointperiod == 0:
            if not os.path.exists(dir_checkpoint):
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')

            torch.save(net.state_dict(), dir_checkpoint + f'relighting_net_one_to_one_{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved!')

    if not os.path.exists('models'):
        os.mkdir('models')
        logging.info('Created trained models directory')
    torch.save(net.state_dict(), 'models/' + 'relighting_net_one_to_one.pth')
    logging.info('Saved trained model!')
    if use_tb:
        writer.close()
    logging.info('End of training relighting net')
    return net


def train_relighting_one_to_any(net, device, epochs, batch_size, lr, val_percent, lrdf, lrdp, chkpointperiod, patchsz,
                     validationFrequency, in_dir_img, gt_dir_img, tr_dir_img, save_cp):
    dir_checkpoint = 'checkpoints_relighting/'
    dataset = DataLoading(in_dir_img, gt_dir=gt_dir_img, target_dir=tr_dir_img)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    if use_tb:
        writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training relighting net:
        Epochs:          {epochs} epochs
        Batch size:      {batch_size}
        Patch size:      {patchsz} x {patchsz}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Validation Frq.: {validationFrequency}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        TensorBoard:     {use_tb}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, lrdp, gamma=lrdf, last_epoch=-1)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['input']
                gts = batch['gt']
                targets = batch['target']
                assert imgs.shape[1] == 3, \
                    f'Network has been defined with 3 input channels, ' \
                    f'but loaded training images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                assert gts.shape[1] == 3, \
                    f'Network has been defined with 3 input channels, ' \
                    f'but loaded GT images have {gts.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                assert targets.shape[1] == 3, \
                    f'Network has been defined with 3 input channels, ' \
                    f'but loaded guide images have {targets.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                gts = gts.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.float32)

                results = net(imgs, t=targets)
                loss = utils.compute_loss(results, gts)
                epoch_loss += loss.item()
                if use_tb:
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(np.ceil(imgs.shape[0]))
                global_step += 1

        if (epoch + 1) % validationFrequency == 0:
            val_score = vald_net(net, val_loader, 'relighting_one_to_any', device)
            logging.info('Validation MAE: {}'.format(val_score))
            if use_tb:
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('Loss/test', val_score, global_step)
                writer.add_images('images', imgs, global_step)
                writer.add_images('result', results, global_step)
                writer.add_images('GT', gts, global_step)

        scheduler.step()

        if save_cp and (epoch + 1) % chkpointperiod == 0:
            if not os.path.exists(dir_checkpoint):
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')

            torch.save(net.state_dict(), dir_checkpoint + f'relighting_net_one_to_any_{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved!')

    if not os.path.exists('models'):
        os.mkdir('models')
        logging.info('Created trained models directory')
    torch.save(net.state_dict(), 'models/' + 'relighting_net_one_to_any.pth')
    logging.info('Saved trained model!')
    if use_tb:
        writer.close()
    logging.info('End of training relighting net')
    return net


def vald_net(net, loader, task, device):
    if task == 'normalization':
        return vald_net_normalization(net, loader, device)
    else:
        return vald_net_relighting(net, loader, device, task)



def vald_net_normalization(net, loader, device):
    net.eval()
    n_val = len(loader) + 1
    score = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['input']
            gts = batch['gt']
            assert imgs.shape[1] == 3, \
                f'Network has been defined with 3 input channels, ' \
                f'but loaded training images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            assert gts.shape[1] == 3, \
                f'Network has been defined with 3 input channels, ' \
                f'but loaded AWB GT images have {gts.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            imgs = imgs.to(device=device, dtype=torch.float32)
            gts = gts.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                results = net(imgs)
                loss = utils.compute_loss(results, gts)
                score = score + loss

            pbar.update(np.ceil(imgs.shape[0]))

        net.train()
        return score / n_val


def vald_net_relighting(net, loader, device, task):
    net.eval()
    n_val = len(loader) + 1
    score = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            if task == 'relighting_one_to_any':
                imgs = batch['input']
                gts = batch['gt']
                targets = batch['target']
                assert imgs.shape[1] == 3, \
                    f'Network has been defined with 3 input channels, ' \
                    f'but loaded training images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                assert gts.shape[1] == 3, \
                    f'Network has been defined with 3 input channels, ' \
                    f'but loaded AWB GT images have {gts.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                assert targets.shape[1] == 3, \
                    f'Network has been defined with 3 input channels, ' \
                    f'but loaded guide images have {targets.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                gts = gts.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    results = net(imgs, t=targets)
                    loss = utils.compute_loss(results, gts)
                    score = score + loss

                pbar.update(np.ceil(imgs.shape[0]))
            else:
                imgs = batch['input']
                gts = batch['gt']
                assert imgs.shape[1] == 3, \
                    f'Network has been defined with 3 input channels, ' \
                    f'but loaded training images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                assert gts.shape[1] == 3, \
                    f'Network has been defined with 3 input channels, ' \
                    f'but loaded AWB GT images have {gts.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                gts = gts.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    results = net(imgs)
                    loss = utils.compute_loss(results, gts)
                    score = score + loss

                pbar.update(np.ceil(imgs.shape[0]))


        net.train()
        return score / n_val


def get_args():
    parser = argparse.ArgumentParser(description='Train deep WB editing network.')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=60,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-l', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-vf', '--validation-frequency', dest='val_frq', type=int, default=10,
                        help='Validation frequency.')
    parser.add_argument('-s', '--patch-size', dest='patchsz', type=int, default=256,
                        help='Size of training patch')
    parser.add_argument('-c', '--checkpoint-period', dest='chkpointperiod', type=int, default=10,
                        help='Number of epochs to save a checkpoint')
    parser.add_argument('-mdldir', '--model_dir', dest='model_dir', type=str, default='./models/',
                        help='Models directory')
    parser.add_argument('-ldf', '--learning-rate-drop-factor', dest='lrdf', type=float, default=0.5,
                        help='Learning rate drop factor')
    parser.add_argument('-ldp', '--learning-rate-drop-period', dest='lrdp', type=int, default=25,
                        help='Learning rate drop period')
    parser.add_argument('-trd', '--training_dir', dest='trdir', default='../',
                        help='Training dataset directory')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Training of Deep Image Relighting')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if os.path.exists(os.path.join(args.model_dir, 'relighting_net_one_to_any.pth')):
        logging.info(f'Model is already trained.')
        sys.exit()

    elif os.path.exists(os.path.join(args.model_dir, 'relighting_net_one_to_one.pth')):
        # load net
        logging.info(f'Found a relighting net one to one model in {os.path.join(args.model_dir)}.')
        norm_net = Relighting.Relighting(task='normalization')
        logging.info("Loading normalization model {}".format(os.path.join(args.model_dir, 'normalization_net.pth')))
        norm_net.load_state_dict(torch.load(os.path.join(args.model_dir, 'normalization_net.pth'), map_location=device))

        net = Relighting.Relighting(task='relighting_one_to_any', device=device)
        if args.load:
            net.load_state_dict(
                torch.load(args.load, map_location=device)
            )
            logging.info(f'Model loaded from {args.load}')
        net.to(device=device)
        net.load_normalization_net(norm_net.norm_net)
        curr_task = 'relighting_one_to_any'

    elif os.path.exists(os.path.join(args.model_dir, 'normalization_net.pth')):
        # load net
        logging.info(f'Found a normalization net model in {os.path.join(args.model_dir)}.')
        norm_net = Relighting.Relighting(task='normalization')
        logging.info("Loading normalization model {}".format(os.path.join(args.model_dir, 'normalization_net.pth')))
        norm_net.load_state_dict(torch.load(os.path.join(args.model_dir, 'normalization_net.pth'), map_location=device))

        net = Relighting.Relighting(task='relighting_one_to_one', device=device)
        if args.load:
            net.load_state_dict(
                torch.load(args.load, map_location=device)
            )
            logging.info(f'Model loaded from {args.load}')
        net.to(device=device)
        net.load_normalization_net(norm_net.norm_net)
        curr_task = 'relighting_one_to_one'

    else:
        # load net
        net = Relighting.Relighting(task='normalization', device=device)
        if args.load:
            net.load_state_dict(
                torch.load(args.load, map_location=device)
            )
            logging.info(f'Model loaded from {args.load}')
        net.to(device=device)
        curr_task = 'normalization'

    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        logging.info('Starting training...')
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  lrdf=args.lrdf,
                  lrdp=args.lrdp,
                  device=device,
                  chkpointperiod=args.chkpointperiod,
                  dataset_dir=args.trdir,
                  val_percent=args.val / 100,
                  validationFrequency=args.val_frq,
                  patchsz=args.patchsz,
                  task=curr_task
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'bkupCheckPoint.pth')
        logging.info('Saved interrupt checkpoint backup')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
