import argparse
import logging
import os
import torch
from src import Relighting
from PIL import Image
import numpy as np
from src import utils



def get_args():
    parser = argparse.ArgumentParser(description='Image Relighting.')
    parser.add_argument('--model_dir', '-m', default='./models',
                        help="Specify the directory of the trained model.", dest='model_dir')
    parser.add_argument('--input', '-i', help='Input image filename', dest='input',
                        default='../../validation_t1/input/Image343.png')
    parser.add_argument('--input_g', '-ig', help='Input ground truth image filename', dest='input_g',
                        default=None)#default='../../train_t3/Image000_2500_N.png')
    parser.add_argument('--input_t', '-it', help='Input target image filename', dest='input_t',
                        default=None)#default='../../train_t3/Image095_2500_N.png')
    parser.add_argument('--output_dir', '-o', default='./results',
                        help='Directory to save the output images', dest='out_dir')
    parser.add_argument('--show', '-v', action='store_true', default=True,
                        help="Visualize images",
                        dest='show')
    parser.add_argument('--save', '-s', action='store_true',
                        help="Save the output images",
                        default=True, dest='save')
    parser.add_argument('--device', '-d', default='cpu',
                        help="Device: cuda or cpu.", dest='device')
    parser.add_argument('-tsk', '--task', dest='task', default='relighting',
                        help='Task: normalization or relighting')

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    if args.device.lower() == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    fn = args.input
    fn_g = args.input_g
    fn_t = args.input_t
    out_dir = args.out_dir
    tosave = args.save
    task = args.task.lower()
    maxSize = 512 - 64

    logging.info(f'Using device {device}')

    if tosave:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    if task == 'normalization':
        if os.path.exists(os.path.join(args.model_dir, 'normalization_net.pth')):
            # load net
            net = Relighting.Relighting(task=task)
            logging.info("Loading model {}".format(os.path.join(args.model_dir, 'normalization_net.pth')))
            net.load_state_dict(torch.load(os.path.join(args.model_dir, 'normalization_net.pth'), map_location=device))
            net.eval()
            net.norm_net.to(device=device)
        elif os.path.exists(os.path.join(args.model_dir, 'relighting_net_one_to_one.pth')):
            # load net
            net = Relighting.Relighting(task='relighting_one_to_one')
            logging.info("Loading model {}".format(os.path.join(args.model_dir, 'relighting_net_one_to_one.pth')))
            net.load_state_dict(torch.load(os.path.join(args.model_dir, 'relighting_net_one_to_one.pth'), map_location=device))
            net.task = 'normalization'
            net.norm_net.to(device=device)
            net.eval()
        elif os.path.exists(os.path.join(args.model_dir, 'relighting_net_one_to_any.pth')):
            # load net
            net = Relighting.Relighting(task='relighting_one_to_any')
            logging.info("Loading model {}".format(os.path.join(args.model_dir, 'relighting_net_one_to_one.pth')))
            net.load_state_dict(torch.load(os.path.join(args.model_dir, 'relighting_net_one_to_one.pth'), map_location=device))
            net.task = 'normalization'
            net.norm_net.to(device=device)
            net.eval()
        else:
            raise Exception('Normalization Model not found!')

        logging.info("Processing image {} ...".format(fn))
        in_img = Image.open(fn)
        # get image size
        d1, d2 = in_img.size
        if d1 > maxSize or d2 > maxSize:
            md = np.max([d1, d2])
            factor = maxSize / md
            in_img = in_img.resize((int(d1 * factor), int(d2 * factor)), 3)
            d1, d2 = in_img.size
        in_img = np.array(in_img) / 255
        if in_img.shape[2] == 4:
            in_img = in_img[:, :, 0:3]

        pad1 = 16 - d1 % 16
        pad2 = 16 - d2 % 16

        temp = np.zeros((d1 + pad1, d2 + pad2, 3)).astype(np.float32)
        temp[:d1, :d2, :] = in_img
        in_img = temp

        tensor_in_img = torch.unsqueeze(torch.from_numpy(in_img.transpose(
            (2, 0, 1))).to(device=device, dtype=torch.float32), dim=0)

        name, _ = os.path.splitext(os.path.split(fn)[1])

        result = net(tensor_in_img)
        result = torch.squeeze(result).cpu().detach().numpy().transpose((1, 2, 0))
        result = result[:d1, :d2, :]
        result = utils.to_image(result)
        in_img = utils.to_image(in_img)
        if tosave:
            result.save(os.path.join(out_dir, name + '.png'))

        if args.show:
            logging.info("Visualizing results for image: {}, close to continue ...".format(fn))
            utils.imshow(in_img, result=result)

    elif task == 'relighting':
        if fn_t is None:
            if os.path.exists(os.path.join(args.model_dir, 'relighting_net_one_to_one.pth')):
                # load net
                net = Relighting.Relighting(task='relighting_one_to_one')
                logging.info("Loading model {}".format(os.path.join(args.model_dir, 'relighting_net_one_to_one.pth')))
                net.load_state_dict(torch.load(os.path.join(args.model_dir, 'relighting_net_one_to_one.pth'), map_location=device))
                net.eval()
                net.norm_net.to(device=device)
                net.relighting_net.to(device=device)
                net.task = 'relighting_one_to_one'
            else:
                raise Exception('Relighting Model not found!')

            logging.info("Processing image {} ...".format(fn))
            in_img = Image.open(fn)
            # get image size
            d1, d2 = in_img.size
            if d1 > maxSize or d2 > maxSize:
                md = np.max([d1, d2])
                factor = maxSize / md
                in_img = in_img.resize((int(d1 * factor), int(d2 * factor)), 3)
                d1, d2 = in_img.size
            pad1 = 16 - d1 % 16
            pad2 = 16 - d2 % 16
            in_img = np.array(in_img) / 255
            if in_img.shape[2] == 4:
                in_img = in_img[:, :, 0:3]
            temp = np.zeros((d1 + pad1, d2 + pad2, 3)).astype(np.float32)
            temp[:d1, :d2, :] = in_img
            in_img = temp
            tensor_in_img = torch.unsqueeze(torch.from_numpy(in_img.transpose(
                (2, 0, 1))).to(device=device, dtype=torch.float32), dim=0)

            name, _ = os.path.splitext(os.path.split(fn)[1])

            result = net(tensor_in_img)
            result = torch.squeeze(result).cpu().detach().numpy().transpose((1, 2, 0))
            result = result[:d1, :d2, :]
            result = utils.to_image(result)
            in_img = utils.to_image(in_img)

            if fn_g is not None:
                gt_img = Image.open(fn_g)
                gt_img = np.array(gt_img) / 255

            if tosave:
                result.save(os.path.join(out_dir, name + '.png'))

            if args.show:
                logging.info("Visualizing results for image: {}, close to continue ...".format(fn))
                if fn_g is not None:
                    utils.imshow(in_img, result=result, gt=gt_img)
                else:
                    utils.imshow(in_img, result=result)

        else:
            if os.path.exists(os.path.join(args.model_dir, 'relighting_net_one_to_any.pth')):
                # load net
                net = Relighting.Relighting(task='relighting_one_to_any')
                logging.info("Loading model {}".format(os.path.join(args.model_dir, 'relighting_net_one_to_any.pth')))
                net.load_state_dict(
                    torch.load(os.path.join(args.model_dir, 'relighting_net_one_to_any.pth'), map_location=device))
                net.eval()
                net.norm_net.to(device=device)
                net.relighting_net.to(device=device)
                net.task = 'relighting_one_to_any'
            else:
                raise Exception('Relighting Model not found!')

            logging.info("Processing image {} ...".format(fn))
            in_img = Image.open(fn)
            # get image size
            d1, d2 = in_img.size
            if d1 > maxSize or d2 > maxSize:
                md = np.max([d1, d2])
                factor = maxSize / md
                in_img = in_img.resize((int(d1 * factor), int(d2 * factor)), 3)
                d1, d2 = in_img.size
            pad1 = 16 - d1 % 16
            pad2 = 16 - d2 % 16
            in_img = np.array(in_img) / 255
            if in_img.shape[2] == 4:
                in_img = in_img[:, :, 0:3]
            temp = np.zeros((d1 + pad1, d2 + pad2, 3)).astype(np.float32)
            temp[:d1, :d2, :] = in_img
            in_img = temp
            d1_temp, d2_temp, _ = temp.shape

            tensor_in_img = torch.unsqueeze(torch.from_numpy(in_img.transpose(
                (2, 0, 1))).to(device=device, dtype=torch.float32), dim=0)

            parts = os.path.split(fn)
            base_name = parts[1]
            name, _ = os.path.splitext(base_name)

            t_img = Image.open(fn_t)
            t_img = t_img.resize((d1_temp, d2_temp), 3)
            t_img = np.array(t_img) / 255

            tensor_t_img = torch.unsqueeze(torch.from_numpy(t_img.transpose(
                (2, 0, 1))).to(device=device, dtype=torch.float32), dim=0)

            name, _ = os.path.splitext(os.path.split(fn)[1])

            result = net(tensor_in_img, t=tensor_t_img)
            result = torch.squeeze(result).cpu().detach().numpy().transpose((1, 2, 0))
            result = result[:d1, :d2, :]
            result = utils.to_image(result)
            in_img = utils.to_image(in_img)

            if fn_g is not None:
                gt_img = Image.open(fn_g)
                gt_img = np.array(gt_img) / 255

            if tosave:
                result.save(os.path.join(out_dir, name + '.png'))

            if args.show:
                logging.info("Visualizing results for image: {}, close to continue ...".format(fn))
                if fn_g is not None:
                    utils.imshow(in_img, result=result, target=t_img, gt=gt_img)
                else:
                    utils.imshow(in_img, result=result, target=t_img)
