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
    parser.add_argument('--input_dr', '-i', help='Input image directory', dest='input_dir',
                        default='../../validation_t1/input/')
    parser.add_argument('--input_dir_t', '-it', help='Input target image directory', dest='input_dir_t',
                        default=None)#default='../../validation_t3/guide/')
    parser.add_argument('--output_dir', '-o', default='./results',
                        help='Directory to save the output images', dest='out_dir')
    parser.add_argument('--save', '-s', action='store_true',
                        help="Save the output images",
                        default=True, dest='save')
    parser.add_argument('--device', '-d', default='cuda',
                        help="Device: cuda or cpu.", dest='device')
    parser.add_argument('-tsk', '--task', dest='task', default='relighting',
                        help='Task: normalization or relighting')
    parser.add_argument('-ens', '--ensemble', dest='ensemble', default=True,
                        help='Ensembling for one-to-one mapping by using the one-to-any model')

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    if args.device.lower() == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    input_dir = args.input_dir
    input_dir_t = args.input_dir_t
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

        imgfiles = []
        valid_images = (".jpg", ".png")
        for fn in os.listdir(input_dir):
            if fn.lower().endswith(valid_images):
                imgfiles.append(os.path.join(input_dir, fn))

        for fn in imgfiles:
            logging.info("Processing image {} ...".format(fn))
            in_img = Image.open(fn)
            # get image size
            d1, d2 = in_img.size
            if d1 > maxSize or d2 > maxSize:
                md = np.max([d1, d2])
                factor = maxSize / md
                in_img = in_img.resize((int(d1 * factor), int(d2 * factor)), 3)
                upscaling_pp = 1
            else:
                upscaling_pp = 0

            if upscaling_pp is 1:
                pp = True
                remove_zeros_pp = 0

            in_img = np.array(in_img) / 255
            d1, d2, c = in_img.shape
            if c == 4:
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
            result.save('temp_r.png')
            original_in_img = Image.open(fn)
            original_in_img.save('temp_i.png')

            if pp is True:
                if os.path.exists('result.png'):
                    os.remove('result.png')
                os.system(f'post_processing.exe "temp_i.png" "{upscaling_pp}" "temp_r.png" "{remove_zeros_pp}"')
                print('Post processing....')
                while not os.path.exists('result.png'):
                    continue
                result = Image.open('result.png')

            if tosave:
                result.save(os.path.join(out_dir, name + '.png'))

    elif task == 'relighting':
        if input_dir_t is None:
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

            if args.ensemble is True:
                if os.path.exists(os.path.join(args.model_dir, 'relighting_net_one_to_any.pth')):
                    # load net
                    net_one_to_any = Relighting.Relighting(task='relighting_one_to_any')
                    logging.info(
                        "Loading model {}".format(os.path.join(args.model_dir, 'relighting_net_one_to_any.pth')))
                    net_one_to_any.load_state_dict(
                        torch.load(os.path.join(args.model_dir, 'relighting_net_one_to_any.pth'), map_location=device))
                    net_one_to_any.eval()
                    net_one_to_any.norm_net.to(device=device)
                    net_one_to_any.relighting_net.to(device=device)
                    net_one_to_any.task = 'relighting_one_to_any'
                else:
                    raise Exception('Relighting Model (one-to-any) not found for ensembling!')

            imgfiles = []
            valid_images = (".jpg", ".png")
            for fn in os.listdir(input_dir):
                if fn.lower().endswith(valid_images):
                    imgfiles.append(os.path.join(input_dir, fn))
            with torch.no_grad():
                for fn in imgfiles:
                    logging.info("Processing image {} ...".format(fn))
                    in_img = Image.open(fn)
                    # get image size
                    d1, d2 = in_img.size
                    if d1 > maxSize or d2 > maxSize:
                        md = np.max([d1, d2])
                        factor = maxSize / md
                        in_img = in_img.resize((int(d1 * factor), int(d2 * factor)), 3)
                        d1, d2 = in_img.size
                        upscaling_pp = 1
                    else:
                        upscaling_pp = 0

                    pp = True

                    remove_zeros_pp = 1
                    color_transfer_pp = 0

                    in_img = np.array(in_img) / 255
                    d1, d2, c = in_img.shape
                    pad1 = 16 - d1 % 16
                    pad2 = 16 - d2 % 16
                    if c == 4:
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
                    result.save('temp_r.png')
                    original_in_img = Image.open(fn)
                    original_in_img.save('temp_i.png')

                    if pp is True:
                        if os.path.exists('result.png'):
                            os.remove('result.png')
                        os.system(f'post_processing.exe "temp_i.png" "{upscaling_pp}" "temp_r.png" "{remove_zeros_pp}"')
                        print('Post processing....')
                        while not os.path.exists('result.png'):
                            continue
                        result = Image.open('result.png')

                    if args.ensemble is True:
                        result = np.array(result)/255 * 0.5
                        d1_temp, d2_temp, _ = in_img.shape
                        original_in_img = Image.open(fn)
                        original_in_img.save('temp_i.png')
                        for i in range(6):
                            t_img = Image.open(f'ref/{i+1}.png')
                            t_img = t_img.resize((d1_temp, d2_temp), 3)
                            t_img = np.array(t_img) / 255
                            if t_img.shape[2] == 4:
                                t_img = t_img[:, :, 0:3]
                            tensor_t_img = torch.unsqueeze(torch.from_numpy(t_img.transpose(
                                (2, 0, 1))).to(device=device, dtype=torch.float32), dim=0)

                            result_t = net_one_to_any(tensor_in_img, t=tensor_t_img)
                            result_t = torch.squeeze(result_t).cpu().detach().numpy().transpose((1, 2, 0))
                            result_t = result_t[:d1, :d2, :]
                            result_t = utils.to_image(result_t)
                            result_t.save('temp_r.png')
                            if pp is True:
                                if os.path.exists('result.png'):
                                    os.remove('result.png')
                                os.system(f'post_processing.exe "temp_i.png" "{upscaling_pp}"'
                                          f' "temp_r.png" "{remove_zeros_pp}"')
                                print('Post processing....')
                                while not os.path.exists('result.png'):
                                    continue
                                result_t = Image.open('result.png')
                                result = result + np.array(result_t)/255 * 1/12

                        result = utils.to_image(result)

                    if tosave:
                        result.save(os.path.join(out_dir, name + '.png'))


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


            imgfiles = []
            valid_images = (".jpg", ".png")
            for fn in os.listdir(input_dir):
                if fn.lower().endswith(valid_images):
                    imgfiles.append(os.path.join(input_dir, fn))
            with torch.no_grad():
                for fn in imgfiles:
                    logging.info("Processing image {} ...".format(fn))
                    in_img = Image.open(fn)
                    # get image size
                    d1, d2 = in_img.size
                    if d1 > maxSize or d2 > maxSize:
                        md = np.max([d1, d2])
                        factor = maxSize / md
                        in_img = in_img.resize((int(d1 * factor), int(d2 * factor)), 3)
                        upscaling_pp = 1
                    else:
                        upscaling_pp = 0

                    pp = True

                    remove_zeros_pp = 1
                    color_transfer_pp = 0
                    in_img = np.array(in_img) / 255
                    d1, d2, c = in_img.shape
                    pad1 = 16 - d1 % 16
                    pad2 = 16 - d2 % 16
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
                    fn_t = os.path.join(input_dir_t, base_name)
                    name, _ = os.path.splitext(base_name)

                    t_img = Image.open(fn_t)
                    t_img = t_img.resize((d1_temp, d2_temp), 3)
                    t_img = np.array(t_img) / 255

                    tensor_t_img = torch.unsqueeze(torch.from_numpy(t_img.transpose(
                        (2, 0, 1))).to(device=device, dtype=torch.float32), dim=0)

                    result = net(tensor_in_img, t=tensor_t_img)
                    result = torch.squeeze(result).cpu().detach().numpy().transpose((1, 2, 0))
                    result = result[:d1, :d2, :]
                    result = utils.to_image(result)

                    result.save('temp_r.png')
                    original_in_img = Image.open(fn)
                    original_in_img.save('temp_i.png')
                    if pp is True:
                        if os.path.exists('result.png'):
                            os.remove('result.png')
                        if color_transfer_pp is 0:
                            os.system(f'post_processing.exe "temp_i.png" "{upscaling_pp}"'
                                      f' "temp_r.png" "{remove_zeros_pp}"')
                        else:
                            os.system(f'post_processing.exe "temp_i.png" "{upscaling_pp}"'
                                      f' "temp_r.png" "{remove_zeros_pp}" "{color_transfer_pp}" "{fn_t}"')
                        print('Post processing....')
                        while not os.path.exists('result.png'):
                            continue
                        result = Image.open('result.png')

                    if tosave:
                        result.save(os.path.join(out_dir, name + '.png'))
