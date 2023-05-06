import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from collections import OrderedDict
from modules.bisenetv2 import BiSeNetV2

from utils import setup_clients, parse_args, modify_command_options, Label2Color, Denormalize, color_map


def remove_module_prefix(ordered_dict):
    updated_dict = OrderedDict()
    for key, value in ordered_dict.items():
        updated_key = key.replace("module.", "")
        updated_dict[updated_key] = value
    return updated_dict


def _plot(img, client_name, img_id=-1, img_type='', path_to_save_folder='', plot=False):
    fig, ax = plt.subplots()
    ax.imshow(transforms.ToPILImage()(img))
    ax.margins(0)
    plt.axis('off')
    if path_to_save_folder != '':
        assert img_type != ''
        plt.savefig(os.path.join(path_to_save_folder, f'{client_name}_{img_type}_{img_id}.png'),
                    bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()


def plot_samples(samples, dataset, client_name, denorm, label2color, path_to_save_folder='', plot=False):

    for index, (img, cfsi_img, lab_img, target, pred) in samples:

        img = (denorm(img) * 255).transpose(1, 2, 0).astype(np.uint8)
        if dataset == 'idda':
            cfsi_img = (denorm(cfsi_img) * 255).transpose(1, 2, 0).astype(np.uint8)
            lab_img = (denorm(lab_img) * 255).transpose(1, 2, 0).astype(np.uint8)
        target = label2color(target).astype(np.uint8)
        pred = label2color(pred).astype(np.uint8)

        _plot(img, client_name, img_id=index, img_type='img', path_to_save_folder=path_to_save_folder, plot=plot)
        _plot(cfsi_img, client_name, img_id=index, img_type='cfsi', path_to_save_folder=path_to_save_folder, plot=plot)
        _plot(lab_img, client_name, img_id=index, img_type='lab', path_to_save_folder=path_to_save_folder, plot=plot)
        _plot(target, client_name, img_id=index, img_type='target', path_to_save_folder=path_to_save_folder, plot=plot)
        _plot(pred, client_name, img_id=index, img_type='pred', path_to_save_folder=path_to_save_folder, plot=plot)


def main():

    load_path = 'path/to/ckpt/checkpoint_name.ckpt'
    indices = [0, 1, 2]  # indices of the samples
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path_to_save_folder = 'path/to/save/folder/'
    plot = False

    parser = parse_args()
    args = parser.parse_args()
    args = modify_command_options(args)

    label2color = Label2Color(cmap=color_map(args.dataset, args.remap))
    if args.dataset == 'idda':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'cityscapes':
        if args.cts_norm:
            mean = [0.3257, 0.3690, 0.3223]
            std = [0.2112, 0.2148, 0.2115]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
    else:
        mean, std = None, None
    denorm = Denormalize(mean=mean, std=std)

    model = BiSeNetV2(args.num_classes, output_aux=args.output_aux, pretrained=False)
    model.cuda()

    checkpoint = torch.load(load_path, map_location='cuda:0')
    model_state = remove_module_prefix(checkpoint["model_state"])
    model.load_state_dict(model_state)
    _, test_clients = setup_clients(args, None, None, model, None, None, 1, None, None, disable_ddp=True)

    for client in test_clients:
        client.dataset.images['x'] = sorted(client.dataset.images['x'])
        client.dataset.images['y'] = sorted(client.dataset.images['y'])

        model.eval()

        images = []
        for index in indices:

            (_, img), target = client.dataset[index]
            cfsi_img, _ = client.dataset.cfsi(index)
            lab_img, _ = client.dataset.lab(index)

            if args.dataset == 'idda':
                cfsi_img = client.dataset.test_transform(cfsi_img).detach().cpu().numpy()
                lab_img = client.dataset.test_transform(lab_img).detach().cpu().numpy()

            img = img.cuda()
            target = target.cuda()

            output, _, _, _, _ = model(img.unsqueeze(0))

            prediction = torch.argmax(output, dim=1)
            prediction = prediction.squeeze(0)

            images.append((index, (
                img.detach().cpu().numpy(),
                cfsi_img,
                lab_img,
                target.detach().cpu(),
                prediction.detach().cpu()
            )))

        plot_samples(images, args.dataset, str(client), denorm, label2color, path_to_save_folder=path_to_save_folder,
                     plot=plot)


if __name__ == '__main__':
    main()
