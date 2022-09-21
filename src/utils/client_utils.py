import os
import cv2
import json
import pickle
import numpy as np

from PIL import Image
from collections import defaultdict

from dataset import get_dataset
from clients import Client, SiloBNClient


def read_dir(data_dir):
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        data.update(cdata['user_data'])

    return data


def read_data(train_data_dir, test_data_dir):
    """
    train_data: Form: {'user_id': {'x': [list of images], 'y': [list of labels]}}
    """
    train_data = read_dir(train_data_dir)
    test_data = read_dir(test_data_dir)

    return train_data, test_data


def define_client(args):
    if args.algorithm == 'FedAvg':
        return Client
    if args.algorithm == 'SiloBN':
        return SiloBNClient
    raise NotImplementedError


def create_clients(args, train_data, test_data, model, world_size, rank, device, logger, writer, num_gpu, train,
                   ckpt_path):
    if args.dataset == 'idda' or args.dataset == 'cityscapes':
        train_transform, test_transform, test_bisenetv2, dataset = get_dataset(args, train)
        transform = []
    else:
        transform, test_bisenetv2, dataset = get_dataset(args, train)
        train_transform = test_transform = []

    clients = []
    users = train_data.keys() if train else test_data.keys()

    client_func = define_client(args)

    for i, user in enumerate(users):

        data = train_data[user] if train else test_data[user]
        batch_size = args.batch_size if train else args.test_batch_size

        if args.dataset == 'cityscapes':
            ds = dataset(data=data, transform=train_transform, test_transform=test_transform,
                         test_bisenetv2=test_bisenetv2, double=args.double_dataset, quadruple=args.quadruple_dataset,
                         use_cv2_transform=args.cv2_transform, dom_gen=args.dom_gen, split_name=args.clients_type)
        elif args.dataset == 'idda':
            ds = dataset(data=data, transform=train_transform, test_transform=test_transform,
                         test_bisenetv2=test_bisenetv2, crop_size=(1856, 1024), remap=args.remap, dom_gen=args.dom_gen,
                         use_cv2_transform=args.cv2_transform, setting_type=args.setting_type,
                         split_type=args.clients_type, user=user)
        else:
            raise NotImplementedError

        client = client_func(user, ds, model, logger, writer, args, batch_size, world_size, rank, num_gpu,
                             device=device, ckpt_path=ckpt_path, name=args.name)

        clients.append(client)

    return clients


def extract_amp_spectrum(img_np):
    fft = np.fft.fft2(img_np, axes=(0, 1))
    return np.abs(fft)


def create_domgen_bank(args, train_data):
    if args.dataset == 'idda':
        base_path = os.path.join('..', 'data', 'idda', 'data', 'IDDAsmall')
    elif args.dataset == 'cityscapes':
        base_path = os.path.join('..', 'data', 'cityscapes', 'data', 'leftImg8bit')
    else:
        raise NotImplementedError
    if args.dom_gen == 'cfsi':
        if not os.path.isdir(
                os.path.join('..', 'data', args.dataset, 'data', args.clients_type, args.setting_type, 'bank_A')):
            os.makedirs(
                os.path.join('..', 'data', args.dataset, 'data', args.clients_type, args.setting_type, 'bank_A'))
            for cid in train_data.keys():
                for x in train_data[cid]['x']:
                    img = Image.open(os.path.join(base_path, x))
                    img_np = np.asarray(img, np.float32)

                    amp = extract_amp_spectrum(img_np)
                    sample = x.split('/')[-1].split('.')[0]
                    np.save(
                        os.path.join('..', 'data', args.dataset, 'data', args.clients_type, args.setting_type,
                                     'bank_A', '{}_amp_{}'.format(cid, sample)), amp)
    elif args.dom_gen == 'lab':
        if not os.path.isdir(
                os.path.join('..', 'data', args.dataset, 'data', args.clients_type, args.setting_type, 'bank_lab')):
            os.makedirs(
                os.path.join('..', 'data', args.dataset, 'data', args.clients_type, args.setting_type, 'bank_lab'))
            for cid in train_data.keys():
                for x in train_data[cid]['x']:
                    to_save = {}
                    sample = x.split('/')[-1].split('.')[0]
                    img = cv2.imread(os.path.join(base_path, x))
                    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                    mean_t = np.mean(img_lab, axis=(0, 1))
                    std_t = np.std(img_lab, axis=(0, 1))
                    to_save['mean'] = mean_t
                    to_save['std'] = std_t
                    file_name = '%s_%s.pkl' % (cid, sample)
                    file_path = os.path.join('..', 'data', args.dataset, 'data', args.clients_type, args.setting_type,
                                             'bank_lab', file_name)
                    with open(file_path, 'wb') as f:
                        pickle.dump(to_save, f)
                    f.close()


def setup_clients(args, logger, writer, model, world_size, rank, num_gpu, device=None, ckpt_path=None):
    if args.dataset == 'cityscapes':
        train_data_dir = os.path.join('..', 'data', args.dataset, 'data', args.clients_type, 'train')
        test_data_dir = os.path.join('..', 'data', args.dataset, 'data', args.clients_type, 'test')
    elif args.dataset == 'idda':
        train_data_dir = os.path.join('..', 'data', args.dataset, 'data', args.clients_type, args.setting_type, 'train')
        test_data_dir = os.path.join('..', 'data', args.dataset, 'data', args.clients_type, args.setting_type, 'test')
    else:
        train_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'train')
        test_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'test')

    train_data, test_data = read_data(train_data_dir, test_data_dir)

    if args.dom_gen is not None:
        create_domgen_bank(args, train_data)

    if args.framework == 'centralized' and args.algorithm == 'FedAvg':
        train_data_all = {'x': [], 'y': []}
        for c in train_data.keys():
            train_data_all['x'].extend(train_data[c]['x'])
            train_data_all['y'].extend(train_data[c]['y'])
        train_data = {'centralized_user': train_data_all}

    train_clients = create_clients(args, train_data, test_data, model, world_size, rank, device, logger, writer,
                                   num_gpu, train=True, ckpt_path=ckpt_path)

    test_clients = create_clients(args, train_data, test_data, model, world_size, rank, device, logger, writer, num_gpu,
                                  train=False, ckpt_path=ckpt_path)

    return train_clients, test_clients
