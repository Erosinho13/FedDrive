from .cityscapes import Cityscapes
from .idda import IDDADataset
import dataset.transform as T
import dataset.transform_cv2 as Tcv
from functools import partial


def get_dataset(args, train=True):
    """ Dataset And Augmentation

    Returns:
        train_transform / test_transform: compose of transforms
        test_bisenetv2: flag. True if train==False and args.model == bisenetv2
        dataset: partial initialization of the dataset according to args.dataset
    """

    if args.dataset == 'cityscapes':

        dataset = partial(Cityscapes, cl19=True)

        if args.model == 'bisenetv2':
            train_transform = []
            test_transform = []
            if args.cts_norm:
                mean = [0.3257, 0.3690, 0.3223]
                std = [0.2112, 0.2148, 0.2115]
            else:
                mean = [0.5, 0.5, 0.5]
                std = [0.5, 0.5, 0.5]
            if not args.cv2_transform:
                if not args.double_dataset and not args.quadruple_dataset:
                    train_transform.append(T.RandomHorizontalFlip(0.5))
                if args.rsrc_transform:
                    train_transform.append(T.RandomScaleRandomCrop(crop_size=(1024, 2048),
                                                                   scale=(0.75, 1.0, 1.25, 1.5, 1.75, 2.0)))
                    train_transform.append(T.Resize(size=(args.h_resize, args.w_resize)))
                elif args.rrc_transform:
                    train_transform.append(T.RandomResizedCrop((args.h_resize, args.w_resize),
                                                               scale=(args.min_scale, args.max_scale)))
                if args.jitter:
                    train_transform.append(T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
                train_transform = train_transform + [T.ToTensor(), T.Normalize(mean=mean, std=std)]
                train_transform = T.Compose(train_transform)
                # ------------------------------------------
                if args.use_test_resize:
                    test_transform.append(T.Resize(size=(512, 1024)))
                test_transform = test_transform + [T.ToTensor(), T.Normalize(mean=mean, std=std)]
                test_transform = T.Compose(test_transform)
            else:
                train_transform.append(
                    Tcv.RandomResizedCrop(scales=(args.min_scale, args.max_scale), size=(args.h_resize, args.w_resize)))
                if not args.double_dataset and not args.quadruple_dataset:
                    train_transform.append(Tcv.RandomHorizontalFlip(0.5))
                if args.jitter:
                    train_transform.append(Tcv.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
                train_transform.append(Tcv.ToTensor(mean=mean, std=std))
                train_transform = Tcv.Compose(train_transform)
                # ------------------------------------------
                if args.use_test_resize:
                    test_transform.append(Tcv.Resize(size=(512, 1024)))
                test_transform = test_transform + [Tcv.ToTensor(mean=mean, std=std)]
                test_transform = Tcv.Compose(test_transform)
        else:
            raise NotImplementedError

    elif args.dataset == 'idda':

        dataset = IDDADataset

        if args.model == 'bisenetv2':
            train_transform = []
            test_transform = []
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            if not args.cv2_transform:
                train_transform.append(T.RandomHorizontalFlip(0.5))
                if args.rsrc_transform:
                    train_transform.append(
                        T.RandomScaleRandomCrop(crop_size=(1024, 1856), scale=(0.75, 1.0, 1.25, 1.5, 1.75, 2.0)))
                    train_transform.append(T.Resize(size=(args.h_resize, args.w_resize)))
                elif args.rrc_transform:
                    train_transform.append(
                        T.RandomResizedCrop((args.h_resize, args.w_resize), scale=(args.min_scale, args.max_scale)))
                if args.jitter:
                    train_transform.append(T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
                train_transform = train_transform + [T.ToTensor(), T.Normalize(mean=mean, std=std)]
                train_transform = T.Compose(train_transform)
                # ------------------------------------------
                if args.use_test_resize:
                    test_transform.append(T.Resize(size=(512, 928)))
                test_transform = test_transform + [T.ToTensor(), T.Normalize(mean=mean, std=std)]
                test_transform = T.Compose(test_transform)
            else:
                train_transform.append(
                    Tcv.RandomResizedCrop(scales=(args.min_scale, args.max_scale), size=(args.h_resize, args.w_resize)))
                train_transform.append(Tcv.RandomHorizontalFlip(0.5))
                if args.jitter:
                    train_transform.append(Tcv.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
                train_transform.append(Tcv.ToTensor(mean=mean, std=std))
                train_transform = Tcv.Compose(train_transform)
                # ------------------------------------------
                if args.use_test_resize:
                    test_transform.append(Tcv.Resize(size=(512, 928)))
                test_transform = test_transform + [Tcv.ToTensor(mean=mean, std=std)]
                test_transform = Tcv.Compose(test_transform)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    test_bisenetv2 = True if (args.model == 'bisenetv2' and not train) else False

    if args.dataset == 'idda' or args.dataset == 'cityscapes':
        return train_transform, test_transform, test_bisenetv2, dataset
    else:
        if train:
            return train_transform, test_bisenetv2, dataset
        else:
            return test_transform, test_bisenetv2, dataset
