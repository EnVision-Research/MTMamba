import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil
import pdb


def get_backbone(p):
    """ Return the backbone """

    if p['backbone'] == 'swin_large':
        from mmseg.models.backbones.swin import SwinTransformer
        backbone_channels = None
        ppath = './pretrained_ckpts/'
        backbone = SwinTransformer(patch_size=4, window_size=12, embed_dims=192, 
                    depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), pretrain_img_size=p.TRAIN.SCALE,
                    pretrained=f'{ppath}/mmseg_swin_large_patch4_window12_384_22kto1k.pth')
        backbone.init_weights()
    else:
        raise NotImplementedError

    return backbone, backbone_channels


def get_model(p):
    """ Return the model """

    if p['model'] == 'MTMamba':
        backbone, backbone_channels = get_backbone(p)
        from models.MTMamba import MTMamba
        model = MTMamba(p, backbone)
    elif p['model'] == 'MTMamba_plus':
        backbone, backbone_channels = get_backbone(p)
        from models.MTMamba_plus import MTMamba_plus
        model = MTMamba_plus(p, backbone)
    else:
        raise NotImplementedError('Unknown model {}'.format(p['model']))
    return model


"""
    Transformations, datasets and dataloaders
"""
def get_transformations(p):
    """ Return transformations for training and evaluationg """
    from data import transforms
    import torchvision

    # Training transformations
    if p['train_db_name'] == 'NYUD' or p['train_db_name'] == 'PASCALContext':
        train_transforms = torchvision.transforms.Compose([ # from ATRC
            transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=p.TRAIN.SCALE, cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TRAIN.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])

        # Testing 
        valid_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TEST.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        return train_transforms, valid_transforms

    else:
        return None, None


def get_train_dataset(p, transforms=None):
    """ Return the train dataset """

    db_name = p['train_db_name']
    print('Preparing train dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p.db_paths['PASCALContext'], download=False, split=['train'], transform=transforms, retname=True,
                                          do_semseg='semseg' in p.TASKS.NAMES,
                                          do_edge='edge' in p.TASKS.NAMES,
                                          do_normals='normals' in p.TASKS.NAMES,
                                          do_sal='sal' in p.TASKS.NAMES,
                                          do_human_parts='human_parts' in p.TASKS.NAMES,
                                          overfit=False)

    if db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_paths['NYUD_MT'], download=False, split='train', transform=transforms, do_edge='edge' in p.TASKS.NAMES, 
                                    do_semseg='semseg' in p.TASKS.NAMES, 
                                    do_normals='normals' in p.TASKS.NAMES, 
                                    do_depth='depth' in p.TASKS.NAMES, overfit=False)

    if db_name == 'Cityscapes':
        from data.cityscapes import CITYSCAPES
        database = CITYSCAPES(p, p.db_paths['Cityscapes'], split=["train"], is_transform=True,
                    img_size=p.TRAIN.SCALE, augmentations=None, 
                    task_list=p.TASKS.NAMES)

    return database


def get_train_dataloader(p, dataset, sampler):
    """ Return the train dataloader """
    collate = collate_mil
    trainloader = DataLoader(dataset, batch_size=p['trBatch'], drop_last=True,
                             num_workers=p['nworkers'], collate_fn=collate, pin_memory=True, sampler=sampler)
    return trainloader


def get_test_dataset(p, transforms=None):
    """ Return the test dataset """

    db_name = p['val_db_name']
    print('Preparing test dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p.db_paths['PASCALContext'], download=False, split=['val'], transform=transforms, retname=True,
                                      do_semseg='semseg' in p.TASKS.NAMES,
                                      do_edge='edge' in p.TASKS.NAMES,
                                      do_normals='normals' in p.TASKS.NAMES,
                                      do_sal='sal' in p.TASKS.NAMES,
                                      do_human_parts='human_parts' in p.TASKS.NAMES,
                                    overfit=False)
    
    elif db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_paths['NYUD_MT'], download=False, split='val', transform=transforms, do_edge='edge' in p.TASKS.NAMES, 
                                    do_semseg='semseg' in p.TASKS.NAMES, 
                                    do_normals='normals' in p.TASKS.NAMES, 
                                    do_depth='depth' in p.TASKS.NAMES)

    elif db_name == 'Cityscapes':
        from data.cityscapes import CITYSCAPES
        database = CITYSCAPES(p, p.db_paths['Cityscapes'], split=["val"], is_transform=True,
                    img_size=p.TEST.SCALE, augmentations=None, 
                    task_list=p.TASKS.NAMES)

    else:
        raise NotImplemented("test_db_name: Choose among PASCALContext and NYUD")

    return database


def get_test_dataloader(p, dataset):
    """ Return the validation dataloader """
    collate = collate_mil
    testloader = DataLoader(dataset, batch_size=p['valBatch'], shuffle=False, drop_last=False,
                            num_workers=p['nworkers'], pin_memory=True, collate_fn=collate)
    return testloader


""" 
    Loss functions 
"""
def get_loss(p, task=None):
    """ Return loss function for a specific task """

    if task == 'edge':
        from losses.loss_functions import BalancedBinaryCrossEntropyLoss
        criterion = BalancedBinaryCrossEntropyLoss(pos_weight=p['edge_w'], ignore_index=p.ignore_index)

    elif task == 'semseg' or task == 'human_parts':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(ignore_index=p.ignore_index)

    elif task == 'normals':
        from losses.loss_functions import L1Loss
        criterion = L1Loss(normalize=True, ignore_index=p.ignore_index)

    elif task == 'sal':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(balanced=True, ignore_index=p.ignore_index) 

    elif task == 'depth':
        from losses.loss_functions import L1Loss
        criterion = L1Loss()

    else:
        criterion = None

    return criterion


def get_criterion(p):
    if p['loss_kwargs']['loss_scheme'] == 'log':
        from losses.loss_schemes import MultiTaskLoss_log
        loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
        loss_weights = p['loss_kwargs']['loss_weights']
        return MultiTaskLoss_log(p, p.TASKS.NAMES, loss_ft, loss_weights)
    else:
        from losses.loss_schemes import MultiTaskLoss
        loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
        loss_weights = p['loss_kwargs']['loss_weights']
        return MultiTaskLoss(p, p.TASKS.NAMES, loss_ft, loss_weights)


"""
    Optimizers and schedulers
"""
def get_optimizer(p, model):
    """ Return optimizer for a given model and setup """

    print('Optimizer uses a single parameter group - (Default)')
    params = model.parameters()

    backbone_params, others_params = [], []
    for pn, pp in model.named_parameters():
        if 'backbone' in pn:
            backbone_params.append(pp)
        else:
            others_params.append(pp)

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
                                params,
                                lr=p['optimizer_kwargs']['lr'],
                                betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=p['optimizer_kwargs']['weight_decay'], amsgrad=False)
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    # get scheduler
    if p.scheduler == 'poly':
        from utils.train_utils import PolynomialLR
        scheduler = PolynomialLR(optimizer, p.max_iter, gamma=0.9, min_lr=0)
    else:
        scheduler = None

    return scheduler, optimizer

