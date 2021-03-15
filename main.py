'''
/* ===========================================================================
** Copyright (C) 2019 Infineon Technologies AG. All rights reserved.
** ===========================================================================
**
** ===========================================================================
** Infineon Technologies AG (INFINEON) is supplying this file for use
** exclusively with Infineon's sensor products. This file can be freely
** distributed within development tools and software supporting such 
** products.
** 
** THIS SOFTWARE IS PROVIDED "AS IS".  NO WARRANTIES, WHETHER EXPRESS, IMPLIED
** OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
** MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE.
** INFINEON SHALL NOT, IN ANY CIRCUMSTANCES, BE LIABLE FOR DIRECT, INDIRECT, 
** INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES, FOR ANY REASON 
** WHATSOEVER.
** ===========================================================================
*/
'''
import os
import sys
import json
import shutil
import numpy as np
import torch
from torch import nn
from parse_arguments import parse_opts
from models import mobilenetv2_3d
from jester import Jester
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from torch_train import train_epoch
from torch_validation import val_epoch
from mean import get_mean, get_std
import test


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name),'%s/%s_best.pth' % (opt.result_path, opt.store_name))

best_prec1 = 0

if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    #opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)

    opt.store_name = '_'.join([opt.dataset, opt.model,
                               opt.modality, str(opt.sample_duration)])

    
    torch.manual_seed(opt.manual_seed)

    # Create a reference for a model
    model = mobilenetv2_3d.get_model(num_classes=opt.n_classes, sample_size=opt.sample_size, width_mult=opt.width_mult)
    parameters = model.parameters()
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    ###########------------------ DATA AUGMENTATION AND IMPORTING FOR TRAINING AND VALIDATION ------------------------------------########
   
    # TRAINING
    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            RandomRotate(),
            RandomResize(),
            crop_method,
            ToTensor(opt.norm_value), 
            norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        # All the augmentations implemented at the function below
        training_data = Jester(opt.video_path,
            opt.annotation_path, 
            opt.train_list,
            opt.modality,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])
    # VALIDATION                   
    if not opt.no_val:
        spatial_transform = Compose([            
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), 
            norm_method
        ])
        temporal_transform = TemporalCenterCrop(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = Jester(
            opt.video_path,
            opt.annotation_path,
            opt.val_list,
            opt.modality,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'prec1', 'prec5'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        print(checkpoint['arch'])
        assert opt.arch == checkpoint['arch']
        best_prec1 = checkpoint['best_prec1']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])  
    


    # ---------------   Setting OPTIMIZER
    if opt.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(parameters, lr=opt.learning_rate, betas=(0.6, 0.999), eps=1e-08, weight_decay=opt.weight_decay, amsgrad=False)
    else:
        if opt.nesterov:
            dampening = 0
        else:
            dampening = 0.9
        optimizer = torch.optim.SGD(parameters,lr=opt.learning_rate,momentum=0.9,dampening=0.9,weight_decay=opt.weight_decay,nesterov=opt.nesterov)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
    # ---------------   Setting Loss Function 
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()
       
    #################### ----------- TRAINING PROCESS ----------------------------#########################
    print('Training started')
    for i in range(opt.begin_epoch, opt.n_epochs + 1): #range of epochs defined
        # Training
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt, train_logger, train_batch_logger)
            state = {
               'epoch': i,
               'arch': opt.arch,
               'state_dict': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'best_prec1': 0
               }
            save_checkpoint(state, False)
            torch.cuda.empty_cache()
        # Validation
        if not opt.no_val:
            validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt, val_logger)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
            save_checkpoint(state, is_best)
            
        # Adjusting Learning rate according to validation loss
        scheduler.step(validation_loss)

  
