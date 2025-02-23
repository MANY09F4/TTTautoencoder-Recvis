# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
import copy
import torch
import models_mae_shared
import os.path
import numpy as np
from scipy import stats
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import timm.optim.optim_factory as optim_factory
import glob
from utils import display_images, apply_mask_to_image


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # output is (B, classes)
    # target is (B)
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def get_prameters_from_args(model, args):
    if args.finetune_mode == 'encoder':
        for name, p in model.named_parameters():
            if name.startswith('decoder'):
                p.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
    elif args.finetune_mode == 'all':
        parameters = model.parameters()
    elif args.finetune_mode == 'encoder_no_cls_no_msk':
        for name, p in model.named_parameters():
            if name.startswith('decoder') or name == 'cls_token' or name == 'mask_token':
                p.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
    return parameters


def _reinitialize_model(base_model, base_optimizer, base_scalar, clone_model, args, device):
    if args.stored_latents:
        # We don't need to change the model, as it is never changed
        base_model.train(True)
        base_model.to(device)
        return base_model, base_optimizer, base_scalar
    clone_model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    clone_model.train(True)
    clone_model.to(device)
    if args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(get_prameters_from_args(clone_model, args), lr=args.lr, momentum=args.optimizer_momentum)
    elif args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
    else:
        assert args.optimizer_type == 'adam_w'
        optimizer = torch.optim.AdamW(get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
    optimizer.zero_grad()
    loss_scaler = NativeScaler()
    if args.load_loss_scalar:
        loss_scaler.load_state_dict(base_scalar.state_dict())
    return clone_model, optimizer, loss_scaler

# def sequential_model(base_model, clone_model, args, device, previous_model_state_dict):

#     clone_model.load_state_dict(previous_model_state_dict)
#     clone_model.train(True)
#     clone_model.to(device)
#     if args.optimizer_type == 'sgd':
#         optimizer = torch.optim.SGD(get_prameters_from_args(clone_model, args), lr=args.lr, momentum=args.optimizer_momentum)
#     elif args.optimizer_type == 'adam':
#         optimizer = torch.optim.Adam(get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
#     else:
#         assert args.optimizer_type == 'adam_w'
#         optimizer = torch.optim.AdamW(get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
#     optimizer.zero_grad()
#     return clone_model, optimizer



def train_on_test(base_model: torch.nn.Module,
                  base_optimizer,
                  base_scalar,
                  dataset_train, dataset_val,
                  device: torch.device,
                  log_writer=None,
                  args=None,
                  num_classes: int = 1000,
                  iter_start: int = 0):
    if args.model == 'mae_vit_small_patch16':
        classifier_depth = 8
        classifier_embed_dim = 512
        classifier_num_heads = 16
    else:
        assert ('mae_vit_huge_patch14' in args.model or args.model == 'mae_vit_large_patch16')
        classifier_embed_dim = 768
        classifier_depth = 12
        classifier_num_heads = 12
    clone_model = models_mae_shared.__dict__[args.model](num_classes=num_classes, head_type=args.head_type,
                                                         norm_pix_loss=args.norm_pix_loss,
                                                         classifier_depth=classifier_depth, classifier_embed_dim=classifier_embed_dim,
                                                         classifier_num_heads=classifier_num_heads,
                                                         rotation_prediction=False)
    # Intialize the model for the current run
    all_results = [list() for i in range(args.steps_per_example)]
    all_losses =  [list() for i in range(args.steps_per_example)]
    metric_logger = misc.MetricLogger(delimiter="  ")
    train_loader = iter(torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=args.num_workers))
    val_loader = iter(torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers))
    accum_iter = args.accum_iter
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    model, optimizer, loss_scaler = _reinitialize_model(base_model, base_optimizer, base_scalar, clone_model, args, device)
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    dataset_len = len(dataset_val)

    if args.print_images :
        s = (args.steps_per_example * accum_iter - 1) / (args.num_print_images - 1)
        indices_to_show = {int(round(i * s)) for i in range(args.num_print_images - 1)}
        indices_to_show.add(args.steps_per_example * accum_iter - 1)

    for data_iter_step in range(iter_start, dataset_len):

        rec_losses = []
        class_losses = []
        reconstructed_imgs = []
        steps = []

        val_data = next(val_loader)
        (test_samples, test_label) = val_data
        test_samples = test_samples.to(device, non_blocking=True)[0]
        test_label = test_label.to(device, non_blocking=True)
        pseudo_labels = None

        # Test time training:

        for step_per_example in range(args.steps_per_example):
            train_data = next(train_loader)
            # Train data are 2 values [image, class]
            mask_ratio = args.mask_ratio
            samples, _ = train_data
            targets_rot, samples_rot = None, None
            samples = samples.to(device, non_blocking=True)[0] # index [0] becuase the data is batched to have size 1.
            loss_dict, pred_patches, _, _, mask = model(samples, None, mask_ratio=mask_ratio)
            loss = torch.stack([loss_dict[l] for l in loss_dict]).sum()
            loss_value = loss.item()
            loss /= accum_iter
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(step_per_example + 1) % accum_iter == 0)
            if (step_per_example + 1) % accum_iter == 0:
                if args.verbose:
                    print(f'datapoint {data_iter_step} iter {step_per_example}: rec_loss {loss_value}')

                all_losses[step_per_example // accum_iter].append(loss_value/accum_iter)
                optimizer.zero_grad()


            metric_logger.update(**{k:v.item() for k,v in loss_dict.items()})
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            # Test:
            if (step_per_example + 1) % accum_iter == 0:
                with torch.no_grad():
                    model.eval()
                    all_pred = []
                    for _ in range(accum_iter):
                        loss_d, _, _, pred,_ = model(test_samples, test_label, mask_ratio=0, reconstruct=False)
                        if args.verbose:
                            cls_loss = loss_d['classification'].item()
                            print(f'datapoint {data_iter_step} iter {step_per_example}: class_loss {cls_loss}')
                        all_pred.extend(list(pred.argmax(axis=1).detach().cpu().numpy()))
                    acc1 = (stats.mode(all_pred).mode[0] == test_label[0].cpu().detach().numpy()) * 100.
                    if (step_per_example + 1) // accum_iter == args.steps_per_example:
                        metric_logger.update(top1_acc=acc1)
                        metric_logger.update(loss=loss_value)
                    all_results[step_per_example // accum_iter].append(acc1)
                    model.train()

            if (args.print_images) and data_iter_step % 10 == 0 :

                if (step_per_example in indices_to_show) :

                    reconstructed_img = model.unpatchify(pred_patches[0].unsqueeze(0))
                    mask1 = mask[0].clone()
                    mask1[mask[0] == 1] = 0
                    mask1[mask[0] == 0] = 1
                    reconstructed_img = apply_mask_to_image(reconstructed_img.squeeze(0), mask1, patch_size=16)

                    reconstructed_imgs.append(reconstructed_img)
                    class_losses.append(cls_loss)
                    rec_losses.append(loss_value)
                    steps.append(step_per_example)

                    if step_per_example == args.steps_per_example * accum_iter - 1 :

                        original = samples[0].clone()
                        patch_size = 16
                        masked_image = apply_mask_to_image(original, mask[0], patch_size)

                        original = samples[0].squeeze().detach().cpu()
                        masked_image = masked_image.squeeze().detach().cpu()
                        reconstructed_img = reconstructed_img.squeeze().detach().cpu()

                        save_dir = '/home/toniomirri/Images_evolution'
                        file_name = f'image_{data_iter_step}.png'

                        display_images(original,masked_image,reconstructed_imgs,save_dir,file_name,rec_losses,class_losses,steps)

        if data_iter_step % 50 == 1:
            print('step: {}, acc {} rec-loss {}'.format(data_iter_step, np.mean(all_results[-1]), loss_value))
        if data_iter_step % 500 == 499 or (data_iter_step == dataset_len - 1):
            with open(os.path.join(args.output_dir, f'results_{data_iter_step}.npy'), 'wb') as f:
                np.save(f, np.array(all_results))
            with open(os.path.join(args.output_dir, f'losses_{data_iter_step}.npy'), 'wb') as f:
                np.save(f, np.array(all_losses))
            all_results = [list() for i in range(args.steps_per_example)]
            all_losses = [list() for i in range(args.steps_per_example)]
        model, optimizer, loss_scaler = _reinitialize_model(base_model, base_optimizer, base_scalar, clone_model, args, device)

    save_accuracy_results(args)
    # gather the stats from all processes
    try:
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    except:
        pass
    return


def train_on_test_online(base_model: torch.nn.Module,
                  base_optimizer,
                  base_scalar,
                  dataset_train, dataset_val,
                  device: torch.device,
                  log_writer=None,
                  args=None,
                  num_classes: int = 1000,
                  iter_start: int = 0):
    if args.model == 'mae_vit_small_patch16':
        classifier_depth = 8
        classifier_embed_dim = 512
        classifier_num_heads = 16
    else:
        assert ('mae_vit_huge_patch14' in args.model or args.model == 'mae_vit_large_patch16')
        classifier_embed_dim = 768
        classifier_depth = 12
        classifier_num_heads = 12
    clone_model = models_mae_shared.__dict__[args.model](num_classes=num_classes, head_type=args.head_type,
                                                         norm_pix_loss=args.norm_pix_loss,
                                                         classifier_depth=classifier_depth, classifier_embed_dim=classifier_embed_dim,
                                                         classifier_num_heads=classifier_num_heads,
                                                         rotation_prediction=False)

    # Intialize the model for the current run
    all_results = [list() for i in range(args.steps_per_example)]
    all_losses =  [list() for i in range(args.steps_per_example)]
    metric_logger = misc.MetricLogger(delimiter="  ")
    train_loader = iter(torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=args.num_workers))
    val_loader = iter(torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers))
    accum_iter = args.accum_iter
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    num_steps_per_example = args.steps_per_example * accum_iter


    model, optimizer, loss_scaler = _reinitialize_model(base_model, base_optimizer, base_scalar, clone_model, args, device)
    if args.reinitialize_first_last_one :
        # state_dict_model_previous = model.state_dict()
        # state_dict_optimizer_previous = optimizer.state_dict()
        torch.save({'model' : model.state_dict()},'/home/toniomirri/checkpoints/last_online_weights.pth')

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    dataset_len = len(dataset_val)

    if args.print_images :
        s = (args.steps_per_example * accum_iter - 1) / (args.num_print_images - 1)
        indices_to_show = {int(round(i * s)) for i in range(args.num_print_images - 1)}
        indices_to_show.add(args.steps_per_example * accum_iter - 1)

    for data_iter_step in range(iter_start, dataset_len):

        rec_losses = []
        class_losses = []
        reconstructed_imgs = []
        steps = []

        val_data = next(val_loader)
        (test_samples, test_label) = val_data
        test_samples = test_samples.to(device, non_blocking=True)[0]
        test_label = test_label.to(device, non_blocking=True)
        pseudo_labels = None

        if args.online_ttt :
            if data_iter_step == iter_start :
                num_steps_per_example = args.steps_first_example
            else :
                num_steps_per_example = args.steps_per_example

        #Reinitialize the model to the first step of the last example
        if args.reinitialize_first_last_one :
            #model = clone_model
            model_checkpoint = torch.load('/home/toniomirri/checkpoints/last_online_weights.pth', map_location='cpu')
            model.load_state_dict(model_checkpoint['model'])
            # model.load_state_dict(copy.deepcopy(state_dict_model_previous))
            # if args.optimizer_type == 'sgd':
            #     optimizer = torch.optim.SGD(get_prameters_from_args(clone_model, args), lr=args.lr, momentum=args.optimizer_momentum)
            # elif args.optimizer_type == 'adam':
            #     optimizer = torch.optim.Adam(get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
            # else:
            #     assert args.optimizer_type == 'adam_w'
            #     optimizer = torch.optim.AdamW(get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
            # optimizer.load_state_dict(state_dict_optimizer_previous)
            optimizer.zero_grad()

        # Test time training:

        for step_per_example in range(num_steps_per_example):
            train_data = next(train_loader)
            # Train data are 2 values [image, class]
            mask_ratio = args.mask_ratio
            samples, _ = train_data
            targets_rot, samples_rot = None, None
            samples = samples.to(device, non_blocking=True)[0] # index [0] becuase the data is batched to have size 1.
            loss_dict, pred_patches, _, _, mask = model(samples, None, mask_ratio=mask_ratio)
            loss = torch.stack([loss_dict[l] for l in loss_dict]).sum()
            loss_value = loss.item()
            loss /= accum_iter
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(step_per_example + 1) % accum_iter == 0)
            if (step_per_example + 1) % accum_iter == 0:
                if args.verbose:
                    print(f'datapoint {data_iter_step} iter {step_per_example}: rec_loss {loss_value}')
                if data_iter_step == iter_start and num_steps_per_example - step_per_example <= args.steps_per_example :
                    all_losses[(args.steps_per_example - (num_steps_per_example - step_per_example)) // accum_iter].append(loss_value/accum_iter)
                elif data_iter_step > 0 :
                    all_losses[step_per_example // accum_iter].append(loss_value/accum_iter)
                optimizer.zero_grad()


            metric_logger.update(**{k:v.item() for k,v in loss_dict.items()})
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            # Test:
            if (step_per_example + 1) % accum_iter == 0:
                with torch.no_grad():
                    model.eval()
                    all_pred = []
                    for _ in range(accum_iter):
                        loss_d, _, _, pred,_ = model(test_samples, test_label, mask_ratio=0, reconstruct=False)
                        if args.verbose:
                            cls_loss = loss_d['classification'].item()
                            print(f'datapoint {data_iter_step} iter {step_per_example}: class_loss {cls_loss}')
                        all_pred.extend(list(pred.argmax(axis=1).detach().cpu().numpy()))
                    acc1 = (stats.mode(all_pred).mode[0] == test_label[0].cpu().detach().numpy()) * 100.
                    if (step_per_example + 1) // accum_iter == num_steps_per_example:
                        metric_logger.update(top1_acc=acc1)
                        metric_logger.update(loss=loss_value)
                    if data_iter_step == iter_start and num_steps_per_example - step_per_example <= args.steps_per_example :
                        all_results[(args.steps_per_example - (num_steps_per_example - step_per_example)) // accum_iter].append(acc1)
                    elif data_iter_step > 0 :
                        all_results[step_per_example // accum_iter].append(acc1)
                    model.train()

            if args.reinitialize_first_last_one and step_per_example == 0 :
                torch.save({'model' : model.state_dict()},'/home/toniomirri/checkpoints/last_online_weights.pth')
                # state_dict_model_previous = model.state_dict()
                # state_dict_optimizer_previous = optimizer.state_dict()

            if (args.print_images) and data_iter_step % 10 == 0 :

                if (step_per_example in indices_to_show) :

                    reconstructed_img = model.unpatchify(pred_patches[0].unsqueeze(0))
                    mask1 = mask[0].clone()
                    mask1[mask[0] == 1] = 0
                    mask1[mask[0] == 0] = 1
                    reconstructed_img = apply_mask_to_image(reconstructed_img.squeeze(0), mask1, patch_size=16)

                    reconstructed_imgs.append(reconstructed_img)
                    class_losses.append(cls_loss)
                    rec_losses.append(loss_value)
                    steps.append(step_per_example)

                    if step_per_example == args.steps_per_example * accum_iter - 1 :

                        original = samples[0].clone()
                        patch_size = 16
                        masked_image = apply_mask_to_image(original, mask[0], patch_size)

                        original = samples[0].squeeze().detach().cpu()
                        masked_image = masked_image.squeeze().detach().cpu()
                        reconstructed_img = reconstructed_img.squeeze().detach().cpu()

                        save_dir = '/home/toniomirri/Images_evolution'
                        file_name = f'image_{data_iter_step}.png'

                        display_images(original,masked_image,reconstructed_imgs,save_dir,file_name,rec_losses,class_losses,steps)

        if data_iter_step % 50 == 1:
            print('step: {}, acc {} rec-loss {}'.format(data_iter_step, np.mean(all_results[-1]), loss_value))
        if data_iter_step % 500 == 499 or (data_iter_step == dataset_len - 1):
            with open(os.path.join(args.output_dir, f'results_{data_iter_step}.npy'), 'wb') as f:
                np.save(f, np.array(all_results))
            with open(os.path.join(args.output_dir, f'losses_{data_iter_step}.npy'), 'wb') as f:
                np.save(f, np.array(all_losses))
            all_results = [list() for i in range(args.steps_per_example)]
            all_losses = [list() for i in range(args.steps_per_example)]

        if data_iter_step % (args.number_of_example_reinitialize - 1) == 0 and args.number_of_example_reinitialize > 0 :
            print(f"Reinitializing model after {args.number_of_example_reinitialize} examples...")
            model, optimizer, loss_scaler = _reinitialize_model(base_model, base_optimizer, base_scalar, clone_model, args, device)

    save_accuracy_results(args)

    if args.save_mae_online : 
        torch.save({'model' : model.state_dict()},'/home/toniomirri/checkpoints/latest_online_weights.pth')

    # gather the stats from all processes
    try:
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    except:
        pass
    return




def save_accuracy_results(args):
    # Initialisation des résultats pour chaque étape
    all_all_results = [list() for i in range(args.steps_per_example)]

    # Calcul dynamique du nombre d'images
    # On récupère les fichiers .npy présents dans le dossier des résultats
    result_files = glob.glob(os.path.join(args.output_dir, 'results_*.npy'))
    if len(result_files) > 0:
        # Charger un des fichiers pour déterminer le nombre d'images
        sample_data = np.load(result_files[0])
        num_images = len(sample_data[0]) * len(result_files)
    else:
        raise ValueError(f"Aucun fichier 'results_*.npy' trouvé dans {args.output_dir}")

    print(f"Nombre total d'images calculé : {num_images}")

    # Chargement des fichiers de résultats
    for file_number, f_name in enumerate(result_files):
        all_data = np.load(f_name)
        for step in range(args.steps_per_example):
            all_all_results[step] += all_data[step].tolist()

    # Indiquer que le modèle est finalisé
    with open(os.path.join(args.output_dir, 'model-final.pth'), 'w') as f:
        f.write(f'Done!\n')

    # Sauvegarde des résultats d'accuracy
    with open(os.path.join(args.output_dir, 'accuracy.txt'), 'a') as f:
        f.write(f'{str(args)}\n')
        for i in range(args.steps_per_example):
            assert len(all_all_results[i]) == num_images, f"Expected {num_images}, but got {len(all_all_results[i])}"
            f.write(f'{i}\t{np.mean(all_all_results[i])}\n')
