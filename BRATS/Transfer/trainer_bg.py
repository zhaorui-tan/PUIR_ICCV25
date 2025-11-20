# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
from torchvision.utils import save_image, make_grid

from monai import data, transforms

from monai.data import decollate_batch
from utils.ops import aug_rand, rot_rand




def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):

    def contrast_loss(x_i, x_j, temp=0.1):
        sim = torch.einsum("nc,kc->nk",[x_i,x_j])
        l_pos = torch.diag(sim).unsqueeze(-1)  # positive logits Nx1
        neg_mask = torch.eye(sim.shape[0], sim.shape[0], dtype=float).cuda()
        l_neg = sim[neg_mask!=1].reshape((sim.shape[0],sim.shape[0]-1)) # negative logits Nx(N-1)
        logits = torch.cat([l_pos,l_neg],dim=1)
        logits /= temp
        labels = torch.zeros(logits.shape[0],dtype=torch.long).cuda()
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return loss

    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        all_modal = []
        for i in ['t1n', 't1c', 't2w',  't2f', ]:
            tmp= batch_data[i].squeeze(1)
            all_modal.append(tmp)
  
        data = torch.stack(all_modal, dim=1)
        target = batch_data['seg']
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        loss = 0.0

        mask = data.ge(-0.35)
        
        with autocast(enabled=args.amp):
            current_semantic_anchor = None
            modal_index = np.array([0,1,2,3])
            np.random.shuffle(modal_index)
            # print(modal_index)
            for i in modal_index:
                step_loss = 0.0
                x1 = data[:,i,...][:,None,...].repeat(1,4,1,1,1)
                x1, rot1 = rot_rand(args, x1)
                x1_augment = aug_rand(args, x1)
                rot1_p, contrastive1_p, semantic, out1 = model(x1_augment.detach())

                with torch.no_grad():
                    x2, rot2 = rot_rand(args, x1)
                    x2_augment = aug_rand(args, x2)
                    _, contrastive2_p, _, _ = model(x2_augment.detach())
                

                rot_loss = torch.nn.CrossEntropyLoss()(rot1_p, rot1.detach())
                con_loss = contrast_loss(contrastive1_p, contrastive2_p.detach())
                step_loss += rot_loss + con_loss
                for i in range(len(out1)):
                    out1[i] = out1[i].rot90(4-rot1[i], (2,3)) 

                if  mask.int().sum() > 0:
                    tmp_loss = torch.nn.MSELoss(reduction='sum')(out1*mask, data.detach()*mask) * 10
                    tmp_loss = tmp_loss / mask.int().sum()
                    tmp_loss += torch.nn.MSELoss(reduction='mean')(out1, data.detach()) 
                    step_loss += tmp_loss
                else:
                    tmp_loss = torch.nn.MSELoss(reduction='mean')(out1*mask, data.detach()*mask) * 10
                    tmp_loss += torch.nn.MSELoss(reduction='mean')(out1, data.detach()) 
                    step_loss += tmp_loss


                if current_semantic_anchor is not None:
                    semantic_loss = loss_func(semantic, current_semantic_anchor.detach())
                    step_loss += semantic_loss
                    current_semantic_anchor = (current_semantic_anchor + semantic)/2
                    current_semantic_anchor = current_semantic_anchor.detach()   
                else:
                    current_semantic_anchor = semantic.detach()

            
                if args.amp:
                    scaler.scale(step_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    step_loss.backward()
                    optimizer.step()
                loss += step_loss
        
        
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=4*args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg

from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    EnsureType,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)
from utils.utils import dice, resample_3d

def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    post_trans_output = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_trans_gt = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    ###
    run_acc = AverageMeter()
    start_time = time.time()

    with torch.no_grad():
        mse_list_case = []
        counter = 0
        for batch_data in loader:
            val_outputs = []
            all_modal = []
            for i in ['t1n', 't1c', 't2w',  't2f', ]:
                tmp= batch_data[i].squeeze(1)
                all_modal.append(tmp)
            data = torch.stack(all_modal, dim=1)
            data = data.cuda(args.rank)
            mse = 0.
            mask = data.ge(-0.35)
            for i in range(4): #['t1c', 't1n', 't2w',  't2f', ]:
                val_output = model_inferer(data[:,i,...].repeat(1,4,1,1,1))
                if  mask.int().sum() > 0:
                    mse = torch.nn.MSELoss(reduction='sum')(val_output*mask, data.detach()*mask) 
                    mse = mse / mask.int().sum()
                else:
                    mse = torch.nn.MSELoss(reduction='mean')(val_output*mask, data.detach()*mask) 

                val_output = val_output*mask + data*(1-mask.int()).bool()
                val_outputs.append(val_output)
            mse_list_case.append(mse.item())
            # print(mse)
            counter += 1
            if counter > 5:
                break

        mse_list_case = np.array(mse_list_case)
        mse_list_case = mse_list_case[mse_list_case!=None]
        if args.distributed:
            MSE_list = distributed_all_gather(
                [torch.from_numpy(mse_list_case.astype(np.float32)).cuda(args.gpu)],out_list=True
            )
            total_mse_list = []
            for dl in MSE_list[0]:
                total_mse_list=total_mse_list+dl
        else:
            total_mse_list = mse_list_case
        if args.rank == 0:
            print(
                "time {:.2f}s".format(time.time() - start_time),
                "MSE {:.4f} total data {}".format(np.mean(total_mse_list),len(total_mse_list))
            )
    
    return np.mean(total_mse_list), val_outputs, data


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_mse_min = 10000.0
    epoch = 0

    print('Begining! start to validate')


    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()


        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )

        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)

        b_new_best = False
        if args.rank == 0 and args.logdir is not None and (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, epoch, args, best_acc=val_mse_min, filename="model_final_{}.pt".format(str(epoch)))
      
        if (epoch) % args.val_every == 0:
            print('start to validate')
            
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_mse, val_outputs, target = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            if args.rank == 0 and writer is not None:
                write_fig(val_outputs, target, writer, epoch)

            val_avg_mse = np.mean(val_mse)

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_mse,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_mse, epoch)
                if val_avg_mse < val_mse_min:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_mse_min, val_avg_mse))
                    val_mse_min = val_avg_mse
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_mse_min, optimizer=optimizer, scheduler=scheduler
                        )
        
        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best MSE: ", val_mse_min)
    return val_mse_min



def write_fig(val_outputs_1, target, writer, epoch):
    val_outputs_1 = [i.detach().cpu().numpy() for i in val_outputs_1]
    target_image = target.detach().cpu().numpy() 

    keys = ['t1n', 't1c', 't2w',  't2f', ]
    num = val_outputs_1[0][0][0].shape[0] // 2
    num1 = val_outputs_1[0][0][0].shape[1] // 2
    num2 = val_outputs_1[0][0][0].shape[2] // 2

    for i in range(4):
        key = keys[i]
        current_for_vis = val_outputs_1[i]
        current_for_vis = [
            (current_for_vis[0,0,num,:,:] -current_for_vis[0,0,:,:,:].min() )/(current_for_vis[0,0,:,:,:].max() - current_for_vis[0,0,:,:,:].min() ), 
            (current_for_vis[0,1,num,:,:] -current_for_vis[0,1,:,:,:].min() )/(current_for_vis[0,1,:,:,:].max() - current_for_vis[0,1,:,:,:].min() ), 
            (current_for_vis[0,2,num,:,:] -current_for_vis[0,2,:,:,:].min() )/(current_for_vis[0,2,:,:,:].max() - current_for_vis[0,2,:,:,:].min() ), 
            (current_for_vis[0,3,num,:,:] -current_for_vis[0,3,:,:,:].min() )/(current_for_vis[0,3,:,:,:].max() - current_for_vis[0,3,:,:,:].min() ), 
            (current_for_vis[0,0,:,num1,:] -current_for_vis[0,0,:,:,:].min() )/(current_for_vis[0,0,:,:,:].max() - current_for_vis[0,0,:,:,:].min() ), 
            (current_for_vis[0,1,:,num1,:] -current_for_vis[0,1,:,:,:].min() )/(current_for_vis[0,1,:,:,:].max() - current_for_vis[0,1,:,:,:].min() ), 
            (current_for_vis[0,2,:,num1,:] -current_for_vis[0,2,:,:,:].min() )/(current_for_vis[0,2,:,:,:].max() - current_for_vis[0,2,:,:,:].min() ), 
            (current_for_vis[0,3,:,num1,:] -current_for_vis[0,3,:,:,:].min() )/(current_for_vis[0,3,:,:,:].max() - current_for_vis[0,3,:,:,:].min() ), 
        ]

        fixed_grid = make_grid(torch.tensor(current_for_vis).unsqueeze(1), nrow=4, value_range=(0, 1), normalize=True)
        writer.add_image(f'GenImages_{key}', fixed_grid, epoch)
        
        current_for_vis = val_outputs_1[i]
        current_for_vis = [
            (current_for_vis[0,0,:,:,num2] -current_for_vis[0,0,:,:,:].min() )/(current_for_vis[0,0,:,:,:].max() - current_for_vis[0,0,:,:,:].min() ), 
            (current_for_vis[0,1,:,:,num2] -current_for_vis[0,1,:,:,:].min() )/(current_for_vis[0,1,:,:,:].max() - current_for_vis[0,1,:,:,:].min() ), 
            (current_for_vis[0,2,:,:,num2] -current_for_vis[0,2,:,:,:].min() )/(current_for_vis[0,2,:,:,:].max() - current_for_vis[0,2,:,:,:].min() ), 
            (current_for_vis[0,3,:,:,num2] -current_for_vis[0,3,:,:,:].min() )/(current_for_vis[0,3,:,:,:].max() - current_for_vis[0,3,:,:,:].min() ), 
            (current_for_vis[0,0,:,:,int(num2*1.5)] -current_for_vis[0,0,:,:,:].min() )/(current_for_vis[0,0,:,:,:].max() - current_for_vis[0,0,:,:,:].min() ),  
            (current_for_vis[0,1,:,:,int(num2*1.5)] -current_for_vis[0,1,:,:,:].min() )/(current_for_vis[0,1,:,:,:].max() - current_for_vis[0,1,:,:,:].min() ),  
            (current_for_vis[0,2,:,:,int(num2*1.5)] -current_for_vis[0,2,:,:,:].min() )/(current_for_vis[0,2,:,:,:].max() - current_for_vis[0,2,:,:,:].min() ), 
            (current_for_vis[0,3,:,:,int(num2*1.5)] -current_for_vis[0,3,:,:,:].min() )/(current_for_vis[0,3,:,:,:].max() - current_for_vis[0,3,:,:,:].min() ), 
            (current_for_vis[0,0,:,:,int(num2*0.5)] -current_for_vis[0,0,:,:,:].min() )/(current_for_vis[0,0,:,:,:].max() - current_for_vis[0,0,:,:,:].min() ),  
            (current_for_vis[0,1,:,:,int(num2*0.5)] -current_for_vis[0,1,:,:,:].min() )/(current_for_vis[0,1,:,:,:].max() - current_for_vis[0,1,:,:,:].min() ),  
            (current_for_vis[0,2,:,:,int(num2*0.5)] -current_for_vis[0,2,:,:,:].min() )/(current_for_vis[0,2,:,:,:].max() - current_for_vis[0,2,:,:,:].min() ), 
            (current_for_vis[0,3,:,:,int(num2*0.5)] -current_for_vis[0,3,:,:,:].min() )/(current_for_vis[0,3,:,:,:].max() - current_for_vis[0,3,:,:,:].min() ), 
        ]

        fixed_grid = make_grid(torch.tensor(current_for_vis).unsqueeze(1), nrow=4, value_range=(0, 1), normalize=True)
        writer.add_image(f'GenImages_{key}_top', fixed_grid, epoch)

    current_for_vis = [
        (target_image[0,0,num,:,:] -target_image[0,0,:,:,:].min() )/(target_image[0,0,:,:,:].max() - target_image[0,0,:,:,:].min() ),  
        (target_image[0,1,num,:,:] -target_image[0,1,:,:,:].min() )/(target_image[0,1,:,:,:].max() - target_image[0,1,:,:,:].min() ),  
        (target_image[0,2,num,:,:] -target_image[0,2,:,:,:].min() )/(target_image[0,2,:,:,:].max() - target_image[0,2,:,:,:].min() ), 
        (target_image[0,3,num,:,:] -target_image[0,3,:,:,:].min() )/(target_image[0,3,:,:,:].max() - target_image[0,3,:,:,:].min() ), 
        (target_image[0,0,:,num1,:] -target_image[0,0,:,:,:].min() )/(target_image[0,0,:,:,:].max() - target_image[0,0,:,:,:].min() ),  
        (target_image[0,1,:,num1,:] -target_image[0,1,:,:,:].min() )/(target_image[0,1,:,:,:].max() - target_image[0,1,:,:,:].min() ),  
        (target_image[0,2,:,num1,:] -target_image[0,2,:,:,:].min() )/(target_image[0,2,:,:,:].max() - target_image[0,2,:,:,:].min() ), 
        (target_image[0,3,:,num1,:] -target_image[0,3,:,:,:].min() )/(target_image[0,3,:,:,:].max() - target_image[0,3,:,:,:].min() ), 
        ]
    fixed_grid = make_grid(torch.tensor(current_for_vis).unsqueeze(1), nrow=4, value_range=(0, 1), normalize=True)
    writer.add_image('Target', fixed_grid, epoch)

    current_for_vis = [
        (target_image[0,0,:,:,num2] -target_image[0,0,:,:,:].min() )/(target_image[0,0,:,:,:].max() - target_image[0,0,:,:,:].min() ),  
        (target_image[0,1,:,:,num2] -target_image[0,1,:,:,:].min() )/(target_image[0,1,:,:,:].max() - target_image[0,1,:,:,:].min() ),  
        (target_image[0,2,:,:,num2] -target_image[0,2,:,:,:].min() )/(target_image[0,2,:,:,:].max() - target_image[0,2,:,:,:].min() ), 
        (target_image[0,3,:,:,num2] -target_image[0,3,:,:,:].min() )/(target_image[0,3,:,:,:].max() - target_image[0,3,:,:,:].min() ), 
        (target_image[0,0,:,:,int(num2*1.5)] -target_image[0,0,:,:,:].min() )/(target_image[0,0,:,:,:].max() - target_image[0,0,:,:,:].min() ),  
        (target_image[0,1,:,:,int(num2*1.5)] -target_image[0,1,:,:,:].min() )/(target_image[0,1,:,:,:].max() - target_image[0,1,:,:,:].min() ),  
        (target_image[0,2,:,:,int(num2*1.5)] -target_image[0,2,:,:,:].min() )/(target_image[0,2,:,:,:].max() - target_image[0,2,:,:,:].min() ), 
        (target_image[0,3,:,:,int(num2*1.5)] -target_image[0,3,:,:,:].min() )/(target_image[0,3,:,:,:].max() - target_image[0,3,:,:,:].min() ), 
        (target_image[0,0,:,:,int(num2*0.5)] -target_image[0,0,:,:,:].min() )/(target_image[0,0,:,:,:].max() - target_image[0,0,:,:,:].min() ),  
        (target_image[0,1,:,:,int(num2*0.5)] -target_image[0,1,:,:,:].min() )/(target_image[0,1,:,:,:].max() - target_image[0,1,:,:,:].min() ),  
        (target_image[0,2,:,:,int(num2*0.5)] -target_image[0,2,:,:,:].min() )/(target_image[0,2,:,:,:].max() - target_image[0,2,:,:,:].min() ), 
        (target_image[0,3,:,:,int(num2*0.5)] -target_image[0,3,:,:,:].min() )/(target_image[0,3,:,:,:].max() - target_image[0,3,:,:,:].min() ), 
    ]
    fixed_grid = make_grid(torch.tensor(current_for_vis).unsqueeze(1), nrow=4, value_range=(0, 1), normalize=True)
    writer.add_image('Target_top', fixed_grid, epoch)