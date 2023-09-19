# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from torch_utils.clc_buffer import CLCBuffer
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        
        # br coef
        #self.P_coeff, self.I_coeff, self.D_coeff = 1.0, 0.1, 0
        self.P_coeff, self.I_coeff, self.D_coeff = 1.0, 0.1, 0
        self.batch_size = 4
        self.cuda = True
        self.use_labels = False
        self.pid_buffer_factor = 100 
        self.rou1 = 0.0001
        self.rou2 = 0.000001
        
        if self.I_coeff > 0.0:
            self.integral_pid_type = 'br';
            if self.integral_pid_type == "accurate":
                self.max0 = torch.nn.ReLU()
            self.clc_buffer_real = CLCBuffer(self.pid_buffer_factor * self.batch_size, use_labels=self.use_labels)
            self.clc_buffer_fake = CLCBuffer(self.pid_buffer_factor * self.batch_size, use_labels=self.use_labels)
        if self.D_coeff > 0.0:
            self.last_real_images = None
            self.last_fake_images = None 
        

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def get_torch_variable(self, arg):
        if (torch.is_tensor(arg)):
            arg = arg.clone().detach().requires_grad_(True)
        else:
            arg = torch.tensor(arg)
            
        if self.cuda:
            arg = arg.cuda()
            
        return arg
        
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()
                
        # Gpl: Apply path length regularization.
        if do_Gpl:
        #if False:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight * float(torch.normal(torch.tensor(1.0), torch.tensor(0.1)))
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain or do_Dr1:
            # loss fake
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = self.P_coeff * torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()
            # loss real
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = self.P_coeff * torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                # gradient penaly
                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)
            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()
            images = real_img
            fake_images = gen_img
            
            if self.I_coeff > 0.0:
            # Get a batch of images (real and fake) extracted randomly (without replacement) among the most recent ones (including the current ones)
                if self.use_labels:
                    self.clc_buffer_real.insert(images.cpu(), y.cpu())
                    self.clc_buffer_fake.insert(fake_images.cpu(), y.cpu())
                    recent_real_images, recent_yreal = self.clc_buffer_real.get(self.batch_size)
                    recent_real_images, recent_yreal = self.get_torch_variable(recent_real_images), self.get_torch_variable(recent_yreal)
                    recent_fake_images, recent_yfake = self.clc_buffer_fake.get(self.batch_size)
                    recent_fake_images, recent_yfake = self.get_torch_variable(recent_fake_images), self.get_torch_variable(recent_yfake)
                    # Pass the samples (real and fake) through the discriminator, and compute the losses
                    integral_d_out_real = self.run_D(recent_real_images, recent_yreal, sync=sync)
                    integral_d_out_fake = self.run_D(recent_fake_images, recent_yfake, sync=sync)
                else:
                    self.clc_buffer_real.insert(images.cpu())
                    self.clc_buffer_fake.insert(fake_images.cpu())
                    recent_real_images = self.get_torch_variable(self.clc_buffer_real.get(self.batch_size))
                    recent_fake_images = self.get_torch_variable(self.clc_buffer_fake.get(self.batch_size))
                    # Pass the samples (real and fake) through the discriminator, and compute the losses
                    integral_d_out_real = torch.nn.functional.softplus(-self.run_D(recent_real_images, None, sync=sync))
                    integral_d_out_fake = torch.nn.functional.softplus(self.run_D(recent_fake_images, None, sync=sync))
                
                integral_loss_real = integral_d_out_real.mean()
                integral_loss_fake = integral_d_out_fake.mean()

                # Compute the overall Integral loss using the formula below
                if self.integral_pid_type == "function":
                    integral_loss = (integral_loss_real + integral_loss_fake) * self.I_coeff
                elif self.integral_pid_type == "square":
                    integral_loss = ((integral_d_out_real ** 2).mean() + (integral_d_out_fake ** 2).mean()) * self.I_coeff
                elif self.integral_pid_type == "abs":
                    integral_loss = (torch.abs(integral_d_out_real).mean() + torch.abs(integral_d_out_fake).mean()) * self.I_coeff
                elif self.integral_pid_type == "accurate":
                    integral_loss = (self.max0(integral_d_out_fake) - (-1 * self.max0(-1 * integral_d_out_real))).mean() * self.I_coeff
                elif self.integral_pid_type == "br":
                    br1 = torch.abs(torch.normal(torch.zeros(1), torch.ones(1)).cuda())
                    br2 = torch.abs(torch.normal(torch.zeros(1), torch.ones(1)).cuda())
                    integral_loss = self.I_coeff * (0.5 * self.rou1 * br1 * (pow(integral_d_out_real,2)).mean() + br2 * (0.25* self.rou2 * pow(integral_d_out_real, 4) + 0.5 * self.rou2 * pow(integral_d_out_real, 2)).mean() + 0.5 * self.rou1 * br1 * (pow(integral_d_out_fake,2)).mean() + br2 * (0.25* self.rou2 * pow(integral_d_out_fake, 4) + 0.5 * self.rou2 * pow(integral_d_out_fake, 2)).mean())
                
                # compute D's gradients in backward pass
                integral_loss.mul(gain).backward()

            ## PID Control (D): if we are using derivative control (i.e. the D coefficient is >0.0), we apply that component here:
            derivative_loss = torch.from_numpy(np.array([0.0]))
            if self.D_coeff > 0.0:
                if self.last_real_images is None:
                    self.last_real_images = images.cpu()
                    self.last_fake_images = fake_images.cpu()
                    if self.use_labels:
                        self.last_y = y
                else:
                    if self.use_labels:
                        derivative_d_out_real_last = self.D(self.get_torch_variable(self.last_real_images), self.get_torch_variable(self.last_y))
                        derivative_d_out_fake_last = self.D(self.get_torch_variable(self.last_fake_images), self.get_torch_variable(self.last_y))
                    else:
                        derivative_d_out_real_last = self.D(self.get_torch_variable(self.last_real_images))
                        derivative_d_out_fake_last = self.D(self.get_torch_variable(self.last_fake_images))
                    derivative_loss_real_last = (2*1-1)*derivative_d_out_real_last.mean()
                    derivative_loss_fake_last = (2*0-1)*derivative_d_out_fake_last.mean()
                    derivative_loss_last = derivative_loss_real_last + derivative_loss_fake_last
                    
                    if self.use_labels:
                        derivative_d_out_real_current = self.run_D(self.get_torch_variable(images),y, sync=sync)
                        derivative_d_out_fake_current = self.run_D(self.get_torch_variable(fake_images), y, sync=sync)
                    else:
                        derivative_d_out_real_current = self.D(self.get_torch_variable(images))
                        derivative_d_out_fake_current = self.D(self.get_torch_variable(fake_images))
                    derivative_loss_real_current = (2*1-1)*derivative_d_out_real_current.mean()
                    derivative_loss_fake_current = (2*0-1)*derivative_d_out_fake_current.mean()
                    derivative_loss_current = derivative_loss_real_current + derivative_loss_fake_current

                    derivative_loss = (derivative_loss_current - derivative_loss_last) * self.D_coeff
                    derivative_loss.mul(gain).backward()

                    self.last_real_images = images
                    self.last_fake_images = fake_images
                    if self.use_labels:
                        self.last_y = y

#----------------------------------------------------------------------------
