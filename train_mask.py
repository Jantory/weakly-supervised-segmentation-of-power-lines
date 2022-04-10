# coding = utf-8
# usr/bin/env python

from __future__ import print_function
import os
import glob
import itertools
import numpy as np
from random import randint
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import torch as T
import torch.nn as nn
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal
from lib.utils import *
# training set:
from datasets_VLIR import ImageDataset
import matplotlib.pyplot as plt
from utils import mask_generator, QueueMask
from tensorboard.compat.proto.types_pb2 import DT_COMPLEX128_REF


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReplicationPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReplicationPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator_S2F(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator_S2F, self).__init__()

        # Initial convolution block
        model = [   nn.ReplicationPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReplicationPad2d(3),
                    nn.Conv2d(64, output_nc, 7) ]
                    #nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return (self.model(x) + x).tanh() #(min=-1, max=1) #just learn a residual


class Generator_F2S(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator_F2S, self).__init__()

        # Initial convolution block
        model = [   nn.ReplicationPad2d(3),
                    nn.Conv2d(input_nc + 1, 64, 7), # + mask
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReplicationPad2d(3),
                    nn.Conv2d(64, output_nc, 7) ]
                    #nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x, m):
        return (self.model(T.cat((x, m), 1)) + x).tanh() #(min=-1, max=1) #just learn a residual


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1) #global avg pool

def main():
    output_path = "Output_mask"
    G_step_over_D = 4
    G_GANloss_w = 1
    G_cycleloss_w = 10
    G_identityloss_w = 1

    batchSize = 16

    start_epoch = 0
    decay_epoch = 100
    n_epochs = 200

    cuda_available = False
    glr = 1e-3
    dlr = 1e-4

    ## create the output folder
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    tsboard = SummaryWriter(os.path.join(output_path, 'TSboard'))

    ## check for cuda
    if T.cuda.is_available():
        cuda_available = True
        print('Cuda is availabe')

    ###### Definition of Variables ######
    ## network
    netG_A2B = Generator_S2F(3, 3, True)
    netG_B2A = Generator_F2S(3, 3, True)
    netD_A = Discriminator(3)
    netD_B = Discriminator(3)

    if cuda_available:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    ## optimizers & LR schedulers
    optimizer_G = T.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=glr,
                               betas=(0.5, 0.999))
    optimizer_D_A = T.optim.Adam(netD_A.parameters(), lr=dlr, betas=(0.5, 0.999))
    optimizer_D_B = T.optim.Adam(netD_B.parameters(), lr=dlr, betas=(0.5, 0.999))

    lr_scheduler_G = T.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)
    lr_scheduler_D_A = T.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)
    lr_scheduler_D_B = T.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)

    ## losses
    criterion_GAN = T.nn.MSELoss()
    criterion_cycle = T.nn.MSELoss()
    criterion_identity = T.nn.MSELoss()

    from torchvision.transforms.transforms import ToTensor
    ## inputs and targets memory allocation
    Tensor = T.cuda.FloatTensor if cuda_available else T.Tensor
    input_A = Tensor(batchSize, 3, 128, 128)
    input_B = Tensor(batchSize, 3, 128, 128)
    mask_wire = Tensor(batchSize, 1, 128, 128)
    target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)
    mask_non_wire = Variable(Tensor(batchSize, 1, 128, 128).fill_(0.0), requires_grad=False)
    masks_path = sorted(glob.glob(os.path.join('masks', '*.bmp')))
    # masks = [T.from_numpy(np.array(Image.open(p))).type(T.FloatTensor).unsqueeze(0) for p in masks_path]

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    transforms_ = [
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    mask_transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
    mask_transforms = transforms.Compose(mask_transforms_)
    masks = [mask_transforms(Image.open(p)) for p in masks_path]

    test_set_size = 40
    dataset = ImageDataset('/home/TUE/20201168/VL_SIM', transforms_=transforms_, unaligned=True, test_set_size=test_set_size)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=1, drop_last=True)

    plt.ioff()
    to_pil = transforms.ToPILImage()

    for i in range(5):
        data = dataset.getTestPair(i)
        fake_B = data['A'].unsqueeze(0).cuda()
        fake_A = data['B'].unsqueeze(0).cuda()
        tsboard.add_images("Test/img%d/fake_B" % i, (fake_B + 1) * 0.5, 0, dataformats='NCHW')
        tsboard.add_images("Test/img%d/fake_A" % i, (fake_A + 1) * 0.5, 0, dataformats='NCHW')
        tsboard.add_images("Test/img%d/mask" % i, mask_non_wire, 0, dataformats='NCHW')

        data = dataset.__getitem__(i * 10)
        fake_B = data['A'].unsqueeze(0).cuda()
        fake_A = data['B'].unsqueeze(0).cuda()
        tsboard.add_images("Train/img%d/fake_B" % i, (fake_B + 1) * 0.5, 0, dataformats='NCHW')
        tsboard.add_images("Train/img%d/fake_A" % i, (fake_A + 1) * 0.5, 0, dataformats='NCHW')
        tsboard.add_images("Train/img%d/mask" % i, mask_non_wire, 0, dataformats='NCHW')

    ###### training ######
    for epoch in range(start_epoch, n_epochs):
        losses = {'loss_G': [], 'loss_identity_A': [], 'loss_identity_B': [], 'loss_GAN_A2B': [], 'loss_GAN_B2A': [],
                  'loss_cycle_ABA': [], 'loss_cycle_BAB': [], 'loss_D_A': [], 'loss_D_B': []}

        for i, batch in enumerate(dataloader):
            ## set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))
            mask = Variable(mask_wire.copy_(masks[randint(0, 4)].unsqueeze(0).repeat(batchSize, 1, 1, 1)))

            #### Generator A2B and B2A ####
            optimizer_G.zero_grad()

            ## identity loss
            same_A = netG_B2A(real_A, mask_non_wire)
            same_B = netG_A2B(real_B)
            loss_identity_A = criterion_identity(same_A, real_A) * G_identityloss_w
            loss_identity_B = criterion_identity(same_B, real_B) * G_identityloss_w

            ## GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake_B = netD_B(fake_B)
            fake_A = netG_B2A(real_B, mask)
            pred_fake_A = netD_A(fake_A)
            loss_GAN_A2B = criterion_GAN(pred_fake_B, target_real) * G_GANloss_w
            loss_GAN_B2A = criterion_GAN(pred_fake_A, target_real) * G_GANloss_w

            ## cycle loss
            recovered_A = netG_B2A(fake_B, mask_generator(real_A, fake_B))
            recovered_B = netG_A2B(fake_A)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * G_cycleloss_w
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * G_cycleloss_w

            ## total loss
            # loss_identity_A = T.tensor(0)
            # loss_identity_B = T.tensor(0)
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            ## save the losses
            losses["loss_G"].append(loss_G.item())
            losses["loss_identity_A"].append(loss_identity_A.item())
            losses["loss_identity_B"].append(loss_identity_B.item())
            losses["loss_GAN_A2B"].append(loss_GAN_A2B.item())
            losses["loss_GAN_B2A"].append(loss_GAN_B2A.item())
            losses["loss_cycle_ABA"].append(loss_cycle_ABA.item())
            losses["loss_cycle_BAB"].append(loss_cycle_BAB.item())

            loss_G.backward()
            optimizer_G.step()

            if i % (G_step_over_D + 1) == 0:
                #### Discriminator A ####
                optimizer_D_A.zero_grad()

                ## real loss
                pred_real_A = netD_A(real_A)
                loss_D_real = criterion_GAN(pred_real_A, target_real)

                ## fake loss
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake_A = netD_A(fake_A.detach())
                loss_D_fake = criterion_GAN(pred_fake_A, target_fake)

                ## total loss
                loss_D_A = (loss_D_real + loss_D_fake) * 0.5

                losses["loss_D_A"].append(loss_D_A.item())

                loss_D_A.backward()
                optimizer_D_A.step()

                #### Discriminator B ####
                optimizer_D_B.zero_grad()

                ## real loss
                pred_real_B = netD_B(real_B)
                loss_D_real = criterion_GAN(pred_real_B, target_real)

                ## fake loss
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake_B = netD_B(fake_B.detach())
                loss_D_fake = criterion_GAN(pred_fake_B, target_fake)

                ## total loss
                loss_D_B = (loss_D_real + loss_D_fake) * 0.5

                losses["loss_D_B"].append(loss_D_B.item())

                loss_D_B.backward()
                optimizer_D_B.step()

        tsboard.add_scalar("Learning Rate/Generator", lr_scheduler_G.get_last_lr()[0], epoch)
        tsboard.add_scalar("Learning Rate/Discriminator", lr_scheduler_D_A.get_last_lr()[0], epoch)
        tsboard.add_scalar("Generator/Total", np.mean(losses["loss_G"][-len(dataloader):]), epoch)
        tsboard.add_scalar("Generator/loss_identity_A", np.mean(losses["loss_identity_A"][-len(dataloader):]), epoch)
        tsboard.add_scalar("Generator/loss_identity_B", np.mean(losses["loss_identity_B"][-len(dataloader):]), epoch)
        tsboard.add_scalar("Generator/loss_GAN_A2B", np.mean(losses["loss_GAN_A2B"][-len(dataloader):]), epoch)
        tsboard.add_scalar("Generator/loss_GAN_B2A", np.mean(losses["loss_GAN_B2A"][-len(dataloader):]), epoch)
        tsboard.add_scalar("Generator/loss_cycle_ABA", np.mean(losses["loss_cycle_ABA"][-len(dataloader):]), epoch)
        tsboard.add_scalar("Generator/loss_cycle_BAB", np.mean(losses["loss_cycle_BAB"][-len(dataloader):]), epoch)

        tsboard.add_scalar("Discriminator/loss_D_A", np.mean(losses["loss_D_A"][-len(dataloader):]), epoch)
        tsboard.add_scalar("Discriminator/loss_D_B", np.mean(losses["loss_D_B"][-len(dataloader):]), epoch)

        #### visualize some images ####
        if (epoch + 1) % 200 == 0 or epoch in [0, 1, 2, 5, 10, 100]:
            for i in range(5):
                data = dataset.getTestPair(i)
                real_A = data['A'].unsqueeze(0).cuda()
                fake_B = netG_A2B(real_A)
                real_B = data['B'].unsqueeze(0).cuda()
                fake_A = netG_B2A(real_B, masks[randint(0, 4)].unsqueeze(0).cuda())

                tsboard.add_images("Test/img%d/fake_B" % i, (fake_B + 1) * 0.5, epoch + 1, dataformats='NCHW')
                tsboard.add_images("Test/img%d/fake_A" % i, (fake_A + 1) * 0.5, epoch + 1, dataformats='NCHW')
                tsboard.add_images("Test/img%d/mask" % i, mask_generator(real_A, fake_B), epoch + 1, dataformats='NCHW')

                data = dataset.__getitem__(i * 10)
                real_A = data['A'].unsqueeze(0).cuda()
                fake_B = netG_A2B(real_A)
                real_B = data['B'].unsqueeze(0).cuda()
                fake_A = netG_B2A(real_B, masks[randint(0, 4)].unsqueeze(0).cuda())

                tsboard.add_images("Train/img%d/fake_B" % i, (fake_B + 1) * 0.5, epoch + 1, dataformats='NCHW')
                tsboard.add_images("Train/img%d/fake_A" % i, (fake_A + 1) * 0.5, epoch + 1, dataformats='NCHW')
                tsboard.add_images("Train/img%d/mask" % i, mask_generator(real_A, fake_B), epoch + 1,
                                   dataformats='NCHW')

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        print("Epoch %d/300 epochs finshed" % (epoch + 1))

if __name__ == '__main__':
    main()
