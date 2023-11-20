import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from Image_Segmentation.image_segmentation.evaluation import EvaluationMatrix
from Image_Segmentation.image_segmentation.model import (AttU_Net, R2AttU_Net,
                                                         R2U_Net, U_Net)
from Image_Segmentation.image_segmentation.utils.visualize import \
    ImageVisualization
from torch import optim
from torch.utils import data


class Solver(object):
    def __init__(self, config):
        # Data
        self.image_size = config.image_size

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.BCELoss()
        self.augmentation_prob = config.augmentation_prob
        self.model_type = config.model_type

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.best_unet_score = 0.

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained_unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' % (
            self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))

        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""

        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=self.img_ch,
                                output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3, output_ch=1, t=self.t)

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)

        # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):

        isexist = os.path.isfile(self.pretrained_unet_path)
        if isexist:
            self.unet.load_state_dict(torch.load(self.pretrained_unet_path))
            # print(f"Loading pretrained U-Net model at {self.pretrained_unet_path}.")

        return isexist

    def train(self, train_loader, valid_loader, test_loader):
        """Train encoder, generator and discriminator."""

        isexist = self.load_pretrained_model()

        if not isexist:
            for epoch in range(self.num_epochs):

                self.unet.train(True)
                epoch_loss = 0.0
                length = 0

                eval_matrix = EvaluationMatrix()
                for i, (images, GT) in enumerate(train_loader):
                    # GT : Ground Truth
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    GT_flat = GT.view(GT.size(0), -1)

                    # SR : Segmentation Result
                    SR = torch.sigmoid(self.unet(images))
                    SR_flat = SR.view(SR.size(0), -1)

                    loss = self.criterion(SR_flat, GT_flat)
                    epoch_loss += loss.item()

                    self.unet.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pred = SR > 0.5
                    eval_matrix.update(pred, GT)

                    length += images.size(0)

                print('Epoch [%d/%d], Loss: %.4f,' % (
                    epoch+1, self.num_epochs,
                    epoch_loss))
                eval_matrix.update_print('[Training] ')

                if epoch % self.val_step == 0:
                    self.validation(valid_loader, epoch)

                # visualize
                vis = ImageVisualization(self.image_size, self.result_path)
                for i in range(3):
                    vis.visualize_pred_gt(epoch, images[i], pred[i], GT[i])
                

        self.test(test_loader)

    def validation(self, valid_loader, epoch):
        '''
        Validation
        '''
        self.unet.eval()

        length = 0
        val_eval_matrix = EvaluationMatrix()
        
        for i, (images, GT) in enumerate(valid_loader):

            images = images.to(self.device)
            GT = GT.to(self.device)

            SR = torch.sigmoid(self.unet(images))

            pred = SR > 0.5
            val_eval_matrix.update(pred, GT)

            length += images.size(0)

        val_eval_matrix.update_print('[Validation] ')

        # TODO: score
        unet_score = val_eval_matrix.F1

        # Save Best U-Net model
        if unet_score > self.best_unet_score:
            self.best_unet_score = unet_score
            best_epoch = epoch
            best_unet = self.unet.state_dict()
            print(f'Best {self.model_type} model score : {round(self.best_unet_score,4)} at epoch: {best_epoch}')
            torch.save(best_unet, self.pretrained_unet_path)

    def test(self, test_loader, visualize=False):
        '''
        test_loader : test data loader
        visualize : if True, save the segmentation result image
        '''
        self.unet.eval()

        self.load_pretrained_model()

        length = 0
        test_eval_matrix = EvaluationMatrix()

        for i, (images, GT) in enumerate(test_loader):

            images = images.to(self.device)
            GT = GT.to(self.device)

            SR = torch.sigmoid(self.unet(images))

            pred = SR > 0.5
            test_eval_matrix.update(pred, GT)

            length += images.size(0)

            if visualize:
                SR = SR[0, 0, ...]
                mask = 255*((SR > 0.5).cpu().numpy().astype(np.uint8))
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                gt = 255*((GT[0, 0, ...] > 0.5).cpu().numpy().astype(np.uint8))
                gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
                vis = np.concatenate([gt, mask], axis=0)
                cv2.imwrite(os.path.join(self.result_path, 'visualization/',
                            f"visualize_img{i}.png"), vis)
                print(f"Saved test samples to {os.path.join(self.result_path, 'visualization/')}.")

        test_eval_matrix.update_print('[Test] ')

    
    def images2masks(self, images):
        '''
        images : list of images
        '''
        dataset = data.TensorDataset(images)
        loader = data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)
        
        loader = tqdm.tqdm(enumerate(loader))
        loader.set_description("Segmenting in batch")
        
        all_masks = []
    
        self.unet.eval()

        self.load_pretrained_model()

        for i, images in enumerate(loader):

            images = images[1][0] # lx: TODO? Q? [1] means get item from stack
            images = images.to(self.device)

            SR = torch.sigmoid(self.unet(images))

            pred = SR > 0.5
            all_masks.append(pred.reshape(images.shape))
        all_masks = torch.cat(all_masks).cpu().numpy().astype(np.uint8)

        return all_masks
