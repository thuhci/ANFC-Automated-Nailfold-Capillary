import argparse
import os

from Image_Segmentation.image_segmentation.data_loader import get_loader
from Image_Segmentation.image_segmentation.solver import Solver
from torch.backends import cudnn


def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s' % config.model_type)
        return


    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)


    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob,
                              img_ch=config.img_ch)
    valid_loader = get_loader(image_path=config.valid_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0.,
                              img_ch=config.img_ch)
    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0.,
                             img_ch=config.img_ch)

    solver = Solver(config)

    # Train and sample the images
    if config.mode == 'train':
        solver.train(train_loader, valid_loader, test_loader)
    elif config.mode == 'test':
        solver.test(test_loader, config.visualize)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # model
    # data parameters
    # data itself
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    # data loading
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    # data augmentation
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    # training parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=30)

    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # test parameters
    parser.add_argument('--val_step', type=int, default=2)

    # log
    parser.add_argument('--log_step', type=int, default=2)

    # running
    parser.add_argument('--mode', type=str, default='train',
                        help='train/test')
    parser.add_argument('--model_type', type=str, default='U_Net',
                        help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./Image_Segmentation/image_segmentation/checkpoints')

    # dataset path
    parser.add_argument('--train_path', type=str,
                        default='./Data_Preprocess/nailfold_data_preprocess/data/segment_dataset/train')
    parser.add_argument('--valid_path', type=str,
                        default='./Data_Preprocess/nailfold_data_preprocess/data/segment_dataset/test') # TODO split a val set
    parser.add_argument('--test_path', type=str,
                        default='./Data_Preprocess/nailfold_data_preprocess/data/segment_dataset/test')
    parser.add_argument('--result_path', type=str, default='./Image_Segmentation/image_segmentation/result')

    # cuda
    parser.add_argument('--cuda_idx', type=int, default=1)

    # visualiza test result
    parser.add_argument('--visualize', type=bool, default=False)

    config = parser.parse_args()
    # with open("./Image_Segmentation/image_segmentation/config.yaml", 'r') as stream:
    #     config = yaml.load(stream, Loader=yaml.FullLoader)
    # config = argparse.Namespace(**config)
    main(config)
