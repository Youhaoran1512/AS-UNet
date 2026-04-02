import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
import albumentations as albu
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter

"""
需要指定参数：--name dsb2018_96_NestedUNet_woDS
"""
save_dir = os.path.join(
    'outputs',
    'data_sum_moxi+bijie(622)_UNet_woDS',
    'moxi_test'
)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='data_sum_moxi+bijie(622)_UNet_woDS',
                        help='model name')


    args = parser.parse_args()

    return args

def save_csv(df, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                              config['input_channels'],
                                              )
    #model = archs.__dict__[config['arch']](config['num_classes'],
    #                                       config['input_channels'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'test', 'moxi_test', 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    #_, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)  # 固定随机种子
    val_img_ids = img_ids
    # print("验证集数量:", len(val_img_ids))  # 输出：93

    val_mask_dir = glob(os.path.join('inputs', config['dataset'], 'test', 'moxi_test', 'masks', '*' + config['mask_ext']))
    val_mask_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_mask_dir]
    # print("验证集数量:", len(val_img_ids))  # 输出：93

    # 数据增强：
    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        mask_ids=val_mask_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'test', 'moxi_test', 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'test', 'moxi_test', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c), 'moxi_test'), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), 'moxi_test', meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meter.avg)

    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, 'test_iou.csv')

    pd.DataFrame([{
        'mean_iou': avg_meter.avg
    }]).to_csv(csv_path, index=False)
    plot_examples(input, target, model, num_examples=3)

    torch.cuda.empty_cache()


def plot_examples(datax, datay, model, num_examples=6):
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18, 4 * num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx + 1]).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1, 2, 0))[:, :, 0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0, :, :].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1, 2, 0))[:, :, 0])
        ax[row_num][2].set_title("Target image")
    plt.show()


if __name__ == '__main__':
    main()
