# Copyright (c) OpenMMLab. All rights reserved.
import os

import mmcv
import numpy as np

from tools.scannet_data_utils import ScanNetData, ScanNetSegData


def create_indoor_info_file(data_path,
                            pkl_prefix='sunrgbd',
                            save_path=None,
                            use_v1=False,
                            workers=4):
    """Create indoor information file.

    Get information of the raw data and save it to the pkl file.

    Args:
        data_path (str): Path of the data.
        pkl_prefix (str, optional): Prefix of the pkl to be saved.
            Default: 'sunrgbd'.
        save_path (str, optional): Path of the pkl to be saved. Default: None.
        use_v1 (bool, optional): Whether to use v1. Default: False.
        workers (int, optional): Number of threads to be used. Default: 4.
    """
    assert os.path.exists(data_path)
    assert pkl_prefix in ['sunrgbd', 'scannet', 's3dis', 'scannet200'], \
        f'unsupported indoor dataset {pkl_prefix}'
    save_path = data_path if save_path is None else save_path
    assert os.path.exists(save_path)

    # generate infos for both detection and segmentation task
    if pkl_prefix in ['sunrgbd', 'scannet', 'scannet200']:
        train_filename = os.path.join(save_path,
                                      f'{pkl_prefix}_infos_train.pkl')
        val_filename = os.path.join(save_path, f'{pkl_prefix}_infos_val.pkl')
        
        if pkl_prefix == 'scannet':
            # ScanNet has a train-val-test split
            train_dataset = ScanNetData(root_path=data_path, split='train')
            val_dataset = ScanNetData(root_path=data_path, split='val')
            test_dataset = ScanNetData(root_path=data_path, split='test')
            test_filename = os.path.join(save_path,
                                         f'{pkl_prefix}_infos_test.pkl')
        else: #scannet200
            # ScanNet has a train-val-test split
            train_dataset = ScanNetData(root_path=data_path, split='train',
                                        scannet200=True, save_path=save_path)
            val_dataset = ScanNetData(root_path=data_path, split='val',
                                        scannet200=True, save_path=save_path)
            test_dataset = ScanNetData(root_path=data_path, split='test',
                                        scannet200=True, save_path=save_path)
            test_filename = os.path.join(save_path,
                                         f'{pkl_prefix}_infos_test.pkl')

        infos_train = train_dataset.get_infos(
            num_workers=workers, has_label=True)
        mmcv.dump(infos_train, train_filename, 'pkl')
        print(f'{pkl_prefix} info train file is saved to {train_filename}')

        infos_val = val_dataset.get_infos(num_workers=workers, has_label=True)
        mmcv.dump(infos_val, val_filename, 'pkl')
        print(f'{pkl_prefix} info val file is saved to {val_filename}')

    if pkl_prefix == 'scannet' or pkl_prefix == 'scannet200':
        infos_test = test_dataset.get_infos(
            num_workers=workers, has_label=False)
        mmcv.dump(infos_test, test_filename, 'pkl')
        print(f'{pkl_prefix} info test file is saved to {test_filename}')

    # generate infos for the semantic segmentation task
    # e.g. re-sampled scene indexes and label weights
    # scene indexes are used to re-sample rooms with different number of points
    # label weights are used to balance classes with different number of points
    if pkl_prefix == 'scannet':
        # label weight computation function is adopted from
        # https://github.com/charlesq34/pointnet2/blob/master/scannet/scannet_dataset.py#L24
        train_dataset = ScanNetSegData(
            data_root=data_path,
            ann_file=train_filename,
            split='train',
            num_points=8192,
            label_weight_func=lambda x: 1.0 / np.log(1.2 + x))
        # TODO: do we need to generate on val set?
        val_dataset = ScanNetSegData(
            data_root=data_path,
            ann_file=val_filename,
            split='val',
            num_points=8192,
            label_weight_func=lambda x: 1.0 / np.log(1.2 + x))
        # no need to generate for test set
        train_dataset.get_seg_infos()
        val_dataset.get_seg_infos()