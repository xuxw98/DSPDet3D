import pickle
import numpy as np
import os
from os import path as osp
import mmcv

num_class = 70
type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
                            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
                            'refrigerator':12, 'showercurtrain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'garbagebin':17,
                            "bag":18, "bottle":19, "bowl":20, "camera":21, "can":22,
                            "cap":23, "clock":24, "keyboard":25, "display":26, "earphone":27,
                            "jar":28, "knife":29, "lamp":30, "laptop":31, "microphone":32,
                            "microwave":33, "mug":34, "printer":35, "remote control":36, "phone":37,
                            "alarm":38, "book":39, "cake":40, "calculator":41, "candle":42,
                            "charger":43, "chessboard":44, "coffee_machine":45, "comb":46, "cutting_board":47,
                            "dishes":48, "doll":49, "eraser":50, "eye_glasses":51, "file_box":52,
                            "fork":53, "fruit":54, "globe":55, "hat":56, "mirror":57,
                            "notebook":58, "pencil":59, "plant":60, "plate":61, "radio":62,
                            "ruler":63, "saucepan":64, "spoon":65, "tea_pot":66, "toaster":67,
                            "vase":68, "vegetables":69}
classes = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                            'window', 'bookshelf','picture', 'counter', 'desk', 'curtain',
                            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin',
                            "bag", "bottle", "bowl", "camera", "can",
                            "cap", "clock", "keyboard", "display", "earphone",
                            "jar", "knife", "lamp", "laptop", "microphone",
                            "microwave", "mug", "printer", "remote control", "phone",
                            "alarm", "book", "cake", "calculator", "candle",
                            "charger", "chessboard", "coffee_machine", "comb", "cutting_board",
                            "dishes", "doll", "eraser", "eye_glasses", "file_box",
                            "fork",  "fruit", "globe", "hat", "mirror",
                            "notebook", "pencil", "plant", "plate", "radio",
                            "ruler", "saucepan", "spoon", "tea_pot", "toaster",
                            "vase", "vegetables"]
cat2label = {cat: classes.index(cat) for cat in classes}
label2cat = {cat2label[t]: t for t in cat2label}
cat_ids = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39,
                                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                                    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                                    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93])
cat_ids2class = {nyu40id: i for i,nyu40id in enumerate(list(cat_ids))}


def dowmsample(mesh_vertices, semantic_labels, instance_labels, instance_bboxes):
    mask = np.in1d(semantic_labels, cat_ids)
    semantic_labels_care = semantic_labels[mask]
    instance_labels_care = instance_labels[mask]
    mesh_vertices_care = mesh_vertices[mask, :]

    mask_back = ~mask
    semantic_labels_dcare = semantic_labels[mask_back]
    instance_labels_dcare = instance_labels[mask_back]
    mesh_vertices_dcare = mesh_vertices[mask_back, :]


    object_mesh = {}
    object_ins = {}
    object_sem = {}
    i = 0
    for i_instance in np.unique(instance_labels_care):
        # find all points belong to that instance
        ind = np.where(instance_labels_care == i_instance)[0]
        object_mesh[i] = mesh_vertices_care[ind, :]
        object_sem[i] = semantic_labels_care[ind]
        object_ins[i] = instance_labels_care[ind]
        i = i + 1

    for i in range(len(np.unique(instance_labels_care))):
        if instance_bboxes[i, 6] in cat_ids[18:]:
            max_area = max(instance_bboxes[i, 3]*instance_bboxes[i, 4], instance_bboxes[i, 3]*instance_bboxes[i, 5], instance_bboxes[i, 4]*instance_bboxes[i, 5])
            scannet_table = 3700
            toscannet_tabel = 1200
            if instance_bboxes[i, 3]*instance_bboxes[i, 4]*instance_bboxes[i, 5] > 0.01 :
                scannet_samll = 6000  #7000
            else:
                scannet_samll = 9000 #10000

            num_points = int(scannet_samll / scannet_table * toscannet_tabel * max_area)  #2950
            if len(object_mesh[i]) > num_points:
                choice =np.random.choice(len(object_mesh[i]), num_points, replace=False)
            else:
                choice = np.random.choice(len(object_mesh[i]), len(object_mesh[i]), replace=False)
            object_mesh[i] = object_mesh[i][choice]
            object_sem[i] = object_sem[i][choice]
            object_ins[i] = object_ins[i][choice]
    semantic_labels_sample = np.array([])
    instance_labels_sample = np.array([])
    # mesh_vertices_sample = []
    for i in range(len(np.unique(instance_labels_care))):
        for j in range(len(object_mesh[i])):
            if i==0 and j==0:
                mesh_vertices_sample = object_mesh[i][j, :].reshape(1, 6)
            else:
                mesh_vertices_sample = np.append(mesh_vertices_sample, object_mesh[i][j, :].reshape(1, 6), axis=0)
            semantic_labels_sample = np.append(semantic_labels_sample, object_sem[i][j])
            instance_labels_sample = np.append(instance_labels_sample, object_ins[i][j])

    assert len(mesh_vertices_dcare) == len(semantic_labels_dcare) == len(instance_labels_dcare) , "backgroud points numbers X="
    for i in range(len(mesh_vertices_dcare)):
        mesh_vertices_sample = np.append(mesh_vertices_sample, mesh_vertices_dcare[i].reshape(1, 6), axis=0)
        semantic_labels_sample = np.append(semantic_labels_sample, semantic_labels_dcare[i])
        instance_labels_sample = np.append(instance_labels_sample, instance_labels_dcare[i])

    choices = np.random.choice(instance_labels_sample.shape[0], instance_labels_sample.shape[0], replace=False)
    mesh_vertices_sample = mesh_vertices_sample[choices, :]
    semantic_labels_sample = semantic_labels_sample[choices].astype(np.int64)
    instance_labels_sample = instance_labels_sample[choices].astype(np.int64)

    return mesh_vertices_sample, semantic_labels_sample, instance_labels_sample





def main():
    root_dir = '/path/DSPDet3D/data/TO-SCENE-down'
    
    train_filenames = os.path.join('/path/DSPDet3D/data/TO-SCENE-down/TO_scannet/meta_data/TO-scannet/', 'train.txt')
    val_filenames = os.path.join('/path/DSPDet3D/data/TO-SCENE-down/TO_scannet/meta_data/TO-scannet/', 'val.txt')
    test_filenames = os.path.join('/path/DSPDet3D/data/TO-SCENE-down/TO_scannet/meta_data/TO-scannet/', 'test.txt')
    with open(train_filenames, 'r') as f:
        train_scan_names = [name + '.npz' for name in f.read().splitlines()]
    print('train_num_scans:', len(train_scan_names))


    with open(val_filenames, 'r') as f:
        val_scan_names = [name + '.npz' for name in f.read().splitlines()]
    print('val_num_scans:', len(val_scan_names))


    with open(test_filenames, 'r') as f:
        test_scan_names = [name + '.npz' for name in f.read().splitlines()]
    print('test_num_scans:', len(test_scan_names))
    ########################
    with open(train_filenames, 'r') as f:
        train_names = [name for name in f.read().splitlines()]


    with open(val_filenames, 'r') as f:
        val_names = [name for name in f.read().splitlines()]


    with open(test_filenames, 'r') as f:
        test_names = [name for name in f.read().splitlines()]



    infos_train = []
    data_path = '/path/DSPDet3D/data/TO-SCENE-down/TO_scannet/train'
    for idx in range(len(train_scan_names)):
        scan_name_npz =train_scan_names[idx]
        scan_name = train_names[idx]
        mesh_vertices = np.hstack((np.load(os.path.join(data_path, scan_name_npz))['xyz'],
                                   np.load(os.path.join(data_path, scan_name_npz))['color'])).astype('float32')
        instance_labels = np.load(os.path.join(data_path, scan_name_npz))['instance_label'].astype('uint32')
        semantic_labels = np.load(os.path.join(data_path, scan_name_npz))['semantic_label'].astype('uint32')
        instance_bboxes = np.load(os.path.join(data_path, scan_name_npz))['bbox'].astype('float64')

        info = dict()
        mesh_vertices, semantic_labels, instance_labels = dowmsample(mesh_vertices, semantic_labels, instance_labels, instance_bboxes)
        print(f'{train_filenames} sample_idx: {scan_name_npz}  {len(mesh_vertices)}')
        #point
        pc_info = {'num_features': 6, 'lidar_idx': scan_name}
        info['point_cloud'] = pc_info
        mmcv.mkdir_or_exist(osp.join(root_dir, 'points'))
        mesh_vertices.tofile(
            osp.join(root_dir, 'points', f'{scan_name}.bin'))
        info['pts_path'] = osp.join('points', f'{scan_name}.bin')
        #instance and semantic label
        mmcv.mkdir_or_exist(osp.join(root_dir, 'instance_mask'))
        mmcv.mkdir_or_exist(osp.join(root_dir, 'semantic_mask'))

        instance_labels.tofile(
            osp.join(root_dir, 'instance_mask',
                     f'{scan_name}.bin'))
        semantic_labels.tofile(
            osp.join(root_dir, 'semantic_mask',
                     f'{scan_name}.bin'))

        info['pts_instance_mask_path'] = osp.join(
            'instance_mask', f'{scan_name}.bin')
        info['pts_semantic_mask_path'] = osp.join(
            'semantic_mask', f'{scan_name}.bin')

        #bbox
        annotations = {}
        aligned_box_label = instance_bboxes
        unaligned_box_label = aligned_box_label
        annotations['gt_num'] = aligned_box_label.shape[0]
        if annotations['gt_num'] != 0:
            aligned_box = aligned_box_label[:, :-1]  # k, 6
            unaligned_box = unaligned_box_label[:, :-1]
            classes = aligned_box_label[:, -1]  # k
            annotations['name'] = np.array([
                label2cat[cat_ids2class[classes[i]]]
                for i in range(annotations['gt_num'])
            ])
            # default names are given to aligned bbox for compatibility
            # we also save unaligned bbox info with marked names
            annotations['location'] = aligned_box[:, :3]
            annotations['dimensions'] = aligned_box[:, 3:6]
            annotations['gt_boxes_upright_depth'] = aligned_box
            annotations['unaligned_location'] = unaligned_box[:, :3]
            annotations['unaligned_dimensions'] = unaligned_box[:, 3:6]
            annotations[
                'unaligned_gt_boxes_upright_depth'] = unaligned_box
            annotations['index'] = np.arange(annotations['gt_num'], dtype=np.int32)
            annotations['class'] = np.array([
                cat_ids2class[classes[i]]
                for i in range(annotations['gt_num'])
            ])
        annotations['axis_align_matrix'] = np.eye(4)
        info['annos'] = annotations
        infos_train.append(info)
    mmcv.dump(infos_train, '/path/DSPDet3D/data/TO-SCENE-down/toscene_infos_train.pkl', 'pkl')

    infos_val = []
    data_path = '/path/DSPDet3D/data/TO-SCENE-down/TO_scannet/val'
    for idx in range(len(val_scan_names)):
        scan_name_npz = val_scan_names[idx]
        scan_name = val_names[idx]
        mesh_vertices = np.hstack((np.load(os.path.join(data_path, scan_name_npz))['xyz'],
                                   np.load(os.path.join(data_path, scan_name_npz))['color'])).astype('float32')
        instance_labels = np.load(os.path.join(data_path, scan_name_npz))['instance_label'].astype('uint32')
        semantic_labels = np.load(os.path.join(data_path, scan_name_npz))['semantic_label'].astype('uint32')
        instance_bboxes = np.load(os.path.join(data_path, scan_name_npz))['bbox'].astype('float64')


        mesh_vertices, semantic_labels, instance_labels = dowmsample(mesh_vertices, semantic_labels, instance_labels, instance_bboxes)
        info = dict()
        print(f'{val_filenames} sample_idx: {scan_name_npz}   {len(mesh_vertices)}')
        # point
        pc_info = {'num_features': 6, 'lidar_idx': scan_name}
        info['point_cloud'] = pc_info
        mmcv.mkdir_or_exist(osp.join(root_dir, 'points'))
        mesh_vertices.tofile(
            osp.join(root_dir, 'points', f'{scan_name}.bin'))
        info['pts_path'] = osp.join('points', f'{scan_name}.bin')
        # instance and semantic label
        mmcv.mkdir_or_exist(osp.join(root_dir, 'instance_mask'))
        mmcv.mkdir_or_exist(osp.join(root_dir, 'semantic_mask'))

        instance_labels.tofile(
            osp.join(root_dir, 'instance_mask',
                     f'{scan_name}.bin'))
        semantic_labels.tofile(
            osp.join(root_dir, 'semantic_mask',
                     f'{scan_name}.bin'))

        info['pts_instance_mask_path'] = osp.join(
            'instance_mask', f'{scan_name}.bin')
        info['pts_semantic_mask_path'] = osp.join(
            'semantic_mask', f'{scan_name}.bin')

        # bbox
        annotations = {}
        aligned_box_label = instance_bboxes
        unaligned_box_label = aligned_box_label
        annotations['gt_num'] = aligned_box_label.shape[0]
        if annotations['gt_num'] != 0:
            aligned_box = aligned_box_label[:, :-1]  # k, 6
            unaligned_box = unaligned_box_label[:, :-1]
            classes = aligned_box_label[:, -1]  # k
            annotations['name'] = np.array([
                label2cat[cat_ids2class[classes[i]]]
                for i in range(annotations['gt_num'])
            ])
            # default names are given to aligned bbox for compatibility
            # we also save unaligned bbox info with marked names
            annotations['location'] = aligned_box[:, :3]
            annotations['dimensions'] = aligned_box[:, 3:6]
            annotations['gt_boxes_upright_depth'] = aligned_box
            annotations['unaligned_location'] = unaligned_box[:, :3]
            annotations['unaligned_dimensions'] = unaligned_box[:, 3:6]
            annotations[
                'unaligned_gt_boxes_upright_depth'] = unaligned_box
            annotations['index'] = np.arange(annotations['gt_num'], dtype=np.int32)
            annotations['class'] = np.array([
                cat_ids2class[classes[i]]
                for i in range(annotations['gt_num'])
            ])
        annotations['axis_align_matrix'] = np.eye(4)
        info['annos'] = annotations
        infos_val.append(info)
    mmcv.dump(infos_val, '/path/DSPDet3D/data/TO-SCENE-down/toscene_infos_val.pkl', 'pkl')

    infos_test = []
    data_path = '/path/DSPDet3D/data/TO-SCENE-down/TO_scannet/test'
    for idx in range(len(test_scan_names)):
        scan_name_npz = test_scan_names[idx]
        scan_name = test_names[idx]
        mesh_vertices = np.hstack((np.load(os.path.join(data_path, scan_name_npz))['xyz'],
                                   np.load(os.path.join(data_path, scan_name_npz))['color'])).astype('float32')

        print(f'{test_filenames} sample_idx: {scan_name_npz}')
        info = dict()

        # point
        pc_info = {'num_features': 6, 'lidar_idx': scan_name}
        info['point_cloud'] = pc_info
        mmcv.mkdir_or_exist(osp.join(root_dir, 'points'))
        mesh_vertices.tofile(
            osp.join(root_dir, 'points', f'{scan_name}.bin'))
        info['pts_path'] = osp.join('points', f'{scan_name}.bin')

        infos_test.append(info)
    mmcv.dump(infos_test, '/path/DSPDet3D/data/TO-SCENE-down/toscene_infos_test.pkl', 'pkl')


if __name__ == '__main__':
    main()

