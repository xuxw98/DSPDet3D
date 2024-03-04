import numpy as np

try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
from mmcv.cnn import bias_init_with_prob
from mmcv.ops import nms3d, nms3d_normal
from mmcv.runner import BaseModule
from torch import nn

from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.models.builder import HEADS, build_loss
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, build_assigner


import pdb

@HEADS.register_module()
class DSPHead(BaseModule):
    def __init__(self,
                 n_classes,
                 in_channels,
                 out_channels,
                 n_reg_outs,
                 voxel_size,
                 pts_prune_threshold,
                 assigner,
                 volume_threshold,
                 r,
                 assign_type='volume',
                 prune_threshold=0,
                 bbox_loss=dict(type='AxisAlignedIoULoss', reduction='none'),
                 cls_loss=dict(type='FocalLoss', reduction='none'),
                 keep_loss=dict(type='FocalLoss', reduction='mean', use_sigmoid=True),               
                 train_cfg=None,
                 test_cfg=None):
        super(DSPHead, self).__init__()
        self.voxel_size = voxel_size
        self.pts_prune_threshold = pts_prune_threshold
        self.assign_type = assign_type
        self.volume_threshold = volume_threshold
        self.r = r
        self.prune_threshold = prune_threshold
        self.keep_loss_weight = keep_loss_weight
        self.assigner = build_assigner(assigner)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.keep_loss = build_loss(keep_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)


    @staticmethod
    def make_block(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels,
                                    kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))


    @staticmethod
    def make_down_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                    stride=2, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))


    @staticmethod
    def make_up_block(in_channels, out_channels, generative=False):
        conv = ME.MinkowskiGenerativeConvolutionTranspose if generative \
            else ME.MinkowskiConvolutionTranspose
        return nn.Sequential(
            conv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))


    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        self.bbox_conv = ME.MinkowskiConvolution(
            out_channels, n_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.keep_conv = nn.ModuleList([
            ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, bias=True, dimension=3),
            ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, bias=True, dimension=3),
            ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, bias=True, dimension=3)
        ])
        self.pruning = ME.MinkowskiPruning()

        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    self.make_up_block(in_channels[i], in_channels[i - 1], generative=True))
            # if i < len(in_channels) - 1:
            self.__setattr__(
                        f'lateral_block_{i}',
                        self.make_block(in_channels[i], in_channels[i]))
            self.__setattr__(
                        f'out_block_{i}',
                        self.make_block(in_channels[i], out_channels))
        # ######only train keep_head  
        # for name, param in self.named_parameters():
        #     if "keep_conv" not in name:
        #         param.requires_grad=False


    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

        for i in range(len(self.keep_conv)):
            nn.init.normal_(self.keep_conv[i].kernel, std=.01)

        for n, m in self.named_modules():
            if ('bbox_conv' not in n) and ('cls_conv' not in n) \
                and ('keep_conv' not in n) and ('loss' not in n):
                if isinstance(m, ME.MinkowskiConvolution):
                    ME.utils.kaiming_normal_(
                        m.kernel, mode='fan_out', nonlinearity='relu')

                if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)       
    

    def _forward_single(self, x):
        reg_final = self.bbox_conv(x).features
        reg_distance = torch.exp(reg_final[:, 3:6])
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)
        scores = self.cls_conv(x)
        cls_pred = scores.features
        prune_training = ME.SparseTensor(
            scores.features.max(dim=1, keepdim=True).values,
            coordinate_map_key=scores.coordinate_map_key,
            coordinate_manager=scores.coordinate_manager)

        bbox_preds, cls_preds, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)
        return bbox_preds, cls_preds, points, prune_training


    def forward(self, x, gt_bboxes, gt_labels, img_metas):

        bboxes_level = []
        bboxes_state = []
        if self.assign_type == 'volume':
            for idx in range(len(img_metas)):
                bbox = gt_bboxes[idx]
                bbox_state = torch.cat((bbox.gravity_center, bbox.tensor[:, 3:]), dim=1)
                bbox_level = torch.zeros([len(bbox), 1])
                downsample_times = [5,4,3]
                for n in range(len(bbox)):
                    bbox_volume = bbox_state[n][3] * bbox_state[n][4] * bbox_state[n][5]
                    for i in range(len(downsample_times)):
                        if bbox_volume > self.volume_threshold * (self.voxel_size * 2 ** downsample_times[i]) ** 3:
                            bbox_level[n] = 3 - i
                            break                  
                bboxes_level.append(bbox_level)
                bbox_state = torch.cat((bbox_level, bbox_state), dim=1)
                bboxes_state.append(bbox_state)
        elif self.assign_type == 'label':
            for idx in range(len(img_metas)):
                bbox = gt_bboxes[idx]
                bbox_label = gt_labels[idx]
                label2level = gt_labels[idx].new_tensor(self.label2level)
                bbox_state = torch.cat((bbox.gravity_center, bbox.tensor[:, 3:]), dim=1)
                bbox_level = label2level[bbox_label].to(bbox_state.device).unsqueeze(1)
                bboxes_level.append(bbox_level)
                bbox_state = torch.cat((bbox_level, bbox_state), dim=1)
                bboxes_state.append(bbox_state)
        bbox_preds, cls_preds, points = [], [], []
        keep_gts = []
        keep_preds, prune_masks = [], []
        prune_mask = None
        inputs = x
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:            
                prune_mask = self._get_keep_voxel(x, i + 2, bboxes_state, img_metas)
                keep_gt = []
                for permutation in out.decomposition_permutations:
                    keep_gt.append(prune_mask[permutation])
                keep_gts.append(keep_gt)
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                coords = x.coordinates.float()
                x_level_features = inputs[i].features_at_coordinates(coords)
                x_level = ME.SparseTensor(features=x_level_features,
                                          coordinate_map_key=x.coordinate_map_key,
                                        coordinate_manager=x.coordinate_manager)
                x = x + x_level
                x = self._prune_training(x, prune_training_keep)

            if i > 0:
                keep_scores = self.keep_conv[i-1](x)
                prune_training_keep = ME.SparseTensor(
                                    -keep_scores.features,
                                    coordinate_map_key=keep_scores.coordinate_map_key,
                                    coordinate_manager=keep_scores.coordinate_manager)
                keep_pred = keep_scores.features
                prune_inference = keep_pred
                keeps = []
                for permutation in x.decomposition_permutations:
                    keeps.append(keep_pred[permutation])
                keep_preds.append(keeps)
            x = self.__getattr__(f'lateral_block_{i}')(x)
            out = self.__getattr__(f'out_block_{i}')(x)
            bbox_pred, cls_pred, point, prune_training = self._forward_single(out)
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)

        return bbox_preds[::-1], cls_preds[::-1], points[::-1], keep_preds[::-1], keep_gts[::-1], bboxes_level
    

    def _prune_inference(self, x, scores):
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """
        with torch.no_grad():
            prune_mask = scores.new_zeros(
                (len(scores)), dtype=torch.bool)

            for permutation in x.decomposition_permutations:
                score = scores[permutation].sigmoid()
                score = 1 - score
                mask = score > self.prune_threshold
                mask = mask.reshape([len(score)])
                prune_mask[permutation[mask]] = True
        if prune_mask.sum() != 0:
            x = self.pruning(x, prune_mask)
        else:
            x = None

        return x


    def _prune_training(self, x, scores):
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """

        with torch.no_grad():
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros(
                (len(interpolated_scores)), dtype=torch.bool)
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_prune_threshold)
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True
        x = self.pruning(x, prune_mask)
        return x


    @torch.no_grad()
    def _get_keep_voxel(self, input, cur_level, bboxes_state, input_metas):
        bboxes = []
        for size in range(len(input_metas)):
            bboxes.append([])
        for idx in range(len(input_metas)):
            for n in range(len(bboxes_state[idx])):
                if bboxes_state[idx][n][0] < (cur_level - 1):    
                    bboxes[idx].append(bboxes_state[idx][n])
        idx = 0
        mask = []
        l0 = self.voxel_size * 2 ** 2  # pool  True :2**3  False:2**2
        for idx, permutation in enumerate(input.decomposition_permutations):
            point = input.coordinates[permutation][:, 1:] * self.voxel_size
            if len(bboxes[idx]) != 0:
                point = input.coordinates[permutation][:, 1:] * self.voxel_size
                boxes = bboxes[idx]
                level = 3
                bboxes_level = [[] for _ in range(level)]
                for n in range(len(boxes)):
                    for l in range(level):
                        if boxes[n][0] == l:
                            bboxes_level[l].append(boxes[n])
                inside_box_conditions = torch.zeros((len(permutation)), dtype=torch.bool).to(point.device)
                for l in range(level):
                    if len(bboxes_level[l]) != 0:
                        point_l = point.unsqueeze(1).expand(len(point), len(bboxes_level[l]), 3)
                        boxes_l = torch.cat(bboxes_level[l]).reshape([-1, 8]).to(point.device)
                        boxes_l = boxes_l.expand(len(point), len(bboxes_level[l]), 8)
                        shift = torch.stack(
                            (point_l[..., 0] - boxes_l[..., 1], point_l[..., 1] - boxes_l[..., 2],
                            point_l[..., 2] - boxes_l[..., 3]),
                            dim=-1).permute(1, 0, 2)
                        shift = rotation_3d_in_axis(
                            shift, -boxes_l[0, :, 7], axis=2).permute(1, 0, 2)
                        centers = boxes_l[..., 1:4] + shift
                        up_level_l = self.r 
                        dx_min = centers[..., 0] - boxes_l[..., 1] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2  
                        dx_max = boxes_l[..., 1] - centers[..., 0] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2 
                        dy_min = centers[..., 1] - boxes_l[..., 2] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2  
                        dy_max = boxes_l[..., 2] - centers[..., 1] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2
                        dz_min = centers[..., 2] - boxes_l[..., 3] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2  
                        dz_max = boxes_l[..., 3] - centers[..., 2] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2


                        distance = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)
                        inside_box_condition = distance.min(dim=-1).values > 0
                        inside_box_condition = inside_box_condition.sum(dim=1)
                        inside_box_condition = inside_box_condition >= 1
                        inside_box_conditions += inside_box_condition
                mask.append(inside_box_conditions)
            else:
                inside_box_conditions = torch.zeros((len(permutation)), dtype=torch.bool).to(point.device)
                mask.append(inside_box_conditions)

        prune_mask = torch.cat(mask)
        prune_mask = prune_mask.to(input.device)
        return prune_mask
    

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)


    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 3] + bbox_pred[:, 4]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)


    def _loss_single(self,
                     bbox_preds,
                     cls_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     bboxes_level,
                     img_meta):
        assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, bboxes_level, img_meta)

        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0


        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)
        cls_loss = self.cls_loss(cls_preds, cls_targets)

        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(
                    self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))
        else:
            bbox_loss = None
        return bbox_loss, cls_loss, pos_mask


    def _loss(self, bbox_preds, cls_preds, points,
              gt_bboxes, gt_labels, img_metas, keep_preds, keep_gts, bboxes_level):
        bbox_losses, cls_losses, pos_masks = [], [], []

        #keep loss
        keep_losses = 0
        for i in range(len(img_metas)):
            k_loss = 0
            keep_pred = [x[i] for x in keep_preds]
            keep_gt = [x[i] for x in keep_gts]
            for j in range(len(keep_preds)):
                pred = keep_pred[j]
                gt = (keep_gt[j]).long()

                if gt.sum() != 0:
                    keep_loss = self.keep_loss(pred, gt, avg_factor=gt.sum())
                    k_loss = torch.mean(keep_loss) / 3 + k_loss
                else:
                    keep_loss = self.keep_loss(pred, gt, avg_factor=len(gt))  
                    k_loss = torch.mean(keep_loss) / 3 + k_loss

            keep_losses = keep_losses + k_loss

        for i in range(len(img_metas)):
            bbox_loss, cls_loss, pos_mask = self._loss_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],
                bboxes_level=bboxes_level[i])
            if bbox_loss is not None:
                bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            pos_masks.append(pos_mask)

        return dict(
            bbox_loss=torch.mean(torch.cat(bbox_losses)),
            cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)),
            keep_loss=self.keep_loss_weight * keep_losses / len(img_metas)) 


    def forward_train(self, x, gt_bboxes, gt_labels, img_metas):
        bbox_preds, cls_preds, points, keep_preds, keep_gts, bboxes_level = self(x, gt_bboxes, gt_labels, img_metas)
        return self._loss(bbox_preds, cls_preds, points,
                          gt_bboxes, gt_labels, img_metas, keep_preds, keep_gts, bboxes_level)


    def _nms(self, bboxes, scores, img_meta):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels


    def _get_bboxes_single(self, bbox_preds, cls_preds, points, img_meta):
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        boxes = self._bbox_pred_to_bbox(points, bbox_preds)
        boxes, scores, labels = self._nms(boxes, scores, img_meta)
        return boxes, scores, labels


    def _get_bboxes(self, bbox_preds, cls_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results


    def forward_test(self, x, img_metas):
        inputs = x
        x = inputs[-1]
        bbox_preds, cls_preds, points = [], [], []
        keep_scores = None
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self._prune_inference(x, prune_inference)
                if x != None:
                    x = self.__getattr__(f'up_block_{i + 1}')(x)
                    coords = x.coordinates.float()
                    x_level_features = inputs[i].features_at_coordinates(coords)
                    x_level = ME.SparseTensor(features=x_level_features,
                                              coordinate_map_key=x.coordinate_map_key,
                                              coordinate_manager=x.coordinate_manager)
                    x = x + x_level
                else:
                    break

            if i > 0:
                keep_scores = self.keep_conv[i-1](x)
                keep_pred = keep_scores.features
                prune_inference = keep_pred

            x = self.__getattr__(f'lateral_block_{i}')(x)
            out = self.__getattr__(f'out_block_{i}')(x)
            bbox_pred, cls_pred, point, prune_training = self._forward_single(out)
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)

        return self._get_bboxes(bbox_preds[::-1], cls_preds[::-1], points[::-1], img_metas)


@BBOX_ASSIGNERS.register_module()
class DSPAssigner:
    def __init__(self, top_pts_threshold):
        # top_pts_threshold: per box
        self.top_pts_threshold = top_pts_threshold

    @torch.no_grad()
    def assign(self, points, gt_bboxes, gt_labels, bboxes_level, img_meta):
        # -> object id or -1 for each point
        float_max = points[0].new_tensor(1e8)
        levels = torch.cat([points[i].new_tensor(i, dtype=torch.long).expand(len(points[i]))
                            for i in range(len(points))])
        points = torch.cat(points)
        n_points = len(points)
        n_boxes = len(gt_bboxes)
        if len(gt_labels) == 0:
            return gt_labels.new_full((n_points,), -1)

        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        boxes = boxes.to(points.device).expand(n_points, n_boxes, 7)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
           
        # condition 1: fix level for label
        bboxes_level = bboxes_level.squeeze(1)
        label_levels = bboxes_level.unsqueeze(0).expand(n_points, n_boxes).to(points.device)
        point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = label_levels == point_levels

        # condition 2: keep topk location per box by center distance
        center = boxes[..., :3]
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        ######3X3X3 limit
        L0 = 0.01 * 2**2
        p = 7   
        level_box_l = p * (L0 * 2 ** bboxes_level).unsqueeze(0).expand(n_points, n_boxes).unsqueeze(2).to(points.device)
        level_box = torch.cat((center,level_box_l),dim=2)
        shift = torch.stack((
            points[..., 0] - level_box[..., 0],
            points[..., 1] - level_box[..., 1],
            points[..., 2] - level_box[..., 2]
        ), dim=-1)
        level_centers = level_box[..., :3] + shift
        dx_min = level_centers[..., 0] - level_box[..., 0] + level_box[..., 3] / 2
        dx_max = level_box[..., 0] + level_box[..., 3] / 2 - level_centers[..., 0]
        dy_min = level_centers[..., 1] - level_box[..., 1] + level_box[..., 3] / 2
        dy_max = level_box[..., 1] + level_box[..., 3] / 2 - level_centers[..., 1]
        dz_min = level_centers[..., 2] - level_box[..., 2] + level_box[..., 3] / 2
        dz_max = level_box[..., 2] + level_box[..., 3] / 2 - level_centers[..., 2]
        level_bbox_targets = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)
        inside_level_bbox_mask = level_bbox_targets[..., :6].min(-1)[0] > 0
        center_distances = torch.where(inside_level_bbox_mask, center_distances, float_max)
        #######
        center_distances = torch.where(level_condition, center_distances, float_max)
        topk_distances = torch.topk(center_distances,
                                    min(self.top_pts_threshold + 1, len(center_distances)),
                                    largest=False, dim=0).values[-1]
        topk_condition = center_distances < topk_distances.unsqueeze(0)

        # condition 3.0: tonly closest object to poin
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        _, min_inds_ = center_distances.min(dim=1)

        # condition 3: min center distance to box per point
        center_distances = torch.where(topk_condition, center_distances, float_max)
        min_values, min_ids = center_distances.min(dim=1)
        min_inds = torch.where(min_values < float_max, min_ids, -1)
        min_inds = torch.where(min_inds == min_inds_, min_ids, -1)

        return min_inds
