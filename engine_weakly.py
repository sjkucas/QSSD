# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import json
import logging
import torch.distributed as dist
import math
import sys
from typing import Iterable

import torch.nn.functional as F
from util.utils import to_device
from tqdm import tqdm
import torch
import util.misc as utils
from datasets import build_evaluator
import warnings
import numpy as np
# 忽略UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


def grouping(arr):
    """
    Group the connected results
    """
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)

def get_proposal_oic(tempseg_list, int_temp_scores, c_pred, c_pred_scores, grid, lamb=0.25,
                     gamma=0.20):  # [0.25, 0.20]
    temp = []
    for i in range(len(tempseg_list)):
        c_temp = []
        temp_list = np.array(tempseg_list[i])[0]

        if temp_list.any():
            grouped_temp_list = grouping(temp_list)

            for j in range(len(grouped_temp_list)):
                if len(grouped_temp_list[j]) < 2:
                    continue

                inner_score = np.mean(int_temp_scores[grouped_temp_list[j], i])

                len_proposal = len(grouped_temp_list[j])

                outer_s = max(0, int(grouped_temp_list[j][0] - lamb * len_proposal))
                outer_e = min(int(int_temp_scores.shape[0] - 1), int(grouped_temp_list[j][-1] + lamb * len_proposal))

                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(
                    range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))

                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(int_temp_scores[outer_temp_list, i])

                c_score = inner_score - outer_score + gamma * c_pred_scores[c_pred[i]]

                t_start = grid[0][(grouped_temp_list[j][0] + 0)] - 4.0/30.0

                t_end = grid[0][(grouped_temp_list[j][-1])] + 4.0/30.0
                c_temp.append([c_pred[i], c_score, t_start, t_end])

            temp.append(c_temp)
    return temp


def gather_results(preds):
    world_size = utils.get_world_size()

    gathered_preds = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_preds, preds)

    # Flatten the gathered dictionaries
    final_preds = {}
    for proc_preds in gathered_preds:
        for video_name, outputs in proc_preds.items():
            if video_name not in final_preds:
                final_preds[video_name] = {'scores': [], 'labels': [], 'segments': []}
            final_preds[video_name]['scores'].append(outputs['scores'])
            final_preds[video_name]['labels'].append(outputs['labels'])
            final_preds[video_name]['segments'].append(outputs['segments'])

    # Concatenate results across processes
    for video_name in final_preds.keys():
        final_preds[video_name]['scores'] = torch.cat(final_preds[video_name]['scores'])
        final_preds[video_name]['labels'] = torch.cat(final_preds[video_name]['labels'])
        final_preds[video_name]['segments'] = torch.cat(final_preds[video_name]['segments'])

    return final_preds


@torch.no_grad()
def inference(model, data_loader, device, args):
    model.eval()
    all_outputs = []
    all_infos = []

    for samples, info in data_loader:
        samples = samples.to(device)
        info = [{k: v.to(device) if not isinstance(v, str) else v for k, v in t.items()} for t in info]

        with torch.amp.autocast('cuda', enabled=args.amp, dtype=torch.bfloat16):
            outputs = model(samples, info)

        all_outputs.append(outputs)
        all_infos.extend(info)

    return all_outputs, all_infos


def train_one_epoch_weakly(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, class_nums=20, args=None, epoch=None):

    model.train()
    criterion.train()

    # train_process
    train_num_correct = 0
    train_num_total = 0

    loss_stack = []
    act_inst_loss_stack = []
    act_cont_loss_stack = []
    act_back_loss_stack = []
    guide_loss_stack = []
    att_loss_stack = []
    feat_loss_stack = []


    _cnt = 0

    for samples, info in tqdm(data_loader):

        samples = samples.to(device, non_blocking=True)
        info = [{
            k: v.to(device) if not isinstance(v, str) else v
            for k, v in t.items()
        } for t in info]
        vid_label = [d['labels'] for d in info]

        vid_label_t = torch.stack([
            F.one_hot(torch.unique(label), num_classes=class_nums).sum(dim=0).clamp_max(1).float()
            for label in vid_label
        ], dim=0)

        vid_label = [torch.cat([torch.unique(label), torch.tensor([20.0]).cuda()]) for label in vid_label]
        # vid_label = [torch.cat([torch.unique(label), torch.tensor([200.0]).cuda()]) for label in vid_label]
        act_inst_cls, act_cont_cls, act_back_cls, \
            act_inst_feat, act_cont_feat, act_back_feat, \
            temp_att, act_inst_cas, act_cas, _, _, pseudo, contrast_pairs = model(samples, info, weakly=args.weakly, vid_label=vid_label)




        loss, loss_dict = criterion(act_inst_cls, act_cont_cls, act_back_cls, vid_label_t, temp_att, \
                                    act_inst_feat, act_cont_feat, act_back_feat, act_inst_cas, act_cas, pseudo, contrast_pairs, epoch)

        optimizer.zero_grad()
        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            fg_score = act_inst_cls[:, :args.num_classes]
            label_np = vid_label_t.cpu().numpy()
            score_np = fg_score.cpu().numpy()

            pred_np = np.zeros_like(score_np)
            pred_np[score_np >= args.cls_threshold] = 1
            pred_np[score_np < args.cls_threshold] = 0
            correct_pred = np.sum(label_np == pred_np, axis=1)

            train_num_correct += np.sum((correct_pred == args.num_classes))
            train_num_total += correct_pred.shape[0]

            loss_stack.append(loss.cpu().item())
            act_inst_loss_stack.append(loss_dict["act_inst_loss"])
            act_cont_loss_stack.append(loss_dict["act_cont_loss"])
            act_back_loss_stack.append(loss_dict["act_back_loss"])

            guide_loss_stack.append(loss_dict["guide_loss"])
            feat_loss_stack.append(loss_dict["feat_loss"])
            att_loss_stack.append(loss_dict["sparse_loss"])

    train_acc = train_num_correct / train_num_total

    train_log_dict = {}
    train_log_dict["train_act_inst_cls_loss"] = np.mean(act_inst_loss_stack)
    train_log_dict["train_act_cont_cls_loss"] = np.mean(act_cont_loss_stack)
    train_log_dict["train_act_back_cls_loss"] = np.mean(act_back_loss_stack)
    train_log_dict["train_guide_loss"] = np.mean(guide_loss_stack)
    train_log_dict["train_feat_loss"] = np.mean(feat_loss_stack)
    train_log_dict["train_att_loss"] = np.mean(att_loss_stack)
    train_log_dict["train_loss"] = np.mean(loss_stack)
    train_log_dict["train_acc"] = train_acc

    print("")
    print("train_act_inst_cls_loss:{:.3f}  train_act_cont_cls_loss:{:.3f}".format(np.mean(act_inst_loss_stack),
                                                                                      np.mean(act_cont_loss_stack)))
    print("train_act_back_cls_loss:{:.3f}  train_att_loss:{:.3f}".format(np.mean(act_back_loss_stack),
                                                                             np.mean(att_loss_stack)))
    print("train_feat_loss:        {:.3f}  train_loss:{:.3f}".format(np.mean(feat_loss_stack), np.mean(loss_stack)))
    print("train acc:{:.3f}".format(train_acc))
    print("-------------------------------------------------------------------------------")

    return train_log_dict



@torch.no_grad()
def evaluate_weakly(model, criterion, postprocessors, data_loader, device, output_dir, wo_class_error=False, args=None, logger=None, class_nums=20):
    model.eval()


    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    action_evaluator = build_evaluator('train', args)
    # action_evaluator = build_evaluator('val', args)
    _cnt = 0

    test_num_correct = 0
    test_num_total = 0
    liebiao=[]
    for samples, info in metric_logger.log_every(data_loader, 50, header, logger=logger):
        samples = samples.to(device, non_blocking=True)
        info = [{
            k: v.to(device) if not isinstance(v, str) else v
            for k, v in t.items()
        } for t in info]

        strides = torch.stack([t['stride'] for t in info], dim=0)
        video_durations = torch.stack([t["video_duration"] for t in info], dim=0)
        feature_durations = torch.stack([t["feature_duration"] for t in info], dim=0)
        offsets = torch.stack([t["offset"] for t in info], dim=0)

        vid_label = [d['labels'] for d in info]

        vid_label_t = torch.stack([
            F.one_hot(torch.unique(label), num_classes=class_nums).sum(dim=0).clamp_max(1).float()
            for label in vid_label
        ], dim=0)

        act_inst_cls, act_cont_cls, act_back_cls, \
        act_inst_feat, act_cont_feat, act_back_feat, \
        temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas, grid = model(samples, info, weakly=args.weakly)
        # loss_dict = criterion(outputs, info)

        temp_cas = act_inst_cas

        fg_score = act_inst_cls[:, :args.action_cls_num]
        label_np = vid_label_t.cpu().numpy()
        score_np = fg_score.cpu().numpy()
        pred_np = np.zeros_like(score_np)
        pred_np[score_np >= args.cls_threshold] = 1
        pred_np[score_np < args.cls_threshold] = 0

        correct_pred = np.sum(label_np == pred_np, axis=1)
        test_num_correct += np.sum((correct_pred == args.action_cls_num))
        test_num_total += correct_pred.shape[0]

        temp_cls_score_np = temp_cas[:, :, :args.action_cls_num].cpu().numpy()
        temp_cls_score_np = np.reshape(temp_cls_score_np, (temp_cas.shape[1], args.action_cls_num, 1))
        temp_att_ins_score_np = temp_att[:, :, 0].unsqueeze(2).expand([-1, -1, args.action_cls_num]).cpu().numpy()
        temp_att_con_score_np = temp_att[:, :, 1].unsqueeze(2).expand([-1, -1, args.action_cls_num]).cpu().numpy()
        temp_att_ins_score_np = np.reshape(temp_att_ins_score_np, (temp_cas.shape[1], args.action_cls_num, 1))
        temp_att_con_score_np = np.reshape(temp_att_con_score_np, (temp_cas.shape[1], args.action_cls_num, 1))

        score_np = np.reshape(score_np, (-1))
        if score_np.max() > args.cls_threshold:
            cls_prediction = np.array(np.where(score_np > args.cls_threshold)[0])
        else:
            cls_prediction = np.array([np.argmax(score_np)], dtype=np.int64)

        temp_cls_score_np = temp_cls_score_np[:, cls_prediction]
        temp_att_ins_score_np = temp_att_ins_score_np[:, cls_prediction]
        temp_att_con_score_np = temp_att_con_score_np[:, cls_prediction]



        int_temp_cls_scores = temp_cls_score_np

        int_temp_att_ins_score_np = temp_att_ins_score_np
        int_temp_att_con_score_np = temp_att_con_score_np

        cas_act_thresh = np.arange(0.15, 0.25, 0.05)
        att_act_thresh = np.arange(0.15, 1.00, 0.05)
        # cas_act_thresh = [0.005, 0.01, 0.015, 0.02]
        # att_act_thresh = [0.005, 0.01, 0.015, 0.02]

        proposal_dict = {}
        # CAS based proposal generation
        # cas_act_thresh = []
        for act_thresh in cas_act_thresh:

            tmp_int_cas = int_temp_cls_scores.copy()
            zero_location = np.where(tmp_int_cas < act_thresh)
            tmp_int_cas[zero_location] = 0

            tmp_seg_list = []
            for c_idx in range(len(cls_prediction)):
                pos = np.where(tmp_int_cas[:, c_idx] >= act_thresh)
                tmp_seg_list.append(pos)


            props_list = get_proposal_oic(tmp_seg_list, (1.0 * tmp_int_cas + 0.0 * int_temp_att_ins_score_np),
                                          cls_prediction, score_np, grid, lamb=0.2, gamma=0.0)
            # props_list = get_proposal_oic(tmp_seg_list, (0.70 * tmp_int_cas + 0.30 * int_temp_att_ins_score_np),
            #                               cls_prediction, score_np,grid, lamb=0.150, gamma=0.0)

            for i in range(len(props_list)):
                if len(props_list[i]) == 0:
                    continue
                class_id = props_list[i][0][0]

                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []

                proposal_dict[class_id] += props_list[i]

        # att_act_thresh = []
        for att_thresh in att_act_thresh:

            tmp_int_att = int_temp_att_ins_score_np.copy()
            zero_location = np.where(tmp_int_att < att_thresh)
            tmp_int_att[zero_location] = 0

            tmp_seg_list = []
            for c_idx in range(len(cls_prediction)):
                pos = np.where(tmp_int_att[:, c_idx] >= att_thresh)
                tmp_seg_list.append(pos)

            props_list = get_proposal_oic(tmp_seg_list, (1.0 * int_temp_cls_scores + 0.0 * tmp_int_att), cls_prediction,
                                          score_np, grid, lamb=0.2, gamma=0.0)
            # props_list = get_proposal_oic(tmp_seg_list, (0.70 * int_temp_cls_scores + 0.30 * tmp_int_att),
            #                               cls_prediction, score_np, grid, lamb=0.150, gamma=0.250)

            for i in range(len(props_list)):
                if len(props_list[i]) == 0:
                    continue
                class_id = props_list[i][0][0]

                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []

                proposal_dict[class_id] += props_list[i]


        scores = []
        labels = []
        segments = []

        for key, data in proposal_dict.items():
            label = torch.stack([torch.tensor(item[0], dtype=torch.int64) for item in data])  # shape: [100]
            score = torch.stack([torch.tensor(item[1], dtype=torch.float32) for item in data])  # shape: [100]
            segment = torch.stack([
                torch.stack([torch.tensor(item[2], dtype=torch.float32), torch.tensor(item[3], dtype=torch.float32)]) for item in data
            ])
            labels.append(label)
            scores.append(score)
            segments.append(segment)
        if len(labels) != 0:
            labels = torch.cat(labels, dim=0).cuda()
        else:
            labels = torch.tensor([]).cuda()

        if len(scores) != 0:
            scores = torch.cat(scores, dim=0).cuda()
        else:
            scores = torch.tensor([]).cuda()

        if len(segments) != 0:
            segments = torch.cat(segments, dim=0).cuda()
        else:
            segments = torch.tensor([]).cuda()
        # labels = torch.cat(labels, dim=0).cuda()
        # scores = torch.cat(scores, dim=0).cuda()
        # segments = torch.cat(segments, dim=0).cuda()
        results= [{
            'scores': scores,
            'labels': labels,
            'segments': segments,
        }]

        # results = postprocessors(outputs, video_durations, feature_durations, strides, offsets, args.duration_thresh)
        preds = {
            t['video_name']: output
            for t, output in zip(info, results)
        }

        #=========================
        # print(preds[list(preds.keys())[0]]['scores'].shape)
        # sub_dict = preds[list(preds.keys())[0]]
        # sub_dict = {k: v.tolist() for k, v in sub_dict.items()}
        # preds1 = {
        #     list(preds.keys())[0]: sub_dict
        # }
        # liebiao.append(preds1)
        #========================
        action_evaluator.update(preds)

        # weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

    # with open("filter.json", "w", encoding="utf-8") as f:
    #     json.dump(liebiao, f, ensure_ascii=False, indent=4)
    print('==============')
    my_dict = {
        "shuju": action_evaluator.all_pred['nms']
    }
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(my_dict, f, ensure_ascii=False, indent=4)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)
    if action_evaluator is not None:
        action_evaluator.synchronize_between_processes()

        action_evaluator.accumulate()

        action_evaluator.summarize()

    stats = {}
    stats.update({f'{k}': meter.global_avg for k,meter in metric_logger.meters.items() if meter.count > 0})

    if action_evaluator is not None:
        for k, v in action_evaluator.stats.items():
            for vk, vv in v.items():
                stats[vk + '_' + k] = vv

        mAP_values = ' '.join([f'{k}: {100*v:.2f}'.format(k, v)
                              for k, v in stats.items() if k.startswith('mAP')])
        logging.info(mAP_values)

        stats['stats_summary'] = action_evaluator.stats_summary

    return stats, action_evaluator


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    final_res = []
    for samples, grids, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        grids = grids.to(device)

        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples, grids)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id),
                        "category_id": l,
                        "bbox": b,
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)

    return final_res
