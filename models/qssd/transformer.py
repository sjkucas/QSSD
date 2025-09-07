
import copy
import math
import sys
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from .ops.temporal_deform_attn import DeformAttn
from .modules import ResizerBackbone, GatedConv
from .utils import get_feature_grids
from .utils import MLP, _get_activation_fn
import matplotlib.pyplot as plt
class TransposeConvUpsample(nn.Module):
    def __init__(self, in_dim=512, upsample_ratio=4):
        super().__init__()
        self.trans_conv = nn.ConvTranspose1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=upsample_ratio,
            stride=upsample_ratio
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.trans_conv(x)  # [B, D, L * upsample_ratio]
        x = x.transpose(1, 2)
        return x
class ConvDownsample(nn.Module):
    def __init__(self, in_dim=512, downsample_ratio=4):  # 4096 / 4 = 1024
        super().__init__()
        self.downsample_ratio = downsample_ratio
        # self.conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim,
        #                       kernel_size=downsample_ratio, stride=downsample_ratio)
        self.pool = nn.AvgPool1d(kernel_size=downsample_ratio, stride=downsample_ratio)
    def forward(self, x, mask):
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)

        B, L = mask.shape
        L_new = x.shape[1]
        mask = mask[:, :L_new * self.downsample_ratio]  # 截断为整数倍
        mask = mask.view(B, L_new, self.downsample_ratio)  # 每组patch
        mask = mask.any(dim=2)
        return x, mask
class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_deform_heads=2, gc_kernel_size=11, dc_level=2, group_conv=True, base_scale=2.0, max_queries=3000,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, enc_dropout=0.1, dec_dropout=0.0, attn_dropout=0.1,
                 activation="relu", return_intermediate_dec=False, fix_encoder_proposals=True,
                 num_feature_levels=4, num_sampling_levels=4, dec_n_points=4, length_ratio=-1,
                 temperature=10000, num_classes=20, query_selection_ratio=0.5,
                 
                 ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_deform_heads = n_deform_heads
        self.base_scale = base_scale
        self.max_queries = max_queries
        self.num_sampling_levels = num_sampling_levels
        self.fix_encoder_proposals = fix_encoder_proposals
        self.length_ratio = length_ratio
        self.query_selection_ratio = query_selection_ratio

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, gc_kernel_size, dc_level, enc_dropout, group_conv
        )

        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)


        decoder_layer = DeformableTransformerDecoderLayer(
            d_model, dim_feedforward, dec_dropout, attn_dropout, activation,
            num_sampling_levels, n_heads, n_deform_heads, dec_n_points
        )

        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers,
            d_model=d_model, return_intermediate=return_intermediate_dec,
            temperature=temperature
        )

        self.level_embed = nn.Embedding(num_feature_levels, d_model)
        self.num_feature_levels = num_feature_levels
        self.enc_proj = ResizerBackbone(d_model, d_model, kernel_size=3, num_feature_levels=num_feature_levels, scale_factor=0.5, with_ln=True)
        self.num_classes = num_classes
        self.one_to_many_cls = nn.Linear(d_model, num_classes)

        self.encoder_out = nn.Linear(d_model, d_model)
        self.encoder_out_norm = nn.LayerNorm(d_model)

        self.content_proj = nn.Linear(d_model, d_model)
        self.content_norm = nn.LayerNorm(d_model)
        scales = [2 ** lvl for lvl in range(num_feature_levels)]
        self.learnable_scale = nn.Parameter(torch.tensor(scales, dtype=torch.float32))

        self.memory_proj = nn.Linear(d_model, d_model)
        self.memory_norm = nn.LayerNorm(d_model)

        #====================weakly=====================
        self.snippet_cls1 = nn.Linear(in_features=d_model, out_features=d_model)
        self.snippet_cls = nn.Linear(in_features=d_model, out_features=(self.num_classes + 1))
        self.dropout = nn.Dropout(0.7)
        #===============================================
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformAttn):
                m._reset_parameters()

    def get_proposal_pos_embed(self, proposals):
        temperature = 10000
        dim_t = torch.arange(self.d_model, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.d_model)
        # N, L, 4, 128
        pos_ct = proposals[:, :, [0]] / dim_t
        pos_w = proposals[:, :, [1]] / dim_t
        pos_ct = torch.stack((pos_ct[:, :, 0::2].sin(), pos_ct[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_ct, pos_w), dim=2)
        # N, L, 4, 64, 2
        return pos

    def get_valid_ratio(self, mask):
        _, T = mask.shape
        valid_T = torch.sum(~mask, 1)
        valid_ratio = valid_T.float() / T
        return valid_ratio    # shape=(bs)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, temporal_lengths, grids, fps, stride):
        N_, S_, C_ = memory.shape
        proposals = []
        _cur = 0

        base_scale = self.base_scale / fps

        for lvl, T_ in enumerate(temporal_lengths):
            timeline = grids[:, _cur:(_cur + T_)]

            scale = torch.ones_like(timeline) * (base_scale[..., None]) * 2 ** lvl
            proposal = torch.stack((timeline, scale), -1).view(N_, -1, 2)
            proposals.append(proposal)
            _cur += T_

        output_proposals = torch.cat(proposals, dim=1)
        output_proposals_valid = ~memory_padding_mask
        output_proposals = torch.cat([
            output_proposals[..., :1],
            output_proposals[..., 1:].log()
        ], dim=-1)

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask[..., None], float(0))
        output_memory = self.encoder_out_norm(self.encoder_out(output_memory))

        return output_memory, output_proposals, output_proposals_valid


    def forward(self, srcs, masks, pos_embeds, grids,
                feature_durations, fps, stride, query_embed=None, weakly=False, vid_label=None):
        '''
        Params:
            srcs: list of Tensor with shape (bs, c, t)
            masks: list of Tensor with shape (bs, t)
            pos_embeds: list of Tensor with shape (bs, c, t)
            query_embed: list of Tensor with shape (nq, 2c)
        Returns:
            hs: list, per layer output of decoder
            init_reference_out: reference points predicted from query embeddings
            inter_references_out: reference points predicted from each decoder layer
            memory: (bs, c, t), final output of the encoder
        '''

        # deformable encoder
        memory = self.encoder(srcs[0].transpose(1, 2), masks[0])  # shape=(bs, t, c)

        enc_input = memory.transpose(1, 2)
        srcs, masks = self.enc_proj(enc_input, masks[0])

        cur_stride = stride
        if self.num_feature_levels > 1:
            for l in range(1, self.num_feature_levels):
                mask = masks[l]
                cur_stride = stride * 2 ** l
                grid = get_feature_grids(mask, fps, cur_stride, cur_stride)
                for i, (g, m, m2) in enumerate(zip(grids[-1], mask, masks[l-1])):
                    g = F.interpolate(g[None, None, ~m2], size=(~m).sum().item(), mode='linear')[0, 0, :]
                    grid[i, :g.size(0)] = g

                grids.append(grid)

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        grid_flatten = []
        temporal_lengths = []

        for lvl, (src, mask, grid) in enumerate(zip(srcs, masks, grids)):
            bs, c, t = src.shape
            temporal_lengths.append(t)
            # (bs, c, t) => (bs, t, c)
            src = src.transpose(1, 2) + self.level_embed.weight[lvl].view(1, 1, -1)
            src_flatten.append(src)
            mask_flatten.append(mask)
            grid_flatten.append(grid)
        src_flatten = torch.cat(src_flatten, dim=1)
        mask_flatten = torch.cat(mask_flatten, dim=1)
        grid_flatten = torch.cat(grid_flatten, dim=1)
        temporal_lengths = torch.as_tensor(temporal_lengths, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((temporal_lengths.new_zeros((1, )), temporal_lengths.cumsum(0)[:-1]), dim=0)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], dim=1)   # (bs, nlevels)

        bs, t, c = memory.shape



        output_memory, output_proposals, output_proposals_valid = self.gen_encoder_output_proposals(
            src_flatten, mask_flatten, temporal_lengths, grid_flatten.detach(), fps, stride
        )

        memory_input = self.memory_norm(self.memory_proj(src_flatten))


        # enc_outputs_class = self.decoder.class_embed[-1](output_memory)[:, :, 0].unsqueeze(-1)

        temp_att = self.decoder.class_embed[-1](output_memory)
        # temp_att = self.decoder.class_embed[-1](output_memory.permute(0, 2, 1)).permute(0, 2, 1)
        enc_outputs_class = temp_att[:, :, 0].unsqueeze(-1)
        enc_outputs_mask = ~output_proposals_valid
        enc_scores = enc_outputs_class.squeeze(-1)

        valid_scores = enc_scores.masked_fill(enc_outputs_mask, float('-50000'))
        valid_scores = torch.cat([
            valid_scores[:, :temporal_lengths[0].item()],
            F.interpolate(valid_scores[:, :temporal_lengths[0]].unsqueeze(1), size=temporal_lengths[1].item(), mode='linear',
                          align_corners=False).squeeze(1),
            F.interpolate(valid_scores[:, :temporal_lengths[0]].unsqueeze(1), size=temporal_lengths[2].item(), mode='linear',
                          align_corners=False).squeeze(1)
        ], dim=1)


        # ================weakly======================
        if weakly:
            if self.training:
                new_output_memory = torch.zeros_like(output_memory[:, :temporal_lengths[0].item(), :])
                new_temp_att = torch.zeros_like(temp_att[:, :temporal_lengths[0].item(), :])
                # output_memory = output_memory[:, :temporal_lengths[0].item(), :]
                # temp_att = temp_att[:, :temporal_lengths[0].item(), :]
                for b in range(output_memory.shape[0]):
                    # 有放回随机采样并排序
                    sample_idxs = torch.randint(0, (~mask_flatten[b][:temporal_lengths[0].item()]).sum().item(), (temporal_lengths[0].item(),), device='cuda')
                    sample_idxs, _ = torch.sort(sample_idxs)
                    new_output_memory[b] = output_memory[b, sample_idxs, :]
                    new_temp_att[b] = temp_att[b, sample_idxs, :]
            else:
                new_output_memory = output_memory[:, :(~mask_flatten[0][:temporal_lengths[0].item()]).sum().item(),:]
                new_temp_att = temp_att[:, :(~mask_flatten[0][:temporal_lengths[0].item()]).sum().item(), :]

            # embeded_feature = output_memory[:, :temporal_lengths[0].item(), :]
            embeded_feature = new_output_memory
            device = embeded_feature.device
            batch_size, temp_len = embeded_feature.shape[0], embeded_feature.shape[1]

            inst_topk_num = max(temp_len // 8, 1)
            cont_topk_num = max(temp_len // 3, 1)
            back_topk_num = max(temp_len // 3, 1)
            # inst_topk_num = max(temp_len // 2, 1)
            # cont_topk_num = max(temp_len // 10, 1)
            # back_topk_num = max(temp_len // 10, 1)


            # temp_att = torch.softmax(temp_att[:, :temporal_lengths[0].item(), :], dim=2)
            temp_att = torch.softmax(new_temp_att, dim=2)
            act_inst_att = temp_att[:, :, 0].unsqueeze(2)
            act_cont_att = temp_att[:, :, 1].unsqueeze(2)
            act_back_att = temp_att[:, :, 2].unsqueeze(2)

            embeded_feature_rev = embeded_feature

            select_idx = torch.ones((batch_size, temp_len, 1), device=device)
            select_idx = self.dropout(select_idx)
            embeded_feature = embeded_feature * select_idx

            act_cas = self.snippet_cls1(self.dropout(embeded_feature))
            act_cas = F.leaky_relu(act_cas, negative_slope=0.01)
            act_cas = self.snippet_cls(act_cas)


            act_inst_cas = act_cas * act_inst_att
            act_cont_cas = act_cas * act_cont_att
            act_back_cas = act_cas * act_back_att



            sorted_inst_cas, _ = torch.sort(act_inst_cas, dim=1, descending=True)
            sorted_cont_cas, _ = torch.sort(act_cont_cas, dim=1, descending=True)
            sorted_back_cas, _ = torch.sort(act_back_cas, dim=1, descending=True)


            act_inst_cls = torch.mean(sorted_inst_cas[:, :inst_topk_num, :], dim=1)
            act_cont_cls = torch.mean(sorted_cont_cas[:, :cont_topk_num, :], dim=1)
            act_back_cls = torch.mean(sorted_back_cas[:, :back_topk_num, :], dim=1)

            act_inst_cls = torch.softmax(act_inst_cls, dim=1)
            act_cont_cls = torch.softmax(act_cont_cls, dim=1)
            act_back_cls = torch.softmax(act_back_cls, dim=1)

            act_inst_cas = torch.softmax(act_inst_cas, dim=2)
            act_cont_cas = torch.softmax(act_cont_cas, dim=2)
            act_back_cas = torch.softmax(act_back_cas, dim=2)

            act_cas = torch.softmax(act_cas, dim=2)


            _, sorted_act_inst_att_idx = torch.sort(act_inst_att, dim=1, descending=True)
            _, sorted_act_cont_att_idx = torch.sort(act_cont_att, dim=1, descending=True)
            _, sorted_act_back_att_idx = torch.sort(act_back_att, dim=1, descending=True)
            act_inst_feat_idx = sorted_act_inst_att_idx[:, :inst_topk_num, :].expand([-1, -1, self.d_model])
            act_cont_feat_idx = sorted_act_cont_att_idx[:, :cont_topk_num, :].expand([-1, -1, self.d_model])
            act_back_feat_idx = sorted_act_back_att_idx[:, :back_topk_num, :].expand([-1, -1, self.d_model])

            act_inst_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_inst_feat_idx), dim=1)
            act_cont_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_cont_feat_idx), dim=1)
            act_back_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_back_feat_idx), dim=1)

            if self.training:
                feat = embeded_feature_rev / torch.norm(embeded_feature_rev, dim=-1, keepdim=True)
                T1 = 100
                actionness_scores = act_inst_att.squeeze(-1)
                sorted_scores, sorted_indices = torch.sort(actionness_scores, dim=1, descending=True)
                topk_indices1 = sorted_indices[:, :T1]
                bottomk_indices1 = sorted_indices[:, -T1:]

                F_fg_A = torch.gather(feat, 1, topk_indices1.unsqueeze(-1).expand(-1, -1, feat.shape[2]))
                F_bg_A = torch.gather(feat, 1, bottomk_indices1.unsqueeze(-1).expand(-1, -1, feat.shape[2]))

                feat_similarity = torch.einsum('bnf,bkf->bnk', feat, F_fg_A)
                feat_sim = torch.mean(feat_similarity, dim=2)
                feat_var = -1 * torch.var(feat_similarity, dim=2)
                A_fg_S = feat_sim + feat_var
                feat_similarity = torch.einsum('bnf,bkf->bnk', feat, F_bg_A)
                feat_sim = torch.mean(feat_similarity, dim=2)
                feat_var = -1 * torch.var(feat_similarity, dim=2)
                A_bg_S = feat_sim + feat_var
                A_fg_S = A_fg_S.unsqueeze(-1)
                A_bg_S = A_bg_S.unsqueeze(-1)
                A_S = torch.cat([A_fg_S, A_bg_S], dim=-1)
                A_S = F.softmax(A_S, dim=-1)[:, :, 0]

                sorted_scores, sorted_indices = torch.sort(A_S, dim=1, descending=True)
                topk_indices2 = sorted_indices[:, :T1]
                bottomk_indices2 = sorted_indices[:, -T1:]


                pseudo = torch.zeros(feat.shape[0], feat.shape[1], dtype=torch.long)
                for b in range(feat.shape[0]):
                    topk_intersection = set(topk_indices1[b].tolist()) & set(topk_indices2[b].tolist())
                    bottomk_intersection = set(bottomk_indices1[b].tolist()) & set(bottomk_indices2[b].tolist())

                    pseudo[b, list(topk_intersection)] = 1
                    pseudo[b, list(bottomk_intersection)] = 2

                easy_act_list = []
                hard_act_list = []
                easy_bg_list = []
                hard_bg_list = []
                for b in range(feat.shape[0]):
                    set1 = set(topk_indices1[b].tolist())
                    set2 = set(topk_indices2[b].tolist())
                    set3 = set(bottomk_indices1[b].tolist())
                    set4 = set(bottomk_indices2[b].tolist())

                    inter = list(set1 & set2)
                    if len(inter) == 0:
                        inter = torch.tensor(random.choices(list(set1), k=50))
                    else:
                        inter = torch.tensor(random.choices(inter, k=50))

                    inter1 = list(set3 & set4)
                    if len(inter1) == 0:
                        inter1 = torch.tensor(random.choices(list(set3), k=20))
                    else:
                        inter1 = torch.tensor(random.choices(inter1, k=20))


                    sym_diff = list(set1 ^ set2)
                    if len(sym_diff) == 0:
                        sym_diff = torch.tensor(random.choices(list(set1), k=50))
                    else:
                        sym_diff = torch.tensor(random.choices(sym_diff, k=50))

                    sym_diff1 = list(set3 ^ set4)
                    if len(sym_diff1) == 0:
                        sym_diff1 = torch.tensor(random.choices(list(set3), k=20))
                    else:
                        sym_diff1 = torch.tensor(random.choices(sym_diff1, k=20))


                    easy_act_list.append(feat[b, inter, :])  # shape [len(inter), 512]
                    easy_bg_list.append(feat[b, inter1, :])
                    hard_act_list.append(feat[b, sym_diff, :])  # shape [len(sym_diff), 512]
                    hard_bg_list.append(feat[b, sym_diff1, :])

                easy_act = torch.stack(easy_act_list, dim=0)
                hard_act = torch.stack(hard_act_list, dim=0)
                easy_bg = torch.stack(easy_bg_list, dim=0)
                hard_bg = torch.stack(hard_bg_list, dim=0)

                contrast_pairs = {
                    'EA': easy_act,
                    'EB': easy_bg,
                    'HA': hard_act,
                    'HB': hard_bg
                }


            if self.training:
                return act_inst_cls, act_cont_cls, act_back_cls, \
                    act_inst_feat, act_cont_feat, act_back_feat, \
                    temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas, pseudo, contrast_pairs

            return act_inst_cls, act_cont_cls, act_back_cls, \
                act_inst_feat, act_cont_feat, act_back_feat, \
                temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas
        # ============================================

        enc_outputs_segment = output_proposals

        if self.fix_encoder_proposals:
            enc_outputs_segment_updated = enc_outputs_segment
        else:
            enc_outputs_segment_updated = self.decoder.segment_embed[-1](output_memory)
            enc_outputs_segment_updated = torch.cat([
                output_proposals[..., :1] + enc_outputs_segment_updated[..., :1] * (output_proposals[..., 1:].exp().sum(dim=-1, keepdim=True).detach()),
                output_proposals[..., 1:] + enc_outputs_segment_updated[..., 1:], 
            ], dim=-1)

        # Top-k selection
        ratio = self.query_selection_ratio
        num_valid_points = (valid_scores > float('-50000')).sum(dim=1)
        num_queries = (num_valid_points * ratio).int()
        num_queries = torch.clamp(num_queries, max=self.max_queries)

        max_num_queries = num_queries.max().item()
        query_mask = torch.ones(bs, max_num_queries, dtype=torch.bool, device=valid_scores.device)
        topk_segments = torch.zeros(bs, max_num_queries, 2, device=enc_outputs_segment.device)
        topk_segments_updated = torch.zeros(bs, max_num_queries, 2, device=enc_outputs_segment.device)
        topk_valid_scores = torch.zeros(bs, max_num_queries, device=valid_scores.device)
        topk_content = torch.zeros(bs, max_num_queries, src_flatten.size(-1), device=valid_scores.device, dtype=torch.float32)
        for i, num_point in enumerate(num_queries):
            cur_valid_score, cur_topk_indices = torch.topk(valid_scores[i], num_point, dim=0)
            topk_valid_scores[i, :num_point] = cur_valid_score
            topk_segments[i, :num_point] = torch.gather(enc_outputs_segment[i], 0, cur_topk_indices.unsqueeze(-1).expand(-1, 2))
            topk_segments_updated[i, :num_point] = torch.gather(enc_outputs_segment_updated[i], 0, cur_topk_indices.unsqueeze(-1).expand(-1, 2))
            # topk_enc_classes[i, :num_point] = torch.gather(enc_classes[i], 0, cur_topk_indices)
            query_mask[i, :num_point] = False
            topk_content[i, :num_point] = torch.gather(src_flatten[i], 0, cur_topk_indices.unsqueeze(-1).expand(-1, src_flatten.size(-1)))
        input_mask = query_mask

        tgt = self.content_norm(self.content_proj(topk_content.detach()))

        hs, inter_grids = self.decoder(
            tgt, topk_segments, feature_durations, fps,
            memory_input, temporal_lengths, level_start_index,
            valid_ratios, mask_flatten, query_embed, input_mask,
            None,
        )

        return (
            hs, inter_grids,
            enc_outputs_class, enc_outputs_segment_updated, enc_outputs_mask, output_proposals, query_mask
        )



class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,d_model=256, d_ffn=1024, gc_kernel_size=11, dc_level=2, dropout=0.1, group_conv=True):
        super().__init__()

        if dc_level == 3:
            self.self_attn = GatedConv(d_model, kernel_size=gc_kernel_size, num_levels=dc_level, hidden_dim=2049, group_conv=group_conv)
        else:
            self.self_attn = GatedConv(d_model, kernel_size=gc_kernel_size, num_levels=dc_level, hidden_dim=d_ffn, group_conv=group_conv)
            # self.self_attn = CycleMixer(d_model, 1024, 32, 4)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout3(src2))
        return src

    def forward(self, src, mask):
        src2 = self.self_attn(src, mask)
        src = self.norm1(src + self.dropout1(src2))
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, mask):
        '''
        src: shape=(bs, t, c)
        temporal_lens: shape=(n_levels). content: [t1, t2, t3, ...]
        level_start_index: shape=(n_levels,). [0, t1, t1+t2, ...]
        valid_ratios: shape=(bs, n_levels).
        '''
        output = src
        for _, layer in enumerate(self.layers):
            output = layer(output, mask)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, attn_dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_deform_heads=8, n_points=4, pre_norm=True):
        super().__init__()

        # cross attention
        self.cross_attn = DeformAttn(d_model, n_levels, n_deform_heads, n_points, dropout=attn_dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # # # boundary-aware cross attention
        self.cross_attn_se = DeformAttn(d_model, n_levels, n_deform_heads, n_points, dropout=attn_dropout, boundary_aware=True)
        self.dropout_se = nn.Dropout(dropout)
        self.norm_se = nn.LayerNorm(d_model)

        # self attention 
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout4(tgt2))
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, temporal_lengths, level_start_index, src_padding_mask=None, query_mask=None, attn_mask=None, iou_bias=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, tgt,
            key_padding_mask=query_mask, attn_mask=attn_mask,
            need_weights=False
        )[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # cross attention
        tgt2, _ = self.cross_attn(
            self.with_pos_embed(tgt, query_pos), reference_points, src,
            temporal_lengths, level_start_index, src_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # # # cross attention_boundary
        tgt2, _ = self.cross_attn_se(
            self.with_pos_embed(tgt, query_pos), reference_points, src,
            temporal_lengths, level_start_index, src_padding_mask
        )
        tgt = tgt + self.dropout_se(tgt2)
        tgt = self.norm_se(tgt)
        tgt = self.forward_ffn(tgt)

        return tgt

class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, d_model=256, return_intermediate=False, temperature=10000
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.segment_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.grid_head = MLP(d_model * 2, d_model, d_model, 2)
        self.temperature = temperature

    def get_proposal_pos_embed(self, proposals):
        scale = 2 * math.pi

        dim_t = torch.arange(self.d_model, dtype=torch.float32, device=proposals.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.d_model)
        # N, L, 4
        proposals = proposals * scale
        # N, L, 4, 128
        pos_ct = proposals[:, :, [0]] / dim_t
        pos_w = proposals[:, :, [1]] / dim_t
        pos_ct = torch.stack((pos_ct[:, :, 0::2].sin(), pos_ct[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_ct, pos_w), dim=2)

        # N, L, 4, 64, 2
        return pos

    def forward(self, tgt, enc_output_segments, feature_durations, fps,
                src, temporal_lens, src_level_start_index, src_valid_ratios,
                src_padding_mask=None, query_pos=None, query_mask=None, attn_mask=None):
        '''
        tgt: [bs, nq, C]
        reference_points: [bs, nq, 1 or 2]
        src: [bs, T, C]
        src_valid_ratios: [bs, levels]
        '''
        output = tgt
        intermediate = []
        intermediate_grids = []
        segment_outputs = enc_output_segments.detach()

        reference_points = torch.cat([
            segment_outputs[..., :1], segment_outputs[..., 1:].exp()
        ], dim=-1)
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points / feature_durations[:, None, None]
            reference_points_input = reference_points_input[:, :, None, :] * src_valid_ratios[:, None, :, None]
            iou_bias = None

            grid_sine_embed = self.get_proposal_pos_embed(reference_points)
            raw_query_pos = self.grid_head(grid_sine_embed) # nq, bs, 256
            query_pos = raw_query_pos

            output = layer(output, query_pos, reference_points_input, src, temporal_lens, src_level_start_index, src_padding_mask, query_mask, attn_mask, iou_bias)

            # segment refinement

            if self.segment_embed is not None:
                segment_outputs_detach = segment_outputs.detach()
                segment_outputs = self.segment_embed[lid](output)
                segment_outputs = torch.cat([
                    segment_outputs_detach[..., :1] + segment_outputs[..., :1] * (segment_outputs_detach[..., 1:].exp().sum(dim=-1, keepdim=True).detach()),
                    segment_outputs_detach[..., 1:] + segment_outputs[..., 1:],
                ], dim=-1)

                new_reference_points = torch.cat([
                    segment_outputs[..., :1], segment_outputs[..., 1:].exp()
                ], dim=-1)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_grids.append(segment_outputs)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_grids)

        return output, segment_outputs

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_deformable_transformer(args):
     return DeformableTransformer(
        d_model=args.hidden_dim,
        n_heads=args.n_heads,
        n_deform_heads=args.n_deform_heads,
        gc_kernel_size=args.gc_kernel_size,
        dc_level=args.dc_level,
        group_conv=args.group_conv,
        base_scale=args.base_scale,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        enc_dropout=args.enc_dropout,
        dec_dropout=args.dec_dropout,
        attn_dropout=args.attn_dropout,
        activation=args.transformer_activation,
        return_intermediate_dec=True,
        max_queries=args.max_queries,
        fix_encoder_proposals=args.fix_encoder_proposals,
        num_feature_levels=args.num_feature_levels,
        num_sampling_levels=args.num_sampling_levels,
        dec_n_points=args.dec_n_points,
        length_ratio=args.length_ratio,
        temperature=args.temperature,
        num_classes=args.num_classes,
        query_selection_ratio=args.query_selection_ratio,
    )
