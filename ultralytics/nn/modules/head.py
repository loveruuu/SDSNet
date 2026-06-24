# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Model head modules
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.yolo.utils.tal import dist2bbox, make_anchors

from .block import DFL, Proto
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init_

__all__ = ['Detect', 'Detect_LSCSBD', 'Segment', 'Segment2', 'Pose', 'Classify', 'RTDETRDecoder']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.layers = nn.ModuleList(nn.Conv2d(n, k, 1) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LQE(nn.Module):
    """
    位置质量估计器 (Location Quality Estimator, LQE)
    用于评估和调整边界框预测的质量分数，结合分布统计信息提升精度
    """
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        """
        初始化 LQE 模块
        参数:
            k: 前 k 个最高概率值的数量，用于统计分析
            hidden_dim: MLP隐藏层维度
            num_layers: MLP层数
            reg_max: 回归的最大值（边界框分布的最大范围）
        """
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        # 定义一个多层感知机（MLP），输入维度为 4*(k+1)，输出为 1
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
        # 初始化最后一层的偏置和权重为 0
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        """
        前向传播
        参数:
            scores: 初始分类得分 [B, num_classes, h, w]
            pred_corners: 预测的边界框角点分布 [B, 4*(reg_max), h, w]
        返回:
            调整后的质量分数
        """
        # 计算 softmax 概率
        B, C, H, W = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, self.reg_max, 4, H, W), dim=1)
        # 提取前 k 个最高概率值及其索引
        prob_topk, _ = prob.topk(self.k, dim=1)
        # 将 top-k 概率及其均值拼接，作为统计特征
        stat = torch.cat([prob_topk, prob_topk.mean(dim=1, keepdim=True)], dim=1)
        # 通过 MLP 计算质量分数调整值
        quality_score = self.reg_conf(stat.reshape(B, -1, H, W))
        # 将初始得分与质量调整值相加
        return scores + quality_score


class Detect_LSCSBD(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3),
                          Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * self.reg_max, 1)
                          ) for x in ch)

        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3),
                          Conv(c3, c3, 3),
                          nn.Conv2d(c3, self.nc, 1)
                          ) for x in ch)

        self.lqe = nn.ModuleList(LQE(4, 64, 2, self.reg_max) for x in ch)

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            pred_corners = self.cv2[i](x[i])
            pred_scores = self.lqe[i](self.cv3[i](x[i]), pred_corners)
            x[i] = torch.cat((pred_corners, pred_scores), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)

        # self.cv2 = nn.ModuleList(
        #     nn.Sequential(DWConv(x, c2, 5), Conv(c2, c2, 1), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        # self.cv3 = nn.ModuleList(nn.Sequential(DWConv(x, c3, 5), Conv(c3, c3, 1), nn.Conv2d(c3, self.nc, 1)) for x in ch)

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, ch)
        self.npr = 32  # intermediate convolutional feature dimension
        self.cv1 = Conv(ch[0], self.npr, k=3)
        self.upsample = nn.ConvTranspose2d(self.npr, self.npr//2, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        # self.cv2 = Conv(self.npr//2, self.npr//4, k=3)
        # self.cv3 = Conv(self.npr//4, self.nc+1) ###### self.nc+1 means add the background
        self.cv2 = Conv(self.npr//2, self.nc+1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        # p = self.cv3(self.cv2(self.upsample(self.cv1(x[0])))) # mask protos
        p = self.cv2(self.upsample(self.cv1(x[0])))
        if self.training:
            return p
        return p


class Segment2(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, ch)
        ###### Jiayuan changed self.nm to self.nc
        self.npr = 32  # intermediate convolutional feature dimension
        self.cv1 = Conv(ch[0], self.npr, k=3)
        self.upsample1 = nn.ConvTranspose2d(self.npr, self.npr // 2, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(self.npr // 2, self.npr // 2, k=3)
        self.upsample2 = nn.ConvTranspose2d(self.npr// 2, self.npr // 4, 2, 2, 0, bias=True)
        self.cv3 = Conv(self.npr//4, self.nc+1) ###### self.nc+1 means add the background
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        y1 = self.cv1(x[0])
        y2 = self.upsample1(y1)
        y3 = self.cv2(y2)
        y3 = self.upsample2(y3)
        y4 = self.cv3(y3)
        if self.training:
            return y4
        return y4


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 1:2].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3].sigmoid_()  # inplace sigmoid
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class RTDETRDecoder(nn.Module):

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hidden_dim=256,
            num_queries=300,
            strides=(8, 16, 32),  # TODO
            nl=3,
            num_decoder_points=4,
            nhead=8,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.,
            act=nn.ReLU(),
            eval_idx=-1,
            # training args
            num_denoising=100,
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        super().__init__()
        assert len(ch) <= nl
        assert len(strides) == len(ch)
        for _ in range(nl - len(strides)):
            strides.append(strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = strides
        self.nl = nl
        self.nc = nc
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers

        # backbone feature projection
        self._build_input_proj_layer(ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, act, nl,
                                                          num_decoder_points)
        self.decoder = DeformableTransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(nc, hidden_dim)
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, nc)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, nc) for _ in range(num_decoder_layers)])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_decoder_layers)])

        self._reset_parameters()

    def forward(self, feats, gt_meta=None):
        # input projection and embedding
        memory, spatial_shapes, _ = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training:
            raise NotImplementedError
            # denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
            #     get_contrastive_denoising_training_group(gt_meta,
            #                                 self.num_classes,
            #                                 self.num_queries,
            #                                 self.denoising_class_embed.weight,
            #                                 self.num_denoising,
            #                                 self.label_noise_ratio,
            #                                 self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask = None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(target,
                                              init_ref_points_unact,
                                              memory,
                                              spatial_shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        if not self.training:
            out_logits = out_logits.sigmoid_()
        return out_bboxes, out_logits  # enc_topk_bboxes, enc_topk_logits, dn_meta

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)

    def _build_input_proj_layer(self, ch):
        self.input_proj = nn.ModuleList()
        for in_channels in ch:
            self.input_proj.append(
                nn.Sequential(nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, bias=False),
                              nn.BatchNorm2d(self.hidden_dim)))
        in_channels = ch[-1]
        for _ in range(self.nl - len(ch)):
            self.input_proj.append(
                nn.Sequential(nn.Conv2D(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
                              nn.BatchNorm2d(self.hidden_dim)))
            in_channels = self.hidden_dim

    def _generate_anchors(self, spatial_shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=torch.float32),
                                            torch.arange(end=w, dtype=torch.float32),
                                            indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)

            valid_WH = torch.tensor([h, w]).to(torch.float32)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = torch.concat(anchors, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors.to(device=device, dtype=dtype), valid_mask.to(device=device)

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.nl > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.nl):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        for feat in proj_feats:
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return feat_flatten, spatial_shapes, level_start_index

    def _get_decoder_input(self, memory, spatial_shapes, denoising_class=None, denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        anchors, valid_mask = self._generate_anchors(spatial_shapes, dtype=memory.dtype, device=memory.device)
        memory = torch.where(valid_mask, memory, 0)
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)  # (bs, h*w, nc)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors  # (bs, h*w, 4)

        # (bs, topk)
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        # extract region proposal boxes
        # (bs, topk_ind)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        topk_ind = topk_ind.view(-1)

        # Unsigmoided
        reference_points_unact = enc_outputs_coord_unact[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        enc_topk_bboxes = torch.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = enc_outputs_class[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            target = output_memory[batch_ind, topk_ind].view(bs, self.num_queries, -1)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits
