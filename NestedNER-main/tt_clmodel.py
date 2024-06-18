import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer,BertModel
import os
import argparse

class MyModel(nn.Module):
    def __init__(self, arg):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model[arg.data_name])
        self.span_embedding = nn.Linear(self.bert.hidden_size + arg.max_span_size, arg.embedding_size)

        self.anchor_embed = nn.Linear(self.bert.hidden_size, embedding_size)
        self.positive_embed = nn.Linear(self.bert.hidden_size, embedding_size)
        self.boundary_embed = nn.Linear(self.bert.hidden_size, embedding_size)

        self.margin = 0.2  # 对比学习的margin

        self.type_loss_weight = nn.Parameter(torch.Tensor([1.0]))  # 类型损失函数的权重
        self.boundary_loss_weight = nn.Parameter(torch.Tensor([1.0]))  # 边界损失函数的权重

    def forward(self, x, attention_mask, anchor_indices, positive_indices, boundary_anchor_indices, boundary_positive_indices, iou_weights):
        outputs = self.bert(x, attention_mask=attention_mask)
        h_cls = outputs[0][:, 0, :]  # 句子内容信息的编码

        # 第二部分 - 获取跨度
        spans = []
        for i in range(len(x)):
            for j in range(i, min(i + cfg.max_span_length, len(x))):
                span_content = outputs[0][i:j+1, :]
                span_width_embed = torch.randn(1, self.bert.hidden_size)  # 跨度宽度嵌入，随机初始化
                span_embedding = torch.cat([h_cls, span_content, span_width_embed], dim=0)
                spans.append(span_embedding)

        spans = torch.stack(spans, dim=0)

        # 第三部分 - 跨度聚类
        spans_embed = F.relu(self.span_embedding(spans))

        # 对比学习损失函数
        anchor_embeds = spans_embed[anchor_indices]
        positive_embeds = spans_embed[positive_indices]
        negative_embeds = spans_embed[~(anchor_indices | positive_indices)]

        boundary_anchor_embeds = spans_embed[boundary_anchor_indices]
        boundary_positive_embeds = spans_embed[boundary_positive_indices]

        loss = self.contrastive_loss(anchor_embeds, positive_embeds, negative_embeds, iou_weights)
        boundary_loss = self.contrastive_loss(boundary_anchor_embeds, boundary_positive_embeds, negative_embeds, iou_weights)

        total_loss = self.type_loss_weight * loss + self.boundary_loss_weight * boundary_loss

        return total_loss

    def contrastive_loss(self, anchor_embeds, positive_embeds, negative_embeds, iou_weights):
        anchor_embeds = F.relu(self.anchor_embed(anchor_embeds))
        positive_embeds = F.relu(self.positive_embed(positive_embeds))
        negative_embeds = F.relu(self.positive_embed(negative_embeds))

        # 计算距离
        anchor_positive_dist = torch.pairwise_distance(anchor_embeds, positive_embeds)
        anchor_negative_dist = torch.pairwise_distance(anchor_embeds.unsqueeze(1), negative_embeds.unsqueeze(0))
        anchor_negative_dist = anchor_negative_dist.min(dim=1)[0]

        # 应用iou权重
        weighted_anchor_positive_dist = anchor_positive_dist * iou_weights
        weighted_anchor_negative_dist = anchor_negative_dist * iou_weights.unsqueeze(1)
        weighted_anchor_negative_dist = weighted_anchor_negative_dist.min(dim=1)[0]

        # 对比学习损失
        loss = torch.clamp(weighted_anchor_positive_dist - weighted_anchor_negative_dist + self.margin, min=0.0).mean()

        return loss

