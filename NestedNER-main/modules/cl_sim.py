import torch
import torch.nn as nn
from transformers import BertModel
from models.base_model import Base_Model

class CL_Model(Base_Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.width_embedding = nn.Embedding(cfg.max_width, cfg.width_embedding_dim)

        self.cls_embedding_dim = self.bert_model.config.hidden_size
        self.span_embedding_dim = self.bert_model.config.hidden_size + self.cls_embedding_dim + cfg.width_embedding_dim

        self.linear = nn.Linear(self.span_embedding_dim, 1)
        self.type_loss_weight = nn.Parameter(torch.FloatTensor([1.0]))
        self.boundary_loss_weight = nn.Parameter(torch.FloatTensor([1.0]))

    def _forward_logits(self, sample):
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']

        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state

        span_embeddings = []
        for i in range(len(input_ids)):
            for j in range(i, min(i + 8, len(input_ids))):
                span_token_embeddings = token_embeddings[i:j + 1]
                max_pooled_embeddings = torch.max(span_token_embeddings, dim=0).values
                cls_embedding = outputs.pooler_output[i]
                width_embedding = self.width_embedding(j - i + 1)

                span_embedding = torch.cat([max_pooled_embeddings, cls_embedding, width_embedding], dim=-1)
                span_embeddings.append(span_embedding)

        span_embeddings = torch.stack(span_embeddings)
        logits = self.linear(span_embeddings)

        return logits

    def forward(self, sample):
        logits = self._forward_logits(sample)
        # 在这里添加对比学习的聚类逻辑
        # 计算类型相同的实体的对比学习损失
        type_loss = compute_type_contrastive_loss(logits, sample['labels'], sample['label_ids'])

        # 计算边界相同的跨度的对比学习损失
        boundary_loss = compute_boundary_contrastive_loss(logits, sample['labels'], sample['label_ids'])

        # 计算总的对比学习损失
        total_loss = self.type_loss_weight * type_loss + self.boundary_loss_weight * boundary_loss

        return total_loss

    def evaluate(self, sample, idx2label):
        logits = self._forward_logits(sample)
        # 在这里添加评估逻辑

        return logits

    def aggregate_log_outputs(self, log_outputs):
        # 在这里添加聚合日志输出的逻辑
        pass


def compute_type_contrastive_loss(logits, labels, label_ids):
    # 计算类型相同的实体的对比学习损失
    # 根据label_ids选取对应的正样本和负样本
    positive_mask = label_ids.unsqueeze(1) == label_ids.unsqueeze(2)
    negative_mask = ~positive_mask

    positive_distances = torch.norm(logits.unsqueeze(1) - logits.unsqueeze(2), dim=-1)  # 计算正样本之间的距离
    negative_distances = torch.norm(logits.unsqueeze(1) - logits.unsqueeze(2), dim=-1)  # 计算负样本之间的距离

    type_contrastive_loss = torch.mean(positive_distances * positive_mask) - torch.mean(
        negative_distances * negative_mask)

    return type_contrastive_loss


def compute_boundary_contrastive_loss(logits, labels, label_ids, anchor_boundaries, positive_boundaries):
    # 计算边界相同的跨度的对比学习损失
    # 根据labels选取对应的正样本和负样本
    positive_mask = labels.unsqueeze(1) == labels.unsqueeze(2)
    negative_mask = ~positive_mask

    positive_distances = torch.norm(logits.unsqueeze(1) - logits.unsqueeze(2), dim=-1)  # 计算正样本之间的距离
    negative_distances = torch.norm(logits.unsqueeze(1) - logits.unsqueeze(2), dim=-1)  # 计算负样本之间的距离

    # 根据边界信息计算正样本和锚点之间的IoU得分
    positive_iou_scores = compute_iou_scores(anchor_boundaries.unsqueeze(1), positive_boundaries.unsqueeze(2))

    # 将IoU得分作为正样本的权重
    positive_weights = positive_iou_scores

    boundary_contrastive_loss = torch.mean(positive_distances * positive_mask * positive_weights) - torch.mean(
        negative_distances * negative_mask)

    return boundary_contrastive_loss


def compute_iou_scores(anchor_boundaries, positive_boundaries):
    # 计算IoU得分
    intersection = torch.min(anchor_boundaries[:, :, 1], positive_boundaries[:, :, 1]) - torch.max(
        anchor_boundaries[:, :, 0], positive_boundaries[:, :, 0])
    intersection = torch.clamp(intersection, min=0)

    union = torch.max(anchor_boundaries[:, :, 1], positive_boundaries[:, :, 1]) - torch.min(anchor_boundaries[:, :, 0],
                                                                                            positive_boundaries[:, :,
                                                                                            0])

    iou_scores = intersection / union

    return iou_scores
