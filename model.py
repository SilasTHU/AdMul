# Author: Silas Cheung (Shenglong Zhang)
import torch
import torch.nn as nn
import argparse
from transformers import AutoModel, AutoConfig


# Gradient Reversal Layer
class GRLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lmbd=0.01):
        ctx.lmbd = torch.tensor(lmbd)
        return x.reshape_as(x)

    @staticmethod
    # 输入为forward输出的梯度
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.lmbd * grad_input.neg(), None


# Copied from transformers.models.bert.modeling_bert.BertPooler
# DeBERTa does not have poolerout, so we implement manually
class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self._init_weights(self.dense)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Base_model(nn.Module):
    def __init__(self, args):
        super(Base_model, self).__init__()
        self.plm = AutoModel.from_pretrained(args.DATA.plm)
        for param in self.plm.parameters():
            param.requires_grad = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.plm.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BERT_base(Base_model):
    def __init__(self, args):
        super(BERT_base, self).__init__(args)
        self.config = AutoConfig.from_pretrained(args.DATA.plm)
        self.dropout = nn.Dropout(args.MODEL.dropout)
        self.wsd_fc = nn.Linear(args.MODEL.embed_dim, args.MODEL.num_classes)
        self.md_fc = nn.Linear(args.MODEL.embed_dim, args.MODEL.num_classes)
        self._init_weights(self.wsd_fc)
        self._init_weights(self.md_fc)

    def forward(self, input_ids, type_ids, att_mask, domains):
        embeddings = self.plm(input_ids, token_type_ids=type_ids, attention_mask=att_mask)
        embeddings = embeddings.pooler_output
        embeddings = self.dropout(embeddings)

        wsd_indices = (domains == 0).nonzero(as_tuple=True)[0]
        md_indices = (domains == 1).nonzero(as_tuple=True)[0]
        wsd_embeddings = torch.index_select(embeddings, 0, wsd_indices)
        md_embeddings = torch.index_select(embeddings, 0, md_indices)

        wsd_out = self.wsd_fc(wsd_embeddings)
        md_out = self.md_fc(md_embeddings)

        return wsd_out, md_out


class DeBERTa_base(Base_model):
    def __init__(self, args):
        super(DeBERTa_base, self).__init__(args)
        self.dropout = nn.Dropout(args.MODEL.dropout)
        self.config = AutoConfig.from_pretrained(args.DATA.plm)
        self.pooler = Pooler(config=self.config)
        self.wsd_fc = nn.Linear(args.MODEL.embed_dim, args.MODEL.num_classes)
        self.md_fc = nn.Linear(args.MODEL.embed_dim, args.MODEL.num_classes)
        self.task_fc = nn.Linear(args.MODEL.embed_dim, args.MODEL.num_tasks)
        self.label_fc0 = nn.Linear(args.MODEL.embed_dim, args.MODEL.num_classes)
        self.label_fc1 = nn.Linear(args.MODEL.embed_dim, args.MODEL.num_classes)
        self._init_weights(self.wsd_fc)
        self._init_weights(self.md_fc)
        self._init_weights(self.task_fc)
        self._init_weights(self.label_fc0)
        self._init_weights(self.label_fc1)

    def forward(self, input_ids, type_ids, att_mask, adv_lmbd=0.01, domain_idx=1):
        if self.training:
            embeddings = self.plm(input_ids, token_type_ids=type_ids, attention_mask=att_mask)
            embeddings = embeddings.last_hidden_state
            pooler_embeddings = self.pooler(embeddings)
            mean_embeddings = torch.mean(embeddings, dim=1)

            half_batch_size = int(0.5 * pooler_embeddings.shape[0])
            wsd_embeddings = pooler_embeddings[:half_batch_size]
            md_embeddings = pooler_embeddings[half_batch_size:]

            wsd_out = self.wsd_fc(wsd_embeddings)
            md_out = self.md_fc(md_embeddings)

            # marginal distribution: global out
            mean_reverse = GRLayer.apply(mean_embeddings, adv_lmbd)
            global_out = self.task_fc(mean_reverse)

            # conditional distribution: local out
            wsd_reverse = mean_reverse[:half_batch_size]
            md_reverse = mean_reverse[half_batch_size:]
            wsd_prob = torch.softmax(wsd_out, dim=1)
            md_prob = torch.softmax(md_out, dim=1)

            # align class 0
            wsd_p0 = wsd_prob[:, 0].reshape(wsd_reverse.shape[0], 1)
            wsd_feature0 = wsd_p0 * wsd_reverse
            md_p0 = md_prob[:, 0].reshape(md_reverse.shape[0], 1)
            md_feature0 = md_p0 * md_reverse
            wsd_local_out0 = self.label_fc0(wsd_feature0)
            md_local_out0 = self.label_fc0(md_feature0)

            # align class 1
            wsd_p1 = wsd_prob[:, 1].reshape(wsd_reverse.shape[0], 1)
            wsd_feature1 = wsd_p1 * wsd_reverse
            md_p1 = md_prob[:, 1].reshape(md_reverse.shape[0], 1)
            md_feature1 = md_p1 * md_reverse
            wsd_local_out1 = self.label_fc1(wsd_feature1)
            md_local_out1 = self.label_fc1(md_feature1)

            wsd_local_out = [wsd_local_out0, wsd_local_out1]
            md_local_out = [md_local_out0, md_local_out1]

            return wsd_out, md_out, global_out, wsd_local_out, md_local_out

        else:
            embeddings = self.plm(input_ids, token_type_ids=type_ids, attention_mask=att_mask)
            embeddings = embeddings.last_hidden_state
            pooler_embeddings = self.pooler(embeddings)

            if domain_idx == 0:
                out = self.wsd_fc(pooler_embeddings)
            else:
                out = self.md_fc(pooler_embeddings)

            return out


if __name__ == "__main__":
    pass