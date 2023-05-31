import torch
import torch.nn as nn
import torch.nn.functional as F

from .bert_model import LayerAttention, BertPredictionHeadTransform
from . import objectives

class LinkTower(nn.Module):
    def __init__(self, config):
        super(LinkTower, self).__init__()
        self.LayerNorm = nn.LayerNorm(config['hidden_size'])

    def forward(self, hidden_states, cross_modal_hidden_states):
        return self.LayerNorm(hidden_states + cross_modal_hidden_states)


class Manager(nn.Module):
    def __init__(self, config, routed_layers=0, layer_index=0):
        super().__init__()
        self.config = config
        if self.config["manager_type"] == 'SAUE':
            self.manager_layer = SAUE(config, routed_layers, layer_index)
        elif self.config["manager_type"] == 'AAUE':
            if layer_index == 0:
                self.manager_layer = SAUE(config, routed_layers, layer_index)
            else:
                self.manager_layer = AAUE(config, routed_layers, layer_index)
        else:
            raise NotImplementedError(f"manager_type {self.config['manager_type']} is not implemented")

    def forward(self, hidden_states, cross_modal_hidden_states, masks, extra_query=None, is_training=False):
        return self.manager_layer(hidden_states, cross_modal_hidden_states, masks, extra_query=extra_query, is_training=is_training)

# Static Aggregation of Uni-Modal Experts
class SAUE(nn.Module):
    def __init__(self, config, routed_layers, layer_index=0):
        '''
        :param config:
        :param routed_layers: the number of routed layers (text/image)
        :param layer_index: the index of the current layer
        '''
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        # num_previous: the number of previous calculated cross-modal layers
        num_previous = 1
        if layer_index == 0:
            num_previous = 0
        self.num_previous = num_previous
        self.routed_layers = routed_layers
        
        # weight init
        self.layer_scores = torch.ones(routed_layers + num_previous)
        self.layer_scores.data.fill_(1 / routed_layers)

        if layer_index != 0:
            # avg for previous layer outputs
            self.layer_scores.data[routed_layers:].fill_(1 / num_previous)

        # weight: scalar or vector
        self.layer_scores = self.layer_scores.unsqueeze(0).unsqueeze(2).unsqueeze(2) # B x N x L x (1 / D)
        if self.config['manager_weight_type'] == 'scalar':
            self.layer_scores = nn.Parameter(self.layer_scores, requires_grad=self.config['manager_learnable'])
        elif self.config['manager_weight_type'] == 'vector':
            self.layer_scores = nn.Parameter(self.layer_scores.repeat(1, 1, 1, self.config['hidden_size']), requires_grad=self.config['manager_learnable'])
        
        # normalization for reps
        self.LayerNorm = nn.ModuleList([nn.LayerNorm(self.config['hidden_size']) for _ in range(routed_layers + num_previous)])
        
        # learnable temperature
        self.softmax_temperature = nn.Parameter(torch.ones(1) * self.config['manager_softmax_temperature'], requires_grad=self.config['manager_softmax_temperature_learnable'])
        self.softmax_temperature_cross = nn.Parameter(torch.ones(1) * self.config['manager_softmax_temperature'], requires_grad=self.config['manager_softmax_temperature_learnable'])
        
    def aggregate_reps(self, layer_scores, reps, masks, is_training=False):
        # layer_scores: 1 x N x 1 x (1 / D)
        # reps: B x N x L x D
        if is_training:
            layer_scores = layer_scores.expand_as(reps) + torch.cat((torch.normal(mean=0, std=1/(self.routed_layers), size=reps[:, :self.routed_layers].shape, device=layer_scores.device), torch.zeros_like(reps[:, self.routed_layers:])), dim=1)
        
        layer_scores = torch.cat((
            torch.softmax(layer_scores[:, :self.routed_layers] / self.softmax_temperature, dim=1), 
            layer_scores[:, self.routed_layers:]
        ), dim=1)
        
        return torch.sum(layer_scores * reps, dim=1)


    def forward(self, hidden_states, cross_modal_hidden_states, masks, extra_query=None, is_training=False):
        hidden_states = torch.stack([self.LayerNorm[i](hidden_states[:, i]) for i in range(hidden_states.shape[1])], dim=1)
        
        layer_scores_ = self.layer_scores
        if self.layer_index == 0:
            hidden_states = self.aggregate_reps(layer_scores_, hidden_states, masks, is_training=is_training)
        else:
            cross_modal_hidden_states = cross_modal_hidden_states.unsqueeze(1)
            cross_modal_hidden_states = torch.stack([self.LayerNorm[i + hidden_states.shape[1]](cross_modal_hidden_states[:, i]) for i in range(cross_modal_hidden_states.shape[1])], dim=1)
            hidden_states = self.aggregate_reps(layer_scores_, torch.cat((hidden_states, cross_modal_hidden_states), dim=1), masks, is_training=is_training)
        
        return hidden_states


# Adaptive Aggregation of Uni-Modal Experts
class AAUE(nn.Module):
    def __init__(self, config, routed_layers, layer_index=0):
        '''
        :param config:
        :param routed_layers: the number of routed layers (text/image)
        :param layer_index: the index of the current layer
        '''
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        # num_previous: the number of previous calculated cross-modal layers
        num_previous = 1
        if layer_index == 0:
            num_previous = 0
        self.num_previous = num_previous
        self.routed_layers = routed_layers

        self.layer_scores_cross = torch.ones(num_previous)
        self.layer_scores_cross.data.fill_(1 / num_previous)
        self.layer_scores_cross = self.layer_scores_cross.unsqueeze(0).unsqueeze(2).unsqueeze(2) # B x N x L x (1 / D)
        if self.config['manager_weight_type'] == 'scalar':
            self.layer_scores_cross = nn.Parameter(self.layer_scores_cross, requires_grad=self.config['manager_learnable'])
        elif self.config['manager_weight_type'] == 'vector':
            self.layer_scores_cross = nn.Parameter(self.layer_scores_cross.repeat(1, 1, 1, self.config['hidden_size']), requires_grad=self.config['manager_learnable'])

        # B x (2 x D) -> B x N
        self.linear_controller = nn.Linear(self.config['hidden_size'] * 2, routed_layers)
        self.fusion_attention = LayerAttention(self.config['hidden_size'], self.config['hidden_size'])

        # normalization for reps
        self.LayerNorm = nn.ModuleList([nn.LayerNorm(self.config['hidden_size']) for _ in range(routed_layers + num_previous)])
        self.extra_query_LayerNorm = nn.LayerNorm(self.config['hidden_size'])

        # learnable temperature
        self.softmax_temperature = nn.Parameter(torch.ones(1) * self.config['manager_softmax_temperature'], requires_grad=self.config['manager_softmax_temperature_learnable'])
        self.softmax_temperature_cross = nn.Parameter(torch.ones(1) * self.config['manager_softmax_temperature'], requires_grad=self.config['manager_softmax_temperature_learnable'])

    def aggregate_reps(self, layer_scores, reps, masks, is_training=False):
        # layer_scores: B x N x L x (1 / D) ; B x N x 1 x (1 / D)
        # reps: B x N x L x D
        if is_training:
            layer_scores = layer_scores.expand_as(reps) + torch.cat((torch.normal(mean=0, std=1/(self.routed_layers), size=reps[:, :self.routed_layers].shape, device=layer_scores.device), torch.zeros_like(reps[:, self.routed_layers:])), dim=1)
        
        layer_scores = torch.cat((
            torch.softmax(layer_scores[:, :self.routed_layers] / self.softmax_temperature, dim=1), 
            layer_scores[:, self.routed_layers:]
        ), dim=1)

        return torch.sum(layer_scores * reps, dim=1)

    def forward(self, hidden_states, cross_modal_hidden_states, masks, extra_query=None, is_training=False):
        # hidden_states: B x N x L x D
        # cross_modal_hidden_states: B x L x D ; B x N x L x D
        # extra_query: B x D
        if self.layer_index == 0:
            raise NotImplementedError('Not implemented yet for the first layer')
            # we don't have cross_modal_hidden_states as our query in the first layer
        
        hidden_states = torch.stack([self.LayerNorm[i](hidden_states[:, i]) for i in range(hidden_states.shape[1])], dim=1)
        extra_query = self.extra_query_LayerNorm(extra_query) # B x L x D ; B x D 
        cross_modal_hidden_states = cross_modal_hidden_states.unsqueeze(1)
        cross_modal_hidden_states = torch.stack([self.LayerNorm[i + hidden_states.shape[1]](cross_modal_hidden_states[:, i]) for i in range(cross_modal_hidden_states.shape[1])], dim=1)
        # B x L x 2D -> B x L x N => B x N x L x 1
        fused_query = torch.cat((torch.softmax(self.fusion_attention(cross_modal_hidden_states[:, -1], extra_query), dim=-1) @ extra_query , cross_modal_hidden_states[:, -1]), dim=-1) # B x L x 2D

        
        layer_scores_generate = self.linear_controller(fused_query).transpose(1, 2).unsqueeze(-1)
        
        if self.config['manager_weight_type'] == 'vector':
            layer_scores_generate = layer_scores_generate.expand(-1, -1, -1, self.config['hidden_size'])
        layer_scores_generate = torch.cat((layer_scores_generate, self.layer_scores_cross.expand(layer_scores_generate.shape[0], -1, layer_scores_generate.shape[2], -1)), dim=1)
        
        hidden_states = torch.cat((hidden_states, cross_modal_hidden_states), dim=1)
        hidden_states = self.aggregate_reps(layer_scores_generate, hidden_states, masks, is_training=is_training)
        
        return hidden_states


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITCHead(nn.Module):
    def __init__(self, hidden_size, embed_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        return self.fc(x)


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        return self.fc(x)


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x
