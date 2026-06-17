import torch
import torch.nn as nn
from base_policy.utils.util import init
from base_policy.components.rnn import RNNLayer
from base_policy.components.popart import PopArt
from base_policy.components.distributions import FixedCategorical
import torch.nn.functional as F


class EntityCategorical(nn.Module):
    def __init__(self, args, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(EntityCategorical, self).__init__()
        self.args = args
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward_i(self, x, available_actions=None, n_enemies=None):
        # the first entity is always its own features
        x = self.linear(x[:, :1]).squeeze()

        if available_actions is not None:
            value = -1e10
            if x.dtype == torch.float16:
                value = max(min(value, 65504), -65504)  # clamp within float16 range 
            x[available_actions == 0] = value
        return x

    def forward(self, x, available_actions=None, n_enemies=None):
        x = [self.forward_i(x_i, available_actions_i, n_enemy) for x_i, available_actions_i, n_enemy in zip(x, available_actions, n_enemies)]
        return [FixedCategorical(logits=x_i) for x_i in x]


class MultiACTLayer(nn.Module):
    def __init__(self, args, action_space, inputs_dim, use_orthogonal, gain):
        super(MultiACTLayer, self).__init__()
        self.args = args
        action_dim = action_space.n
        self.action_out = EntityCategorical(args, inputs_dim, action_dim, use_orthogonal, gain)

    def mode(self, action_logits):
        '''
        action_logits is a list of class: FixedCategorical
        '''
        return torch.cat([action_logit.mode() for action_logit in action_logits])

    def sample(self, action_logits):
        '''
        action_logits is a list of class: FixedCategorical
        '''
        return torch.cat([action_logit.sample() for action_logit in action_logits])
    
    def log_probs(self, action_logits, actions, bs_na_list=None):
        '''
        action_logits is a list of class: FixedCategorical
        actions is a list of tensor(action)
        '''
        action_list = torch.split(actions, bs_na_list, dim=0) if bs_na_list is not None else actions
        return torch.cat([action_logit.log_probs(action) for action_logit, action in zip(action_logits, action_list)])

    def dist_entropy_(self, action_logits, active_masks):
        '''
        action_logits is a list of class: FixedCategorical
        active_masks is a list of tensor(action_masks)
        '''
        dist_entropy = []
        if active_masks is not None:
            for action_logit, active_mask in zip(action_logits, active_masks):
                dist_entropy.append(
                    (action_logit.entropy()*active_mask.squeeze(-1)).sum() / active_mask.sum()
                )
            # dist_entropy = \
            #     torch.cat([action_logit.entropy()*active_mask.squeeze(-1) for action_logit, active_mask in zip(action_logits, active_masks)]).sum() /\
            #         torch.cat(active_masks).sum()
        else:
            dist_entropy.extend(
                [action_logit.entropy().mean() for action_logit in action_logits]
            )
            # dist_entropy = torch.cat([action_logit.entropy() for action_logit in action_logits]).mean()
        return dist_entropy

    def forward(self, x, available_actions=None, deterministic=False, n_enemies=None):
        bs_na_list = [x_i.size(0) for x_i in x]
        action_logits = self.action_out(x, available_actions, n_enemies)
        actions = self.mode(action_logits) if deterministic else self.sample(action_logits) 
        action_log_probs = self.log_probs(action_logits, actions, bs_na_list)
        
        return actions, action_log_probs
    
    def forward4skills(self, x, available_actions=None, deterministic=False, n_enemies=None):
        bs_na_list = [x_i.size(0) for x_i in x]
        action_logits = self.action_out(x, available_actions, n_enemies)
        actions = self.mode(action_logits) if deterministic else self.sample(action_logits) 
        action_log_probs = self.log_probs(action_logits, actions, bs_na_list)
        
        # return probs rather than logits
        action_probs_list = [act_logits.probs for act_logits in action_logits]
        return actions, action_log_probs, action_probs_list

    def forward4subtask(self, x, available_actions=None, deterministic=False, n_enemies=None):
        bs_na_list = [x_i.size(0) for x_i in x]
        action_logits = self.action_out(x, available_actions, n_enemies)
        actions = self.mode(action_logits) if deterministic else self.sample(action_logits) 
        action_log_probs = self.log_probs(action_logits, actions, bs_na_list)
        
        # return probs rather than logits
        pi_list = [act_logits.probs for act_logits in action_logits]

        # clip the values
        return actions, action_log_probs, pi_list

    def get_probs(self, x, available_actions=None, n_enemies=None):
        raise NotImplementedError
        action_logits = self.action_out(x, available_actions, n_enemies)
        action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None, n_enemies=None):
        action_logits = self.action_out(x, available_actions, n_enemies)
        action_log_probs = self.log_probs(action_logits, action)
        dist_entropy = self.dist_entropy_(action_logits, active_masks)
        
        return action_log_probs, dist_entropy

    def evaluate_preds(self, x, available_actions=None, deterministic=False, n_enemies=None):
        bs_na_list = [x_i.size(0) for x_i in x]
        action_logits = self.action_out(x, available_actions, n_enemies)
        # If each `logit` is a Categorical, get its full log prob vector
        action_log_probs = [logit.logits.log_softmax(dim=-1) for logit in action_logits]  # shape: [B, A]
        # Step 3: Concatenate into one tensor
        log_probs_cat = torch.cat(action_log_probs, dim=0)  # shape: [sum(B_i), A]
        # Step 4: Split back to match original batch structure
        action_log_probs_split = list(torch.split(log_probs_cat, bs_na_list, dim=0))  # list of [B_i, A]
        return action_log_probs_split