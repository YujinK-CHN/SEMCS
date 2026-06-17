import torch
import torch.nn as nn

"""RNN modules for entity based Transformers."""


class RNNEntityLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNEntityLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x_list, hxs_list, mask_list):
        x_new_list, hxs_new_list = [], []
        for x_i, h_i, mask_i in zip(x_list, hxs_list, mask_list):
            [x_bs, n_entity, _] = x_i.size()
            [hxs_bs, rnn_layers, _] = h_i.size()
            assert mask_i.size(0) == x_bs

            x_i = x_i.reshape(x_bs*n_entity, -1)
            h_i = h_i.unsqueeze(1).repeat(1, n_entity, 1, 1).view(hxs_bs*n_entity, rnn_layers, -1)
            mask_i = mask_i.unsqueeze(1).repeat(1, n_entity, 1).view(x_bs*n_entity, -1)

            [xe_bs, _] = x_i.size()
            [he_bs, rnn_layers, _] = h_i.size()
            
            if x_bs == hxs_bs:
                x_i, h_i = self.rnn(x_i.unsqueeze(0),
                                (h_i * mask_i.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous())
                x_i = x_i.squeeze(0)
                h_i = h_i.transpose(0, 1)
            else:
                # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
                N = he_bs
                T = int(xe_bs / N)

                # unflatten
                x_i = x_i.view(T, N, -1)

                # Same deal with masks
                mask_i = mask_i.view(T, N)

                # Let's figure out which steps in the sequence have a zero for any agent
                # We will always assume t=0 has a zero in it as that makes the logic cleaner
                has_zeros = ((mask_i[1:] == 0.0)
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

                # +1 to correct the masks[1:]
                if has_zeros.dim() == 0:
                    # Deal with scalar
                    has_zeros = [has_zeros.item() + 1]
                else:
                    has_zeros = (has_zeros + 1).numpy().tolist()

                # add t=0 and t=T to the list
                has_zeros = [0] + has_zeros + [T]

                h_i = h_i.transpose(0, 1)

                outputs = []
                for i in range(len(has_zeros) - 1):
                    # We can now process steps that don't have any zeros in masks together!
                    # This is much faster
                    start_idx = has_zeros[i]
                    end_idx = has_zeros[i + 1]
                    temp = (h_i * mask_i[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous()
                    rnn_scores, h_i = self.rnn(x_i[start_idx:end_idx], temp)
                    outputs.append(rnn_scores)

                # assert len(outputs) == T
                # x is a (T, N, -1) tensor
                x_i = torch.cat(outputs, dim=0)

                # flatten
                x_i = x_i.reshape(T * N, -1)
                h_i = h_i.transpose(0, 1)
                                
            x_i = self.norm(x_i)
            
            x_i = x_i.reshape(x_bs, n_entity, -1)
            h_i = h_i[::n_entity, :, :]
            
            x_new_list.append(x_i)
            hxs_new_list.append(h_i)
        
        return x_new_list, hxs_new_list
