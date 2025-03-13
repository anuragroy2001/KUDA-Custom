import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, obs_dim, action_dim, n_history, architecture, block_center=False, agent_center=False):

        super(MLP, self).__init__()

        self.state_dim = obs_dim
        self.action_dim = action_dim
        self.block_center = block_center
        self.agent_center = agent_center
        assert not (self.block_center and self.agent_center), "Cannot have both block_center and agent_center"

        # how many steps in history we are using as input
        self.n_history = n_history

        self.layers = []
        input_dim = (self.state_dim + self.action_dim) * self.n_history
        for i in architecture:
            self.layers.append(nn.Linear(input_dim, i))
            self.layers.append(nn.ReLU())
            input_dim = i
        self.layers.append(nn.Linear(input_dim, self.state_dim))
        self.model = nn.Sequential(*self.layers)

        # N = 3  # corresponds to three probabilities: ReLU, ID or Zero

    def forward(self,
                state,  # [B, n_history, obs_dim]
                action,  # [B, n_history, action_dim]
                ):  # torch.FloatTensor B x state_dim
        """
        input['observation'].shape is [B, n_history, obs_dim]
        input['action'].shape is [B, n_history, action_dim]
        """

        B, n_history, state_dim = state.shape

        if self.block_center:
            assert n_history == 1, "block_center only works for n_history=1"
            tc,tl,tr,bt = state[..., 0:2],state[..., 2:4],state[..., 4:6],state[..., 6:8]
            block_angle = [
                torch.arctan2((tl - tc)[...,1],(tl - tc)[...,0]),
                torch.arctan2((tc - tr)[...,1],(tc - tr)[...,0]),
                torch.arctan2((bt - tc)[...,1],(bt - tc)[...,0])+torch.pi*3/2,
                torch.arctan2((tl - tr)[...,1],(tl - tr)[...,0]),
                ][torch.randint(4,(1,))]

            transition_matrix = torch.zeros(B,n_history, 2, 2).to(state.device)
            transition_matrix[..., 0, 0] =  torch.cos(block_angle)
            transition_matrix[..., 0, 1] = -torch.sin(block_angle)
            transition_matrix[..., 1, 0] =  torch.sin(block_angle)
            transition_matrix[..., 1, 1] =  torch.cos(block_angle)

            local_state = ((state.view(B,n_history,-1,2) - tc[:,:,None,:]) @ transition_matrix).view(B,n_history,-1)
            local_action = ((action - tc)[:,:,None,:] @ transition_matrix).view(B,n_history,-1)

            state = local_state
            action = local_action

        elif self.agent_center:
            raise NotImplementedError("not tested yet")
            agent_pos = state[..., -2:]
            state = state - agent_pos[:,:,None,:]
            action = action - agent_pos[:,:,None,:]


        # flatten the observation and action inputs
        # then concatenate them
        # thus of shape (B, n_history * obs_dim + n_history * action_dim)
        x = torch.cat([state.view(B, -1), action.view(B, -1)], 1).float()

        output = self.model(x)

        # output: B x state_dim
        # always predict the residual, i.e. the change in state
        output = output + state[:, -1]

        if self.block_center:
            output = (output.view(B,n_history,-1,2) @ transition_matrix.inverse() + tc[:,:,None,:]).view(B,-1)
        
        elif self.agent_center:
            raise NotImplementedError("not tested yet")
            output = output + agent_pos[:,:,None,:]
        return output

    def rollout_model(self,
                      state_init,  # [B, n_history, obs_dim]
                      action_seq,  # [B, n_history + n_rollout - 1, action_dim]
                      grad=False,
                      ):
        """
        Rolls out the dynamics model for the given number of steps
        """

        assert len(state_init.shape) == 3
        assert len(action_seq.shape) == 3

        B, n_history, obs_dim = state_init.shape
        _, n_tmp, action_dim = action_seq.shape

        assert n_history == 1, "TODO: check the normalization is reasonable for n_history > 1"

        # if state_init and action_seq have same size in dim=1
        # then we are just doing 1 step prediction
        n_rollout = n_tmp - n_history + 1
        assert n_rollout > 0, "n_rollout = %d must be greater than 0" % (
            n_rollout)

        if grad:
            state_cur = state_init.requires_grad_(True)
        else:
            state_cur = state_init.clone().detach()
        state_pred_list = []

        for i in range(n_rollout):

            # [B, n_history, action_dim]
            actions_cur = action_seq[:, i:i + n_history].clone().detach()
            # state_cur is [B, n_history, obs_dim]
            # # save previous absolute position
            # prev_absolute_position = state_cur[:, 0, :2].clone().detach()

            # [B, obs_dim]
            obs_pred = self.forward(state_cur, actions_cur)


            # [B, n_history-1, action_dim] + [B, 1, action_dim] --> [B, n_history, action_dim]
            state_cur = torch.cat(
                [state_cur[:, 1:].float(), obs_pred.unsqueeze(1)], 1)
            state_pred_list.append(obs_pred)

        # [B, n_rollout, obs_dim]
        state_pred_tensor = torch.stack(state_pred_list, axis=1)

        return state_pred_tensor
