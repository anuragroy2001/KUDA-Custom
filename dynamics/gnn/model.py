import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
        self.output_size = output_size

    def forward(self, x):
        s = x.size()
        x = self.model(x.view(-1, s[-1]))
        return x.view(list(s[:-1]) + [self.output_size])


class Propagator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Propagator, self).__init__()

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.output_size = output_size

    def forward(self, x, res=None):
        s_x = x.size()

        x = self.linear(x.view(-1, s_x[-1]))

        if res is not None:
            s_res = res.size()
            x += res.view(-1, s_res[-1])

        x = self.relu(x).view(list(s_x[:-1]) + [self.output_size])
        return x


class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.output_size = output_size

    def forward(self, x):
        s_x = x.size()

        x = x.view(-1, s_x[-1])
        x = self.relu(self.linear_0(x))
        x = self.relu(self.linear_1(x))

        return self.linear_2(x).view(list(s_x[:-1]) + [self.output_size])


class DynamicsPredictor(nn.Module):
    def __init__(self, model_config, device):

        super(DynamicsPredictor, self).__init__()

        self.model_config = model_config
        self.device = device

        self.nf_particle = model_config['nf_particle']
        self.nf_relation = model_config['nf_relation']
        self.nf_effect = model_config['nf_effect']
        self.out_dim = model_config['out_dim']
        self.state_normalize = False

        self.quat_offset = torch.FloatTensor([1., 0., 0., 0.]).to(device)

        # self.dt = torch.FloatTensor([args.dt]).to(args.device)
        # self.mean_p = torch.FloatTensor(args.mean_p).to(args.device)
        # self.std_p = torch.FloatTensor(args.std_p).to(args.device)
        # self.mean_d = torch.FloatTensor(args.mean_d).to(args.device)
        # self.std_d = torch.FloatTensor(args.std_d).to(args.device)
        self.eps = 1e-6
        self.motion_clamp = 100  # TODO hyperparameter

        # ParticleEncoder
        # args.mem_dim = args.nf_effect * args.mem_nlayer
        # input_dim = args.attr_dim + args.density_dim + args.n_his * args.state_dim \
        #         + args.n_his * args.offset_dim + args.action_dim + phys_dim  # + args.mem_dim

        input_dim = model_config['n_his'] * model_config['state_dim'] + \
                    model_config['n_his'] * model_config['offset_dim'] + \
                    model_config['attr_dim'] + \
                    model_config['action_dim'] + \
                    model_config['density_dim']

        self.particle_encoder = Encoder(input_dim, self.nf_particle, self.nf_effect)

        # RelationEncoder
        if model_config['rel_particle_dim'] == -1:
            model_config['rel_particle_dim'] = input_dim

        # rel_input_dim = args.rel_particle_dim * 2 + args.rel_attr_dim * 2 + args.rel_can_attr_dim \
        #         + args.rel_group_dim + args.n_his * args.rel_distance_dim + args.rel_density_dim \
        #         + args.rel_canonical_distance_dim + args.rel_canonical_attr_dim

        rel_input_dim = model_config['rel_particle_dim'] * 2 + \
                        model_config['rel_attr_dim'] * 2 + \
                        model_config['rel_group_dim'] + \
                        model_config['rel_distance_dim'] * model_config['n_his'] + \
                        model_config['rel_density_dim'] # + \
                        # model_config['rel_can_attr_dim'] + \
                        # model_config['rel_canonical_distance_dim'] + \
                        # model_config['rel_canonical_attr_dim']
        self.relation_encoder = Encoder(rel_input_dim, self.nf_relation, self.nf_effect)

        # ParticlePropagator
        self.particle_propagator = Propagator(self.nf_effect * 2, self.nf_effect)

        # RelationPropagator
        self.relation_propagator = Propagator(self.nf_effect * 3, self.nf_effect)
        # NOTE: dyn-res-manip adds a particle density to the relation propagator, 
        # which I think is unnecessary since the density is already included in the particle encoder.

        # ParticlePredictor
        # self.non_rigid_out_dim = kwargs["non_rigid_out_dim"]
        # self.rigid_out_dim = kwargs["rigid_out_dim"]
        # if kwargs["predict_non_rigid"]:
        #     self.non_rigid_predictor = ParticlePredictor(self.nf_effect, self.nf_effect, kwargs["non_rigid_out_dim"])
        # else:
        #     self.non_rigid_predictor = None
        # if kwargs["predict_rigid"]:
        #     self.rigid_predictor = ParticlePredictor(self.nf_effect, self.nf_effect, kwargs["rigid_out_dim"])
        # else:
        #     self.rigid_predictor = None

        self.non_rigid_predictor = ParticlePredictor(self.nf_effect, self.nf_effect, self.out_dim)
        # self.rigid_predictor = None
        
        if model_config['verbose']:
            print("DynamicsPredictor initialized")
            print("particle input dim: {}, relation input dim: {}".format(input_dim, rel_input_dim))
            # print("non-rigid output dim: {}, rigid output dim: {}".format(kwargs["non_rigid_out_dim"], kwargs["rigid_out_dim"]))

    # @profile
    def forward(self, state, attrs, Rr, Rs, p_instance, 
            action=None, particle_den=None, obj_mask=None, **kwargs):  
            # physics_param=None, canonical_state=None, rel_can_attrs=None, 
            # verbose=False, phys_encoded_noise=0.1, memory=None, offset=None

        n_his = self.model_config['n_his']

        B, N = attrs.size(0), attrs.size(1)  # batch size, total particle num
        n_instance = p_instance.size(2)  # number of instances
        n_p = p_instance.size(1)  # number of object particles (that need prediction)
        n_s = attrs.size(1) - n_p  # number of shape particles that do not need prediction
        n_rel = Rr.size(1)  # number of relations
        state_dim = state.size(3)  # state dimension

        # attrs: B x N x attr_dim
        # state: B x n_his x N x state_dim
        # Rr, Rs: B x n_rel x N
        # memory: B x mem_nlayer x N x nf_memory
        # p_rigid: B x n_instance (deprecated)
        # p_instance: B x n_particle x n_instance
        # obj_mask: B x n_particle

        # Rr_t, Rs_t: B x N x n_rel
        Rr_t = Rr.transpose(1, 2).contiguous()
        Rs_t = Rs.transpose(1, 2).contiguous()

        # particle belongings and rigidness
        # p_rigid_per_particle: B x n_p x 1
        # p_rigid_per_particle = torch.sum(p_instance * p_rigid[:, None, :], 2, keepdim=True)

        # state_res: B x (n_his - 1) x N x state_dim, state_cur: B x 1 x N x state_dim     
        state_res = state[:, 1:] - state[:, :-1]
        state_cur = state[:, -1:]

        if self.state_normalize:
            # *_p: absolute scale to model scale, *_d: residual scale to model scale
            mean_p, std_p, mean_d, std_d = self.mean_p, self.std_p, self.mean_d, self.std_d
            state_res_norm = (state_res - mean_d) / std_d
            state_res_norm[:, :, :n_p, :] = 0  # n_p = 301?
            state_cur_norm = (state_cur - mean_p) / std_p
        else:
            state_res_norm = state_res
            state_cur_norm = state_cur

        # state_norm: B x n_his x N x state_dim
        # [0, n_his - 1): state_residual
        # [n_his - 1, n_his): the current position
        state_norm = torch.cat([state_res_norm, state_cur_norm], 1)
        state_norm_t = state_norm.transpose(1, 2).contiguous().view(B, N, n_his * state_dim)

        # p_inputs: B x N x attr_dim
        p_inputs = attrs

        if self.model_config['state_dim'] > 0:
            # add state to attr
            # p_inputs: B x N x (attr_dim + n_his * state_dim)
            p_inputs = torch.cat([attrs, state_norm_t], 2)

        # instance_center: B x n_instance x (n_his * state_dim)
        instance_center = p_instance.transpose(1, 2).bmm(state_norm_t[:, :n_p])
        instance_center /= torch.sum(p_instance, 1).unsqueeze(-1) + self.eps

        # other inputs
        if self.model_config['offset_dim'] > 0:
            raise NotImplementedError
            # add offset to center-of-mass for rigids to attr
            # offset: B x N x (n_his * state_dim)
            # offset = torch.zeros(B, N, n_his * state_dim).to(self.device)

            # # c_per_particle: B x n_p x (n_his * state_dim)
            # # particle offset: B x n_p x (n_his * state_dim)
            # c_per_particle = p_instance.bmm(instance_center)
            # c = (1 - p_rigid_per_particle) * state_norm_t[:, :n_p] + p_rigid_per_particle * c_per_particle
            # offset[:, :n_p] = state_norm_t[:, :n_p] - c

            # # p_inputs: B x N x (attr_dim + 2 * n_his * state_dim)
            # p_inputs = torch.cat([p_inputs, offset], 2)

        # if memory is not None:
        #     # memory_t: B x N x (mem_nlayer * nf_memory)
        #     memory_t = memory.transpose(1, 2).contiguous().view(B, N, -1)
        #     # p_inputs: B x N x (... + mem_nlayer * nf_memory)
        #     p_inputs = torch.cat([p_inputs, memory_t], 2)

        # TODO what if n_instance is not fixed?
        # group info
        # if p_instance is not None:
        #     # g: B x N x n_instance
        #     # for particle i in group j, g[:, i, j] = 1, otherwise 0
        #     g = p_instance
        #     g_s = torch.zeros(B, n_s, n_instance).to(args.device)
        #     g = torch.cat([g, g_s], 1)
        #     # p_inputs: B x N x (... + n_instance)
        #     p_inputs = torch.cat([p_inputs, g], 2)

        # action
        if self.model_config['action_dim'] > 0:
            assert action is not None
            # action: B x N x action_dim
            # action_on_particles = False
            # if action_on_particles:
            #     action_s = torch.zeros(B, n_s, args.action_dim).to(args.device)
            #     action = torch.cat([action, action_s], 1)
            # else:  # action on shape particles
            #     action_p = torch.zeros(B, n_p, args.action_dim).to(args.device)
            #     action = torch.cat([action_p, action], 1)    

            # p_inputs: B x N x (... + action_dim)
            p_inputs = torch.cat([p_inputs, action], 2)

        if self.model_config['density_dim'] > 0:
            assert particle_den is not None
            # particle_den = particle_den / 5000.

            # particle_den: B x N x 1
            particle_den = particle_den[:, None, None].repeat(1, n_p, 1)
            particle_den_s = torch.zeros(B, n_s, 1).to(self.device)
            particle_den = torch.cat([particle_den, particle_den_s], 1)

            # p_inputs: B x N x (... + density_dim)
            p_inputs = torch.cat([p_inputs, particle_den], 2)
        # Finished preparing p_inputs

        # Preparing rel_inputs
        rel_inputs = torch.empty((B, n_rel, 0), dtype=torch.float32).to(self.device)
        if self.model_config['rel_particle_dim'] > 0:
            assert self.model_config['rel_particle_dim'] == p_inputs.size(2)
            # p_inputs_r: B x n_rel x -1
            # p_inputs_s: B x n_rel x -1
            p_inputs_r = Rr.bmm(p_inputs)
            p_inputs_s = Rs.bmm(p_inputs)

            # rel_inputs: B x n_rel x (2 x rel_particle_dim)
            rel_inputs = torch.cat([rel_inputs, p_inputs_r, p_inputs_s], 2)

        if self.model_config['rel_attr_dim'] > 0:
            assert self.model_config['rel_attr_dim'] == attrs.size(2)
            # attr_r: B x n_rel x attr_dim
            # attr_s: B x n_rel x attr_dim
            attrs_r = Rr.bmm(attrs)
            attrs_s = Rs.bmm(attrs)

            # rel_inputs: B x n_rel x (... + 2 x rel_attr_dim)
            rel_inputs = torch.cat([rel_inputs, attrs_r, attrs_s], 2)
        
        # if self.model_config['rel_can_attr_dim'] > 0:
        #     assert rel_can_attrs is not None
        #     # rel_can_attrs: B x n_rel x rel_can_attr_dim
            
        #     # rel_inputs: B x n_rel x (... + rel_can_attr_dim)
        #     rel_inputs = torch.cat([rel_inputs, rel_can_attrs], 2)

        if self.model_config['rel_group_dim'] > 0:
            assert self.model_config['rel_group_dim'] == 1
            # receiver_group, sender_group
            # group_r: B x n_rel x -1
            # group_s: B x n_rel x -1
            g = torch.cat([p_instance, torch.zeros(B, n_s, n_instance).to(self.device)], 1)
            group_r = Rr.bmm(g)
            group_s = Rs.bmm(g)
            group_diff = torch.sum(torch.abs(group_r - group_s), 2, keepdim=True)

            # rel_inputs: B x n_rel x (... + 1)
            rel_inputs = torch.cat([rel_inputs, group_diff], 2)
        
        if self.model_config['rel_distance_dim'] > 0:
            # assert self.model_config['rel_distance_dim'] == 3
            # receiver_pos, sender_pos
            # pos_r: B x n_rel x -1
            # pos_s: B x n_rel x -1
            pos_r = Rr.bmm(state_norm_t)
            pos_s = Rs.bmm(state_norm_t)
            pos_diff = pos_r - pos_s

            # rel_inputs: B x n_rel x (... + 3)
            rel_inputs = torch.cat([rel_inputs, pos_diff], 2)
        
        # if self.model_config['rel_canonical_distance_dim'] > 0 or self.model_config['rel_canonical_attr_dim > 0']:
        #     assert canonical_state is not None
        #     can_dim = canonical_state.shape[2]
        #     canonical_state_s = torch.zeros(B, n_s, can_dim).to(self.device)
        #     canonical_state = torch.cat([canonical_state, canonical_state_s], 1)
            
        #     # canonical_state: B x N x 1
        #     # can_pos_r: B x n_rel x 1
        #     # can_pos_s: B x n_rel x 1
        #     can_pos_r = Rr.bmm(canonical_state)
        #     can_pos_s = Rs.bmm(canonical_state)
        #     can_pos_diff = can_pos_r - can_pos_s

        # if args.rel_canonical_distance_dim > 0:
        #     # rel_inputs: B x n_rel x (... + 1)
        #     rel_inputs = torch.cat([rel_inputs, can_pos_diff], 2)

        # if args.rel_canonical_attr_dim > 0:
        #     assert args.rel_attr_dim is not None
        #     # attrs_mask: B x n_rel x 1
        #     attrs_mask = (attrs_r - attrs_s).abs().sum(2, keepdim=True) == 0
        #     can_pos_diff = can_pos_diff * attrs_mask.float()
        #     can_pos_binary = can_pos_diff.abs() > args.rel_canonical_thresh

        #     # rel_inputs: B x n_rel x (... + 1)
        #     rel_inputs = torch.cat([rel_inputs, can_pos_binary.float()], 2)

        if self.model_config['rel_density_dim'] > 0:
            assert self.model_config['rel_density_dim'] == 1
            # receiver_density, sender_density
            # dens_r: B x n_rel x -1
            # dens_s: B x n_rel x -1
            dens_r = Rr.bmm(particle_den)
            dens_s = Rs.bmm(particle_den)
            dens_diff = dens_r - dens_s

            # rel_inputs: B x n_rel x (... + 1)
            rel_inputs = torch.cat([rel_inputs, dens_diff], 2)

        # particle encode
        particle_encode = self.particle_encoder(p_inputs)
        particle_effect = particle_encode
        if self.model_config['verbose']:
            print("particle encode:", particle_encode.size())

        # calculate relation encoding
        relation_encode = self.relation_encoder(rel_inputs)
        if self.model_config['verbose']:
            print("relation encode:", relation_encode.size())

        for i in range(self.model_config['pstep']):
            if self.model_config['verbose']:
                print("pstep", i)

            # effect_r, effect_s: B x n_rel x nf
            effect_r = Rr.bmm(particle_effect)
            effect_s = Rs.bmm(particle_effect)

            # calculate relation effect
            # effect_rel: B x n_rel x nf
            effect_rel = self.relation_propagator(
                torch.cat([relation_encode, effect_r, effect_s], 2))
            if self.model_config['verbose']:
                print("relation effect:", effect_rel.size())

            # calculate particle effect by aggregating relation effect
            # effect_rel_agg: B x N x nf
            effect_rel_agg = Rr_t.bmm(effect_rel)

            # calculate particle effect
            # particle_effect: B x N x nf
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_rel_agg], 2),
                res=particle_effect)
            if self.model_config['verbose']:
                 print("particle effect:", particle_effect.size())

        # non-rigid motion
        # if self.non_rigid_predictor is not None:
        # non_rigid_motion: B x n_p x state_dim
        non_rigid_motion = self.non_rigid_predictor(particle_effect[:, :n_p].contiguous())

        # rigid motion
        # if self.rigid_predictor is not None and self.rigid_out_dim == 7:
        #     # assert args.state_dim != 7
        #     # aggregate effects of particles in the same instance
        #     # instance effect: B x n_instance x nf_effect
        #     instance_effect = p_instance.transpose(1, 2).bmm(particle_effect[:, :n_p])

        #     # instance_rigid_params: (B * n_instance) x 7
        #     instance_rigid_params = self.rigid_predictor(instance_effect).view(B * n_instance, 7)

        #     # decode rotation
        #     # R: (B * n_instance) x 3 x 3
        #     R = rotation_matrix_from_quaternion(instance_rigid_params[:, :4] + self.quat_offset)
        #     if self.model_config['verbose']:
        #         print("Rotation matrix", R.size(), "should be (B x n_instance, 3, 3)")

        #     # decode translation
        #     b = instance_rigid_params[:, 4:]
        #     if self.state_normalize:  # denormalize
        #         b = b * std_d + mean_d
        #     b = b.view(B * n_instance, 1, state_dim)
        #     if self.model_config['verbose']:
        #         print("b", b.size(), "should be (B x n_instance, 1, state_dim)")

        #     # current particle state
        #     # p_0: B x 1 x n_p x state_dim -> (B * n_instance) x n_p x state_dim
        #     p_0 = state[:, -1:, :n_p]
        #     p_0 = p_0.repeat(1, n_instance, 1, 1).view(B * n_instance, n_p, state_dim)
        #     if self.model_config['verbose']:
        #         print("p_0", p_0.size(), "should be (B x n_instance, n_p, state_dim)")

        #     # current per-instance center state
        #     c = instance_center[:, :, -3:]
        #     if self.state_normalize:  # denormalize
        #         c = c * std_p + mean_p
        #     c = c.view(B * n_instance, 1, state_dim)
        #     if self.model_config['verbose']:
        #         print("c", c.size(), "should be (B x n_instance, 1, state_dim)")

        #     # updated state after rigid motion
        #     p_1 = torch.bmm(p_0 - c, R) + b + c
        #     if self.model_config['verbose']:
        #         print("p_1", p_1.size(), "should be (B x n_instance, n_p, state_dim)")

        #     # compute difference for per-particle rigid motion
        #     # rigid_motion: B x n_instance x n_p x state_dim
        #     rigid_motion = (p_1 - p_0).view(B, n_instance, n_p, state_dim)
        #     if self.state_normalize:  # normalize
        #         rigid_motion = (rigid_motion - mean_d) / std_d
        
        # if self.rigid_predictor is not None and self.rigid_out_dim == 3:
        #     # assert args.state_dim == 3
        #     # rigid_motion: B x n_p x state_dim
        #     rigid_motion = self.rigid_predictor(particle_effect[:, :n_p].contiguous())

        # aggregate motions
        # rigid_part = p_rigid_per_particle[..., 0].bool()
        # non_rigid_part = ~rigid_part
        
        # rigid_part = obj_mask & rigid_part
        # non_rigid_part = obj_mask & non_rigid_part

        # pred_motion = torch.zeros(B, n_p, state_dim).to(self.device)
        # if self.non_rigid_predictor is not None:
        #     pred_motion[non_rigid_part] = non_rigid_motion[non_rigid_part]
        # if self.rigid_predictor is not None and self.rigid_out_dim == 7:
        #     pred_motion[rigid_part] = torch.sum(p_instance.transpose(1, 2)[..., None] * rigid_motion, 1)[rigid_part]
        # if self.rigid_predictor is not None and self.rigid_out_dim == 3:
        #     pred_motion[rigid_part] = rigid_motion[rigid_part]

        pred_motion = non_rigid_motion

        if self.state_normalize:  # denormalize
            pred_motion = pred_motion * std_d + mean_d
        pred_pos = state[:, -1, :n_p] + torch.clamp(pred_motion, max=self.motion_clamp, min=-self.motion_clamp)
        if self.model_config['verbose']:
            print('pred_pos', pred_pos.size())

        # pred_pos (denormalized): B x n_p x state_dim
        # pred_motion (denormalized): B x n_p x state_dim

        # debug
        # orig_pos = state[:, -1, :n_p]
        # obj_mask = torch.zeros_like(orig_pos[:, :, 0]).bool()
        # obj_mask[:, :20] = True
        # _, R_pred, t_pred = umeyama_algorithm(orig_pos, pred_pos, obj_mask, fixed_scale=True)
        # pred_pos_ume = orig_pos.bmm(R_pred.transpose(1, 2)) + t_pred
        # assert torch.allclose(pred_pos[:, :20], pred_pos_ume[:, :20], atol=1e-5), f"umeyama algorithm failed, max diff: {torch.max(torch.abs(pred_pos[:, :20] - pred_pos_ume[:, :20]))}"

        return pred_pos, pred_motion


class Model(nn.Module):
    def __init__(self, args, **kwargs):

        super(Model, self).__init__()

        self.args = args

        # self.dt = torch.FloatTensor([args.dt]).to(args.device)
        # self.mean_p = torch.FloatTensor(args.mean_p).to(args.device)
        # self.std_p = torch.FloatTensor(args.std_p).to(args.device)
        # self.mean_d = torch.FloatTensor(args.mean_d).to(args.device)
        # self.std_d = torch.FloatTensor(args.std_d).to(args.device)

        # PropNet to predict forward dynamics
        self.dynamics_predictor = DynamicsPredictor(args, **kwargs)

    def init_memory(self, B, N):
        """
        memory  (B, mem_layer, N, nf_memory)
        """
        mem = torch.zeros(B, self.args.mem_nlayer, N, self.args.nf_effect).to(self.args.device)
        return mem

    def predict_dynamics(self, **inputs):
        """
        return:
        ret - predicted position of all particles, shape (n_particles, 3)
        """
        ret = self.dynamics_predictor(**inputs, verbose=self.args.verbose_model)
        return ret


class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def chamfer_distance(self, x, y):
        # x: [B, N, D]
        # y: [B, M, D]
        x = x[:, :, None, :].repeat(1, 1, y.size(1), 1) # x: [B, N, M, D]
        y = y[:, None, :, :].repeat(1, x.size(1), 1, 1) # y: [B, N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=3)    # dis: [B, N, M]
        dis_xy = torch.mean(torch.min(dis, dim=2)[0])   # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=1)[0])   # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        # pred: [B, N, D]
        # label: [B, M, D]
        return self.chamfer_distance(pred, label)


class EarthMoverLoss(torch.nn.Module):
    def __init__(self):
        super(EarthMoverLoss, self).__init__()

    def em_distance(self, x, y):
        # x: [B, N, D]
        # y: [B, M, D]
        x_ = x[:, :, None, :].repeat(1, 1, y.size(1), 1)  # x: [B, N, M, D]
        y_ = y[:, None, :, :].repeat(1, x.size(1), 1, 1)  # y: [B, N, M, D]
        dis = torch.norm(torch.add(x_, -y_), 2, dim=3)  # dis: [B, N, M]
        x_list = []
        y_list = []
        # x.requires_grad = True
        # y.requires_grad = True
        for i in range(dis.shape[0]):
            cost_matrix = dis[i].detach().cpu().numpy()
            try:
                ind1, ind2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
            except:
                # pdb.set_trace()
                print("Error in linear sum assignment!")
            x_list.append(x[i, ind1])
            y_list.append(y[i, ind2])
            # x[i] = x[i, ind1]
            # y[i] = y[i, ind2]
        new_x = torch.stack(x_list)
        new_y = torch.stack(y_list)
        # print(f"EMD new_x shape: {new_x.shape}")
        # print(f"MAX: {torch.max(torch.norm(torch.add(new_x, -new_y), 2, dim=2))}")
        emd = torch.mean(torch.norm(torch.add(new_x, -new_y), 2, dim=2))
        return emd

    def __call__(self, pred, label):
        # pred: [B, N, D]
        # label: [B, M, D]
        return self.em_distance(pred, label)


class HausdorffLoss(torch.nn.Module):
    def __init__(self):
        super(HausdorffLoss, self).__init__()

    def hausdorff_distance(self, x, y):
        # x: [B, N, D]
        # y: [B, M, D]
        x = x[:, :, None, :].repeat(1, 1, y.size(1), 1) # x: [B, N, M, D]
        y = y[:, None, :, :].repeat(1, x.size(1), 1, 1) # y: [B, N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=3)    # dis: [B, N, M]
        # print(dis.shape)
        dis_xy = torch.max(torch.min(dis, dim=2)[0])   # dis_xy: mean over N
        dis_yx = torch.max(torch.min(dis, dim=1)[0])   # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        # pred: [B, N, D]
        # label: [B, M, D]
        return self.hausdorff_distance(pred, label)
