from domains import Box
import torch.nn.functional as F
import diffsat as dl2
import numpy as np
import torch


def transform_network_output(o, network_output):
    if network_output == 'logits':
        pass
    elif network_output == 'prob':
        o = [F.softmax(zo) for zo in o]
    elif network_output == 'logprob':
        o = [F.log_sofmtax(zo) for zo in o]
    return o


class Constraint:
    def eval_z(self, z_batches):
        if self.use_cuda:
            z_inputs = [torch.cuda.FloatTensor(z_batch) for z_batch in z_batches]
        else:
            z_inputs = [torch.FloatTensor(z_batch) for z_batch in z_batches]

        for z_input in z_inputs:
            z_input.requires_grad_(True)
        z_outputs = [self.net(z_input) for z_input in z_inputs]
        for z_out in z_outputs:
            z_out.requires_grad_(True)
        return z_inputs, z_outputs

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        assert False

    def loss(self, x_batches, y_batches, z_batches, args):
        if z_batches is not None:
            z_inp, z_out = self.eval_z(z_batches)
        else:
            z_inp, z_out = None, None

        constr = self.get_condition(z_inp, z_out, x_batches, y_batches)
        
        neg_losses = dl2.Negate(constr).loss(args)
        pos_losses = constr.loss(args)
        sat = constr.satisfy(args)
            
        return neg_losses, pos_losses, sat, z_inp


class RobustnessConstraint(Constraint):
    def __init__(self, net, eps, delta, use_cuda=True, network_output='logits'):
        self.net = net
        self.network_output = network_output
        self.eps = eps
        self.delta = delta
        self.use_cuda = use_cuda
        self.n_tvars = 1
        self.n_gvars = 1
        self.name = 'RobustnessG'

    def params(self):
        return {'eps': self.eps, 'network_output': self.network_output}

    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 1
        n_batch = x_batches[0].size()[0]

        return [[Box(np.clip(x_batches[0][i].cpu().numpy() - self.eps, 0, 1),
                     np.clip(x_batches[0][i].cpu().numpy() + self.eps, 0, 1))
                for i in range(n_batch)]]

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        n_batch = x_batches[0].size()[0]
        z_out = transform_network_output(z_out, self.network_output)[0]
        # z_logits = F.log_softmax(z_out[0], dim=1)
        
        pred = z_out[np.arange(n_batch), y_batches[0]]

        limit = torch.FloatTensor([0.3])
        if self.use_cuda:
            limit = limit.cuda()
        return dl2.GEQ(pred, torch.log(limit))
