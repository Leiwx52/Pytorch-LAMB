import math
import re
import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor


class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning:
    Training BERT in 76 minutes`__.
    Arguments:
        net: Network that includes all params to be optimized. Note that
            the second args, `params` should be in `net`.
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        clamp_value: clamp weight_norm in (0,clamp_value) (default: 10)
            set to a high value to avoid it (e.g 10e3)
        exclude_from_weight_decay: List of regex patterns of
              variables excluded from weight decay. Variables whose name
              contain a substring matching the pattern will be excluded.
        exclude_from_layer_adaptation: List of regex patterns of
              variables excluded from layer adaptation. Variables whose name
              contain a substring matching the pattern will be excluded.
        adam: always use trust ratio = 1, which turns this
            into Adam. Useful for comparison purposes. (default: False)
        debias: debias adam by (1 - beta**step) (default: False)
    Example:
        >>> from LAMB import Lamb
        >>> optimizer = Lamb(model, model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ https://arxiv.org/abs/1904.00962
    Note:
        + Reference code: 
        #1 https://github.com/cybertronai/pytorch-lamb
        #2 https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/lamb.py
        #3 https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py [Official]

        + This is different from some Pytorch optimizers, which does not need to pass a `net` argument.
        Adapt to `exculde_from_weight_decay` and `exclude_from_layer_adaptation` by including this args.
        See Reference code #3 or https://github.com/fastalgo/imagenet_resnet50_lamb/blob/master/optimization.py

    """

    def __init__(
        self,
        net,
        params,
        lr = 1e-3,
        betas = (0.9, 0.999),
        eps = 1e-6,
        weight_decay = 0,
        clamp_value = 10,
        exclude_from_weight_decay = None,
        exclude_from_layer_adaptation = None,
        adam = False,
        debias = False,
    ):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if clamp_value < 0.0:
            raise ValueError('Invalid clamp value: {}'.format(clamp_value))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.net = net
        self.clamp_value = clamp_value  # clamp_value see reference code in Pytorch, they clamp value of the weight norm
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if
        # the arg is None.
        # Borrow from official tensorflow implementation
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay
        self.adam = adam
        self.debias = debias

        super(Lamb, self).__init__(params, defaults)
        self._init_paraName()
    
    def _init_paraName(self):
        r'''
        Get all parameters's name in `self.net` and then store in `self.state`
        Do it in initialzation.
        '''
        for name,para in self.net.named_parameters():
            module_top2bottom = name.split('.')
            cursor = self.net
            for i in range(len(module_top2bottom)-1):
                cursor = getattr(cursor, module_top2bottom[i])
            bottom_m_name = repr(cursor).split('(')[0]
            this_para_name = '.'.join([bottom_m_name, module_top2bottom[-1]])
            
            # Adding name for each parameter
            # e.g. Conv2d.weight, BatchNorm2d.bias, etc.
            # This is for `exclude_weight_dacay` and `exculde_layer_adaptation`
            self.state[para]['para_name'] = this_para_name
    
    def _do_layer_adaptation(self, para):
        r"""Whether to do layer-wise learning rate adaptation for
        `para`.
        """
        para_name = self.state[para]['para_name']
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, para_name, re.I) is not None:
                    return False
        return True
    
    def _do_use_weight_decay(self, para):
        r"""Whether to use L2 weight decay for `param`."""
        para_name = self.state[para]['para_name']
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, para_name, re.I) is not None:
                    return False
        return True

    def step(self, closure=None):
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        'Lamb does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                # Modified len(state) == 0 to len(state)<=1 because para_name is added to
                # self.state in initialization stage
                if len(state) <= 1:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                if self.debias:
                    bias_correction = math.sqrt(1 - beta2 ** state['step'])
                    bias_correction /= 1 - beta1 ** state['step']
                else:
                    bias_correction = 1

                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] * bias_correction

                weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0 and self._do_use_weight_decay(p):
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = torch.norm(adam_step)

                if weight_norm == 0 or adam_norm == 0 or not self._do_layer_adaptation(p):
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss

if __name__ == "__main__":
    '''
    A toy model.
    '''
    import torchvision
    resnet = torchvision.models.resnet18(pretrained=False)
    optim = Lamb(resnet, 
            resnet.parameters(), 
            lr=0.01, 
            exclude_from_weight_decay=['Conv', 'bias'], 
            exclude_from_layer_adaptation=['BatchNorm']
        )
    criterion = torch.nn.CrossEntropyLoss()

    resnet.zero_grad()
    inp = torch.randn(1,3,224,224)
    outp = resnet(inp)
    target = torch.ones(1,).long()
    loss = criterion(outp, target)
    loss.backward()
    optim.step()

    for p in optim.state:
        state = optim.state[p]
        print(state['para_name'], state['trust_ratio'])

