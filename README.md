# Pytorch-LAMB
[Pytorch Implementation] Lamb 

[Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962)

## Notes

+ `exculde_from_weight_decay` and `exclude_from_layer_adaptation` is implemented to exclude *weight decay* and *layer-wise adaptation* for some layers. Reference code can be found at [official tensorflow implementation](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py) and [here](https://github.com/fastalgo/imagenet_resnet50_lamb/blob/master/optimization.py).
+ I pass `net`/`model` which the parameters belong as an argument, to configurate name for each parameter. This is for matching the names with list of regex patterns in `exculde_from_weight_decay` and `exclude_from_layer_adaptation`.
+ Add gradient clipping into Lamb. If this has already been done outside the scope of `optimizer.step()` in your code, <u>**REMEMBER**</u> to set `grad_clip_norm = None` and `grad_clip_value = None`.

## Usage

```py
>>> from LAMB import Lamb
>>> optimizer = optim.Lamb(model, model.parameters(), lr=0.1)
>>> optimizer.zero_grad()
>>> loss_fn(model(input), target).backward()
>>> optimizer.step()
```

```python
import torchvision
resnet = torchvision.models.resnet18(pretrained=False)
optim = Lamb(resnet, 
             resnet.parameters(), 
             lr=0.01, 
             exclude_from_layer_adaptation=['BatchNorm'], 
             grad_clip_norm=1.0
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
```

## Reference

- [Pytorch-lamb](https://github.com/cybertronai/pytorch-lamb)

- [pytorch-optimizer](https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/lamb.py)
- [addons[official]](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py)
- [imagenet_resnet50_lamb](https://github.com/fastalgo/imagenet_resnet50_lamb/blob/master/optimization.py)