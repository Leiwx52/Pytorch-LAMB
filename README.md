# Pytorch-LAMB
[Pytorch Implementation] Lamb 

[Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962)



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
```

