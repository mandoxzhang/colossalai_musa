# ColossalAI offload on Musa
<b> With a few simple code changes, you can use ColossalAI offload on musa </b>

### Installation
Download From Source

```shell
git clone -b feature/musa https://github.com/mandoxzhang/ColossalAI.git
cd ColossalAI

# install colossalai
MUSA_CPU=1 pip install .
```

### Example

```python
import os
os.environ["PVR_GPUIDX"] = str(4)
os.environ["musa_MAX_MEM_USAGE_GB"] = "14"
try:
    import musa_torch_extension
except ImportError:
    pass

import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.zero import ZeroOptimizer
from colossalai.nn.parallel import GeminiDDP

import psutil
import random


colossalai.launch(config={},
                  rank=0,
                  world_size=1,
                  host='127.0.0.1',
                  port=random.randint(10023, 45221),
                  backend='gloo')

device = get_current_device()

with ColoInitContext(device=device, dtype=torch.float):
    model = GPT2LMHeadModel(configuration)
    numel = sum([p.numel() for p in model.parameters()])
    print(f'model parameter nueml {numel}')

PLACEMENT_POLICY = 'cpu'
model = GeminiDDP(model, device=get_current_device(), placement_policy=PLACEMENT_POLICY, pin_memory=True)
optimizer = HybridAdam(model.parameters(), lr=args.learning_rate)
optimizer = ZeroOptimizer(optimizer, model, initial_scale=1)

for epoch in range(num_epochs):
    for data in data_loader:

        loss = model(data)
        optimizer.backward(loss) ## not loss.backward()
        optimizer.step()

        print(f'cpu memory has used {psutil.Process().memory_info().rss / 1024**2:.2f} MB')
```

### Result
For every 0.1 billion parameters, ColossalAI can offload about 1.11G parameters from gpu to cpu.
