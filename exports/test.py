import torch, glob
from torchvision import models
import torch.nn as nn

pths = glob.glob("kirc*/*model*.pth")
sds = [torch.load(p, map_location='cpu') for p in pths]

m = models.resnet18(pretrained=True)
m.fc = nn.Linear(m.fc.in_features, 2)
m = m.state_dict()
m = {"module."+k: v for k, v in m.items()}
sds.append(m)

# A = [sd["module.conv1.weight"][0,0] for sd in sds]
# [torch.all(torch.eq(A[0], a)) for a in A]

#AA = [[sd[k] for sd in sds] for k in sd.keys()
#res = [[torch.all(torch.eq(A[0], a)) for a in A] for A in AA]
#print(res)

for k in sds[0].keys():
	print(k)
	A = [sd[k] for sd in sds]
	res = [torch.all(torch.eq(A[0], a)) for a in A]
	print(res)
