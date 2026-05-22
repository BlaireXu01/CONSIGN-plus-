import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location = 'data/')
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=5, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices),len(test_indices)))

net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
epochs = 10
for epoch in range(epochs):
    # 可以在每个 Epoch 开始时打印一下
    print(f"--- Starting Epoch {epoch+1}/{epochs} ---")
    
    for step, (patch, mask, _) in enumerate(train_loader): 
        patch = patch.to(device)
        mask = mask.to(device)
        # 2. 【在此处插入 NaN 检查】
        # if torch.isnan(patch).any() or torch.isnan(mask).any():
        #     print(f"Warning: Input data contains NaN at Epoch {epoch+1}, Step {step}!")
        #     continue # 跳过这个 batch，不参与训练
        mask = torch.unsqueeze(mask,1)
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
        loss = -elbo + 1e-5 * reg_loss
        optimizer.zero_grad()
        loss.backward()
        # --- 加入下面这一行 ---
        # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0) 
        # -
        optimizer.step()
        
        # 添加这部分：每隔 100 个 step 打印一次进度和 loss
        if step % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}")

print("Training finished!")
torch.save(net.state_dict(), 'prob_unet_lidc_10_noclip.pth')
print("Model saved to prob_unet_lidc.pth")