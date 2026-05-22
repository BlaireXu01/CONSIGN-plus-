import torch
import numpy as np
import pickle
import os
from load_LIDC_data import LIDC_IDRI
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from probabilistic_unet import ProbabilisticUnet
import torch.nn.functional as F

def generate_consign_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = LIDC_IDRI(dataset_location='data/')
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    test_indices = indices[:split] 
    test_sampler = SubsetRandomSampler(test_indices)
    
    test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)

    net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
    net.to(device)
    net.load_state_dict(torch.load('prob_unet_lidc_10.pth'))
    net.eval()

    N_samples = 50
    all_smx = []
    all_labels = []
    all_imgs = []

    with torch.no_grad():
        for step, (patch, mask, _) in enumerate(test_loader):
            patch = patch.to(device)
            
            net.forward(patch, None, training=False)
            
            image_samples = []
            for _ in range(N_samples):
                logits = net.sample(testing=False) 
                
                prob_fg = torch.sigmoid(logits) 
                prob_bg = 1.0 - prob_fg
                probs = torch.cat((prob_bg, prob_fg), dim=1) 

                image_samples.append(probs.cpu().numpy()[0])
        
            all_smx.append(np.stack(image_samples))
            all_labels.append(mask.cpu().numpy()[0])
            all_imgs.append(patch.cpu().numpy()[0, 0])
            
            if step % 10 == 0:
                print(f"Processed {step}/{len(test_loader)} images")

    all_smx = np.array(all_smx)       # [N_images, N_samples, 2, 128, 128]
    all_labels = np.array(all_labels).astype(int) # [N_images, 128, 128]
    all_imgs = np.array(all_imgs)     # [N_images, 128, 128]

    save_dir = '../softmax/LIDC/'
    os.makedirs(save_dir, exist_ok=True)

    print("saving .pkl...")
    with open(os.path.join(save_dir, 'smx_LIDC.pkl'), 'wb') as f:
        pickle.dump(all_smx, f)
    with open(os.path.join(save_dir, 'labels_LIDC.pkl'), 'wb') as f:
        pickle.dump(all_labels, f)
    with open(os.path.join(save_dir, 'imgs.pkl'), 'wb') as f:
        pickle.dump(all_imgs, f)
        
    print("All files are saved successfully.")

if __name__ == "__main__":
    generate_consign_data()