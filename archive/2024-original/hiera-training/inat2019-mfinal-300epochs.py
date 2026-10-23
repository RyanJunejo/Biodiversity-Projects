import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
from PIL import Image
import numpy as np
from RandAugment import RandAugment
from pathlib import Path





source_dir = '/scratch/ssd004/scratch/junejory/'
train_dir = '/scratch/ssd004/scratch/junejory/train2019'
val_dir = '/scratch/ssd004/scratch/junejory/val2019'
train_json = '/scratch/ssd004/scratch/junejory/train2019.json'
val_json = '/scratch/ssd004/scratch/junejory/val2019.json'
categories_json = '/scratch/ssd004/scratch/junejory/categories.json'
rsource_dir = '/datasets/inat_comp/2019/train_val2019'

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


train_data = load_json(train_json)
val_data = load_json(val_json)

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str or pathlib.Path): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

tp = Path(rsource_dir)
image_path = tp
if image_path.is_dir():
    print(f"{image_path} directory exists.")



# Write transform for image
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=9, magnitude=1), #9 Augmentations with Magnitude 0.5
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

def get_image_data(annotations):
    images = {img['id']: img['file_name'] for img in annotations['images']}
    img_to_annotations = {ann['image_id']: ann['category_id'] for ann in annotations['annotations']}
    return images, img_to_annotations

train_images, train_img_to_annotations = get_image_data(train_data)
val_images, val_img_to_annotations = get_image_data(val_data)

# Custom Dataset class for iNaturalist
class INatDataset(Dataset):
    def __init__(self, image_dict, img_to_annotations, root_dir, transform=None):
        self.image_dict = image_dict
        self.img_to_annotations = img_to_annotations
        self.root_dir = root_dir
        self.transform = data_transform

    def __len__(self):
        return len(self.image_dict)

    def __getitem__(self, idx):
        img_id = list(self.image_dict.keys())[idx]
        img_name = self.image_dict[img_id]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.img_to_annotations[img_id]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define root directory and create dataset instances
root_dir = '/datasets/inat_comp/2019/'
train_dataset = INatDataset(train_images, train_img_to_annotations, root_dir, transform=data_transform)
val_dataset = INatDataset(val_images, val_img_to_annotations, root_dir, transform=data_transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers = 8)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers = 8)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch import Tensor
# Define the DropPath function
def drop_path(x: Tensor, keep_prob: float = 1.0, inplace: bool = False) -> Tensor:
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
    mask.div_(keep_prob)
    if inplace:
        x.mul_(mask)
    else:
        x = x * mask
    return x

# Define the DropPath module
class DropPath(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0:
            x = drop_path(x, self.p, self.inplace)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

# Modify the Bottleneck block to include DropPath
class ConvBnAct(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, kernel_size=1):
        super().__init__(
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )

class BottleNeck(nn.Module):
    def __init__(self, in_features: int, out_features: int, reduction: int = 4, drop_prob: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnAct(in_features, out_features // reduction, kernel_size=1),
            ConvBnAct(out_features // reduction, out_features // reduction, kernel_size=3),
            ConvBnAct(out_features // reduction, out_features, kernel_size=1),
        )
        self.drop_path = DropPath(p=drop_prob)
        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x = self.drop_path(x)
        return x + res


cat_data = load_json('/scratch/ssd004/scratch/junejory/categories.json')
num_classes = len(cat_data)
print(num_classes)
model = torch.hub.load("facebookresearch/hiera", model="hiera_tiny_224", pretrained=True, checkpoint="mae_in1k")
for name, module in model.named_children():
    if isinstance(module, BottleNeck):
        setattr(model, name, BottleNeck(module.in_features, module.out_features, drop_prob=0.1))
model.head.projection = nn.Linear(in_features=model.head.projection.in_features, out_features=num_classes)
model = model.to(device)


layer_names = []
for idx, (name, param) in enumerate(model.named_parameters()):
    layer_names.append(name)
    #print(f'{idx}: {name}')
layer_names.reverse()
#layer_names[0:5]

# learning rate
lr      = 3e-3    #adjusted so first layer is 2e-3
lr_mult = 0.65

# placeholder
parameters      = []
prev_group_name = layer_names[0].split('.')[0]

# store params & learning rates
for idx, name in enumerate(layer_names):
    
    # parameter group name
    if name == "pos_embed":
        pass
    else:
        cur_group_name = name.split('.')[1]
        # update learning rate
        if cur_group_name == "weight":
            pass
        elif cur_group_name != prev_group_name:
            lr *= lr_mult
        prev_group_name = cur_group_name
    
    # display info
    #print(f'{idx}: lr = {lr:.6f}, {name}')
    
    # append layer parameters
    parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                    'lr':     lr}]

finetune_optimizer = optim.AdamW(parameters, lr=2e-3, betas=(0.9, 0.999), weight_decay=0.05)
scheduler2 = CosineAnnealingLR(finetune_optimizer, 300, eta_min=0)
scheduler1 = ConstantLR(finetune_optimizer, factor=0.1, total_iters=5)
finetune_scheduler = SequentialLR(finetune_optimizer, schedulers=[scheduler1, scheduler2], milestones=[5])
finetune_epochs = 300
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Additional augmentations
from torchvision.transforms import v2
mixup_alpha = 0.8
cutmix_alpha = 1.0
cutmix = v2.CutMix(num_classes=num_classes)
mixup = v2.MixUp(num_classes=num_classes)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

writer = SummaryWriter(log_dir=f'/scratch/ssd004/scratch/junejory/logs/hiera_transfer_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

# Fine-tuning
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

# Set up checkpointing
checkpoint_dir = os.path.join('/scratch/ssd004/scratch/junejory/checkpoint', os.getenv('USER'), os.getenv('SLURM_JOB_ID'))
os.makedirs(checkpoint_dir, exist_ok=True)

# Function to save checkpoint
def save_checkpoint(state, epoch, filename=None):
    if filename is None:
        filename = f'checkpoint19_{epoch}.pth'
    torch.save(state, os.path.join(checkpoint_dir, filename))

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch']
    return start_epoch


#wandb
import wandb
wandb.init(project="hiera_inat2019_transfer_learning", config={"learning_rate": 2e-3,"epochs": finetune_epochs, "training_batch_size": 512})



# Load checkpoint if exists
# checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
# if os.path.exists(checkpoint_path):
#     start_epoch = load_checkpoint(checkpoint_path, model, finetune_optimizer, finetune_scheduler, scaler)
# else:
start_epoch = 0

# Fine-tuning loop
for epoch in range(start_epoch, finetune_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with autocast():
            inputs, labels = cutmix_or_mixup(inputs, labels)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        finetune_optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(finetune_optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{finetune_epochs}, Loss: {epoch_loss:.4f}')
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    wandb.log({"epoch": epoch+1, "train_loss": epoch_loss})
    
     # Validation loop
    model.eval()
    val_loss = 0.0
    corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

    val_loss = val_loss / len(val_loader.dataset)
    accuracy = corrects.double() / len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    writer.add_scalar('Loss/validation', val_loss, epoch)
    writer.add_scalar('Accuracy/validation', accuracy, epoch)
    
    wandb.log({"val_loss": val_loss, "val_accuracy": accuracy})
    finetune_scheduler.step()

    epoch_time = time.time() - start_time
    print(f'Epoch {epoch+1} completed in {epoch_time//60:.0f}m {epoch_time%60:.0f}s')

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        print("Saving right now")
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': finetune_optimizer.state_dict(),
            'scheduler_state_dict': finetune_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, epoch+1)
    
    
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch+1} took {elapsed_time:.2f} seconds")

writer.close()
wandb.finish()

torch.save(model.state_dict(), 'hiera_inat2019_transfer_learned_300epochs.pth')

print("I'm done cooking")