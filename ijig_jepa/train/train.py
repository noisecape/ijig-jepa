import numpy
import torch
import torch.amp
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, v2
from tqdm.auto import tqdm

from ijig_jepa.dataset.flickr30k import Flickr30k
from ijig_jepa.models.ijig_jepa import IJigJepa


def train(params:dict)-> None:

    # Experiment
    exp_name = params['experiment']['exp_name']
    exp_parent_folder = params['experiment']['exp_parent_folder']
    save_every = params['experiment']['save_every']

    # Model
    model_name = params['model']['model_name']
    in_channels = params['model']['in_channels']
    patch_size = params['model']['patch_size']
    emb_size = params['model']['emb_size']
    img_size = params['model']['img_size']
    depth = params['model']['depth']
    out_dim = params['model']['out_dim']
    shuffle_patches = params['model']['shuffle_patches']
    drop_p = params['model']['drop_p']
    forward_expansion = params['model']['forward_expansion']
    mask = params['model']['mask']
    forward_drop_p = params['model']['forward_drop_p']
    num_heads = params['model']['num_heads']
    dropout = params['model']['dropout']

    # Optimization
    epoch = params['optimization']['epoch']
    lr = float(params['optimization']['lr'])
    device = params['optimization']['device']
    use_bf16 = params['optimization']['use_bf16']
    momentum = params['optimization']['momentum']

    # Dataset
    dataset_name = params['dataset']['dataset_name']
    batch_size = params['dataset']['batch_size']
    num_workers = params['dataset']['num_workers']
    pin_memory = params['dataset']['pin_memory']

    train_dst = Flickr30k(
        split='train', 
        transforms=Compose([v2.ToImage(), v2.Resize((224, 224)), v2.ToDtype(torch.float, scale=True)])
        )
    
    val_dst = Flickr30k(
        split='val',
        transforms=Compose([v2.ToImage(), v2.Resize((224, 224)), v2.ToDtype(torch.float, scale=True)])
    )

    test_dst = Flickr30k(
        split='test',
        transforms=Compose([v2.ToImage(), v2.Resize((224, 224)), v2.ToDtype(torch.float, scale=True)])
    )

    train_dl = DataLoader(
        train_dst,
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    val_dl = DataLoader(
        val_dst,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    ijig_jepa = IJigJepa(
        in_channels=in_channels,
        patch_size=patch_size,
        emb_size=emb_size,
        img_size=img_size,
        depth=depth,
        out_dim=out_dim,
        shuffle_patches=shuffle_patches,
        drop_p=drop_p,
        forward_expansion=forward_expansion,
        forward_drop_p=forward_drop_p,
        num_heads=num_heads,
        dropout=dropout,
        mask=mask
    ).to(device)

    optimizer = AdamW(ijig_jepa.parameters(), lr)

    scaler = torch.cuda.amp.GradScaler() if use_bf16 else None

    for e in range(0, epoch):

        loss_history = 0
        batch_loop = tqdm(train_dl, total=len(train_dl))
        
        for data in batch_loop:

            imgs, _ = data
            imgs = imgs.to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bf16):
                
                z, h = ijig_jepa(imgs)
                loss = F.smooth_l1_loss(z, h)

            #  Step 2. Backward & step
            if use_bf16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()


            # Step 3. momentum update of target encoder
            with torch.no_grad():

                for param_c, param_t in zip(ijig_jepa.context_encoder.parameters(), ijig_jepa.target_encoder.parameters()):
                    param_t.data.mul_(momentum).add_((1.-momentum) * param_c.detach().data)

            loss_history += loss.item()
            batch_loop.set_description(f"Loss: {loss.item():.4f}")
        
        print(loss_history/len(train_dl))