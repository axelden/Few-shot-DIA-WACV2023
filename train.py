import torch
import torch.nn as nn
from einops import rearrange
import time
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from BibDataset2 import Bib_trainset, Bib_validset
from sklearn.metrics import accuracy_score
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50, lraspp_mobilenet_v3_large
import segmentation_models_pytorch as smp
import os

###################################################################
################## HYPERPARAM AND COSTANTS ########################
#############6######################################################
#IMG_W =1120
#IMG_H = 1344
##### good ones####
IMG_W =1344
IMG_H = 2016
####################
BACKBONE  = "resnet50"
PATCH_SIZE = 336
BATCH_SIZE = 1
PATCH_SPLIT= True
GEN_CROPS=True
N_CROPS = 10
# N_PATCHES = int(IMG_SIZE / PATCH_SIZE)
EPOCHS = 200
minloss = 1e10
LR = 0.001
LOAD_MODEL = False

for DS in ["CB55", "CS18", "CS863"]:
    ###################################################################
    ####################### DATA LOADING ##############################
    ###################################################################
    # imgs path
    data_path = f'fewshot_data/{DS}/img-{DS}'
    data_path_GT = f'fewshot_data/{DS}/pixel-level-gt-{DS}'
    tf = transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_H,IMG_W))
    ])

    train_data = Bib_trainset(data_path, data_path_GT, tf, gen_crops=GEN_CROPS, n_crops=N_CROPS, patch_split=PATCH_SPLIT, patch_size=(PATCH_SIZE, PATCH_SIZE))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    valid_data = Bib_validset(data_path, data_path_GT, tf, gen_crops=GEN_CROPS,  n_crops=N_CROPS, patch_split=PATCH_SPLIT, patch_size=(PATCH_SIZE, PATCH_SIZE))
    valid_loader = DataLoader(valid_data, batch_size=1)

    ###################################################################
    ##################### MODEL DEFINITION ############################
    ###################################################################
    model = smp.DeepLabV3Plus(encoder_name=BACKBONE, encoder_weights=None, classes=4)
    model = model.to("cuda")

    ce_weights = {"CB55": np.sqrt([1. / 82, 1. / 8.36, 1. / 0.55, 1 / 8.68]),
                  "CS18": np.sqrt([1. / 85, 1. / 6.78, 1. / 1.47, 1 / 6.59]),
                  "CS863": np.sqrt([1. / 78, 1. / 6.35, 1. / 1.83, 1 / 14]),
                  }

    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(ce_weights[DS]).float())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.00001)
    device = "cuda"
    model.to(device)


    MODEL_LOAD_PATH = f'./trained_models/{DS}/_img{IMG_W}_Psize{PATCH_SIZE}_deeplabv3.pt'
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH))

    ###################################################################
    ###################### MODEL TRAINING #############################
    ###################################################################
    print("\nstarting model training...")
    epoch_last_update = 0
    minloss = 1e10
    RANDOM_CROPS = []
    for i in range(EPOCHS):
        # after the 50 epochs buffer if the model doesn't improve over 20 epochs stop the training
        if i - epoch_last_update > 20 and i > 50:
            break

        t0 = time.time()
        loss = []
        val_loss = []
        accuracy = []
        val_accuracy = []
        for it, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # loading the data
            patches = batch["images"].to(device)
            gts = batch["gt"]

            if PATCH_SPLIT:
                patches = rearrange(patches, "b n c h w -> (b n) c h w")
                gts = rearrange(gts, "b n h w -> (b n) h w")

            # running the model
            pred_cls = model(patches.float()).cpu()

            # computing the losses
            gts = gts.flatten()
            c_loss = ce_loss(torch.softmax(pred_cls.permute(0,2,3,1).flatten(0,2), 1), gts.long())

            total_loss = c_loss
            total_loss.backward()
            loss.append(total_loss.detach().cpu().detach())

            #computing accuracy
            acc = accuracy_score(pred_cls.detach().argmax(1).flatten(), gts)
            accuracy.append(acc)

            optimizer.step()

        ###################################################################
        ###################### MODEL VALIDAT. #############################
        ###################################################################
        model.eval()
        with torch.no_grad():
            for it, batch in enumerate(valid_loader):
                # loading the data
                patches = batch["images"].to(device)
                gts = batch["gt"]

                if PATCH_SPLIT:
                    patches = rearrange(patches, "b n c h w -> (b n) c h w")
                    gts = rearrange(gts, "b n h w -> (b n) h w")

                # running the model
                pred_cls = model(patches.float()).cpu()

                # computing the losses
                gts = gts.flatten()
                val_c_loss = ce_loss(torch.softmax(pred_cls.permute(0, 2, 3, 1).flatten(0, 2), 1),
                                 gts.long())
                val_total_loss = val_c_loss

                # computing accuracy
                val_acc = accuracy_score(pred_cls.detach().argmax(1).flatten(), gts.flatten())
                val_accuracy.append(val_acc)

                val_loss.append(val_total_loss.detach().cpu())

        model.train()
        mean_val_loss = np.mean(val_loss)
        t1 = time.time() - t0
        print(f'-- Epoch: {i:3}, Time: {t1:6.2f} --')
        print(f'Loss: {np.mean(loss):10.8f}')
        print(f'VAL Loss: {mean_val_loss:10.8f}')
        print(f'Accuracy: {np.mean(accuracy):10.8f}')
        print(f'VAL accuracy: {np.mean(val_accuracy):10.8f}')

        # save the best performing model
        if mean_val_loss <= minloss:
            epoch_last_update = i
            print("BEST!")
            minloss = mean_val_loss
            os.makedirs('./trained_models', exist_ok=True)
            os.makedirs(f'./trained_models/{DS}', exist_ok=True)
            torch.save(model.state_dict(),
                      f'./trained_models/{DS}/_imgW={IMG_W}_imgH={IMG_H}_dlv3+336_bb={BACKBONE}_fewshot_dilation.pt')