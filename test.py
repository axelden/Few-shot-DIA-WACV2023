import torch
from einops import rearrange
import time
import numpy as np
from BibDataset2 import Bib_testset
from torchvision import transforms
from torch.utils.data import DataLoader
from function import saveIMGRes, get_scores, removeSmallCC
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix
###################################################################
################## HYPERPARAM AND COSTANTS ########################
#############6######################################################
import warnings
warnings.filterwarnings('ignore')
#IMG_W =1120
# PATCH_SIZE =224
#IMG_H = 1344
IMG_W =1344
IMG_H = 2016
PATCH_SIZE=336
PATCH_SPLIT=True
GEN_CROPS=False
WINDOW_STEP = PATCH_SIZE
H_PATCHES = int(IMG_H//PATCH_SIZE)
W_PATCHES = int(IMG_W//PATCH_SIZE)
DS = 'CB55'
BACKBONE = "resnet50"
date_string = time.strftime("%d%m%Y-%H-%M-%S")
REFINED_MASKS = True
SAVE_IMGS = False

###################################################################
####################### DATA LOADING ##############################
###################################################################
tf = transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((IMG_H,IMG_W)),
])

device = "cuda"

###################################################################
###################################################################
model = smp.DeepLabV3Plus(encoder_name=BACKBONE, encoder_weights=None, classes=4).to(device)
model.eval()
LOAD_MODEL = True

for DS in ["CB55", "CS18", "CS863"]:
    ###################################################################
    ####################### DATA LOADING ##############################
    ###################################################################
    SAVE_PATH = f"results/{DS}/"
    data_path = f'fewshot_data/{DS}/img-{DS}'
    data_path_GT = f'fewshot_data/{DS}/pixel-level-gt-{DS}'

    MODEL_LOAD_PATH = f'./trained_models/{DS}/_imgW={IMG_W}_imgH={IMG_H}_dlv3+336_bb=resnet50_fewshot_dilation.pt'
    test_data = Bib_testset(data_path, data_path_GT, tf, gen_crops=GEN_CROPS, patch_split=PATCH_SPLIT, patch_size=(PATCH_SIZE, PATCH_SIZE))
    test_loader = DataLoader(test_data, batch_size=1)
    model.to(device)
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH))
    print("\nstarting...")
    all_gts =np.array([])
    all_preds =np.array([])
    for i, batch in enumerate(test_loader):
        t0 = time.time()
        data = []
        # loading and formatting the data
        patches = batch["images"].to(device)
        gts = batch["gt"]
        refinement_mask = batch["bin_mask"]
        if PATCH_SPLIT:
            patches = rearrange(patches, "b n c h w -> (b n) c h w")
            gts = rearrange(gts, "b n h w -> (b n) h w")
            refinement_mask = rearrange(refinement_mask, "b n h w -> (b n) h w")

        # running the model and obtaining the predicted classes
        out = model(patches.float())
        patch_predictions = out.detach().cpu().argmax(1)

        # mask refinement process
        if REFINED_MASKS:
            patch_predictions = (patch_predictions*refinement_mask)

        all_gts = np.concatenate([all_gts, gts.numpy().flatten()])
        all_preds = np.concatenate([all_preds, patch_predictions.numpy().astype(int).flatten()])

        t1 = time.time() - t0
        # print(f'-- Epoch: {i:3}, Time: {t1:6.2f} --')
        if SAVE_IMGS:
            print("printing imgs")
            bin_mask_rgb = np.repeat(refinement_mask.unsqueeze(1), 3, 1)
            masked_patches = patches.cpu() * bin_mask_rgb
            masked_imgs = rearrange(masked_patches, "(b h w) c p1 p2 -> b c (h p1) (w p2)", h=H_PATCHES, w=W_PATCHES)
            imgs_predictions = rearrange(patch_predictions, "(b h w c) p1 p2 -> b c (h p1) (w p2)", c=1, h=H_PATCHES, w=W_PATCHES)
            saveIMGRes(masked_imgs, imgs_predictions.numpy(), SAVE_PATH)

    print(f"############## {DS} Scores ##############")
    w_acc, w_prec, w_rec, w_f1_sc, w_iou_sc = get_scores(all_gts, all_preds, average="weighted")
    w_acc_macro, w_prec_macro, w_rec_macro, w_f1_sc_macro, w_iou_sc_macro = get_scores(all_gts, all_preds,
                                                                                       average="macro")
    print("weighted average scores:")
    # print(np.mean(weight_acc))
    print(np.mean(w_prec))
    print(np.mean(w_rec))
    print(np.mean(w_iou_sc))
    print(np.mean(w_f1_sc))
    print("macro average scores:")
    # print(np.mean(weight_acc))
    print(np.mean(w_prec_macro))
    print(np.mean(w_rec_macro))
    print(np.mean(w_iou_sc_macro))
    print(np.mean(w_f1_sc_macro))

    # cm = np.round(confusion_matrix(all_gts, all_preds, normalize="true"), 2)
    # print(cm)



