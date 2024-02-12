import torch
import argparse
from torchvision import transforms, datasets
import torch.utils.data as data
from models.vit.masked_encoder import MaskedEncoder
from modules import RankMe, fit_powerlaw
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    encoder = model_path
    encoder_sd = torch.load(encoder, map_location=DEVICE)
    if "state_dict" in encoder_sd:
        encoder_sd = encoder_sd["state_dict"]
    embed_dim = encoder_sd["cls_token"].shape[2]
    patch_size = encoder_sd["patch_embed.proj.weight"].shape[2]
    if embed_dim == 768:
        depth = 12
        attention_heads = 12
    elif embed_dim == 1024:
        depth = 24
        attention_heads = 16
    elif embed_dim == 1280:
        depth = 32
        attention_heads = 16
    backbone = MaskedEncoder(
        patch_size=patch_size,
        embedding_dim=embed_dim,
        depth=depth,
        attention_heads=attention_heads,
        input_shape=(3, 224, 224),
    )
    backbone.load_state_dict(encoder_sd)
    backbone = backbone.to(DEVICE)
    backbone.eval()
    return backbone, embed_dim

def extract_features(model, data_loader, embed_dim):
    print("[INFO] Extracting features")
    N = len(data_loader)
    cov = torch.zeros(embed_dim, embed_dim)
    embeddings = torch.zeros(BATCH_SIZE, embed_dim)
    labels = torch.zeros(BATCH_SIZE, N)
    k = 0
    for x in tqdm(data_loader):
        inputs, batchLabels = x
        features = model.features(inputs.to(DEVICE))
        features = features[:, 0]
        features = features.detach().cpu()
        cov += torch.mm(features.T, features) / N  # compute covariance matrix
        if k == 0:
            embeddings = features
            labels = batchLabels
        else:
            embeddings = torch.concat([embeddings, features])
            labels = torch.concat([labels, batchLabels])
        k += 1
    return cov, embeddings, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dataset", type=str, help="The name of the dataset to be used")
    parser.add_argument("--batch_size", type=int, default=10, help="The batch size for inference")
    parser.add_argument("--model", type=str, help="The path to the model checkpoint")
    args = parser.parse_args()
    
    VAL_PATH = args.val_dataset
    BATCH_SIZE = args.batch_size
    
    TRANSFORM_IMG = transforms.Compose([
            transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    val_data = datasets.ImageFolder(root=VAL_PATH, transform=TRANSFORM_IMG)
    val_data_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=8)
    model, embed_dim = load_model(args.model)
    
    cov, embeddings, labels = extract_features(model, val_data_loader, embed_dim)

    embeddings_rank = RankMe(embeddings)
    _,eigenspectrum,_ = torch.linalg.svd(cov)
    alpha, _, _ = fit_powerlaw(eigenspectrum.real, 10,embed_dim)
    print("[INFO] Calculating RankMe")
    print(f"RankMe: {embeddings_rank}")
    print("[INFO] Calculating Eigenspectrum decay")
    print(f'Eigenspectrum decay α: {alpha}')