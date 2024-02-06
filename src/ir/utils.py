import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from .pdt import PDT
import os 



def find_threshold(distance_metric):
    if distance_metric == 'euclidean':
        return 1
    if distance_metric == "euclidean_l2":
        return 1
    elif distance_metric =='cosine':
        return .5
    else:
        raise ValueError(f"Distance metric must be one of: 'euclidean', 'euclidean_l2', 'cosine' but got {distance_metric} instead")
    
def get_all():
    def bgr_to_rgb_lambda(x):
        return x[[2, 1, 0], :, :]
    bgr_to_rgb = transforms.Lambda(lambda x: bgr_to_rgb_lambda(x))

    transform = transforms.Compose([
        transforms.ToTensor(),
        bgr_to_rgb,
        transforms.Resize((112, 112)),
        transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path_to_checkpoint = os.path.join(os.getcwd(), 'src', 'ir', 'checkpoints', 'checkpoint.pt')
    checkpoint = torch.load(path_to_checkpoint, map_location=torch.device(device))
    translator=PDT(pool_features=6, use_se=False, use_bias=False, use_cbam=True)
    translator.load_state_dict(checkpoint['model_state_dict'])
    translator.eval()
    translator.requires_grad_(False)
    translator.to(device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    resnet.requires_grad_(False)
    resnet.to(device)
    return transform, translator, resnet