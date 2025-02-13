import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

from clip import clip
from .swin import SwinTransformer


def load_clip_to_cpu():

    backbone_name = "ViT-B/16"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts):
        x = self.token_embedding(prompts.cuda()).type(self.dtype) 
      
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), prompts.argmax(dim=-1)] @ self.text_projection

        return x

    
class CLIP_text(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

        prompts = [
            "a photo of a person making a surprise facial expression.", 
            "a photo of a person making a fear facial expression.", 
            "a photo of a person making a disgust facial expression.", 
            "a photo of a person making a happiness facial expression.", 
            "a photo of a person making a sadness facial expression.", 
            "a photo of a person making an anger facial expression.", 
            "a photo of a person making a neutral facial expression." 
            ]
        
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts]) 
  
        self.prompts = prompts
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self):
        feature_t = self.text_encoder(self.prompts) 
        logit_scale = self.logit_scale.exp()
        return feature_t, logit_scale


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 512)

    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class SwinT_clip(nn.Module):
    def __init__(self):
        super(SwinT_clip, self).__init__()
        model = SwinTransformer()
        dict_checkpoint = torch.load('./models/start_0.pt')
        model.load_state_dict(dict_checkpoint["state_dict_backbone"], strict=True)    
        self.encoder = model
        self.MLP = MLP()
    
    def forward(self, image, feature_t, scale):
        feature_e = self.encoder.forward(image)
        feature_n = self.MLP(feature_e)

        feature_t = feature_t/ feature_t.norm(dim=-1, keepdim=True)    
        feature_image_e = feature_e / feature_e.norm(dim=-1, keepdim=True)
        logits_e = scale * feature_image_e @ feature_t.t().float()

        return feature_e, feature_n, logits_e 