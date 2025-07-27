import torch
import torch.nn as nn
from models.multimodal.rgb_encoder import RGBEncoder
from models.multimodal.depth_encoder import DepthEncoder
from models.multimodal.geometry_encoder import GeometryEncoder
from models.multimodal.fusion import CrossModalFusion
from models.multimodal.classifier import ClassificationHead


class SolidFusionPipeline(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.rgb_encoder = RGBEncoder(out_dim=512)
        self.depth_encoder = DepthEncoder(out_dim=512)
        self.geometry_encoder = GeometryEncoder(out_dim=512)
        self.fusion = CrossModalFusion(embed_dim=512, num_tokens=3)
        self.classifier = ClassificationHead(in_dim=512 * 3, num_classes=num_classes)

    def forward(self, rgb, depth, pcl):

        rgb_feat = self.rgb_encoder(rgb)       
        depth_feat = self.depth_encoder(depth) 
        geom_feat = self.geometry_encoder(pcl) 

        fused = self.fusion([rgb_feat, depth_feat, geom_feat])  
        logits = self.classifier(fused)  

        return logits
