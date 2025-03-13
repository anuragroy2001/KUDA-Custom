import os
import sys

sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.spatracker.predictor import SpaTrackerPredictor
from models.spatracker.utils.visualizer import Visualizer


class SpaTrackerWrapper:
    def __init__(
        self,
        checkpoint='dynamics/tracker/checkpoints/spaT_final.pth',
        downsample=1.,
        fps=1,
        s_lenth=12,
        crop=False,
        crop_factor=1.,
        backward=True,
        device='cuda',
    ):
        self.downsample = downsample
        self.fps = fps
        self.s_lenth = s_lenth
        self.crop = crop
        self.backward = backward
        self.device = device

        self.transform = transforms.Compose([
            transforms.CenterCrop((int(384*crop_factor),
                                   int(512*crop_factor))),
        ])
        self.model = SpaTrackerPredictor(
            checkpoint=checkpoint,
            interp_shape=(384, 512),
            seq_length=s_lenth,
        )
        self.model.to(device)

    def __call__(
        self,
        video, # (T, H, W, 3)
        video_depth, # (T, H, W)
        queries, # (N, 2) of (x, y)
        debug=False,
        save_dir=None,
    ):
        """
        Returns:
            - pred_tracks: (T, N, 2)
            - pred_visibility: (T, N, 1)
        """
        # prepare inputs
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        if self.crop:
            video = self.transform(video)
        _, T, _, H, W = video.shape
        if H > W:
            downsample = max(self.downsample, 640//H)
        elif H < W:
            downsample = max(self.downsample, 960//W)
        else:
            downsample = max(self.downsample, 640//H)
        video = F.interpolate(video[0], scale_factor=downsample,
                              mode='bilinear', align_corners=True)[None]
        # downsample video length
        idx = torch.range(0, T-1, self.fps).long()
        video = video[:, idx]
        video_depth = torch.from_numpy(video_depth)[:, None][idx].float()
        num_queries = queries.shape[0]
        queries = np.concatenate([np.zeros((num_queries, 1)), queries], axis=-1)[None]
        queries = torch.from_numpy(queries).float()
        video, video_depth, queries = video.to(self.device), video_depth.to(self.device), queries.to(self.device)

        # forward
        pred_tracks, pred_visibility, T_Firsts = self.model(
            video, video_depth=video_depth, queries=queries,
            backward_tracking=self.backward, wind_length=self.s_lenth,
        )

        if debug == True:
            vis = Visualizer(save_dir=save_dir, grayscale=True,
                             fps=15, pad_value=0, linewidth=2,
                             tracks_leave_trace=1)
            msk_query = (T_Firsts == 0)
            pred_tracks = pred_tracks[:,:,msk_query.squeeze()]
            pred_visibility = pred_visibility[:,:,msk_query.squeeze()]
            try:
                video_vis = vis.visualize(video=video, tracks=pred_tracks[..., :2],
                                    visibility=pred_visibility,
                                    filename="spatracker")
            except:
                video_vis = None


        return pred_tracks[0].cpu().numpy()[..., :2], pred_visibility[0].cpu().numpy()
