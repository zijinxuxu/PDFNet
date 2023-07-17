import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

from lib.models.networks.intaghand_encoder import load_encoder
from lib.models.networks.intaghand_decoder import load_decoder

class HandNET_GCN(nn.Module):
    def __init__(self, encoder, mid_model, decoder):
        super(HandNET_GCN, self).__init__()
        self.encoder = encoder
        self.mid_model = mid_model
        self.decoder = decoder

    def forward(self, img, choose, cloud, depth, ind, K_new, valid):
        hms, mask, dp, ret, img_fmaps, hms_fmaps, dp_fmaps = self.encoder(img, depth, ind, choose, cloud, K_new, valid)
        if 'point2mano_left' in ret:
            return ret
        if False:
            global_feature, fmaps = self.mid_model(img_fmaps, hms_fmaps, dp_fmaps)
            result, paramsDict, handDictList, otherInfo = self.decoder(global_feature, fmaps)
        else:
            global_feature_left, global_feature_right, fmaps = self.mid_model(img_fmaps, hms_fmaps, dp_fmaps)
            result, paramsDict, handDictList, otherInfo = self.decoder(global_feature_left, global_feature_right, fmaps)

        if hms is not None:
            otherInfo['hms'] = hms
        if mask is not None:
            otherInfo['mask'] = mask
        if dp is not None:
            otherInfo['dense'] = dp
        if ret is not None:
            otherInfo['ret'] = ret     
        if True:
            converter = {}
            for hand_type in ['left', 'right']:
                converter[hand_type] = self.decoder.converter[hand_type]    
            otherInfo['converter_left'] = converter['left']
            otherInfo['converter_right'] = converter['right']
        return result, paramsDict, handDictList, otherInfo


def load_model_intag(cfg):
    encoder, mid_model = load_encoder(cfg)
    decoder = load_decoder(cfg, mid_model.get_info())
    model = HandNET_GCN(encoder, mid_model, decoder)

    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # path = os.path.join(abspath, str(cfg.MODEL_PARAM.MODEL_PRETRAIN_PATH))
    # if os.path.exists(path):
    #     state = torch.load(path, map_location='cpu')
    #     print('load model params from {}'.format(path))
    #     try:
    #         model.load_state_dict(state)
    #     except:
    #         state2 = {}
    #         for k, v in state.items():
    #             state2[k[7:]] = v
    #         model.load_state_dict(state2)

    return model
