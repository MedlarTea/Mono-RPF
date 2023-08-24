from .base_identifier import BaseIdentifier
import torch
import time
from ..utils import *




class PartIdentifier(BaseIdentifier):
    def __init__(self, params):
        self.target_id = -1
        self.target_conf = -1

        self.params = params
        self.state = None
        self.classifier = None

        ### Counters ###
        self.feature_extraction_times = AverageMeter()
        self.identification_times = AverageMeter()

        ### set hyperparameter for reid model ###
        losses_weights = {
            GLOBAL: {'id': 1., 'tr': 0.},
            FOREGROUND: {'id': 0., 'tr': 0.},
            CONCAT_PARTS: {'id': 1., 'tr': 0.},  # hanjing
            PARTS: {'id': 0., 'tr': 1.}
        }

        ### initialize the reid model ###
        

        ### initialize the classifier ###
        rpf_model.reid.head.GiLt = GiLtLoss(losses_weights=losses_weights, use_visibility_scores=True, triplet_margin=self.params.triplet_margin, ce_smooth=self.params.ce_smooth,loss_name=self.params.triplet_loss, use_gpu=True)
        self.classifier = name_match.classifiers[self.params.agent](params=self.params, reid_model=rpf_model.reid)
        self.state = InitialState(self.params)
    
