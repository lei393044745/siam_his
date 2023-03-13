import matplotlib.pyplot as plt
import matplotlib
import json
from tqdm import tqdm
from PIL import Image
from .otb import ExperimentOTB
import os
from ..datasets import Whispers

from ..utils.metrics import rect_iou, center_error

class ExperimentWhispers(ExperimentOTB):
  def __init__(self, root_dir,result_dir='results', report_dir='reports', subset='train',type='RGB'):

    self.dataset = Whispers(root_dir, subset=subset, type=type)

    self.result_dir = os.path.join(result_dir, 'whispers')

    self.report_dir = os.path.join(report_dir, 'whispers')

    self.nbins_iou = 21
    self.nbins_ce = 51