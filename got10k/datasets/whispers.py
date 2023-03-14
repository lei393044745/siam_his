from __future__ import absolute_import, print_function, unicode_literals

import os
import glob
import numpy as np
import io
import cv2
from itertools import chain


class Whispers(object):
  test_seq_name = ['ball','basketball','board','book','bus','bus2','campus',
                  'car','car2','car3','card','coin','coke',
                  'drive','excavator','face','face2','forest',
                  'forest2','fruit','hand','kangaroo','paper',
                  'pedestrain','pedestrian2','player',
                  'playground','rider1','rider2','rubik',
                  'student','toy1','toy2','trucker','worker',]
  train_seq_name = ['automobile','automobile10','automobile11','automobile12','automobile13','automobile14','automobile2','automobile3','automobile4','automobile5','automobile6','automobile7','automobile8','automobile9',
                    'basketball','board','bus','bus2',
                    'car1','car10','car2','car3','car4','car5','car6','car7','car8','car9',
                    'kangaroo',
                    'pedestrian','pedestrian2','pedestrian3','pedestrian4',
                    'rider1','rider2','rider3','rider4',
                    'taxi','toy','toy2',]
  
  sequence = {
    'train': train_seq_name,
    'test': test_seq_name
  }

  def __init__(self, root_dir, subset='train', type='HSI'):
    super(Whispers, self).__init__()
    self.root_dir = root_dir
    valid_seqs = self.sequence[subset]
    self.type = type
    self.anno_files = sorted(list(chain.from_iterable(glob.glob(
            os.path.join(root_dir, subset, s, type,'groundtruth*.txt')) for s in valid_seqs)))
    self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
    self.seq_names = [d.split('\\')[-2] for d in self.seq_dirs]
    
  def __getitem__(self, index):
    img_files = sorted(glob.glob(
          os.path.join(self.seq_dirs[index], '*.jpg' if self.type != 'HSI' else '*.png')))
    seq_name = self.seq_names[index]
    with open(self.anno_files[index], 'r') as f:
      anno = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))
    return img_files, anno
    
  def __len__(self):
    return len(self.seq_names)

  def X2Cube(self, img,B=[4, 4],skip = [4, 4],bandNumber=16):
    img = cv2.imread(img, -1)
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M//4, N//4,bandNumber )
    return DataCube
    
# s = Whispers(root_dir='D:\\BaiduNetdiskDownload\\whispers', type='HSI')
