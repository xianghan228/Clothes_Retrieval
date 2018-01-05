import caffe
import numpy as np

class ConcateLayer1(caffe.Layer):

  def setup(self,bottom,top):
    self.bottom_size = bottom[0].data.shape
    top[0].reshape(self.bottom_size[0],self.bottom_size[1]*3)

  def forward(self, bottom, top):
    top[0].data[...] = np.concatenate((bottom[0].data[...],bottom[1].data[...],bottom[2].data[...]),axis = 1)

  def backward(self, top, propagate_down, bottom):
    pass

  def reshape(self, bottom, top):
    pass
  
class ConcateLayer2(caffe.Layer):

  def setup(self,bottom,top):
    self.bottom_size = bottom[0].data.shape
    top[0].reshape(self.bottom_size[0],self.bottom_size[1]*2)

  def forward(self, bottom, top):
    top[0].data[...] = np.concatenate((bottom[0].data[...],bottom[1].data[...]),axis = 1)

  def backward(self, top, propagate_down, bottom):
    pass

  def reshape(self, bottom, top):
    pass
