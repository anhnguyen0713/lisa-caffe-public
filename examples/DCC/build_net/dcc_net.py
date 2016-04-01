from __future__ import print_function
from utils.config import *
from utils.python_utils import *
import sys
sys.path.insert(0, '../../python/')
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import pdb
from caffe_net import *

class dcc(caffe_net):

  def __init__(self, vocab='', num_features=471):
    self.len_vocab = len(open_txt(vocab)) + 1 
    self.num_features = num_features

  def init_net(self):
    self.n = caffe.NetSpec()
    self.silence_count = 0

  def build_image(self, image_data='data'):
    self.n.tops['reshape-data'] = L.Reshape(self.n.tops[image_data], shape=dict(dim=[1, -1, self.num_features])) 

  def build_caption(self, input_sentence='input_sentence', image_data='reshape-data', t=20):
    nl_1 = self.learning_params([[0, 0]])
    nl_2 = self.learning_params([[0, 0]])
    nl_3 = self.learning_params([[0, 0]])
    l_wb = self.learning_params([[1,1],[1,2]])
    l_w = self.learning_params([[1, 1]])
    self.n.tops['embedding'] = self.embed(self.n.tops[input_sentence], 512, input_dim=self.len_vocab, bias_term=False, learning_param=nl_1)
    self.n.tops['embedding2'] = L.InnerProduct(self.n.tops['embedding'], num_output=512, param=nl_2, axis=2)
    self.n.tops['lstm1'] = L.LSTM(self.n.tops['embedding2'], self.n.tops['cont_sentence'], param=nl_3, recurrent_param=dict(num_output=512)) 
    self.n.tops['concat-lm'] = L.Concat(self.n.tops['embedding2'], self.n.tops['lstm1'], axis=2)
    self.n.tops['predict-lm'] = L.InnerProduct(self.n.tops['concat-lm'], num_output=self.len_vocab, param=l_wb, axis=2, weight_filler=self.uniform_weight_filler(-0.08, 0.08), bias_filler=self.constant_filler(0))
    self.n.tops['tile-data'] = L.Tile(self.n.tops[image_data], axis=0, tiles=t)
    self.n.tops['predict-im'] = L.InnerProduct(self.n.tops['tile-data'], num_output=self.len_vocab, param=l_w, axis=2, weight_filler=self.uniform_weight_filler(-0.08, 0.08), bias_term=False) 
    self.n.tops['predict-multimodal'] = L.Eltwise(self.n.tops['predict-lm'], self.n.tops['predict-im'], operation=1) 

  def build_caption_unroll(self, input_sentence='input_sentence', image_data='data', T=20):
    #split input and cont
    input_slices = L.Slice(self.n.inputs[input_sentence], axis=0, ntop=20)
    cont_slices = L.Slice(self.n.inputs['cont_sentence'], axis=0, ntop=20)
    self.rename_tops(input_slices, ['%s_%d' %(input_sentence, t) for t in range(T)])
    self.rename_tops(cont_slices, ['cont_sentence_%d' %t for t in range(T)])

    #init hidden units
    self.n.tops['lstm1_h0'] = self.dummy_data_layer([1, 1000, 512], 0)
    self.n.tops['lstm1_c0'] = self.dummy_data_layer([1, 1000, 512], 0)

    embedding_lp = self.named_params(['embed_w'], [[0,0]])
    embedding2_lp = self.named_params(['embed_w'], [[0,0], [0,0]])
    predict_lm_lp = self.named_params(['predict-lm_w'], [[1,1], [1,2]])

    self.n.tops['predict-im'] = L.InnerProduct(self.n.tops[image_data], num_output=self.len_vocab, param=l_w, axis=2, weight_filler=self.uniform_weight_filler(-0.08, 0.08), bias_term=False) 

    for t in range(0, T):
      input_sentence_t = '%s_%d' %(input_sentence, t)
      cont_sentence_t = 'cont_sentence_%d' %t
      embedding1_t = 'embedding1_%d' %t   
      embedding2_t = 'embedding1_%d' %t   
      prev_hidden_unit = 'lstm1_h%d' %t
      prev_cell_unit = 'lstm1_c%d' %t
      hidden_unit = 'lstm1_h%d' %(t+1)
      cell_unit = 'lstm1_c%d' %(t+1)
      concat_lm_t = 'concat-lm_%d' %t
      predict_lm_t = 'predict-lm_%d' %t
      predict_multimodal_t = 'predict-multimodal_%d' %t

      self.n.tops[embedding1_t] = self.embed(self.n.tops[input_sentence_t], 512, input_dim=self.len_vocab, bias_term=False, learning_param=embedding_lp)
      self.n.tops[embedding2_t] = L.InnerProduct(self.n.tops['embedding'], num_output=512, param=embedding2_lp, axis=2)
      self.n.tops[hidden_unit], self.n.tops[cell_unit] = self.lstm_unit('lstm1', 
                                             self.n.tops[embedding2_t],
                                             self.n.tops[cont_sentence_t],
                                             h = self.n.tops[prev_hidden_unit],
                                             c = self.n.tops[prev_cell_unit],
                                             batch_size = self.N, timestep=t,
                                             weight_lr_mult=0, bias_lr_mult=0,
                                             weigth_decay_mult=0, bias_decay_mult=0)
      self.n.tops[concat_lm_t] = L.Concat(self.n.tops[embedding2], 
                                          self.n.tops[hidden_unit], axis=2)
      self.n.tops[predict_lm_t] = L.InnerProduct(self.n.tops[concat_lm_t], num_output=self.len_vocab, param=predict_lm_lp, axis=2, weight_filler=self.uniform_weight_filler(-0.08, 0.08), bias_filler=self.constant_filler(0)) 
      self.n.tops[predict_multimodal_t] = L.Eltwise(self.n.tops[predict_lm_t], self.n.tops['predict-im'], operation=1) 

    self.n.tops['predict-multimodal'] = L.Concat([self.n.tops['predict-multimodal_%t'] for t in range(T)], axis=0)

  def build_train_caption_net(self, feature_param_str, hdf_source, save_name='', unroll=False):
    self.init_net()
    feature_param_str['top_names'] = ['data', 'labels']
    self.N = feature_param_str['batch_size']
    hdf_top_names = ['cont_sentence', 'input_sentence', 'target_sentence']
    self.python_input_layer('python_data_layers', 'featureDataLayer', feature_param_str)
    hdf_tops = L.HDF5Data(source=hdf_source, batch_size=20, ntop=3)
    self.rename_tops(hdf_tops, hdf_top_names)
    self.silence(self.n.tops['labels'])
    
    self.build_image()
    self.build_caption() 

    self.n.tops['cross-entropy-loss'] = self.softmax_loss(self.n.tops['predict-multimodal'], self.n.tops['target_sentence'], axis=2, loss_weight=20) 

    self.write_net(models_root+save_name)

  def build_deploy_caption_net(self, save_name=''):
    self.init_net()
    self.n.tops['data'] = self.dummy_data_layer([10, self.num_features])
    self.build_image()
    
    self.write_net(models_root+save_name)

  def build_wtd_caption_net(self, save_name=''):
    self.init_net()
    self.n.tops['cont_sentence'] = self.dummy_data_layer([1, 1000])
    self.n.tops['input_sentence'] = self.dummy_data_layer([1, 1000])
    self.n.tops['image_features'] = self.dummy_data_layer([1, 1000, self.num_features])
    self.build_caption(image_data='image_features', t=1)
    self.n.tops['predict'] = self.softmax(self.n.tops['predict-multimodal'], axis=2) 
    
    self.write_net(models_root+save_name)

  
  def build_caption_net_reinfroce(self, feature_param_str, reward=None, reward_param_str=None):
    pass 

