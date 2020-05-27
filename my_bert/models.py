# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:28:04 2020

@author: huzhen
"""

import numpy as np
from my_bert.layers import *
from my_bert.snippets import delete_arguments
from keras.models import Model
import json


class Transformer(object):#各种bert系列模型的基类
    def __init__(self,
                 vocab_size,              #词表大小
                 hidden_size,             #编码维度
                 num_hidden_layers,       #transformer层数
                 num_attention_heads,     #Attention头数
                 intermediate_size,       #FeedForward隐层维度
                 hidden_act,              #FeedForward隐层激活函数
                 dropout_rate,            #dropout比例
                 embedding_size=None,     #是否制定embedding_size
                 attention_key_size=None, #Attention中Q,K的head_size
                 sequence_length=None,    #是否固定序列的长度
                 keep_tokens=None,        #要保留的词ID列表
                 layers=None,             #外部传入的layer层
                 name=None,               #模型name
                 **kwargs
                 ):
        if keep_tokens is None:
            self.vocab_size = vocab_size
        else:
            self.vocab_size = len(keep_tokens)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // self.num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.attention_mask = None
        self.position_bias = None
        self.layers = {} if layers is None else layers  #猜测，为了方便重复调用层
        self.name = name
        self.built = False
        
    def build(self,
              layer_norm_cond=None,
              layer_norm_cond_hidden_size=None,
              layer_norm_cond_hidden_act=None,
              additional_input_layers=None,
              **kwargs
              ):
        if self.built == True:
            return None
        inputs = self.get_inputs()
        self.set_inputs(inputs,additional_input_layers)
        self.layer_norm_conds = [
                                    layer_norm_cond,
                                    layer_norm_cond_hidden_size,
                                    layer_norm_cond_hidden_act or 'linear'
                              ]
        outputs = self.call(inputs)
        self.set_outputs(outputs)
        self.model = Model(self.inputs,self.outputs,name=self.name)
        self.build = True
        
    def call(self,inputs):
        outputs = self.apply_embedding(inputs)
        for i in range(self.num_hidden_layers):
            outputs = self.apply_main_layers(outputs,i)
        outputs = self.apply_final_layers(outputs)
        
        return outputs
        
        
    def apply(self,inputs,layer=None,arguments=None,**kwargs):
        if layer is Dropout and self.dropout_rate == 0:
            return inputs
        
        arguments = arguments or {}
        
        name = kwargs['name']
        if name not in self.layers:
            layer = layer(**kwargs)
            name = layer.name
            self.layers[name] = layer
        return self.layers[name](inputs,**arguments)
    
        
        
    def apply_embeddings(self,inputs):
        raise NotImplementedError
        
    def apply_main_layers(self,inputs,index):
        raise NotImplementedError
    
    def apply_final_layers(self,inputs):
        raise NotImplementedError
        
    def compute_attention_mask(self,inputs=None):
        return self.attention_mask
    
    def compute_position_bias(self,inputs=None):
        return self.position_bias
    
    @property
    def initializer(self):
        return keras.initializers.TruncatedNormal(stddev=0.02)
        
    def get_inputs(self):
        raise NotImplementedError
        
    def set_inputs(self,inputs,additional_input_layers=None):
        if inputs is None:
            inputs = []
        elif isinstance(inputs,list):
            inputs = [inputs]
        inputs = inputs[:]
        
        if additional_input_layers is not None:
            if not isinstance(additional_input_layers,list):
                additional_input_layers = [additional_input_layers]
            inputs.extend(additional_input_layers)
        self.inputs = inputs
        if len(inputs) > 1:
            self.input = inputs
        else:
            self.input = inputs[0]
        
    def set_outputs(self,outputs):
        if not isinstance(outputs,list):
            outputs = [outputs]
        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs) > 1:
            self.output = outputs
        else:
            self.output = outputs[0]
        
    def simplify(self,inputs):
        inputs = [i for i in inputs if i is not None]
        if len(inputs) == 1:
            inputs = inputs[0]
        return inputs
    
    def load_variable(self,checkpoint,name):
        tf.train.load_variable(checkpoint,name)
        
    def create_variable(self,name,value):
        return tf.Variable(value,name=name)
    
    def variable_mapping(self):
        """
        构建keras层与checkpoint变量名之间的映射
        """
        return {}
    
    