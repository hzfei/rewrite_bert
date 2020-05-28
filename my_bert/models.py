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
    
    def load_weights_from_checkpoint(self,checkpoint,mapping=None):
        mapping = mapping or self.variable_mapping()
        mapping = {k:v for k,v in mapping.items() if k in self.layers}
        
        weights_value_pairs = []
        
        for layer,variables in mapping.items():
            layer = self.layers[layer]
            weights = layer.traiable_weights
            values = [self.load_variable(checkpoint,v) for v in variables]
            
            if isinstance(layer, MultiHeadAttention):#这个以后再看...
                """如果key_size不等于head_size，则可以通过
                正交矩阵将相应的权重投影到合适的shape。
                """
                count = 2
                if layer.use_bias:
                    count += 2
                heads = self.num_attention_heads
                head_size = self.attention_head_size
                key_size = self.attention_key_size
                W = np.linalg.qr(np.random.randn(key_size, head_size))[0].T
                if layer.attention_scale:
                    W = W * key_size**0.25 / head_size**0.25
                for i in range(count):
                    w, v = weights[i], values[i]
                    w_shape, v_shape = K.int_shape(w), v.shape
                    if w_shape[-1] != v_shape[-1]:
                        pre_shape = w_shape[:-1]
                        v = v.reshape(pre_shape + (heads, head_size))
                        v = np.dot(v, W)
                        v = v.reshape(pre_shape + (heads * key_size,))
                        values[i] = v
            
            weights_value_pairs.extend(zip(weights,values))
        K.batch_set_value(weights_value_pairs)
        
    def save_weights_as_checkpoint(self, filename, mapping=None):#以后再看这个方法...
        """根据mapping将权重保存为checkpoint格式
        """
        mapping = mapping or self.variable_mapping()
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        with tf.Graph().as_default():
            for layer, variables in mapping.items():
                layer = self.layers[layer]
                values = K.batch_get_value(layer.trainable_weights)
                for name, value in zip(variables, values):
                    self.create_variable(name, value)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.save(sess, filename, write_meta_graph=False)

class Bert(Transformer):
    def __init__(self,
                 max_position,
                 with_pool=False,
                 with_nsp=False,
                 with_mlm=False,
                 custom_position_ids=False,
                 **kwargs):
        super(Bert,self).__init__(**kwargs)
        self.max_position = max_position
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.custom_position_ids = custom_position_ids
    
    def get_inputs(self):
        x_in = Input(shape=(self.sequence_length,),name='Input-Token')
        y_in = Input(shape=(self.sequence_length,),name='Input-Segment')
        if self.custom_position_ids:
            p_in = Input(shape=(self.sequence_length,),name='Input-Position')
            return [x_in,y_in,p_in]
        return [x_in,y_in]
    
    def apply_embedding(self,inputs):
        x,s = inputs[:2]
        z = self.layer_norm_conds[0]
        if self.custom_position_ids:
            p = inputs[2]
        else:
            p = None
        x = self.apply(
                        inputs=x,
                        layer=Embeddding,
                        input_dim = self.vocab_size,
                        output_dim = self.embedding_size,
                        embedding_initializer = self.initializer,
                        mask_zero=True,
                        name='Embedding-Token'
                       )
        
        s = self.apply(
                        inputs=s,
                        layer=Embedding,
                        input_dim = 2,
                        output_dim = self.embedding_size,
                        embedding_initializer = self.initializer,
                        name = 'Embedding-Segment'
                      )
        
        x = self.apply(inputs=[x,s],layer=Add,name='Embedding-Token_Segment')
        
        x = self.apply(
                        inputs=x,
                        layer=PositionEmbedding,
                        input_dim=self.max_position,
                        output_dim = self.embedding_size,
                        merge_mode = 'add',
                        embedding_initializer = self.initializer,
                        custom_position_ids = self.custom_position_ids,
                        name='Embedding-Position'
                       )
        
        x = self.apply(
                        inputs=x,
                        layer=LayerNormalization,
                        conditional=(z is not None),
                        hidden_units=self.layer_norm_conds[1],
                        hidden_activation=self.layer_norm_conds[2],
                        hidden_initializer=self.initializer,
                        name='Embedding-Norm'
                      )
        
        x = self.apply(
                        inputs=x,
                        layer=Dropout,
                        rate=self.dropout_rate,
                        name='Embedding-Dropout'
                
                     )
        
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                            inputs=x,
                            layers=Dense,
                            units=self.hidden_size,
                            kernel_initializer=self.initializer,
                            name='Embedding-Mapping'
                          )
        return x
    
    def apply_main_layers(self,inputs,index):
        x = inputs
        z = self.layer_norm_conds[0]
        
        attention_name = 'Transfomer-{}-MultiHeadSelfAttention'.format(index)
        feed_forward_name = 'Transformer-{}-FeedForward'.format(index)
        attention_mask = self.compute_attention_mask(index)
        
        xi,x,arguments = x,[x,x,x],{'a_mask':None}
        
        if attention_mask is not None:
            arguments['a_mask'] = True
            x.append(attention_mask)
        
        x = self.apply(
                        inputs=x,
                        layer=MultiHeadAttention,
                        arguments=arguments,
                        heads=self.num_attention_heads,
                        head_size = self.attention_head_size,
                        key_size=self.attention_key_size,
                        kernel_initializer=self.initializer,
                        name=attention_name
                      )
        
        x = self.apply(
                        inputs=x,
                        layer=Dropout,
                        rate=self.dropout_rate,
                        name='{}-Dropout'.format(attention_name)
                      )
        
        x = self.apply(
                        inputs=[xi,x],
                        layer=Add,
                        name='{}-Add'.format(attention_name)
                    )
        
        x = self.apply(
                        inputs=self.simplify([x,z]),
                        layer=LayerNormalization,
                        layer=LayerNormalization,
                        conditional=(z is not None),
                        hidden_units=self.layer_norm_conds[1],
                        hidden_activation=self.layer_norm_conds[2],
                        hidden_initializer=self.initializer,
                        name='%s-Norm' % attention_name
                        )
        
        xi = x
        
        x = self.apply(
                        inputs=x,
                        layer=FeedForward,
                        units=self.intermediate_size,
                        activation=self.hidden_act,
                        kernel_initializer=self.initializer,
                        name=feed_forward_name
                       )
        
        x = self.apply(
                        inputs=x,
                        layer=Dropout,
                        rate=self.dropout_rate,
                        name="{}-Dropout".format(feed_forward_name)
                       )
        
        x = self.apply(
                        inputs=[xi,x],
                        layer=Add,
                        name='{}-Add'.format(feed_forward_name)
                      )
        
        x = self.apply(
                        inputs=self.simplify([x, z]),
                        layer=LayerNormalization,
                        conditional=(z is not None),
                        hidden_units=self.layer_norm_conds[1],
                        hidden_activation=self.layer_norm_conds[2],
                        hidden_initializer=self.initializer,
                        name='%s-Norm' % feed_forward_name
                     )
        
        return x
    
    def apply_final_layers(self,inputs):
        x = inputs
        z = self.layer_norm_conds[0]
        outputs = [x]
        
        if self.with_pool or self.with_nsp:
            x = outputs[0]
            x = self.apply(
                            inputs=x,
                            layer=Lambda,
                            function=lambda x:x[:,0],
                            name='Pooler'
                            )
            pool_activation = 'tanh' if self.with_pool is True else self.with_pool
            
            x = self.apply(
                            inputs=x,
                            layer=Dense,
                            units=self.hidden_size,
                            activation=pool_activation,
                            kernel_initializer=self.initializer,
                            name='Pooler-Dense'
                         )
            
            if self.with_nsp:
                x = self.apply(
                                inputs=x,
                                layer=Dense,
                                units=2,
                                activation='softmax',
                                kernel_initializer=self.initializer,
                                name='NSP-Proba'
                              )
            outputs.append(x)