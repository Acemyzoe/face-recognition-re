#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
__author__ = 'Acemyzoe'
'''
tf_model to tensorrt
'''
import tensorflow as tf

def trt(path):
    converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=path)
    converter.convert()
    converter.save(path+'_opt')

if __name__ =='__main__':
    trt('./face_model')
