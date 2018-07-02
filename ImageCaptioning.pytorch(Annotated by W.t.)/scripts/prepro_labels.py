# -*- coding: utf-8 -*-
"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""
#几大功能总结:
# 建立vocab,包含词频超过阈值的单词,同时对剩余的词的个数等进行计数,可以查看
# 对caption进行编码,最后形成行数为总的单词个数的数组,同时记录每幅图片起始和终止词的位置

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
import skimage.io

def build_vocab(imgs, params):
  #把参数里有关词频的量导进来
  #imgs里包含图片特征和描述
  count_thr = params['word_count_threshold']

  # count up the number of words   数了所有的词及对应个数
  counts = {}
  for img in imgs:
    for sent in img['sentences']: #5个句子中的一个
      for w in sent['tokens']:#每个句子中的一个个词
        counts[w] = counts.get(w, 0) + 1#数所有句子中相同词的有多少个

  cw = sorted([(count,w) for w,count in counts.items()], reverse=True) #按词频从大到小排序
  print('top words and their counts:')
  print('\n'.join(map(str,cw[:20]))) #前20个 换成字符 一个一个竖着输出来

  # print some stats
  total_words = sum(counts.values()) #计算总共有多少词
  print('total words:', total_words)
  bad_words = [w for w,n in counts.items() if n <= count_thr] #词个数小于阈值的 bad_words
  vocab = [w for w,n in counts.items() if n > count_thr]  #个数大于阈值的 vocab
  bad_count = sum(counts[w] for w in bad_words) #对bad_words里的词的出现次数求和
  #bad_words个数 占比
  print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
  #vocab中词个数
  print('number of words in vocab would be %d' % (len(vocab), ))
  #bad_words词频数 占比
  print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))


  # lets look at the distribution of lengths as well
  sent_lengths = {}  #数句子的长度,保存在字典里
  for img in imgs:
    for sent in img['sentences']: #5个句子中的一个
      txt = sent['tokens']
      nw = len(txt) #得到的句子有几个词
      sent_lengths[nw] = sent_lengths.get(nw, 0) + 1 #特定的句子长度数各有几个句子
  max_len = max(sent_lengths.keys()) #最长的句子长度
  print('max length sentence in raw data: ', max_len)
  print('sentence length distribution (count, number of words):')
  sum_len = sum(sent_lengths.values()) #总共句子个数
  for i in range(max_len+1): #输出各种长度的句子有多少个占比如何
    print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

  # lets now produce the final annotations
  if bad_count > 0: #如果有词频数不够的词的话，就统一用UNK代替
    # additional special UNK token we will use below to map infrequent words to
    print('inserting the special UNK token')
    vocab.append('UNK')
  
  for img in imgs: #对每张图片的5句子,看没个词是不是超过阈值,超过就放到caption,没超过就放UNK
    img['final_captions'] = []
    for sent in img['sentences']:
      txt = sent['tokens']
      caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
      img['final_captions'].append(caption)

  return vocab

def encode_captions(imgs, params, wtoi):
  """ 
  encode all captions into one large array, which will be 1-indexed.
  also produces label_start_ix and label_end_ix which store 1-indexed 
  and inclusive (Lua-style) pointers to the first and last caption for
  each image in the dataset.
  """

  max_length = params['max_length']
  N = len(imgs) #图片个数
  M = sum(len(img['final_captions']) for img in imgs) # total number of captions

  label_arrays = []
  label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
  label_end_ix = np.zeros(N, dtype='uint32')
  label_length = np.zeros(M, dtype='uint32')
  caption_counter = 0
  counter = 1
  for i,img in enumerate(imgs):#对每个图片
    n = len(img['final_captions'])#n是每个图片描述个数 如果有n<=0的,输出错误
    assert n > 0, 'error: some image has no captions'
    Li = np.zeros((n, max_length), dtype='uint32')
    for j,s in enumerate(img['final_captions']):#每个描述
      label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence 每个单词的长度
      caption_counter += 1
      for k,w in enumerate(s):
        if k < max_length:
          Li[j,k] = wtoi[w] #第j个caption的第k个单词, 把这个单词的编码存在li里

    # note: word indices are 1-indexed, and captions are padded with zeros
    label_arrays.append(Li) #由矩阵构成的list
    #记录第i个图片开始和结束词的位置
    label_start_ix[i] = counter
    label_end_ix[i] = counter + n - 1
    
    counter += n
  
  L = np.concatenate(label_arrays, axis=0) # put all the labels together 数组纵向拼接 L（616767， 16）
  assert L.shape[0] == M, 'lengths don\'t match? that\'s weird' #行数为总的caption数
  assert np.all(label_length > 0), 'error: some caption had no words?'

  print('encoded captions to array of size ', L.shape)
  return L, label_start_ix, label_end_ix, label_length


def main(params):

  imgs = json.load(open(params['input_json'], 'r'))
  imgs = imgs['images']

  seed(123) # make reproducible
  
  # create the vocab #建立字典
  vocab = build_vocab(imgs, params)
  itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table #从1开始,每个数对应一个词
  wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table #反过来,每个词对应一个数
  
  # encode captions in large arrays, ready to ship to hdf5 file
  L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

  # create output h5 file 输出caption的相关信息
  N = len(imgs)
  f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
  f_lb.create_dataset("labels", dtype='uint32', data=L)
  f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
  f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
  f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
  f_lb.close()

  # create output json file
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab 存储图像到词
  out['images'] = []
  for i,img in enumerate(imgs):
    
    jimg = {}
    jimg['split'] = img['split']
    if 'filename' in img: jimg['file_path'] = os.path.join(img['filepath'], img['filename']) # copy it over, might need
    if 'cocoid' in img: jimg['id'] = img['cocoid'] # copy over & mantain an id, if present (e.g. coco ids, useful)
    
    out['images'].append(jimg) #把没个图的编号,路径等作为一个字典导入
  
  json.dump(out, open(params['output_json'], 'w')) #把输出存入到output_json
  print('wrote ', params['output_json'])

if __name__ == "__main__":

  parser = argparse.ArgumentParser() #设置参数解析器，规定params的各项参数

  # input json
  parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
  parser.add_argument('--output_json', default='data.json', help='output json file')
  parser.add_argument('--output_h5', default='data', help='output h5 file')

  # options
  parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict 返回设置的这些参数对应的属性
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)
