# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random

import torch.utils.data as data

import multiprocessing


class DataLoader(data.Dataset): #自定义数据集DataLoader
    #上面整个部分都是附加功能
    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split,
                                                    self, split == 'train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def read_files(self): #读图片特征
        self.feats_fc = h5py.File(os.path.join(
            self.opt.input_fc_dir, 'feats_fc.h5'), 'r')
        self.feats_att = h5py.File(os.path.join(
            self.opt.input_att_dir, 'feats_att.h5'), 'r')

    def get_data(self, ix):
        self.read_files()
        index = str(self.info['images'][ix]['id'])
        if self.use_att: #以np形式输出图片特征
            return (np.array(self.feats_fc[index]).astype('float32'),
                    np.array(self.feats_att[index]).astype('float32'), ix)
        else:
            return (np.array(self.feats_fc[index]).astype('float32'),
                    np.zeros((1, 1, 1)).astype('float32'), ix)

    #自定义数据集的核心部分
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.use_att = getattr(opt, 'use_att', True)

        # load json file which contains additional information about dataset 导入之前创建的dataset有关的信息
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        # open the hdf5 file 导入事先完成的图像和描述信息
        print('DataLoader loading h5 file: ', opt.input_fc_dir,
              opt.input_att_dir, opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r',
                                       driver='core')

        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length) #输出最长的词的长度
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0] #两个矩阵的长度就代表图片的个数
        print('read %d image features' % (self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])): #第ix个图片
            img = self.info['images'][ix]
            #根据train,val,test先进行编号划分
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0:  # restval
                self.split_ix['train'].append(ix)
        #输出各个集合的图片数目
        print('assigned %d images to split train' % len(self.split_ix['train']))
        print('assigned %d images to split val' % len(self.split_ix['val']))
        print('assigned %d images to split test' % len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {}  # The three prefetch process
        #针对不同的类别,各自导入dataloader
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split,
                                                        self,
                                                        split == 'train')
            # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = []
        att_batch = []
        #设置batch的大小
        label_batch = np.zeros(
            [batch_size * seq_per_img, self.seq_length + 2], dtype='int') #（50,18）
        mask_batch = np.zeros(
            [batch_size * seq_per_img, self.seq_length + 2], dtype='float32')

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            # fetch image 把图片数据输进去，提取特征
            tmp_fc, tmp_att,\
                ix, tmp_wrapped = self._prefetch_process[split].get()
            fc_batch += [tmp_fc] * seq_per_img #list len=5
            att_batch += [tmp_att] * seq_per_img

            # fetch the sequence labels #导入对应的句子信息
            ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1 + 1  # number of captions available for this image 图片对应的描述词个数
            assert ncap > 0, 'an image does not have any label.'

            if ncap < seq_per_img: #如果描述不够,就剩余用0填充
                # we need to subsample (with replacement)
                seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
                for q in range(seq_per_img):
                    ixl = random.randint(ix1, ix2) #从开头到结尾抽一个整数,代表第几个词
                    seq[q, :] = self.h5_label_file['labels'][ixl,
                                                             :self.seq_length]
            else: #若足够或者过多,就按顺序抽5个句子
                ixl = random.randint(ix1, ix2 - seq_per_img + 1)
                seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img,
                                                   :self.seq_length]
            #前面相当于固定了描述词的大小,然后设置batch就会简单很多
            label_batch[i * seq_per_img: (i + 1) * seq_per_img,
                        1: self.seq_length + 1] = seq

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(
                self.h5_label_file['labels'][self.label_start_ix[ix] - 1:
                                             self.label_end_ix[ix]])

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, label_batch))) #找到不为0的有几个，设置mask
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        data = {}
        data['fc_feats'] = np.stack(fc_batch)
        data['att_feats'] = np.stack(att_batch)
        data['labels'] = label_batch
        data['gts'] = gts
        data['masks'] = mask_batch
        data['bounds'] = {'it_pos_now': self.iterators[split],
                          'it_max': len(self.split_ix[split]),
                          'wrapped': wrapped}
        data['infos'] = infos

        return data

    # It's not coherent to make DataLoader a subclass of Dataset,
    # but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according
    # the index. However, it's minimum change to switch to pytorch data loading
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        return self.get_data(ix)

    def __len__(self):
        return len(self.info['images'])


class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name,
        caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases:
        1. not hasattr(self, 'split_loader'): Resume from previous training.
        Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in
         the get_minibatch_inds already.
        """
        # batch_size is 0, the merge is done in DataLoader class
        #用dataloader构建
        sampler = self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]#用来绘制样本数据
        self.split_loader = iter(
            data.DataLoader(dataset=self.dataloader,
                            batch_size=1,
                            sampler=sampler,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=multiprocessing.cpu_count(),#利用cpu线程分成几批来处理
                            collate_fn=lambda x: x[0]))#整理数据

    def _get_next_minibatch_inds(self): #设置下一次的迭代
        max_index = len(self.dataloader.split_ix[self.split]) #总长度
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri] #找到对应那一次的图片编号

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:#是否打乱数据
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader') #如果没有dataset,就reset
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()
        assert tmp[2] == ix, "ix not equal"

        return tmp + [wrapped]
