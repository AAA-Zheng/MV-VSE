from __future__ import print_function
import os
import pickle

import numpy
from data import get_test_loader
import time
import numpy as np
from vocab import Vocabulary, deserialize_vocab  # NOQA
import torch
from model import VSE
from collections import OrderedDict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    img_embs_list = []
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            images, captions, lengths, ids = batch_data
            # make sure val logger is used
            model.logger = val_logger

            # compute the embeddings
            img_emb_list, cap_emb = model.forward_emb(images, captions, lengths)

            for j, img_emb in enumerate(img_emb_list):
                if img_embs is None:
                    if img_emb.dim() == 3:
                        img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
                    else:
                        img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))

                    for _ in img_emb_list:
                        img_embs_list.append(img_embs.copy())

                img_embs_list[j][ids] = img_emb.data.cpu().numpy().copy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if cap_embs is None:
                cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
            cap_embs[ids, :] = cap_emb.data.cpu().numpy().copy()

            if i % log_step == 0:
                logging('Test: [{0}/{1}]\t'
                        '{e_log}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                    i, len(data_loader.dataset) // data_loader.batch_size + 1, batch_time=batch_time,
                    e_log=str(model.logger)))
            del images, captions

    return img_embs_list, cap_embs


def evalrank(model_path, data_path=None, split='dev', fold5=False, save_path=None):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # construct model
    model = VSE(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs_list, cap_embs = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs_list[0].shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        start = time.time()
        sims_list = []
        for img_embs in img_embs_list:
            img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
            sims = compute_sim(img_embs, cap_embs)
            sims_list.append(sims)
        sims = np.array(sims_list)
        sims = np.max(sims, axis=0)

        diff = sims_list[0] - sims_list[1]
        view1 = np.where(diff > 0, np.ones_like(diff), np.zeros_like(diff))
        view1 = view1.sum() / 5000000
        print(view1, 1 - view1)

        npts = cap_embs.shape[0] // 5

        if save_path is not None:
            np.save(save_path, {'npts': npts, 'sims': sims})
            print('Save the similarity into {}'.format(save_path))

        end = time.time()
        print("calculate similarity time: {}".format(end - start))

        # no cross-validation, full evaluation
        r, rt = i2t(npts, sims, return_ranks=True)
        ri, rti = t2i(npts, sims, return_ranks=True)

        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
        print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            start = time.time()
            sims_list = []
            for img_embs in img_embs_list:
                img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
                img_embs_shard = img_embs[i * 1000:(i + 1) * 1000]
                cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
                sims = compute_sim(img_embs_shard, cap_embs_shard)
                sims_list.append(sims)
            sims = np.array(sims_list)
            sims = np.max(sims, axis=0)
            end = time.time()
            print("calculate similarity time: {}".format(end - start))

            npts = img_embs_shard.shape[0]
            r, rt0 = i2t(npts, sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(npts, sims, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f" % rsum)
            results += [list(r) + list(ri) + [rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % mean_metrics[10])
        print("Image to text: %.1f, %.1f, %.1f, %.1f %.1f" %
              mean_metrics[:5])
        print("Text to image: %.1f, %.1f, %.1f, %.1f %.1f" %
              mean_metrics[5:10])


def compute_sim(images, captions):
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities


def i2t(npts, sims, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(npts, sims, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

