
"""VSE model"""

import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import OrderedDict
from torch.nn.functional import max_pool1d
import math


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class EncoderImage(nn.Module):

    def __init__(self, opt):
        super(EncoderImage, self).__init__()
        self.embed_size = opt.embed_size
        self.k = opt.k
        self.fc_list = nn.ModuleList([nn.Linear(opt.img_dim, opt.embed_size) for _ in range(opt.k)])
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for fc in self.fc_list:
            r = np.sqrt(6.) / np.sqrt(fc.in_features + fc.out_features)
            fc.weight.data.uniform_(-r, r)
            fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        emb_list = []
        for fc in self.fc_list:
            emb = fc(images)
            emb = emb.permute(0, 2, 1)
            emb = max_pool1d(emb, emb.size(2)).squeeze(2)
            emb = l2norm(emb, dim=-1)
            emb_list.append(emb)

        return emb_list

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)


class EncoderText(nn.Module):

    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.embed_size = opt.embed_size
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.rnn = nn.GRU(opt.word_dim, opt.embed_size, batch_first=True)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out, dim=-1)

        return out


class TripletLoss(nn.Module):

    def __init__(self, opt):
        super(TripletLoss, self).__init__()
        self.margin = opt.margin
        self.k = opt.k
        self.weight = opt.weight
        self.batch_size = opt.batch_size
        self.pos_mask = torch.eye(self.batch_size).cuda()
        self.neg_mask = 1 - self.pos_mask

    def forward(self, v_list, t):

        batch_size = t.size(0)

        if batch_size != self.batch_size:
            pos_mask = torch.eye(batch_size)
            pos_mask = pos_mask.cuda()
            neg_mask = 1 - pos_mask
        else:
            neg_mask = self.neg_mask

        # calculate multi-view similarity score
        scores_list = []
        for v in v_list:
            scores = v.mm(t.t())
            scores_list.append(scores)

        # calculate image embedding similarity
        view_sim = torch.tensor(0)
        if self.k > 1:
            view_sim_list = []
            for i in range(self.k):
                for j in range(i+1, self.k):
                    sims = v_list[i].mm(v_list[j].t())
                    sim = sims.diag().mean()
                    view_sim_list.append(sim)
            view_sim_list = torch.stack(view_sim_list, dim=0)

        # max score
        comb_scores = torch.stack(scores_list, dim=0)
        (max_scores, max_id) = comb_scores.max(0)

        # multi-view up loss
        loss_list = []
        for scores in scores_list:
            pos_scores = scores.diag().view(batch_size, 1)
            pos_scores_t = pos_scores.expand_as(scores)
            pos_scores_v = pos_scores.t().expand_as(scores)
            loss_t = (max_scores - pos_scores_t + self.margin).clamp(min=0)
            loss_v = (max_scores - pos_scores_v + self.margin).clamp(min=0)
            loss_t = loss_t * neg_mask
            loss_v = loss_v * neg_mask
            loss_t = loss_t.max(dim=1)[0]
            loss_v = loss_v.max(dim=0)[0]
            loss_t = loss_t.mean()
            loss_v = loss_v.mean()
            loss = (loss_t + loss_v) / 2
            loss_list.append(loss)

        loss_list = torch.stack(loss_list, dim=0)
        up_loss = loss_list.mean()

        # multi-view low loss
        loss_list = []
        for scores in scores_list:
            max_pos_scores = max_scores.diag().view(batch_size, 1)
            max_pos_scores_t = max_pos_scores.expand_as(scores)
            max_pos_scores_v = max_pos_scores.t().expand_as(scores)
            loss_t = (scores - max_pos_scores_t + self.margin).clamp(min=0)
            loss_v = (scores - max_pos_scores_v + self.margin).clamp(min=0)
            loss_t = loss_t * neg_mask
            loss_v = loss_v * neg_mask
            loss_t = loss_t.max(dim=1)[0]
            loss_v = loss_v.max(dim=0)[0]
            loss_t = loss_t.mean()
            loss_v = loss_v.mean()
            loss = (loss_t + loss_v) / 2
            loss_list.append(loss)

        loss_list = torch.stack(loss_list, dim=0)
        low_loss = loss_list.mean()

        loss = self.weight * up_loss + (1 - self.weight) * low_loss

        return loss, up_loss, low_loss


class UnifiedLoss(nn.Module):

    def __init__(self, opt):
        super(UnifiedLoss, self).__init__()
        self.margin = opt.margin
        self.tau = opt.tau
        self.k = opt.k
        self.weight = opt.weight
        self.batch_size = opt.batch_size
        self.pos_mask = torch.eye(self.batch_size).cuda()
        self.neg_mask = 1 - self.pos_mask

    def forward(self, v_list, t):

        batch_size = t.size(0)

        if batch_size != self.batch_size:
            pos_mask = torch.eye(batch_size)
            pos_mask = pos_mask.cuda()
            neg_mask = 1 - pos_mask
        else:
            pos_mask = self.pos_mask
            neg_mask = self.neg_mask

        # calculate multi-view similarity score
        scores_list = []
        for v in v_list:
            scores = v.mm(t.t())
            scores_list.append(scores)

        # calculate image embedding similarity
        view_sim = torch.tensor(0)
        if self.k > 1:
            view_sim_list = []
            for i in range(self.k):
                for j in range(i+1, self.k):
                    sims = v_list[i].mm(v_list[j].t())
                    sim = sims.diag().mean()
                    view_sim_list.append(sim)
            view_sim_list = torch.stack(view_sim_list, dim=0)

        # max score
        comb_scores = torch.stack(scores_list, dim=0)
        (max_scores, max_id) = comb_scores.max(0)

        # multi-view up loss
        loss_list = []
        for scores in scores_list:
            pos_scores = scores.diag().view(batch_size, 1)
            pos_scores_t = pos_scores.expand_as(scores)
            pos_scores_v = pos_scores.t().expand_as(scores)
            loss_t = max_scores - pos_scores_t + self.margin
            loss_v = max_scores - pos_scores_v + self.margin
            loss_t = loss_t * neg_mask - pos_mask
            loss_v = loss_v * neg_mask - pos_mask
            loss_t = torch.logsumexp(loss_t / self.tau, dim=1) * self.tau
            loss_v = torch.logsumexp(loss_v / self.tau, dim=0) * self.tau
            loss_t = torch.nn.functional.softplus(loss_t, beta=1 / self.tau)
            loss_v = torch.nn.functional.softplus(loss_v, beta=1 / self.tau)
            loss_t = loss_t.mean()
            loss_v = loss_v.mean()
            loss = (loss_t + loss_v) / 2
            loss_list.append(loss)

        loss_list = torch.stack(loss_list, dim=0)
        up_loss = loss_list.mean()

        # multi-view low loss
        loss_list = []
        for scores in scores_list:
            max_pos_scores = max_scores.diag().view(batch_size, 1)
            max_pos_scores_t = max_pos_scores.expand_as(scores)
            max_pos_scores_v = max_pos_scores.t().expand_as(scores)
            loss_t = scores - max_pos_scores_t + self.margin
            loss_v = scores - max_pos_scores_v + self.margin
            loss_t = loss_t * neg_mask - pos_mask
            loss_v = loss_v * neg_mask - pos_mask
            loss_t = torch.logsumexp(loss_t / self.tau, dim=1) * self.tau
            loss_v = torch.logsumexp(loss_v / self.tau, dim=0) * self.tau
            loss_t = torch.nn.functional.softplus(loss_t, beta=1 / self.tau)
            loss_v = torch.nn.functional.softplus(loss_v, beta=1 / self.tau)
            loss_t = loss_t.mean()
            loss_v = loss_v.mean()
            loss = (loss_t + loss_v) / 2
            loss_list.append(loss)

        loss_list = torch.stack(loss_list, dim=0)
        low_loss = loss_list.mean()

        loss = self.weight * up_loss + (1 - self.weight) * low_loss

        return loss, up_loss, low_loss


class VSE(object):

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt)
        self.txt_enc = EncoderText(opt)

        print(self.img_enc)
        print(self.txt_enc)

        total_num = sum(param.numel() for param in self.img_enc.parameters())
        print("Image Encoder Params: %.2fM" % (total_num / 1e6))
        total_num = sum(param.numel() for param in self.txt_enc.parameters())
        print("Text Encoder Params: %.2fM" % (total_num / 1e6))

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.loss = opt.loss
        if self.loss == 'triplet':
            self.criterion = TripletLoss(opt)
        if self.loss == 'unified_max':
            self.criterion = UnifiedLoss(opt)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings
        """
        images = images.cuda()
        captions = captions.cuda()
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)

        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss, loss_1, loss_2, view_sim, percentage = self.criterion(img_emb, cap_emb)
        self.logger.update('L', loss.item(), cap_emb.size(0))
        self.logger.update('L1', loss_1.item(), cap_emb.size(0))
        self.logger.update('L2', loss_2.item(), cap_emb.size(0))

        return loss

    def train_emb(self, images, captions, lengths, ids, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        # compute gradient and do SGD step
        loss.backward()
        clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
