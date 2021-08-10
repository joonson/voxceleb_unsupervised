#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import loadWAV
from ConfModel import *

class SpeakerNet(nn.Module):

    def __init__(self, model, optimizer, scheduler, trainfunc, **kwargs):
        super(SpeakerNet, self).__init__();

        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs).cuda();

        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs).cuda();

        self.__S2E__ = ConfModelBC(num_layers=1, **kwargs).cuda();

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(list(self.__S__.parameters()) + list(self.__L__.parameters()), **kwargs);
        self.__optimizerS2E__ = Optimizer(self.__S2E__.parameters(), **kwargs);

        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        assert self.lr_step in ['epoch', 'iteration']
        self.torchfb    = transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=kwargs['n_mels']).cuda();

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Train network
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, alpha=1, num_steps=1):

        self.train();

        stepsize = loader.batch_size;

        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0;    # EER or accuracy

        criterion   = torch.nn.CrossEntropyLoss()
        conf_labels = torch.LongTensor([1]*stepsize+[0]*stepsize).cuda()

        tstart = time.time()
        
        for data in loader:

            self.zero_grad();

            data = data.transpose(0,1)

            feat = []
            for inp in data:
                outp      = self.__S__.forward(torch.FloatTensor(inp).cuda())
                feat.append(outp)

            feat = torch.stack(feat,dim=1).squeeze()

            if alpha > 0:

                out_a_ = feat[:,0,:].detach()
                out_s_ = feat[:,1,:].detach()
                out_p_ = feat[:,2,:].detach()

                # # ==================== TRAIN DISCRIMINATOR ====================

                for ii in range(0,num_steps):

                    conf_input = torch.cat((torch.cat((out_a_,out_s_),1),torch.cat((out_a_,out_p_),1)),0)

                    conf_output = self.__S2E__(conf_input)

                    dloss1  = criterion(conf_output, conf_labels)

                    dloss1.backward();
                    self.__optimizerS2E__.step();
                    self.__optimizerS2E__.zero_grad();

                    # print(dloss1)

                # # ==================== TRAIN NORMAL AND BACKPROP THROUGH DISCRIMINATOR ====================

                conf_input = torch.cat((torch.cat((feat[:,0,:],feat[:,1,:]),1),torch.cat((feat[:,0,:],feat[:,2,:]),1)),0)

                conf_output = self.__S2E__(conf_input)

                conf_loss   = criterion(conf_output, conf_labels)

                nloss, prec1 = self.__L__.forward(feat[:,[0,2],:],None)

                nloss += conf_loss * alpha

            else:
                conf_loss   = 0
                nloss, prec1 = self.__L__.forward(feat,None)

            loss    += nloss.detach().cpu();
            top1    += prec1
            counter += 1;
            index   += stepsize;

            nloss.backward();
            self.__optimizer__.step();

            telapsed = time.time() - tstart
            tstart = time.time()

            sys.stdout.write("\rProcessing (%d) "%(index));
            sys.stdout.write("Loss %f EER/TAcc %2.3f%% - %.2f Hz "%(loss/counter, top1/counter, stepsize/telapsed));
            sys.stdout.flush();

            if self.lr_step == 'iteration': self.__scheduler__.step()

        if self.lr_step == 'epoch': self.__scheduler__.step()

        sys.stdout.write("\n");
        
        return (loss/counter, top1/counter);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, listfilename, print_interval=100, test_path='', num_eval=10, eval_frames=None):
        
        self.eval();
        
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()

        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line): 
                    break;

                data = line.split();

                ## Append random label if missing
                if len(data) == 2: data = [random.randint(0,1)] + data

                files.append(data[1])
                files.append(data[2])
                lines.append(line)

        setfiles = list(set(files))
        setfiles.sort()

        ## Save all features to file
        for idx, file in enumerate(setfiles):

            inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()

            with torch.no_grad():

                feat = self.torchfb(inp1)

                ref_feat = self.__S__.forward(feat).detach().cpu()

            filename = '%06d.wav'%idx

            feats[file]     = ref_feat

            telapsed = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));

        print('')
        all_scores = [];
        all_labels = [];
        all_trials = [];
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split();

            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0,1)] + data

            ref_feat = feats[data[1]].cuda()
            com_feat = feats[data[2]].cuda()

            if self.__L__.test_normalize:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();

            score = -1 * numpy.mean(dist);

            all_scores.append(score);  
            all_labels.append(int(data[0]));
            all_trials.append(data[1]+" "+data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed));
                sys.stdout.flush();

        print('\n')

        return (all_scores, all_labels, all_trials);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        
        torch.save(self.state_dict(), path);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.state_dict();
        loaded_state = torch.load(path);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);

