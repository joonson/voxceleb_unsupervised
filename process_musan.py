#! /usr/bin/python
# -*- encoding: utf-8 -*-
# splits musan clips to chunks of 5 seconds at 3 second interval
# the first argument should be the parent directory of musan_v1

import pdb, glob, os, sys
from scipy.io import wavfile

files = glob.glob('%s/musan/*/*/*.wav'%sys.argv[1])

audlen = 16000*5
audstr = 16000*3

for idx,file in enumerate(files):
	fs,aud = wavfile.read(file)
	writedir = os.path.splitext(file.replace('/musan/','/musan_split/'))[0]
	os.makedirs(writedir)
	for st in range(0,len(aud)-audlen,audstr):
		wavfile.write(writedir+'/%05d.wav'%(st/fs),fs,aud[st:st+audlen])

	print(idx,file)