# -*- coding: utf-8 -*-
from subprocess import Popen, PIPE, STDOUT
import os
from PIL import Image
import eyed3
import errno

from sliceSpectrogram import createSlicesFromSpectrograms
from audioFilesTools import isMono, getGenre
from config import rawDataPath
from config import spectrogramsPath
from config import pixelPerSecond

#Tweakable parameters
desiredSize = 128

#Define
currentPath = os.path.dirname(os.path.realpath(__file__)) 

#Remove logs
eyed3.log.setLevel("ERROR")

#Create spectrogram from mp3 files
# def createSpectrogram(filename,newFilename):
# 	#Create temporary mono track if needed
# 	if isMono(rawDataPath+filename):
# 		command = "cp '{}' '/tmp/{}.wav'".format(rawDataPath+filename,newFilename)
# 	else:
# 		command = "sox '{}' '/tmp/{}.wav' remix 1,2".format(rawDataPath+filename,newFilename)
# 	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
# 	output, errors = p.communicate()
# 	if errors:
# 		print errors

# 	#Create spectrogram
# 	filename.replace(".wav","")
# 	command = "sox '/tmp/{}.wav' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(newFilename,pixelPerSecond,spectrogramsPath+newFilename)
# 	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
# 	output, errors = p.communicate()
# 	if errors:
# 		print errors

# 	#Remove tmp mono track
# 	os.remove("/tmp/{}.wav".format(newFilename))

#Create spectrogram from mp3 files
def createClassSpectrogram(filename,classname):
	#Create temporary mono track if needed

	filepath = rawDataPath + classname + "/" + filename


	# if isMono(filepath):
	# 	command = "cp '{}' '/tmp/{}.wav'".format(filepath, filename)
	# else:
	# 	command = "sox '{}' '/tmp/{}.wav' remix 1,2".format(filepath, filename)
	# p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
	# output, errors = p.communicate()
	# if errors:
	# 	print errors

	#Create spectrogram
	command = "sox '{}' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(
				filepath,
				pixelPerSecond,
				spectrogramsPath + classname + "/" + filename.replace(".wav",""))
	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
	output, errors = p.communicate()
	if errors:
		print errors

	#Remove tmp mono track
	# os.remove("/tmp/{}.wav".format(filename))



#Creates .png whole spectrograms from mp3 files
def createSpectrogramsFromAudio():
	genresID = dict()


	classes = next(os.walk(rawDataPath))[1] # get all directories

	for classname in classes:
		print "Processing class {}".format(classname)
		
		rawfiles    = os.listdir(rawDataPath + classname)
		soundfiles  = [file for file in rawfiles if file.endswith(".wav")]
		classlength = len(soundfiles)

		print "Found {} files!".format(classlength)

		classFeaturePath = spectrogramsPath + classname
		# no exists-checking, because the check would introduce a race condition 
		try:
			os.makedirs(os.path.dirname(classFeaturePath))
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise

		#Rename files according to genre
		for index, filename in enumerate(soundfiles):
			print "{}/{} - {}: creating spectrogram for file {}...".format(index+1, classlength, classname, filename)
			createClassSpectrogram(filename, classname)



	# files = [file for file in files if file.endswith(".wav")]
	# nbFiles = len(files)
	# print "Found {} files!".format(nbFiles)

	# #Create path if not existing
	# if not os.path.exists(os.path.dirname(spectrogramsPath)):
	# 	try:
	# 		os.makedirs(os.path.dirname(spectrogramsPath))
	# 	except OSError as exc: # Guard against race condition
	# 		if exc.errno != errno.EEXIST:
	# 			raise

	# #Rename files according to genre
	# for index,filename in enumerate(files):
	# 	print "Creating spectrogram for file {}/{}...".format(index+1,nbFiles)
	# 	fileGenre = getGenre(rawDataPath+filename)
	# 	genresID[fileGenre] = genresID[fileGenre] + 1 if fileGenre in genresID else 1
	# 	fileID = genresID[fileGenre]
	# 	newFilename = fileGenre+"_"+str(fileID)
	# 	createSpectrogram(filename,newFilename)

#Whole pipeline .mp3 -> .png slices
def createSlicesFromAudio():
	print "Creating spectrograms..."
	createSpectrogramsFromAudio()
	print "Spectrograms created!"

	print "Creating slices..."
	createSlicesFromSpectrograms(desiredSize)
	print "Slices created!"