# Import Pillow:
from PIL import Image
import os.path

from config import spectrogramsPath, slicesPath

#Slices all spectrograms
def createSlicesFromSpectrograms(desiredSize):
	# for filename in os.listdir(spectrogramsPath):
	# 	if filename.endswith(".png"):
	# 		sliceSpectrogram(filename,desiredSize)

	classes = next(os.walk(spectrogramsPath))[1] # get all directories
	for classname in classes:
		for filename in os.listdir(spectrogramsPath + classname):
			if filename.endswith(".png"):
				sliceSpectrogram(filename, desiredSize, classname)

#Creates slices from spectrogram
#TODO Improvement - Make sure we don't miss the end of the song
def sliceSpectrogram(filename, desiredSize,classname):
#	genre = filename.split("_")[0] 	#Ex. Dubstep_19.png

	# Load the full spectrogram
	img = Image.open(spectrogramsPath + classname + "/" + filename)

	#Compute approximate number of 128x128 samples
	width, height = img.size
	nbSamples = int(width/desiredSize)

	# #Create path if not existing
	# slicePath = slicesPath+"{}/".format(genre);
	# if not os.path.exists(os.path.dirname(slicePath)):
	# 	try:
	# 		os.makedirs(os.path.dirname(slicePath))
	# 	except OSError as exc: # Guard against race condition
	# 		if exc.errno != errno.EEXIST:
	# 			raise

	filename = filename.replace(".png", "")
	filename = filename.replace(".wav", "")
	
	for i in range(nbSamples):
		#Extract and save 128x128 sample
		startPixel = i*desiredSize
		imgTmp = img.crop((startPixel, 1, startPixel + desiredSize, desiredSize + 1))
		imgTmp.save(slicesPath + classname + "/" + filename + "_slice_{}.png".format(i))

	print "Created {} slices in class {} for {}".format(nbSamples, classname, filename)
