#Define paths for files
spectrogramsPath = "Data/Spectrograms/"
slicesPath = "Data/Slices/"
datasetPath = "Data/Dataset/"
rawDataPath = "Data/Raw/"

checkpoint_path = "models"

#Spectrogram resolution
pixelPerSecond = 50

#Slice parameters
sliceSize = 128

#Dataset parameters
filesPerGenre = 59
validationRatio = 0.3
testRatio = 0.1

#Model parameters
batchSize = 25
learningRate = 0.001
nbEpoch = 3000
