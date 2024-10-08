#------------------------------------------------------------

import subprocess
import logging
import zipfile
import sys, os

os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"


sys.path.append('./help_scripts')
sys.path.append('./dataset')

import help_scripts.utils as utils
import dataset.preprocessing_audiofeats as audioprocess
import dataset.preprocessing_videofeats as videoprocess
import progressbar

#------------------------------------------------------------

trainDataDownloader = './help_scripts/train_val_getDataDirect.py'
testDataDownloader = './help_scripts/test_getDataDirect.py'
audioPreprocessor = './dataset/preprocessing_audiofeats.py'
videoreprocessor = './dataset/preprocessing_videofeats.py'

#----------------------------------
# -- prepare the logger
FORMAT = '%(asctime)-15s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
fileHandler = logging.FileHandler("{0}/{1}.log".format('.', 'setup'))
fileHandler.setFormatter(logFormatter)
fileHandler.setLevel(0)
logger.setLevel(0)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
consoleHandler.setLevel(0)
logger.addHandler(consoleHandler)
logger.info('downloading the training and validation data')

# download the training, validation, test data
#subprocess.call(['python', trainDataDownloader])
#subprocess.call(['python', testDataDownloader])

# extract the zip files of train, validation, test to appropriate directories
# folders = ['train', 'validation'] #, 'test']
folders = ['train']

def newProgressBar():
	bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ', 
										progressbar.Bar(),
										' (', progressbar.ETA(), ') ',
							])
	return bar

for folder in folders:
	destfolder = os.path.join("./dataset", folder)
	passwd = ''
	# if folder == 'test':
	# 	passwd = '.chalearnLAPFirstImpressionsFirstRoundECCVWorkshop2016.'
		
	# allfiles = os.listdir(os.path.join("./data", folder + 'zip'))
	allfiles = os.listdir(os.path.join("./dataset", folder))
	bar = newProgressBar()
	
	logger.info('processing folder ' + folder)
	for file in bar(allfiles):
		if file.endswith(".zip"):
			# currentzipfile = os.path.join("./data", folder + 'zip', file)
			currentzipfile = os.path.join("./dataset", folder, file)
			filename = os.path.splitext(os.path.basename(currentzipfile))[0]
			utils.mkdirs(os.path.join(destfolder, filename))
			with zipfile.ZipFile(currentzipfile,"r") as zip_ref:
				zip_ref.extractall(os.path.join(destfolder, filename), pwd=passwd)
				
	audioprocess.audioPreprocess(destfolder)
	# videoprocess.videoPreprocess(destfolder)

#subprocess.call(['python', audioPreprocessor])
#subprocess.call(['python', videoreprocessor])
