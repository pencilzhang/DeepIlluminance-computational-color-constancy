# -------------------------------------------------------------------
# Create the LMDB for the data instances
# Both train and validation lmdbs can be created using this 
# The file is adapted from BVLC Caffe, and requires Caffe tools
# Author: Sukrit Shankar 
# -------------------------------------------------------------------

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Please set the appropriate paths
EXAMPLE=LMDB      		# Path where the output LMDB is stored
DATA=list/train_list/ 			# Path where the data.txt file is present
TOOLS=/root/caffe-master/build/tools   				# Caffe dependency to access the convert_imageset utility 
DATA_ROOT=data/bright_dark_pixel_patch/    	# Path prefix for each entry in data.txt
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

# ----------------------------
# Checks for DATA_ROOT Path
if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: DATA_ROOT is not a path to a directory: $DATA_ROOT"
  echo "Set the DATA_ROOT variable to the path where the data instances are stored."
  exit 1
fi

# ------------------------------
# Creating LMDB
 echo "Creating data lmdb..."
 GLOG_logtostderr=1 $TOOLS/convert_imageset \
    $DATA_ROOT \
    $DATA/all.txt \
    $EXAMPLE/all_lmdb

# ------------------------------
echo "Done."



