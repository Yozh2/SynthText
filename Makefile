# Data paths for directories
DATA_PATH=data
IMAGES_PATH=$(DATA_PATH)/images
IMAGES_RAW_PATH=$(IMAGES_PATH)/raw
DEPTHS_PATH=$(IMAGES_PATH)/depths
SEGS_PATH=$(IMAGES_PATH)/segs
DEPTHS_PATH=$(IMAGES_PATH)/labels

# Paths for datasets
IMAGES_DATASET_PATH=$(IMAGES_RAW_PATH)
DEPTHS_DATASET_PATH=$(DEPTHS_PATH)/depths.h5
SEGS_DATASET_PATH=$(SEGS_PATH)/segs.h5
LABELS_DATASET_PATH=$(LABELS)/labels.h5

# Paths for preparation script directories
DEPTH_NN_DIR=prep_scripts/FCRN_depth_prediction
SEG_NN_DIR=prep_scripts/pytorch_hed
LABEL_DIR=prep_sctipts
COLLECT_DIR=prep_scripts

# Names of executables
DEPTH_EXECUTABLE=fcrn_predict.py
SEG_EXECUTABLE=hed.py
LABEL_EXECUTABLE=floodFill.py
COLLECT_EXECUTABLE=collect_dataset.py

all: clear prepare collect

clear: clear_depths clear_segs clear_labels

clear_images:
	rm -rf $(IMAGES_DATASET_PATH)

clear_depths:
	rm -rf $(DEPTHS_DATASET_PATH)

clear_segs:
	rm -rf $(SEGS_DATASET_PATH)

clear_labels:
	rm -rf $(LABELS_DATASET_PATH)

prepare: prepare_depth prepare_seg prepare_label

prepare_depth:
	cd $(DEPTH_NN_DIR) && python $(DEPTH_EXECUTABLE) -v

prepare_seg:
	cd $(SEG_NN_DIR) && python $(SEG_EXECUTABLE) -v

prepare_label:
	cd $(LABEL_DIR) && python $(LABEL_EXECUTABLE) -v

collect:
	cd $(COLLECT_DIR) && python $(COLLECT_EXECUTABLE) -v
