# Flags (comment values to disable flags)
VERBOSE= -v
VISUAL= #--viz

# Data paths for directories
SYNTHTEXT_DATA_PATH=./data
DATA_PATH=./data/images
IMAGES_PATH=$(DATA_PATH)
IMAGES_RAW_PATH=$(IMAGES_PATH)/raw 
DEPTHS_PATH=$(IMAGES_PATH)/depths
SEGS_PATH=$(IMAGES_PATH)/segs
LABELS_PATH=$(IMAGES_PATH)/labels
RESULTS_PATH=$(DATA_PATH)/results

# Paths for datasets
TXT_DATASET_PATH=$(SYNTHTEXT_DATA_PATH)/newsgroup/newsgroup.txt
IMAGES_DATASET_PATH=$(IMAGES_RAW_PATH)
DEPTHS_DATASET_PATH=$(DEPTHS_PATH)/depths.h5
SEGS_DATASET_PATH=$(SEGS_PATH)/segs.h5
LABELS_DATASET_PATH=$(LABELS_PATH)/labels.h5
INPUT_DATASET_PATH=$(DATA_PATH)/dset.h5
OUTPUT_DATASET_PATH=$(RESULTS_PATH)/SynthText.h5

# Paths for preparation script directories
DEPTH_NN_DIR=prep_scripts/FCRN_depth_prediction
SEG_NN_DIR=prep_scripts/pytorch_hed
LABEL_DIR=prep_scripts
COLLECT_DIR=prep_scripts

# Names of executables
DEPTH_EXECUTABLE=fcrn_predict.py
SEG_EXECUTABLE=hed.py
LABEL_EXECUTABLE=floodFill.py
COLLECT_EXECUTABLE=collect_dataset.py

dataset: clear prepare collect

all: clear prepare collect run

clean: clear

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
	python $(DEPTH_NN_DIR)/$(DEPTH_EXECUTABLE) $(VERBOSE) --inp $(IMAGES_RAW_PATH) --out $(DEPTHS_PATH)

prepare_seg:
	python $(SEG_NN_DIR)/$(SEG_EXECUTABLE) $(VERBOSE) --inp $(IMAGES_RAW_PATH) --out $(SEGS_PATH)

prepare_label:
	python $(LABEL_DIR)/$(LABEL_EXECUTABLE) $(VERBOSE) --inp $(SEGS_DATASET_PATH) --out $(LABELS_DATASET_PATH)

collect:
	python $(COLLECT_DIR)/$(COLLECT_EXECUTABLE) --images $(IMAGES_RAW_PATH) --depths $(DEPTHS_DATASET_PATH) --labels $(LABELS_DATASET_PATH) --out $(INPUT_DATASET_PATH)

run:
	python gen.py $(VISUAL) --inp $(INPUT_DATASET_PATH) --out $(OUTPUT_DATASET_PATH) --data $(SYNTHTEXT_DATA_PATH) --txtdata $(TXT_DATASET_PATH)
