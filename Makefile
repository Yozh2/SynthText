DATA_PATH=data
IMAGES_PATH=$(DATA_PATH)/images
DEPTH_DATASET_PATH=$(IMAGES_PATH)/depths/

.PHONY collect
all: clear prepare collect

clear: clear_depth clear_seg clear_label

clear_depth:
	rm -rf $(DEPTH_DATASET_PATH)

clear_seg:
	rm -rf $