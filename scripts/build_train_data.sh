OUTPUT_DIRECTORY=data
TRAIN_DIRECTORY=${OUTPUT_DIRECTORY}/raw-data/train
VALIDATION_DIRECTORY=${OUTPUT_DIRECTORY}/raw-data/validation
LABELS_FILE=${OUTPUT_DIRECTORY}/raw-data/labels.txt

cd inception
WORK_DIR=scripts/
BUILD_SCRIPT="${WORK_DIR}/build_image_data.py"
"${BUILD_SCRIPT}" \
  --train_directory="${TRAIN_DIRECTORY}" \
  --validation_directory="${VALIDATION_DIRECTORY}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}"
