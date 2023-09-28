# Variables
VENV_NAME?=venv
PYTHON=${VENV_NAME}/bin/python
DATASET_URL=https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset

.PHONY: all clean data venv train test

# Default target will be 'all'
all: venv data train

# Set up virtual environment
venv: ${VENV_NAME}/bin/activate
${VENV_NAME}/bin/activate:
	test -d ${VENV_NAME} || virtualenv ${VENV_NAME}
	${PYTHON} -m pip install -U pip
	${PYTHON} -m pip install -r requirements.txt
	touch ${VENV_NAME}/bin/activate

# Download dataset
data:
	kaggle datasets download -d smaranjitghose/corn-or-maize-leaf-disease-dataset -p data/

# Train the model
train:
	${PYTHON} src/train_model.py

# Test the model
test:
	${PYTHON} src/test_model.py

# Clean up cache files and virtual environment
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf ${VENV_NAME}
	rm -rf data/
