
.PHONY: requirements dev_requirements clean data build_documentation serve_documentation

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = 02460_Project3
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_venv:
	$(PYTHON_INTERPRETER) -m venv venv

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

vae-main = main.py
gan-main = src/graphgan.py

DEVICE = cpu

train-gan:
	$(PYTHON_INTERPRETER) $(gan-main) train --n-epochs 10000 \
		--gen-lr  0.00005 \
		--disc-lr 0.00005 \
		--gen-train-steps  1 \
		--disc-train-steps 5 \
		--mp-rounds 5 \
		--batch-size 10
# $(PYTHON_INTERPRETER) $(gan-main) train --n-epochs 25000 --gen-lr 0.0001 --disc-lr 0.0001 --disc-train-steps 3 ##! cfg1 with modified loss: quite okay settings

sample-gan:
	$(PYTHON_INTERPRETER) $(gan-main) sample

histograms:
	$(PYTHON_INTERPRETER) src/evaluate_samples.py