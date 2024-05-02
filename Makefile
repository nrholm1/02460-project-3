
.PHONY: requirements dev_requirements clean data build_documentation serve_documentation

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = 02460_Project3
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python3

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

train-vae:
	$(PYTHON_INTERPRETER) $(vae-main) train

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


VAE_MODEL_PATH = models/VAE_weights.pt
GAN_MODEL_PATH = models/ep1500_GraphGAN.pt


results-sample: 
	$(PYTHON_INTERPRETER) src/sample_histograms.py --sample --vae-model-path $(VAE_MODEL_PATH) \
	 --gan-model-path $(GAN_MODEL_PATH) --num-samples 1000 --vae-embedding-dim 7 --vae-M 2 --vae-n-message-passing-rounds 5

results-histogram:
	$(PYTHON_INTERPRETER) src/sample_histograms.py --histogram --vae-model-path $(VAE_MODEL_PATH) \
	 --gan-model-path $(GAN_MODEL_PATH) --vae-embedding-dim 7 --vae-M 2 --vae-n-message-passing-rounds 5

results-table:
	$(PYTHON_INTERPRETER) src/sample_histograms.py --table --vae-model-path $(VAE_MODEL_PATH) \
	 --gan-model-path $(GAN_MODEL_PATH) --vae-embedding-dim 7 --vae-M 2 --vae-n-message-passing-rounds 5

results-grid:
	$(PYTHON_INTERPRETER) src/sample_histograms.py --plot-grid vae --vae-model-path $(VAE_MODEL_PATH) \
	 --gan-model-path $(GAN_MODEL_PATH) --vae-embedding-dim 7 --vae-M 2 --vae-n-message-passing-rounds 5

results-plot-mean:
	$(PYTHON_INTERPRETER) src/sample_histograms.py --plot-mean vae --vae-model-path $(VAE_MODEL_PATH) \
	 --gan-model-path $(GAN_MODEL_PATH) --vae-embedding-dim 7 --vae-M 2 --vae-n-message-passing-rounds 5