# Makefile

.PHONY: setup lint test build run clean

DOCKER_IMAGE_NAME := million-song-mlops
DOCKER_TAG := latest

setup:
	pip install -r requirements.txt

lint:
	flake8 src tests

test:
	pytest tests

build:
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) .

run:
	docker run -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
			   -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
			   $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache

tf-init:
	cd terraform && terraform init

tf-plan:
	cd terraform && terraform plan

tf-apply:
	cd terraform && terraform apply -auto-approve

tf-destroy:
	cd terraform && terraform destroy -auto-approve