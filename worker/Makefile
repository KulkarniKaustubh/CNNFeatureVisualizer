VERSION=v1
DOCKERUSER=kaustubhkulkarni

build:
	docker build -f Dockerfile -t cnn-worker-gke-2 .

push:
	docker tag cnn-worker-gke-1 $(DOCKERUSER)/cnn-worker-gke-2:$(VERSION)
	docker push $(DOCKERUSER)/cnn-worker-gke-2:$(VERSION)
	docker tag cnn-worker-gke-1 $(DOCKERUSER)/cnn-worker-gke-2:latest
	docker push $(DOCKERUSER)/cnn-worker-gke-2:latest
