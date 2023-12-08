VERSION=v1
DOCKERUSER=jayantduneja1234

build:
	docker build -f Dockerfile -t cnn-worker-gke .

push:
	docker tag cnn-worker-gke $(DOCKERUSER)/cnn-worker-gke:$(VERSION)
	docker push $(DOCKERUSER)/cnn-worker-gke:$(VERSION)
	docker tag cnn-worker-gke $(DOCKERUSER)/cnn-worker-gke:latest
	docker push $(DOCKERUSER)/cnn-worker-gke:latest
