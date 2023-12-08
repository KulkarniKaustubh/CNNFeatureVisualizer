VERSION=v1
DOCKERUSER=jayantduneja1234

build:
	docker build --no-cache -f Dockerfile -t cnn-worker-gke-1 .

push:
	docker tag cnn-worker-gke-1 $(DOCKERUSER)/cnn-worker-gke-1:$(VERSION)
	docker push $(DOCKERUSER)/cnn-worker-gke-1:$(VERSION)
	docker tag cnn-worker-gke-1 $(DOCKERUSER)/cnn-worker-gke-1:latest
	docker push $(DOCKERUSER)/cnn-worker-gke-1:latest
