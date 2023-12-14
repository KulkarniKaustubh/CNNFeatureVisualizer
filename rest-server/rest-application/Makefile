VERSION=v1
DOCKERUSER=kaustubhkulkarni

build:
	docker build -f Dockerfile -t rest-server-gke .

push:
	docker tag rest-server-gke $(DOCKERUSER)/rest-server-gke-1:$(VERSION)
	docker push $(DOCKERUSER)/rest-server-gke-1:$(VERSION)
	docker tag rest-server-gke $(DOCKERUSER)/rest-server-gke-1:latest
	docker push $(DOCKERUSER)/rest-server-gke-1:latest
