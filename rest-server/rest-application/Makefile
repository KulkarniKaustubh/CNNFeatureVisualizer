VERSION=v1
DOCKERUSER=jayantduneja1234

build:
	docker build -f Dockerfile -t rest-server-gke .

push:
	docker tag rest-server-gke $(DOCKERUSER)/rest-server-gke:$(VERSION)
	docker push $(DOCKERUSER)/rest-server-gke:$(VERSION)
	docker tag rest-server-gke $(DOCKERUSER)/rest-server-gke:latest
	docker push $(DOCKERUSER)/rest-server-gke:latest
