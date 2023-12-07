VERSION=v1
DOCKERUSER=kaustubhkulkarni

build:
	docker build -f Dockerfile -t cnn-worker .

push:
	docker tag cnn-worker $(DOCKERUSER)/cnn-worker:$(VERSION)
	docker push $(DOCKERUSER)/cnn-worker:$(VERSION)
	docker tag cnn-worker $(DOCKERUSER)/cnn-worker:latest
	docker push $(DOCKERUSER)/cnn-worker:latest
