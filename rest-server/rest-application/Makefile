VERSION=v1
DOCKERUSER=jayantduneja1234

build:
	docker build -f Dockerfile -t rest-server-temp .

push:
	docker tag rest-server-temp $(DOCKERUSER)/rest-server-temp:$(VERSION)
	docker push $(DOCKERUSER)/rest-server-temp:$(VERSION)
	docker tag rest-server-temp $(DOCKERUSER)/rest-server-temp:latest
	docker push $(DOCKERUSER)/rest-server-temp:latest
