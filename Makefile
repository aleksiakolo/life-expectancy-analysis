IMAGE_NAME=lifeexp

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --rm \
		-v "$$(pwd)/data:/app/data" \
		-v "$$(pwd)/artifacts:/app/artifacts" \
		$(IMAGE_NAME)

panel:
	docker run --rm \
		-v "$$(pwd)/data:/app/data" \
		-v "$$(pwd)/artifacts:/app/artifacts" \
		$(IMAGE_NAME) lifeexp build-panel --config configs/default.yaml
