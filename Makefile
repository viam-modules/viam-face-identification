module.tar.gz:
	tar czf $@ --exclude='.DS_Store' *.sh .env src/*.py src/models/*.py src/models/checkpoints/*.pt src/models/checkpoints/*.onnx requirements.txt 

test:
	python -m pytest tests/*

lint:
	python -m pylint src --disable=W0719,R0902,R0913,R0914,R0903,R0917
