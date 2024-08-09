module.tar.gz:
	tar czf $@ --exclude='.DS_Store' *.sh .env src/*.py src/models/*.py src/models/checkpoints/*.pt src/models/checkpoints/*.onnx requirements.txt 

test:
	python3 -m pytest tests/*

lint:
	python3 -m pylint src --disable=W0719,R0902,R0913,R0914,R0903
