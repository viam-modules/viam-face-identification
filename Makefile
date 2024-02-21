module.tar.gz:
	tar czf $@ --exclude='.DS_Store' *.sh .env src/*.py src/models/*.py src/models/checkpoints/*.pt src/models/checkpoints/*.onnx requirements.txt 
