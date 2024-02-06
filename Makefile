module.tar.gz:
	tar czf $@ --exclude='.DS_Store' *.sh .env src/*.py src/ir/*.py src/ir/checkpoints/checkpoint.pt requirements.txt 
