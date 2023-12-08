module.tar.gz:
	tar czf $@ --exclude='.DS_Store' *.sh .env src/*.py requirements.txt 
