# detector_fraude

Detector de fraudes em sistemas financeiros utilizando Spark MLLib.

Projeto da interface web: https://github.com/gabrielmldantas/detector_fraude_web

Base de dados disponível em: https://www.kaggle.com/ntnu-testimon/paysim1

Baixe a base de dados e coloque o CSV no diretório data, ajustando o nome da base no método load_data se necessário.

A versão utilizada do Python foi a 3.7. Instale as dependências com pip install -r requirements.txt.
Se o python utilizado não for o padrão da máquina, exporte as variáveis de ambiente PYSPARK_PYTHON e PYSPARK_DRIVER_PYTHON com o caminho do interpretador Python correto.
Se as dependências não foram instaladas globalmente (virtualenv, por exemplo), exporte a variável PYTHONPATH com o caminho onde estão as dependências (no virtualenv, por exemplo, <caminho_do_virtualenv>/lib/python3.7/site-packages).

Crie os diretórios stream, result e checkpoints na raiz do projeto e execute o detector.py.
