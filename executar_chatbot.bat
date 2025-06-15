@echo off
cd /d "%~dp0"

echo Ativando o ambiente virtual...
call venv\Scripts\activate

echo -----------------------------------------
echo Treinando o modelo (train.py)...
python train.py

echo -----------------------------------------
echo Abrindo o navegador para acessar o chatbot...

start "" http://127.0.0.1:5000/

echo -----------------------------------------
echo Iniciando o chatbot (app.py)...
python app.py

echo -----------------------------------------
echo Processo concluído.
pause

echo Pressione qualquer tecla para sair...
deactivate