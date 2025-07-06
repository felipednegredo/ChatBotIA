# Desenvolvimento de chatbot aplicado ao ambiente acadêmico do IFRS-Campus Sertão

## 🚀 Sistema Semântico Moderno

Este chatbot utiliza tecnologias modernas de busca semântica com **ChromaDB** e **all-MiniLM-L6-v2** para proporcionar respostas mais precisas e compreensão contextual avançada.

### 🛠️ Tecnologias Utilizadas

**Backend:**
- Python 3.8+
- Flask (API Web)
- ChromaDB (Base de dados vetorial)
- Sentence-Transformers (Embeddings)
- Transformers (Hugging Face)

**Frontend:**
- JavaScript
- HTML5/CSS3

**Processamento de Linguagem:**
- all-MiniLM-L6-v2 (Modelo de embeddings)
- Busca semântica por similaridade

## 📦 Instalação e Configuração

### 1. Instalação Manual
```bash
# Instalar dependências
pip install -r requirements.txt

# Inicializar base de conhecimento
python knowledge_base.py

# Testar sistema
python chat.py
```

### 3. Executar o Servidor
```bash
# Iniciar servidor Flask
python app.py

# Acessar: http://localhost:5000
```

## 🔍 Como Funciona

### Sistema de Busca Semântica

1. **Carregamento de Dados:** As intenções do `intents.json` são convertidas em embeddings usando all-MiniLM-L6-v2
2. **Armazenamento:** Os embeddings são armazenados no ChromaDB com persistência local
3. **Busca:** Mensagens do usuário são convertidas em embeddings e comparadas com a base
4. **Similaridade:** Usa distância cosine para encontrar a resposta mais relevante
5. **Threshold:** Apenas respostas com similaridade > 60% são retornadas

### Arquitetura do Sistema

```
User Input → Normalize Text → Generate Embedding → ChromaDB Search → Similarity Check → Response
```

## 📊 Endpoints da API

### `POST /predict`
Processa mensagem do usuário e retorna resposta
```json
{
  "message": "como estão as aulas hoje?"
}
```

### `GET /stats`
Retorna estatísticas da base de conhecimento
```json
{
  "total_documents": 150,
  "collection_name": "chatbot_intents",
  "persist_directory": "./chroma_db"
}
```

## 🧪 Testes e Avaliação

### Teste Interativo
```bash
python chat.py
```

### Exemplo de Uso
```
Você: como estão as aulas hoje?
Bot IFRS: As aulas estão funcionando normalmente conforme cronograma acadêmico...

Você: qual o cardápio do restaurante?
Bot IFRS: Aqui está o cardápio de hoje: [imagem do cardápio]
```

## 📁 Estrutura do Projeto

```
ChatBotIA/
├── 🧠 knowledge_base.py      # Gerenciamento ChromaDB
├── 💬 chat.py                # Sistema de chat semântico  
├── ⚙️ setup_semantic.py      # Script de configuração
├── 🌐 app.py                 # API Flask
├── � requirements.txt       # Dependências
├── 📝 intents.json           # Base de conhecimento
├── 🎨 static/                # Arquivos estáticos
├── 🌐 templates/             # Templates HTML
└── 💾 chroma_db/             # Base vetorial (criada automaticamente)
```

## ⚡ Vantagens do Sistema

### 🎯 Precisão Melhorada
- Compreensão semântica vs palavras-chave
- Melhor tratamento de sinônimos e variações
- Similaridade contextual

### 🚀 Performance
- Busca vetorial otimizada
- Sem necessidade de re-treinamento
- Resposta em tempo real

### 🔧 Manutenibilidade  
- Fácil adição de novas intenções
- Configuração via JSON
- Logs detalhados

### 📈 Escalabilidade
- ChromaDB suporta milhões de documentos
- Armazenamento persistente
- API RESTful

## 🐛 Solução de Problemas

### Erro de Dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Modelo não Encontrado
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Base de Dados Corrompida
```bash
python clean_database.py
```

## 🤝 Contribuições
Contribuições são bem-vindas! Por favor, abra uma issue antes de enviar pull requests.

---

## Demonstração:
![1 (1)](https://github.com/Renan1102/TCC/assets/103040108/fa2c66c4-9f88-4a46-a874-17f26e3dae6d)
![1](https://github.com/Renan1102/TCC/assets/103040108/5c6dc449-ffac-4f73-a2f0-c4d4eb5a40a1)

