#!/usr/bin/env python3
"""
Script para configurar e inicializar o sistema de chatbot semântico
"""

import subprocess
import sys
import os
import json

def install_requirements():
    """Instala as dependências necessárias"""
    print("📦 Instalando dependências...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False

def download_model():
    """Baixa o modelo all-MiniLM-L6-v2"""
    print("🤖 Baixando modelo all-MiniLM-L6-v2...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Modelo baixado e carregado com sucesso!")
        return True
    except ImportError:
        print("❌ sentence-transformers não está instalado. Execute install_requirements() primeiro.")
        return False
    except Exception as e:
        print(f"❌ Erro ao baixar modelo: {e}")
        return False

def initialize_knowledge_base():
    """Inicializa a base de conhecimento"""
    print("📚 Inicializando base de conhecimento...")
    try:
        from knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        
        # Verificar se o arquivo intents.json existe
        if not os.path.exists('intents.json'):
            print("❌ Arquivo intents.json não encontrado!")
            return False
        
        # Carregar intenções
        kb.load_intents_from_json('intents.json')
        
        # Mostrar estatísticas
        stats = kb.get_collection_stats()
        print(f"✅ Base de conhecimento inicializada!")
        print(f"   📊 Total de documentos: {stats['total_documents']}")
        print(f"   📁 Diretório: {stats['persist_directory']}")
        
        return True
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro ao inicializar base de conhecimento: {e}")
        return False

def test_semantic_search():
    """Testa o sistema de busca semântica"""
    print("🔍 Testando busca semântica...")
    try:
        from chat_semantic import SemanticChatBot
        
        bot = SemanticChatBot()
        
        # Testes de exemplo
        test_queries = [
            "como estão as aulas hoje?",
            "qual o cardápio?",
            "onde fica a biblioteca?",
            "olá, tudo bem?"
        ]
        
        print("\n🧪 Executando testes:")
        for query in test_queries:
            response = bot.get_response(query)
            print(f"   👤 '{query}'")
            print(f"   🤖 {response[:100]}{'...' if len(response) > 100 else ''}")
            print()
        
        print("✅ Testes de busca semântica concluídos!")
        return True
    except Exception as e:
        print(f"❌ Erro nos testes: {e}")
        return False

def main():
    """Função principal"""
    print("🚀 Configuração do Sistema de Chatbot Semântico")
    print("=" * 50)
    
    # Verificar se estamos no diretório correto
    if not os.path.exists('intents.json'):
        print("❌ Execute este script no diretório do projeto (onde está o intents.json)")
        return False
    
    # Passo 1: Instalar dependências
    if not install_requirements():
        return False
    
    # Passo 2: Baixar modelo
    if not download_model():
        return False
    
    # Passo 3: Inicializar base de conhecimento
    if not initialize_knowledge_base():
        return False
    
    # Passo 4: Testes
    if not test_semantic_search():
        return False
    
    print("\n🎉 Sistema configurado com sucesso!")
    print("\n📋 Próximos passos:")
    print("   1. Execute 'python app.py' para iniciar o servidor web")
    print("   2. Execute 'python chat_semantic.py' para teste interativo")
    print("   3. Acesse http://localhost:5000 no navegador")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
