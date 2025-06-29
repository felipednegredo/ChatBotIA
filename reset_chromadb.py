#!/usr/bin/env python3
"""
Script para limpar e resetar a base de dados ChromaDB
Use este script quando houver conflitos ou problemas com a base existente
"""

import os
import shutil
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_chromadb():
    """
    Remove completamente a base de dados ChromaDB
    """
    chroma_path = "./chroma_db"
    
    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path)
            logger.info(f"✅ Base de dados ChromaDB removida: {chroma_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao remover base de dados: {e}")
            return False
    else:
        logger.info("ℹ️ Base de dados ChromaDB não existe")
        return True

def reinitialize_system():
    """
    Reinicializa completamente o sistema
    """
    logger.info("🔄 Reinicializando sistema...")
    
    try:
        # Importar e criar nova base
        from knowledge_base import KnowledgeBase
        
        kb = KnowledgeBase()
        
        # Carregar intenções
        if os.path.exists('intents.json'):
            kb.load_intents_from_json('intents.json')
            stats = kb.get_collection_stats()
            logger.info(f"✅ Sistema reinicializado com sucesso!")
            logger.info(f"📊 Documentos carregados: {stats['total_documents']}")
            return True
        else:
            logger.error("❌ Arquivo intents.json não encontrado!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro ao reinicializar: {e}")
        return False

def main():
    """
    Função principal
    """
    print("🧹 Limpeza e Reset do ChromaDB")
    print("=" * 40)
    
    # Perguntar confirmação
    response = input("⚠️  Isto vai remover toda a base de dados. Continuar? (s/n): ").strip().lower()
    
    if response not in ['s', 'sim', 'y', 'yes']:
        print("❌ Operação cancelada")
        return False
    
    # Limpar base existente
    if not clean_chromadb():
        return False
    
    # Reinicializar sistema
    if not reinitialize_system():
        return False
    
    print("\n🎉 Reset concluído com sucesso!")
    print("📋 Agora você pode executar:")
    print("   python app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
