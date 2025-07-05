import json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import os
import logging
import warnings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings desnecessários do ChromaDB
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")
warnings.filterwarnings("ignore", category=FutureWarning, module="chromadb")

# Configurar logging do ChromaDB para reduzir mensagens desnecessárias
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

class KnowledgeBase:
    def __init__(self, persist_directory="./chroma_db"):
        """
        Inicializa a base de conhecimento usando ChromaDB e all-MiniLM-L6-v2
        """
        self.persist_directory = persist_directory
        
        # Configurar ChromaDB com persistência e telemetria desabilitada
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Inicializar o modelo de embeddings usando ChromaDB embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Nome da coleção
        self.collection_name = "chatbot_intents"
        
        # Criar ou obter a coleção
        try:
            # Tentar obter coleção existente
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Coleção '{self.collection_name}' carregada com sucesso!")
        except:
            # Coleção não existe, criar nova
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Nova coleção '{self.collection_name}' criada!")
            except Exception as e:
                if "already exists" in str(e):
                    # Se coleção existe mas com configuração diferente, deletar e recriar
                    logger.warning(f"Coleção existe com configuração diferente. Recriando...")
                    self.client.delete_collection(self.collection_name)
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=self.embedding_function
                    )
                    logger.info(f"Coleção '{self.collection_name}' recriada com sucesso!")
                else:
                    logger.error(f"Erro ao criar coleção: {e}")
                    raise
    
    def load_intents_from_json(self, json_file_path):
        """
        Carrega as intenções do arquivo JSON e armazena no ChromaDB
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Verificar se já existem documentos na coleção
            existing_count = self.collection.count()
            if existing_count > 0:
                logger.info(f"Base de conhecimento já possui {existing_count} documentos. Carregamento ignorado.")
                return
            
            documents = []
            metadatas = []
            ids = []
            
            for i, intent in enumerate(data['intents']):
                tag = intent['tag']
                responses = intent['responses']
                patterns = intent['patterns']
                
                # Adicionar cada padrão como um documento separado
                for j, pattern in enumerate(patterns):
                    documents.append(pattern)
                    metadatas.append({
                        'tag': tag,
                        'responses': json.dumps(responses, ensure_ascii=False), # Converter lista para JSON string
                        'intent_id': i,
                        'pattern_id': j,
                        'link': intent.get('link', '')
                    })
                    ids.append(f"{tag}_{i}_{j}")
            
            # Adicionar documentos à coleção em lotes
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            
            logger.info(f"Carregados {len(documents)} padrões de {len(data['intents'])} intenções na base de conhecimento.")
            
        except Exception as e:
            logger.error(f"Erro ao carregar intenções: {str(e)}")
            raise
    
    def search_similar_intent(self, query, n_results=3, similarity_threshold=0.7):
        """
        Busca intenções similares usando busca semântica
        """
        try:
            # Realizar busca semântica
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                return None
            
            # Analisar resultados
            best_result = None
            best_distance = float('inf')
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                # Converter distância para similaridade (ChromaDB usa distância cosine)
                similarity = 1 - distance
                
                logger.info(f"Resultado {i+1}: '{doc}' - Similaridade: {similarity:.3f}")
                
                if similarity >= similarity_threshold and distance < best_distance:
                    best_distance = distance
                    best_result = {
                        'document': doc,
                        'metadata': metadata,
                        'similarity': similarity,
                        'distance': distance
                    }
            
            return best_result
            
        except Exception as e:
            logger.error(f"Erro na busca semântica: {str(e)}")
            return None
    
    def get_collection_stats(self):
        """
        Retorna estatísticas da coleção
        """
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {str(e)}")
            return None
    
    def reset_collection(self):
        """
        Remove todos os documentos da coleção (útil para recarregar dados)
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info("Coleção resetada com sucesso!")
        except Exception as e:
            logger.error(f"Erro ao resetar coleção: {str(e)}")
    
    def get_all_intents(self):
        """
        Retorna todas as intenções da base de conhecimento
        """
        try:
            # Obter todos os documentos da coleção
            results = self.collection.get(
                include=['metadatas']
            )
            
            all_intents = []
            seen_tags = set()
            
            for metadata in results['metadatas']:
                tag = metadata['tag']
                if tag not in seen_tags:
                    seen_tags.add(tag)
                    
                    # Decodificar padrões do JSON
                    patterns = json.loads(metadata['patterns'])
                    
                    intent_data = {
                        'tag': tag,
                        'patterns': patterns,
                        'responses': json.loads(metadata['responses'])
                    }
                    
                    all_intents.append(intent_data)
            
            logger.info(f"Retornadas {len(all_intents)} intenções da base de conhecimento")
            return all_intents
            
        except Exception as e:
            logger.error(f"Erro ao obter todas as intenções: {str(e)}")
            return []

    def get_intents_from_json(self, json_file_path='intents.json'):
        """
        Carrega intenções diretamente do arquivo JSON (alternativa rápida)
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return data.get('intents', [])
        except Exception as e:
            logger.error(f"Erro ao carregar intenções do JSON: {str(e)}")
            return []

if __name__ == "__main__":
    # Teste da classe
    kb = KnowledgeBase()
    
    # Carregar intenções do arquivo JSON
    kb.load_intents_from_json('intents.json')
    
    # Mostrar estatísticas
    stats = kb.get_collection_stats()
    print(f"Estatísticas: {stats}")
    
    # Teste de busca
    query = "como estão as aulas hoje?"
    result = kb.search_similar_intent(query)
    
    if result:
        print(f"\nMelhor resultado para '{query}':")
        print(f"Documento: {result['document']}")
        print(f"Tag: {result['metadata']['tag']}")
        print(f"Similaridade: {result['similarity']:.3f}")
    else:
        print(f"Nenhum resultado encontrado para '{query}'")
