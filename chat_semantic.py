import random
import json
import unicodedata
import requests
import logging
from knowledge_base import KnowledgeBase

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticChatBot:
    def __init__(self, intents_file='intents.json'):
        """
        Inicializa o chatbot semântico usando ChromaDB e all-MiniLM-L6-v2
        """
        self.bot_name = "Bot IFRS"
        self.knowledge_base = KnowledgeBase()
        
        # Carregar intenções na base de conhecimento
        try:
            self.knowledge_base.load_intents_from_json(intents_file)
            stats = self.knowledge_base.get_collection_stats()
            logger.info(f"Base de conhecimento inicializada: {stats}")
        except Exception as e:
            logger.error(f"Erro ao inicializar base de conhecimento: {str(e)}")
            raise
    
    def normalize_text(self, text):
        """
        Remove acentos e converte texto para minúsculas
        """
        return ''.join(c for c in unicodedata.normalize('NFD', text) 
                      if unicodedata.category(c) != 'Mn').lower()
    
    def get_response(self, user_message, similarity_threshold=0.6):
        """
        Gera resposta usando busca semântica
        """
        try:
            # Normalizar mensagem do usuário
            normalized_message = self.normalize_text(user_message)
            
            logger.info(f"Processando mensagem: '{user_message}' -> '{normalized_message}'")
            
            # Buscar intenção similar
            result = self.knowledge_base.search_similar_intent(
                normalized_message, 
                n_results=3,
                similarity_threshold=similarity_threshold
            )
            
            if result:
                metadata = result['metadata']
                similarity = result['similarity']
                
                logger.info(f"Intenção encontrada: '{metadata['tag']}' (similaridade: {similarity:.3f})")
                
                # Decodificar respostas do JSON
                responses = json.loads(metadata['responses'])
                response = random.choice(responses)
                
                # Adicionar link se existir
                if metadata.get('link'):
                    if metadata['tag'] == 'cardapio':
                        # Para cardápio, obter link dinâmico
                        dynamic_link = self.get_dynamic_link(metadata['link'])
                        response += f" <a target='_blank' href='https://ifrs.edu.br/sertao/assistencia-estudantil/restaurante/cardapio/'><img src='{dynamic_link}' width='400' height='300'></a>"
                    else:
                        # Para outros links estáticos
                        response += f" {metadata['link']}"
                
                return response
            else:
                logger.info("Nenhuma intenção similar encontrada")
                return self.get_fallback_response()
                
        except Exception as e:
            logger.error(f"Erro ao processar mensagem: {str(e)}")
            return "Desculpe, ocorreu um erro interno. Tente novamente."
    
    def get_fallback_response(self):
        """
        Resposta padrão quando nenhuma intenção é encontrada
        """
        fallback_responses = [
            "Desculpe, não entendi sua pergunta ou ela não está contemplada nesta interação. Você pode <strong>reformular</strong> de outra maneira ou utilizar a <strong>barra de pesquisa</strong>.",
            "Não consegui encontrar uma resposta adequada. Tente reformular sua pergunta de forma diferente.",
            "Ainda estou aprendendo! Pode tentar fazer a pergunta de outra forma?"
        ]
        return random.choice(fallback_responses)
    
    def get_dynamic_link(self, static_link):
        """
        Obtém link dinâmico para o cardápio
        """
        try:
            response = requests.get(
                'https://ifrs.edu.br/sertao/wp-json/wp/v2/media/31517?_fields=source_url',
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()['source_url']
            else:
                logger.warning(f"Falha ao obter link dinâmico. Status: {response.status_code}")
                return static_link
                
        except Exception as e:
            logger.error(f"Erro ao obter link dinâmico: {str(e)}")
            return static_link
    
    def get_statistics(self):
        """
        Retorna estatísticas da base de conhecimento
        """
        return self.knowledge_base.get_collection_stats()
    
    def reset_knowledge_base(self):
        """
        Reseta a base de conhecimento (útil para desenvolvimento)
        """
        self.knowledge_base.reset_collection()
        self.knowledge_base.load_intents_from_json('intents.json')

# Função de compatibilidade com o código existente
def get_response(msg):
    """
    Função de compatibilidade para manter a interface atual
    """
    global semantic_bot
    
    if 'semantic_bot' not in globals():
        semantic_bot = SemanticChatBot()
    
    return semantic_bot.get_response(msg)

if __name__ == "__main__":
    # Teste interativo
    bot = SemanticChatBot()
    
    print(f"=== {bot.bot_name} - Versão Semântica ===")
    print("Digite 'sair' para encerrar, 'stats' para estatísticas, 'reset' para resetar base")
    
    while True:
        user_input = input("\nVocê: ").strip()
        
        if user_input.lower() == 'sair':
            print("Até logo!")
            break
        elif user_input.lower() == 'stats':
            stats = bot.get_statistics()
            print(f"Estatísticas: {stats}")
            continue
        elif user_input.lower() == 'reset':
            bot.reset_knowledge_base()
            print("Base de conhecimento resetada!")
            continue
        elif not user_input:
            continue
        
        response = bot.get_response(user_input)
        print(f"{bot.bot_name}: {response}")
