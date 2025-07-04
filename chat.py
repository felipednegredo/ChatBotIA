import random
import json
import unicodedata
import requests
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from knowledge_base import KnowledgeBase
from sentence_transformers import SentenceTransformer

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Representa um turno de conversa com embedding vetorial"""
    user_message: str
    bot_response: str
    timestamp: datetime
    user_embedding: List[float]
    combined_embedding: List[float]  # Embedding da pergunta + resposta
    intent_tag: str = None
    similarity_score: float = 0.0

class ShortTermMemory:
    """Sistema de memória de curto prazo com cache vetorial"""
    
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def add_turn(self, user_message: str, bot_response: str, intent_tag: str = None, similarity_score: float = 0.0):
        """Adiciona um novo turno à memória com embeddings"""
        try:
            # Gerar embeddings
            user_embedding = self.embedding_model.encode(user_message).tolist()
            combined_text = f"Pergunta: {user_message} Resposta: {bot_response}"
            combined_embedding = self.embedding_model.encode(combined_text).tolist()
            
            turn = ConversationTurn(
                user_message=user_message,
                bot_response=bot_response,
                timestamp=datetime.now(),
                user_embedding=user_embedding,
                combined_embedding=combined_embedding,
                intent_tag=intent_tag,
                similarity_score=similarity_score
            )
            
            self.turns.append(turn)
            
            # Manter apenas os últimos N turnos
            if len(self.turns) > self.max_turns:
                self.turns.pop(0)
                
            logger.info(f"Turno adicionado à memória. Total: {len(self.turns)} turnos")
            
        except Exception as e:
            logger.error(f"Erro ao adicionar turno à memória: {str(e)}")
    
    def get_relevant_context(self, current_message: str, max_relevant: int = 3) -> List[ConversationTurn]:
        """Busca turnos relevantes baseado na similaridade vetorial"""
        if not self.turns:
            return []
        
        try:
            # Gerar embedding da mensagem atual
            current_embedding = self.embedding_model.encode(current_message)
            
            # Calcular similaridades
            similarities = []
            for turn in self.turns:
                # Calcular similaridade com a mensagem do usuário anterior
                user_similarity = self._cosine_similarity(current_embedding, turn.user_embedding)
                # Calcular similaridade com o contexto completo
                context_similarity = self._cosine_similarity(current_embedding, turn.combined_embedding)
                
                # Usar a maior similaridade
                max_similarity = max(user_similarity, context_similarity)
                similarities.append((turn, max_similarity))
            
            # Ordenar por similaridade e retornar os mais relevantes
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Filtrar por threshold de relevância
            relevant_turns = []
            for turn, similarity in similarities:
                if similarity > 0.3 and len(relevant_turns) < max_relevant:  # Threshold de relevância
                    relevant_turns.append(turn)
            
            logger.info(f"Encontrados {len(relevant_turns)} turnos relevantes para: '{current_message}'")
            return relevant_turns
            
        except Exception as e:
            logger.error(f"Erro ao buscar contexto relevante: {str(e)}")
            return []
    
    def get_recent_context_text(self, max_turns: int = 3) -> str:
        """Retorna texto do contexto recente para injeção no prompt"""
        recent_turns = self.turns[-max_turns:] if self.turns else []
        
        if not recent_turns:
            return ""
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"Usuário anterior: {turn.user_message}")
            context_parts.append(f"Bot anterior: {turn.bot_response}")
        
        return " ".join(context_parts)
    
    def get_contextual_keywords(self, current_message: str) -> List[str]:
        """Extrai palavras-chave relevantes do contexto"""
        relevant_turns = self.get_relevant_context(current_message, max_relevant=2)
        
        keywords = []
        for turn in relevant_turns:
            # Extrair palavras importantes da mensagem do usuário
            user_words = turn.user_message.lower().split()
            important_words = [w for w in user_words if len(w) > 3 and w not in 
                             ['para', 'como', 'onde', 'quando', 'porque', 'qual', 'quem', 'que', 'isso', 'esta', 'esse']]
            keywords.extend(important_words[:2])  # Máximo 2 palavras por turno
            
            # Adicionar intent_tag se disponível
            if turn.intent_tag:
                tag_words = turn.intent_tag.replace('_', ' ').split()
                keywords.extend(tag_words)
        
        # Remover duplicatas e limitar
        return list(set(keywords))[:5]
    
    def _cosine_similarity(self, a, b):
        """Calcula similaridade do cosseno entre dois vetores"""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def clear(self):
        """Limpa a memória de curto prazo"""
        self.turns = []
        logger.info("Memória de curto prazo limpa")
    
    def get_memory_stats(self) -> Dict:
        """Retorna estatísticas da memória"""
        return {
            'total_turns': len(self.turns),
            'memory_age_minutes': (datetime.now() - self.turns[0].timestamp).total_seconds() / 60 if self.turns else 0,
            'recent_intents': [turn.intent_tag for turn in self.turns[-3:] if turn.intent_tag],
            'avg_similarity': sum(turn.similarity_score for turn in self.turns) / len(self.turns) if self.turns else 0
        }

class SemanticChatBot:
    def __init__(self, intents_file='intents.json'):
        """
        Inicializa o chatbot semântico usando ChromaDB e all-MiniLM-L6-v2
        """
        self.bot_name = "Bot IFRS"
        self.knowledge_base = KnowledgeBase()
        
        # Inicializar memória de curto prazo
        self.short_term_memory = ShortTermMemory(max_turns=5)
        
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
    
    def get_response_with_context(self, user_message, conversation_history=None, similarity_threshold=0.6):
        """
        Gera resposta usando busca semântica considerando o contexto da conversa
        Com sistema de sugestões quando há baixa certeza e memória de curto prazo
        """
        try:
            # Normalizar mensagem do usuário
            normalized_message = self.normalize_text(user_message)
            
            logger.info(f"Processando mensagem: '{user_message}' -> '{normalized_message}'")
            
            # 1. USAR MEMÓRIA DE CURTO PRAZO PARA ENRIQUECER O CONTEXTO
            relevant_memory = self.short_term_memory.get_relevant_context(user_message, max_relevant=3)
            contextual_keywords = self.short_term_memory.get_contextual_keywords(user_message)
            
            if relevant_memory:
                logger.info(f"Memória de curto prazo: {len(relevant_memory)} turnos relevantes encontrados")
                logger.info(f"Palavras-chave contextuais: {contextual_keywords}")
            
            # 2. ENRIQUECER QUERY COM CONTEXTO DA MEMÓRIA
            enriched_query = self._enrich_query_with_memory(normalized_message, relevant_memory, contextual_keywords)
            
            if conversation_history:
                logger.info(f"Contexto da conversa incluído: {len(conversation_history)} mensagens anteriores")
                # Analisar contexto para debug
                context_analysis = self._analyze_conversation_context(conversation_history)
                
                # Verificar se o usuário está selecionando uma opção numérica
                option_response = self._check_option_selection(user_message, conversation_history)
                if option_response:
                    # Armazenar na memória antes de retornar
                    self.short_term_memory.add_turn(user_message, option_response)
                    return option_response
            
            # 3. BUSCAR COM QUERY ENRIQUECIDA PELA MEMÓRIA
            logger.info(f"Tentativa 1: Busca com query enriquecida pela memória (threshold: {similarity_threshold})")
            result = self.knowledge_base.search_similar_intent(
                enriched_query, 
                n_results=5,
                similarity_threshold=similarity_threshold
            )
            
            # 4. FALLBACK: Se não encontrou com memória, tentar com query original
            if not result:
                logger.info("Tentativa 2: Busca com mensagem original...")
                result = self.knowledge_base.search_similar_intent(
                    normalized_message, 
                    n_results=5,
                    similarity_threshold=similarity_threshold
                )
            
            # 5. FALLBACK: Se não encontrou resultado satisfatório, tentar com contexto da conversa
            if not result and conversation_history:
                logger.info("Tentativa 3: Busca com contexto da conversa...")
                enriched_message = self._enrich_message_with_context(user_message, conversation_history)
                result = self.knowledge_base.search_similar_intent(
                    enriched_message, 
                    n_results=5,
                    similarity_threshold=similarity_threshold - 0.1
                )
            
            # 6. FALLBACK: Usar palavras-chave do contexto histórico
            if not result and conversation_history:
                logger.info("Tentativa 4: Busca com palavras-chave do contexto...")
                context_keywords = self._extract_context_keywords(conversation_history)
                if context_keywords:
                    keyword_query = f"{user_message} {' '.join(context_keywords)}"
                    normalized_keyword_query = self.normalize_text(keyword_query)
                    logger.info(f"Query com palavras-chave: '{normalized_keyword_query}'")
                    result = self.knowledge_base.search_similar_intent(
                        normalized_keyword_query, 
                        n_results=5,
                        similarity_threshold=similarity_threshold - 0.2
                    )
            
            # 7. ÚLTIMO FALLBACK: Busca com threshold muito baixo
            if not result:
                logger.info("Tentativa 5: Busca com threshold muito baixo...")
                low_threshold_result = self.knowledge_base.search_similar_intent(
                    normalized_message, 
                    n_results=5,
                    similarity_threshold=-0.5  # Threshold muito baixo para pegar qualquer resultado
                )
                if low_threshold_result:
                    logger.info(f"Resultado encontrado com threshold baixo: similaridade {low_threshold_result['similarity']:.3f}")
                    # Se encontrou resultado com threshold baixo, oferecer opções
                    if low_threshold_result['similarity'] < 0.5:
                        logger.info("Oferecendo opções devido à baixa similaridade...")
                        return self._offer_similar_options(normalized_message, low_threshold_result, conversation_history)
                    else:
                        result = low_threshold_result
            
            if result:
                metadata = result['metadata']
                similarity = result['similarity']
                
                logger.info(f"Intenção encontrada: '{metadata['tag']}' (similaridade: {similarity:.3f})")
                
                # Verificar se a similaridade é baixa e oferecer opções
                if similarity < 0.5:  # Threshold de incerteza
                    logger.info(f"Similaridade baixa ({similarity:.3f}), oferecendo opções alternativas...")
                    return self._offer_similar_options(normalized_message, result, conversation_history)
                
                # Decodificar respostas do JSON
                responses = json.loads(metadata['responses'])
                response = random.choice(responses)
                
                # Personalizar resposta baseada no contexto
                response = self._personalize_response_with_context(response, conversation_history)
                
                # Adicionar link se existir
                if metadata.get('link'):
                    if metadata['tag'] == 'cardapio':
                        # Para cardápio, obter link dinâmico
                        dynamic_link = self.get_dynamic_link(metadata['link'])
                        
                        if dynamic_link and dynamic_link != metadata['link']:
                            # Link dinâmico obtido com sucesso
                            response += f"""
                            <div class="cardapio-container">
                                <div class="cardapio-header">
                                    <h3>📍 Cardápio do Restaurante IFRS</h3>
                                    <p>Confira as opções de hoje:</p>
                                </div>
                                <div class="cardapio-image-wrapper">
                                    <img src='{dynamic_link}' alt='Cardápio do dia' class='cardapio-image' onclick='openCardapioModal(this)' onerror='this.style.display="none"; this.parentElement.innerHTML="<p style=\"text-align: center; padding: 20px; color: #666;\">Imagem do cardápio não disponível no momento</p>";'>
                                    <div class="cardapio-overlay">
                                        <span>Clique para ampliar</span>
                                    </div>
                                </div>
                                <div class="cardapio-actions">
                                    <a href='https://ifrs.edu.br/sertao/assistencia-estudantil/restaurante/cardapio/' target='_blank' class='cardapio-link'>
                                        🔗 Ver no site oficial
                                    </a>
                                </div>
                            </div>
                            """
                        else:
                            # Fallback quando não conseguir obter o link dinâmico
                            response += f"""
                            <div class="cardapio-container">
                                <div class="cardapio-header">
                                    <h3>📍 Cardápio do Restaurante IFRS</h3>
                                    <p>Informações sobre o cardápio:</p>
                                </div>
                                <div class="cardapio-fallback">
                                    <p>🍽️ O cardápio não está disponível para visualização no momento.</p>
                                    <p>Você pode consultar diretamente no site oficial ou comparecer ao restaurante.</p>
                                </div>
                                <div class="cardapio-actions">
                                    <a href='https://ifrs.edu.br/sertao/assistencia-estudantil/restaurante/cardapio/' target='_blank' class='cardapio-link'>
                                        🔗 Ver no site oficial
                                    </a>
                                </div>
                            </div>
                            """
                    else:
                        # Para outros links estáticos
                        response += f" {metadata['link']}"
                
                # Adicionar à memória de curto prazo
                self.short_term_memory.add_turn(user_message, response, metadata['tag'], similarity)
                
                return response
            else:
                logger.info("Nenhuma intenção similar encontrada em todas as tentativas")
                # Última tentativa: oferecer opções baseadas em busca ampla
                try:
                    logger.info("Tentativa final: Busca ampla para sugestões...")
                    final_options = self._offer_similar_options(normalized_message, None, conversation_history)
                    if "Não tenho certeza do que você está procurando" in final_options:
                        return final_options
                except:
                    pass
                
                return self._get_contextual_fallback_response(conversation_history)
                
        except Exception as e:
            logger.error(f"Erro ao processar mensagem: {str(e)}")
            return "Desculpe, ocorreu um erro interno. Tente novamente."

    def _extract_context_keywords(self, conversation_history):
        """
        Extrai palavras-chave relevantes do contexto da conversa buscando no ChromaDB
        """
        keywords = []
        
        # Coletar todas as mensagens do usuário no histórico
        user_messages = []
        for msg in conversation_history[-3:]:  # Últimas 3 mensagens
            if msg['type'] == 'user':
                user_messages.append(msg['message'])
        
        if not user_messages:
            return keywords
        
        # Para cada mensagem do usuário, tentar buscar no ChromaDB
        for message in user_messages:
            try:
                # Normalizar a mensagem
                normalized_msg = self.normalize_text(message)
                
                # Buscar diretamente no ChromaDB
                results = self.knowledge_base.collection.query(
                    query_texts=[normalized_msg],
                    n_results=2,  # Pegar 2 resultados mais similares
                    include=['documents', 'metadatas', 'distances']
                )
                
                if results['documents'] and results['documents'][0]:
                    for doc, metadata, distance in zip(
                        results['documents'][0],
                        results['metadatas'][0], 
                        results['distances'][0]
                    ):
                        # Extrair palavras da tag da intenção encontrada
                        intent_tag = metadata.get('tag', '')
                        
                        # Dividir a tag em palavras e adicionar às keywords
                        tag_words = intent_tag.replace('_', ' ').split()
                        for word in tag_words:
                            normalized_word = self.normalize_text(word)
                            if len(normalized_word) > 2 and normalized_word not in keywords:
                                keywords.append(normalized_word)
                        
                        # Também extrair palavras importantes do próprio documento encontrado
                        doc_words = doc.lower().split()
                        important_doc_words = [w for w in doc_words if len(w) > 3 and w not in 
                                             ['para', 'como', 'onde', 'quando', 'porque', 'qual', 'quem', 'que', 'isso', 'esta', 'esse', 'sao', 'tem', 'esta', 'voce', 'quero', 'preciso']]
                        
                        for word in important_doc_words[:2]:  # Máximo 2 palavras por documento
                            normalized_word = self.normalize_text(word)
                            if normalized_word not in keywords:
                                keywords.append(normalized_word)
                            
            except Exception as e:
                logger.debug(f"Erro ao extrair keywords da mensagem '{message}': {str(e)}")
                continue
        
        # Limitar a 5 palavras-chave para não sobrecarregar
        keywords = list(set(keywords))[:5]  # Remove duplicatas e limita
        
        logger.info(f"Palavras-chave extraídas do ChromaDB: {keywords}")
        return keywords

    def _enrich_message_with_context(self, current_message, conversation_history):
        """
        Enriquece a mensagem atual com palavras-chave do contexto buscando no ChromaDB
        """
        if not conversation_history:
            return self.normalize_text(current_message)
        
        # Extrair palavras importantes das últimas mensagens do usuário
        context_keywords = []
        for msg in conversation_history[-3:]:  # Apenas últimas 3 mensagens
            if msg['type'] == 'user':
                # Buscar termos similares no ChromaDB para cada palavra importante
                words = msg['message'].lower().split()
                important_words = [w for w in words if len(w) > 3 and w not in 
                                 ['para', 'como', 'onde', 'quando', 'porque', 'qual', 'quem', 'que', 'isso', 'esta', 'esse']]
                
                for word in important_words[:2]:  # Máximo 2 palavras por mensagem
                    # Buscar termos similares no ChromaDB
                    similar_terms = self._get_similar_terms_from_db(word)
                    context_keywords.extend(similar_terms)
        
        # Combinar mensagem atual com palavras-chave do contexto
        if context_keywords:
            # Remover duplicatas e limitar
            unique_keywords = list(set(context_keywords))[:3]
            enriched = f"{current_message} {' '.join(unique_keywords)}"
            logger.info(f"Mensagem enriquecida com ChromaDB: '{current_message}' -> '{enriched}'")
        else:
            enriched = current_message
        
        return self.normalize_text(enriched)

    def _build_context_message(self, current_message, conversation_history):
        """
        Constrói uma mensagem combinando o contexto da conversa com a mensagem atual
        """
        if not conversation_history:
            return self.normalize_text(current_message)
        
        # Extrair apenas as mensagens de texto das últimas interações
        context_parts = []
        for msg in conversation_history[-5:]:  # Últimas 5 mensagens
            if msg['type'] == 'user':
                context_parts.append(f"Usuário anterior: {msg['message']}")
            elif msg['type'] == 'bot':
                # Remover tags HTML da resposta do bot para contexto
                clean_response = self._clean_html_tags(msg['message'])
                context_parts.append(f"Bot anterior: {clean_response}")
        
        # Combinar contexto com mensagem atual
        context_text = " ".join(context_parts)
        full_message = f"{context_text} Pergunta atual: {current_message}"
        
        return self.normalize_text(full_message)

    def _clean_html_tags(self, text):
        """
        Remove tags HTML básicas do texto
        """
        import re
        # Remove tags HTML comuns
        clean = re.sub(r'<[^>]+>', '', text)
        return clean.strip()

    def _personalize_response_with_context(self, response, conversation_history):
        """
        Personaliza a resposta baseada no contexto da conversa
        """
        if not conversation_history:
            return response
        
        # Verificar se o usuário já fez perguntas similares recentemente
        recent_user_messages = [msg['message'].lower() for msg in conversation_history[-3:] if msg['type'] == 'user']
        
        # Detectar padrões de continuação
        if len(recent_user_messages) > 1:
            current_msg = recent_user_messages[-1] if recent_user_messages else ""
            
            # Palavras que indicam continuação
            continuacao_words = ['também', 'ainda', 'mais', 'outro', 'outra', 'além', 'adicionalmente']
            pergunta_words = ['e', 'qual', 'como', 'onde', 'quando']
            
            if any(word in current_msg for word in continuacao_words):
                response = f"Além disso, {response.lower()}"
            elif any(word in current_msg for word in pergunta_words) and len(current_msg.split()) <= 3:
                response = f"Sobre isso, {response.lower()}"
            elif any('obrigad' in msg for msg in recent_user_messages):
                response = f"Fico feliz em ajudar! {response}"
        
        return response

    def _get_contextual_fallback_response(self, conversation_history):
        """
        Resposta padrão contextualizada quando nenhuma intenção é encontrada
        Usa memória de curto prazo para contexto mais relevante
        """
        fallback_responses = [
            "Desculpe, não entendi sua pergunta ou ela não está contemplada nesta interação. Você pode <strong>reformular</strong> de outra maneira ou utilizar a <strong>barra de pesquisa</strong>.",
            "Não consegui encontrar uma resposta adequada. Tente reformular sua pergunta de forma diferente.",
            "Ainda estou aprendendo! Pode tentar fazer a pergunta de outra forma?"
        ]
        
        # Usar memória de curto prazo para contexto mais relevante
        memory_stats = self.short_term_memory.get_memory_stats()
        if memory_stats['total_turns'] > 0:
            # Sugerir tópicos baseados nos intents recentes da memória
            recent_intents = memory_stats['recent_intents']
            if recent_intents:
                intent_topics = [intent.replace('_', ' ') for intent in recent_intents]
                unique_topics = list(set(intent_topics))
                
                contextual_responses = [
                    f"Com base em nossa conversa sobre {', '.join(unique_topics)}, não consegui entender completamente sua última pergunta. Pode reformular?",
                    f"Considerando que falamos sobre {', '.join(unique_topics)}, preciso que você seja mais específico. Pode tentar de outra forma?",
                ]
                fallback_responses.extend(contextual_responses)
        
        # Fallback para contexto da conversa se memória não tiver info suficiente
        elif conversation_history:
            recent_topics = []
            for msg in conversation_history[-2:]:
                if msg['type'] == 'user':
                    words = msg['message'].lower().split()
                    important = [w for w in words if len(w) > 4]
                    recent_topics.extend(important[:1])  # Uma palavra importante por mensagem
            
            if recent_topics:
                contextual_responses = [
                    f"Com base em nossa conversa sobre {', '.join(set(recent_topics))}, não consegui entender completamente sua última pergunta. Pode reformular?",
                    f"Considerando que falamos sobre {', '.join(set(recent_topics))}, preciso que você seja mais específico. Pode tentar de outra forma?",
                ]
                fallback_responses.extend(contextual_responses)
        
        return random.choice(fallback_responses)
    
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
        Obtém link dinâmico para o cardápio com fallback melhorado
        """
        try:
            response = requests.get(
                'https://ifrs.edu.br/sertao/wp-json/wp/v2/media/31517?_fields=source_url',
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'source_url' in data and data['source_url']:
                    logger.info("Link dinâmico do cardápio obtido com sucesso")
                    return data['source_url']
                else:
                    logger.warning("Resposta da API não contém source_url válido")
                    return static_link
            else:
                logger.warning(f"Falha ao obter link dinâmico. Status: {response.status_code}")
                return static_link
                
        except requests.exceptions.Timeout:
            logger.error("Timeout ao obter link dinâmico do cardápio")
            return static_link
        except requests.exceptions.ConnectionError:
            logger.error("Erro de conexão ao obter link dinâmico do cardápio")
            return static_link
        except Exception as e:
            logger.error(f"Erro inesperado ao obter link dinâmico: {str(e)}")
            return static_link
    
    def get_statistics(self):
        """
        Retorna estatísticas da base de conhecimento e memória de curto prazo
        """
        base_stats = self.knowledge_base.get_collection_stats()
        memory_stats = self.get_memory_stats()
        
        # Combinar estatísticas
        combined_stats = {
            **base_stats,
            'short_term_memory': memory_stats,
            'memory_context_summary': self.get_memory_context_summary()
        }
        
        return combined_stats
    
    def reset_knowledge_base(self):
        """
        Reseta a base de conhecimento (útil para desenvolvimento)
        """
        self.knowledge_base.reset_collection()
        self.knowledge_base.load_intents_from_json('intents.json')
    
    def get_response(self, user_message, similarity_threshold=0.6):
        """
        Gera resposta usando busca semântica (método original para compatibilidade)
        """
        return self.get_response_with_context(user_message, None, similarity_threshold)
    
    def _analyze_conversation_context(self, conversation_history):
        """
        Analisa o contexto da conversa para extrair informações úteis
        """
        if not conversation_history:
            return {}
        
        analysis = {
            'topics': [],
            'user_intent_pattern': [],
            'last_successful_topic': None
        }
        
        for msg in conversation_history:
            if msg['type'] == 'user':
                # Extrair tópicos mencionados
                words = msg['message'].lower().split()
                topics = [w for w in words if len(w) > 3 and w not in 
                         ['para', 'como', 'onde', 'quando', 'porque', 'qual', 'quem', 'que', 'isso', 'esta', 'esse']]
                analysis['topics'].extend(topics)
                analysis['user_intent_pattern'].append(msg['message'])
        
        # Remover duplicatas e manter apenas tópicos únicos
        analysis['topics'] = list(set(analysis['topics']))
        
        logger.info(f"Análise do contexto: {analysis}")
        return analysis

    def _get_similar_terms_from_db(self, term, max_results=3):
        """
        Busca termos similares no ChromaDB para enriquecer o contexto
        """
        try:
            # Buscar termos similares no banco
            results = self.knowledge_base.collection.query(
                query_texts=[self.normalize_text(term)],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            similar_terms = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                ):
                    # Converter distância para similaridade
                    similarity = 1 - distance
                    
                    if similarity > -0.5:  # Threshold muito baixo
                        tag = metadata.get('tag', '')
                        
                        # Extrair palavras do tag
                        tag_words = tag.replace('_', ' ').split()
                        for word in tag_words:
                            normalized_word = self.normalize_text(word)
                            if len(normalized_word) > 2 and normalized_word not in similar_terms:
                                similar_terms.append(normalized_word)
                
                logger.debug(f"Termos similares encontrados para '{term}': {similar_terms}")
            
            return similar_terms[:3]  # Máximo 3 termos
            
        except Exception as e:
            logger.debug(f"Erro ao buscar termos similares para '{term}': {str(e)}")
            return []

    def _offer_similar_options(self, normalized_message, best_result, conversation_history):
        """
        Oferece opções similares quando não há certeza sobre a intenção
        """
        try:
            # Buscar múltiplas opções com threshold baixo
            results = self.knowledge_base.collection.query(
                query_texts=[normalized_message],
                n_results=4,  # Buscar 4 opções
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                return self._get_contextual_fallback_response(conversation_history)
            
            # Processar resultados e criar opções
            options = []
            seen_tags = set()
            
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            ):
                similarity = 1 - distance
                tag = metadata.get('tag', '')
                
                # Evitar tags duplicadas e muito baixas similaridades
                if tag not in seen_tags and similarity > 0.1:
                    seen_tags.add(tag)
                    
                    # Criar descrição amigável da opção
                    option_description = self._create_option_description(tag, doc)
                    options.append({
                        'tag': tag,
                        'description': option_description,
                        'similarity': similarity,
                        'document': doc
                    })
            
            # Limitar a 3 melhores opções
            options = options[:3]
            
            if not options:
                return self._get_contextual_fallback_response(conversation_history)
            
            logger.info(f"Oferecendo {len(options)} opções similares")
            
            # Criar resposta com opções (incluir tags como comentários HTML para referência)
            response = "Não tenho certeza do que você está procurando. Você quis dizer alguma dessas opções?<br><br>"
            
            for i, option in enumerate(options, 1):
                # Incluir tag como comentário HTML para referência posterior
                response += f"<strong>{i}.</strong> {option['description']}<!-- TAG:{option['tag']} --><br>"
            
            response += "<br>Você pode <strong>digitar o número</strong> da opção desejada (ex: 1, 2, 3) ou reformular sua pergunta sendo mais específico."
            
            return response
            
        except Exception as e:
            logger.error(f"Erro ao oferecer opções similares: {str(e)}")
            return self._get_contextual_fallback_response(conversation_history)

    def _create_option_description(self, tag, document):
        """
        Cria uma descrição amigável para uma opção baseada na tag e documento,
        priorizando busca no ChromaDB ao invés de mapeamentos hardcoded
        """
        try:
            # Primeiro, tentar extrair descrição do próprio documento se for informativo
            if len(document) > 10 and len(document) < 100:
                # Se o documento é uma frase informativa curta, usar ele
                return document.strip()
            
            # Buscar no ChromaDB por uma descrição mais específica da tag
            search_queries = self._generate_expanded_queries_from_tag(tag)
            
            for query in search_queries[:2]:  # Tentar apenas 2 queries para não sobrecarregar
                try:
                    results = self.knowledge_base.collection.query(
                        query_texts=[query],
                        n_results=1,
                        include=['documents', 'metadatas']
                    )
                    
                    if (results['documents'] and results['documents'][0] and 
                        results['metadatas'] and results['metadatas'][0]):
                        
                        found_doc = results['documents'][0][0]
                        found_metadata = results['metadatas'][0][0]
                        
                        # Se encontrou a mesma tag, usar o documento como descrição
                        if found_metadata.get('tag') == tag and len(found_doc) < 100:
                            return found_doc.strip()
                
                except Exception as e:
                    logger.debug(f"Erro ao buscar descrição para tag '{tag}' com query '{query}': {str(e)}")
                    continue
            
            # Fallback: formatar a tag de forma amigável
            formatted_tag = self._format_tag_as_suggestion(tag)
            
            # Se a tag formatada for muito pequena, complementar com parte do documento
            if len(formatted_tag) < 10 and len(document) > 20:
                return f"{formatted_tag} - {document[:50]}..." if len(document) > 50 else f"{formatted_tag} - {document}"
            
            return formatted_tag
            
        except Exception as e:
            logger.error(f"Erro ao criar descrição para tag '{tag}': {str(e)}")
            # Fallback final: tag formatada simples
            return tag.replace('_', ' ').title()

    def _check_option_selection(self, user_message, conversation_history):
        """
        Verifica se o usuário está selecionando uma opção numérica das sugestões anteriores
        """
        user_input = user_message.strip()
        
        # Verificar se a mensagem é um número
        if not (user_input.isdigit() and 1 <= int(user_input) <= 5):
            return None
        
        # Se não há contexto de conversa suficiente, usar busca inteligente
        if not conversation_history or len(conversation_history) < 2:
            return self._handle_number_without_context(int(user_input))
        
        # Procurar a última mensagem do bot que contém opções
        for msg in reversed(conversation_history):
            if msg['type'] == 'bot' and 'Não tenho certeza do que você está procurando' in msg['message']:
                logger.info(f"Detectada seleção de opção: {user_input}")
                return self._process_option_selection(int(user_input), msg['message'], conversation_history)
        
        # Se chegou aqui, o usuário digitou um número mas não há opções recentes
        return self._handle_number_without_context(int(user_input))

    def _handle_number_without_context(self, number):
        """
        Lida com casos onde o usuário digita um número mas não há opções para selecionar.
        Busca sugestões inteligentes do ChromaDB baseado em tópicos comuns.
        """
        try:
            # Buscar tópicos mais populares/comuns no ChromaDB
            common_topics = self._get_common_topics_from_chromadb()
            
            if number <= len(common_topics):
                # Se o número corresponde a um tópico comum, sugerir diretamente
                suggested_topic = common_topics[number - 1]
                return f"Não há opções para selecionar no momento, mas interpreto que pode estar interessado em <strong>{suggested_topic}</strong>. Tente perguntar especificamente sobre este tópico!"
            else:
                # Número não corresponde a tópicos comuns
                if common_topics:
                    response = "Não há opções para selecionar no momento. Aqui estão alguns tópicos que posso ajudar:\n"
                    for i, topic in enumerate(common_topics[:5], 1):
                        response += f"{i}. <strong>{topic}</strong>\n"
                    response += "\nDigite sua pergunta sobre qualquer um desses tópicos ou outro assunto!"
                    return response
                else:
                    return "Não há opções para selecionar no momento. Faça uma pergunta específica sobre qualquer tópico acadêmico e eu tentarei ajudar!"
        
        except Exception as e:
            logger.error(f"Erro ao lidar com número sem contexto: {str(e)}")
            return "Não há opções para selecionar no momento. Faça uma pergunta específica e eu tentarei ajudar!"

    def _get_common_topics_from_chromadb(self):
        """
        Busca tópicos comuns/populares do ChromaDB para sugestões
        """
        try:
            # Buscar uma amostra de documentos para identificar tópicos comuns
            results = self.knowledge_base.collection.query(
                query_texts=["informações gerais"],
                n_results=10,
                include=['metadatas', 'documents']
            )
            
            if not results['metadatas'] or not results['metadatas'][0]:
                return []
            
            # Extrair e formatar tags únicas
            seen_topics = set()
            topics = []
            
            for metadata in results['metadatas'][0]:
                tag = metadata.get('tag', '')
                if tag and tag not in seen_topics:
                    seen_topics.add(tag)
                    formatted_topic = self._format_tag_as_suggestion(tag)
                    topics.append(formatted_topic)
            
            # Ordenar por relevância (tópicos mais comuns primeiro)
            priority_keywords = ['horário', 'restaurante', 'auxílio', 'matrícula', 'biblioteca']
            priority_topics = []
            other_topics = []
            
            for topic in topics:
                if any(keyword.lower() in topic.lower() for keyword in priority_keywords):
                    priority_topics.append(topic)
                else:
                    other_topics.append(topic)
            
            # Combinar listas priorizando tópicos importantes
            return (priority_topics + other_topics)[:5]  # Máximo 5 sugestões
            
        except Exception as e:
            logger.error(f"Erro ao buscar tópicos comuns: {str(e)}")
            return []

    def _process_option_selection(self, option_number, bot_message, conversation_history):
        """
        Processa a seleção de uma opção específica
        """
        try:
            # Extrair as tags das opções da mensagem anterior
            options_data = self._extract_options_from_message(bot_message)
            
            if not options_data or option_number > len(options_data):
                return "Opção inválida. Tente novamente."
            
            # Obter a tag da opção selecionada
            selected_option = options_data[option_number - 1]
            selected_tag = selected_option['tag']
            
            logger.info(f"Usuário selecionou opção {option_number}: {selected_tag}")
            
            # Buscar resposta específica para esta tag
            result = self._get_response_by_tag(selected_tag)
            
            if result:
                logger.info(f"Fornecendo resposta para tag selecionada: {selected_tag}")
                return result
            else:
                # Fallback inteligente: buscar sugestões relacionadas no ChromaDB
                logger.warning(f"Tag '{selected_tag}' não retornou resposta específica")
                return self._generate_intelligent_fallback(selected_tag, selected_option['description'])
                
        except Exception as e:
            logger.error(f"Erro ao processar seleção de opção: {str(e)}")
            return "Houve um erro ao processar sua seleção. Tente novamente."

    def _extract_options_from_message(self, bot_message):
        """
        Extrai informações das opções da mensagem do bot usando os comentários HTML
        """
        import re
        
        try:
            # Extrair tags dos comentários HTML
            pattern = r'<!-- TAG:([^>]+) -->'
            matches = re.findall(pattern, bot_message)
            
            if matches:
                options = []
                for i, tag in enumerate(matches, 1):
                    # Extrair descrição da linha correspondente
                    line_pattern = f'<strong>{i}\\.</strong>\\s*([^<]+)<!--'
                    desc_match = re.search(line_pattern, bot_message)
                    description = desc_match.group(1).strip() if desc_match else f"Opção {i}"
                    
                    options.append({
                        'tag': tag,
                        'description': description
                    })
                
                logger.info(f"Extraídas {len(options)} opções das tags HTML")
                return options
            
        except Exception as e:
            logger.error(f"Erro ao extrair opções da mensagem: {str(e)}")
        
        # Fallback para mapeamento comum se não conseguir extrair
        logger.info("Usando mapeamento de fallback para opções")
        common_mappings = [
            {'tag': 'esclarecer_horarios', 'description': 'Horários de aulas e funcionamento'},
            {'tag': 'horarios_restaurante', 'description': 'Horários do restaurante'},
            {'tag': 'sala_aula', 'description': 'Localização de salas de aula'},
            {'tag': 'tipos_auxilio', 'description': 'Tipos de auxílio estudantil disponíveis'},
            {'tag': 'matricula', 'description': 'Processo de matrícula e documentos'}
        ]
        
        return common_mappings

    def _get_response_by_tag(self, tag):
        """
        Busca uma resposta específica baseada na tag usando busca inteligente no ChromaDB
        """
        try:
            # Primeiro, buscar por tag exata
            results = self.knowledge_base.collection.query(
                query_texts=[tag],
                n_results=1,
                include=['documents', 'metadatas', 'distances']
            )
            
            if results['documents'] and results['documents'][0]:
                metadata = results['metadatas'][0][0]
                
                if metadata.get('tag') == tag:
                    # Decodificar e retornar resposta
                    responses = json.loads(metadata['responses'])
                    response = random.choice(responses)
                    
                    # Adicionar link se existir
                    if metadata.get('link'):
                        response += f" {metadata['link']}"
                    
                    logger.info(f"Resposta encontrada para tag '{tag}': {response[:100]}...")
                    return response
            
            # Se não encontrar por tag exata, fazer busca expandida baseada na tag
            expanded_queries = self._generate_expanded_queries_from_tag(tag)
            
            for query in expanded_queries:
                logger.debug(f"Tentando busca expandida para '{tag}' com query: '{query}'")
                result = self.knowledge_base.search_similar_intent(
                    query,
                    n_results=1,
                    similarity_threshold=0.4
                )
                
                if result:
                    responses = json.loads(result['metadata']['responses'])
                    response = random.choice(responses)
                    
                    if result['metadata'].get('link'):
                        response += f" {result['metadata']['link']}"
                    
                    logger.info(f"Resposta encontrada via busca expandida para '{tag}': {response[:100]}...")
                    return response
            
            # Última tentativa: busca genérica com a tag formatada
            formatted_tag = tag.replace('_', ' ')
            logger.debug(f"Última tentativa para '{tag}' com busca genérica: '{formatted_tag}'")
            result = self.knowledge_base.search_similar_intent(
                formatted_tag,
                n_results=1,
                similarity_threshold=0.3
            )
            
            if result:
                responses = json.loads(result['metadata']['responses'])
                response = random.choice(responses)
                
                if result['metadata'].get('link'):
                    response += f" {result['metadata']['link']}"
                
                logger.info(f"Resposta encontrada via busca genérica para '{tag}': {response[:100]}...")
                return response
            
            logger.warning(f"Nenhuma resposta encontrada para tag '{tag}'")
            return None
            
        except Exception as e:
            logger.error(f"Erro ao buscar resposta por tag {tag}: {str(e)}")
            return None

    def _generate_expanded_queries_from_tag(self, tag):
        """
        Gera queries expandidas baseadas na tag, transformando-a em termos mais descritivos
        """
        # Transformar a tag em palavras
        tag_words = tag.replace('_', ' ').split()
        
        # Gerar variações da query baseada nas palavras da tag
        queries = []
        
        # Primeira variação: palavras da tag separadas
        if len(tag_words) > 1:
            queries.append(' '.join(tag_words))
        
        # Gerar variações genéricas baseadas nas palavras da tag
        queries.extend([
            f"informações {' '.join(tag_words)}",
            f"como {' '.join(tag_words)}",
            f"sobre {' '.join(tag_words)}"
        ])
        
        # Limitar a 3-4 queries para não sobrecarregar
        return queries[:4]

    def _generate_intelligent_fallback(self, selected_tag, description):
        """
        Gera um fallback inteligente buscando tópicos relacionados no ChromaDB
        """
        try:
            # Buscar tópicos relacionados no ChromaDB baseado na tag
            tag_words = selected_tag.replace('_', ' ').split()
            search_queries = []
            
            # Gerar queries de busca baseadas nas palavras da tag
            for word in tag_words:
                if len(word) > 3:  # Apenas palavras significativas
                    search_queries.append(word)
            
            # Buscar conteúdos relacionados
            related_topics = set()
            for query in search_queries[:3]:  # Limitar a 3 buscas
                try:
                    results = self.knowledge_base.search_similar_intent(
                        query,
                        n_results=5,
                        similarity_threshold=0.3
                    )
                    
                    if results:
                        # Se search_similar_intent retorna um único resultado
                        if isinstance(results, dict):
                            tag = results['metadata'].get('tag', '')
                            if tag and tag != selected_tag:
                                related_topics.add(self._format_tag_as_suggestion(tag))
                        
                    # Buscar múltiplos resultados usando query direta
                    multi_results = self.knowledge_base.collection.query(
                        query_texts=[query],
                        n_results=3,
                        include=['metadatas']
                    )
                    
                    if multi_results['metadatas'] and multi_results['metadatas'][0]:
                        for metadata in multi_results['metadatas'][0]:
                            tag = metadata.get('tag', '')
                            if tag and tag != selected_tag:
                                related_topics.add(self._format_tag_as_suggestion(tag))
                
                except Exception as e:
                    logger.debug(f"Erro ao buscar tópicos relacionados para '{query}': {str(e)}")
                    continue
            
            # Construir resposta com sugestões
            if related_topics:
                # Limitar a 3-4 sugestões
                suggestions = list(related_topics)[:4]
                response = f"Selecionei '{description}'. Para obter informações mais específicas, você pode tentar perguntar sobre:\n"
                
                for suggestion in suggestions:
                    response += f"• <strong>{suggestion}</strong>\n"
                
                response += "\nOu seja mais específico sobre o que precisa saber."
                return response
            else:
                # Fallback genérico se não encontrar tópicos relacionados
                return f"Selecionei '{description}'. Para obter informações mais específicas sobre este tópico, tente reformular sua pergunta com mais detalhes. Por exemplo: <strong>'{description.lower()} como funciona?'</strong> ou <strong>'preciso de ajuda com {description.lower()}'</strong>"
        
        except Exception as e:
            logger.error(f"Erro ao gerar fallback inteligente para '{selected_tag}': {str(e)}")
            return f"Selecionei '{description}', mas preciso de mais detalhes para ajudar melhor. Pode reformular sua pergunta?"

    def _format_tag_as_suggestion(self, tag):
        """
        Formata uma tag como sugestão amigável para o usuário
        """
        # Remover underscore e capitalizar
        formatted = tag.replace('_', ' ').title()
        
        return formatted

    def _enrich_query_with_memory(self, query: str, relevant_memory: List[ConversationTurn], contextual_keywords: List[str]) -> str:
        """
        Enriquece a query com informações da memória de curto prazo
        """
        enriched_parts = [query]
        
        # Adicionar palavras-chave do contexto da memória
        if contextual_keywords:
            enriched_parts.extend(contextual_keywords[:3])  # Máximo 3 palavras-chave
            logger.info(f"Adicionadas palavras-chave da memória: {contextual_keywords[:3]}")
        
        # Adicionar contexto dos turnos mais relevantes
        if relevant_memory:
            for turn in relevant_memory[:2]:  # Máximo 2 turnos mais relevantes
                # Adicionar intent_tag se disponível
                if turn.intent_tag:
                    enriched_parts.append(turn.intent_tag.replace('_', ' '))
                
                # Adicionar palavras importantes da mensagem anterior
                important_words = self._extract_important_words(turn.user_message)
                enriched_parts.extend(important_words[:2])  # Máximo 2 palavras por turno
        
        enriched_query = " ".join(enriched_parts)
        
        if enriched_query != query:
            logger.info(f"Query enriquecida: '{query}' -> '{enriched_query}'")
        
        return enriched_query
    
    def _extract_important_words(self, text: str) -> List[str]:
        """
        Extrai palavras importantes de um texto
        """
        stop_words = ['para', 'como', 'onde', 'quando', 'porque', 'qual', 'quem', 'que', 'isso', 'esta', 'esse', 'uma', 'um', 'de', 'da', 'do', 'na', 'no', 'em', 'com', 'por', 'se', 'mais', 'muito', 'bem', 'ja', 'so', 'ate', 'mas', 'ou', 'sim', 'nao']
        
        words = self.normalize_text(text).split()
        important_words = []
        
        for word in words:
            if len(word) > 3 and word not in stop_words:
                important_words.append(word)
        
        return important_words
    
    def clear_short_term_memory(self):
        """
        Limpa a memória de curto prazo
        """
        self.short_term_memory.clear()
        logger.info("Memória de curto prazo limpa")
    
    def get_memory_stats(self) -> Dict:
        """
        Retorna estatísticas da memória de curto prazo
        """
        return self.short_term_memory.get_memory_stats()
    
    def get_memory_context_summary(self) -> str:
        """
        Retorna um resumo do contexto atual da memória
        """
        stats = self.get_memory_stats()
        if stats['total_turns'] == 0:
            return "Nenhum contexto na memória."
        
        summary = f"Memória: {stats['total_turns']} turnos"
        if stats['recent_intents']:
            summary += f", tópicos recentes: {', '.join(stats['recent_intents'])}"
        if stats['avg_similarity'] > 0:
            summary += f", similaridade média: {stats['avg_similarity']:.2f}"
        
        return summary

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
