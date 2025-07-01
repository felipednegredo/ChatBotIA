#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste do sistema de seleção numérica de opções
"""

from chat import SemanticChatBot
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_option_selection():
    """
    Testa o sistema de seleção numérica de opções
    """
    print("=== Teste: Sistema de Seleção Numérica ===")
    
    bot = SemanticChatBot()
    conversation_history = []
    
    def add_to_history(user_msg, bot_response):
        conversation_history.append({'type': 'user', 'message': user_msg})
        conversation_history.append({'type': 'bot', 'message': bot_response})
    
    # Primeiro, fazer uma pergunta ambígua que deve gerar opções
    print("\n1. Pergunta ambígua: 'aula'")
    print("-" * 40)
    
    user_msg1 = "aula"
    response1 = bot.get_response_with_context(user_msg1, conversation_history)
    print(f"Usuário: {user_msg1}")
    print(f"Bot: {response1}")
    add_to_history(user_msg1, response1)
    
    # Verificar se a resposta contém opções numeradas
    if "1." in response1 and "2." in response1:
        print("\n✅ Sistema ofereceu opções numeradas!")
        
        # Agora testar seleção numérica
        print("\n2. Seleção da opção 1")
        print("-" * 40)
        
        user_msg2 = "1"
        response2 = bot.get_response_with_context(user_msg2, conversation_history)
        print(f"Usuário: {user_msg2}")
        print(f"Bot: {response2}")
        
        # Verificar se a resposta não é sobre seleção de opção inválida
        if "Opção inválida" not in response2 and "Houve um erro" not in response2:
            print("\n✅ Sistema interpretou seleção numérica corretamente!")
        else:
            print("\n❌ Problema na interpretação da seleção numérica")
    else:
        print("\n❌ Sistema não ofereceu opções numeradas")

def test_edge_cases():
    """
    Testa casos extremos
    """
    print("\n\n=== Teste: Casos Extremos ===")
    
    bot = SemanticChatBot()
    
    # Teste 1: Número sem contexto de opções
    print("\n1. Número sem contexto de opções")
    response = bot.get_response_with_context("1", [])
    print(f"Resposta para '1' sem contexto: {response[:100]}...")
    
    # Teste 2: Número inválido (muito alto)
    print("\n2. Número inválido (muito alto)")
    fake_history = [
        {'type': 'user', 'message': 'teste'},
        {'type': 'bot', 'message': 'Não tenho certeza do que você está procurando. Você quis dizer alguma dessas opções?<br><br><strong>1.</strong> Opção 1<br><strong>2.</strong> Opção 2<br>'}
    ]
    response = bot.get_response_with_context("10", fake_history)
    print(f"Resposta para '10' (inválido): {response}")

if __name__ == "__main__":
    try:
        test_option_selection()
        test_edge_cases()
        print("\n✅ Teste concluído!")
    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
