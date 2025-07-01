#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemplo de funcionamento do sistema de sugestões de opções
"""

from chat import SemanticChatBot
import logging

# Configurar logging para ver o processo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def demo_suggestion_system():
    """
    Demonstra o funcionamento do sistema de sugestões
    """
    print("=== Demo: Sistema de Sugestões de Opções ===")
    print("Testando mensagens com baixa certeza...\n")
    
    bot = SemanticChatBot()
    
    # Casos de teste que devem acionar o sistema de sugestões
    test_cases = [
        "aula",           # Ambíguo - pode ser horário, sala, etc.
        "ajuda",          # Muito genérico
        "informação",     # Muito vago
        "documento",      # Pode ser vários tipos
        "auxilio",        # Pode encontrar, mas com baixa certeza
    ]
    
    for i, message in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Teste {i}: '{message}'")
        print(f"{'='*50}")
        
        response = bot.get_response_with_context(message)
        
        print(f"\nResposta:")
        print(response)
        print("\n" + "-"*50)

def demo_normal_vs_suggestion():
    """
    Compara resposta normal vs sugestões
    """
    print("\n\n=== Comparação: Normal vs Sugestões ===")
    
    bot = SemanticChatBot()
    
    # Mensagem específica (deve dar resposta direta)
    print("\n1. Mensagem ESPECÍFICA: 'horário das aulas'")
    response1 = bot.get_response_with_context("horário das aulas")
    print(f"Resposta: {response1[:100]}...")
    
    # Mensagem ambígua (deve dar sugestões)
    print("\n2. Mensagem AMBÍGUA: 'aula'")
    response2 = bot.get_response_with_context("aula")
    print(f"Resposta: {response2}")

if __name__ == "__main__":
    try:
        demo_suggestion_system()
        demo_normal_vs_suggestion()
        print("\n✅ Demo concluída!")
    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
