# 🎯 Sistema de Sugestões de Opções

## Funcionalidade Implementada

Quando o chatbot **não tem certeza** sobre a intenção do usuário (similaridade < 0.5), ele agora oferece **opções similares** em vez de dar uma resposta genérica.

## Como Funciona

### 🔍 **Detecção de Incerteza**

```python
# Se a similaridade for baixa (< 0.5)
if similarity < 0.5:
    return self._offer_similar_options(normalized_message, result, conversation_history)
```

### 📋 **Geração de Opções**

1. **Busca ampla** no ChromaDB (4 resultados)
2. **Remove duplicatas** por tag
3. **Filtra** similaridades muito baixas (< 0.1)
4. **Cria descrições amigáveis** para cada opção
5. **Apresenta as 3 melhores** opções

### 💬 **Formato da Resposta**

```html
Não tenho certeza do que você está procurando. Você quis dizer alguma dessas opções?

1. Horários de aulas e funcionamento
2. Localização de salas de aula  
3. Informações sobre cursos oferecidos

Você pode reformular sua pergunta ou escolher uma das opções acima sendo mais específico.
```

## Exemplos de Uso

### ❌ **Antes (resposta genérica):**
```
Usuário: "aula"
Bot: "Desculpe, não entendi sua pergunta..."
```

### ✅ **Agora (sugestões úteis):**
```
Usuário: "aula"
Bot: "Não tenho certeza do que você está procurando. Você quis dizer:
1. Horários de aulas e funcionamento
2. Localização de salas de aula
3. Informações sobre cursos oferecidos"
```

## Configurações

### 🎚️ **Thresholds de Ativação**

- **Threshold principal**: 0.6 (padrão)
- **Threshold de incerteza**: 0.5 (ativa sugestões)
- **Threshold mínimo**: 0.1 (filtra opções)

### 📝 **Tags Mapeadas**

O sistema inclui descrições amigáveis para tags comuns:

```python
tag_descriptions = {
    'horario': 'Horários de aulas e funcionamento',
    'cardapio': 'Cardápio do restaurante universitário',
    'tipos_auxilio': 'Tipos de auxílio estudantil disponíveis',
    'matricula': 'Processo de matrícula e documentos',
    # ... mais 15 tags mapeadas
}
```

## Casos de Ativação

### 🎯 **Quando Oferece Sugestões**

1. **Mensagens ambíguas**: "aula", "ajuda", "informação"
2. **Termos genéricos**: "documento", "auxílio", "curso"
3. **Baixa similaridade**: Qualquer resultado < 0.5
4. **Nenhum resultado**: Quando não encontra nada com threshold alto

### ✅ **Quando Dá Resposta Direta**

1. **Alta similaridade**: Resultado >= 0.5
2. **Mensagens específicas**: "horário das aulas", "cardápio do restaurante"
3. **Contexto claro**: Quando há histórico relevante

## Logs do Sistema

```
INFO:chat:Similaridade baixa (0.449), oferecendo opções alternativas...
INFO:chat:Oferecendo 3 opções similares
```

## Benefícios

### 🎯 **Melhor UX**
- Usuário recebe **opções úteis** em vez de mensagem de erro
- **Orientação clara** sobre como reformular a pergunta

### 🧠 **Inteligência Aprimorada**
- Sistema **aprende** das ambiguidades
- **Adapta** as sugestões ao contexto da conversa

### 📊 **Redução de Frustração**
- Menos "não entendi"
- Mais **engajamento** do usuário
- **Descoberta** de funcionalidades

## Para Testar

```bash
python demo_suggestions.py
```

### Teste no Chatbot:
- Digite "aula" → deve oferecer opções
- Digite "horário das aulas" → resposta direta
- Digite "ajuda" → múltiplas sugestões

O sistema agora é muito mais **inteligente e útil** quando há ambiguidade! 🚀
