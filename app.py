from flask import Flask, render_template, request, jsonify, session
import uuid
from datetime import datetime

# Importar o sistema semântico
from chat import SemanticChatBot

app = Flask(__name__)
app.secret_key = 'your-secret-key-for-sessions'  # Necessário para sessões

# Inicializar o chatbot semântico
semantic_bot = SemanticChatBot()

# Dicionário para armazenar histórico de conversas por sessão
conversation_history = {}


@app.get("/")
def index_get():
    # Gerar um ID único de sessão se não existir
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        conversation_history[session['session_id']] = []
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    print(f"Mensagem recebida: {text}")
    
    # Obter ou criar ID da sessão
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_id = session['session_id']
    
    # Garantir que o histórico existe para esta sessão
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # Obter histórico da conversa (últimas 5 mensagens)
    history = conversation_history[session_id]
    recent_history = history[-5:] if len(history) > 5 else history
    
    # Usar o novo sistema semântico com contexto
    response = semantic_bot.get_response_with_context(text, recent_history)
    
    # Adicionar mensagem do usuário e resposta ao histórico
    conversation_history[session_id].append({
        'type': 'user',
        'message': text,
        'timestamp': datetime.now().isoformat()
    })
    conversation_history[session_id].append({
        'type': 'bot',
        'message': response,
        'timestamp': datetime.now().isoformat()
    })
    
    # Manter apenas as últimas 20 mensagens para não sobrecarregar a memória
    if len(conversation_history[session_id]) > 20:
        conversation_history[session_id] = conversation_history[session_id][-20:]
    
    message = {"answer": response}
    return jsonify(message)


@app.get("/stats")
def get_stats():
    """Endpoint para obter estatísticas da base de conhecimento"""
    try:
        stats = semantic_bot.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/clear_history")
def clear_history():
    """Endpoint para limpar o histórico da conversa"""
    try:
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        session_id = session['session_id']
        
        # Garantir que o histórico existe e então limpar
        conversation_history[session_id] = []
        
        return jsonify({"status": "success", "message": "Histórico limpo com sucesso"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("=== Bot IFRS - Sistema Semântico ===")
    print("Inicializando servidor Flask...")
    app.run(debug=True)