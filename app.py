from flask import Flask, render_template, request, jsonify

# Importar o sistema semântico
from chat import SemanticChatBot

app = Flask(__name__)

# Inicializar o chatbot semântico
semantic_bot = SemanticChatBot()


@app.get("/")
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    print(f"Mensagem recebida: {text}")
    # Usar o novo sistema semântico
    response = semantic_bot.get_response(text)
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


if __name__ == "__main__":
    print("=== Bot IFRS - Sistema Semântico ===")
    print("Inicializando servidor Flask...")
    app.run(debug=True)