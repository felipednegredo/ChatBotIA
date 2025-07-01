class Chatbox{
    constructor(){
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }

        this.state = false;
        this.messages = [];
        this.isTyping = false; // Controle para evitar mensagens duplicadas
    }

    display(){
        const {openButton, chatBox, sendButton} = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox))

        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) =>{
            if(key == "Enter"){
                this.onSendButton(chatBox)
            }
        })

        // Adicionar mensagem de boas-vindas contextual
        if (this.messages.length === 0) {
            let welcomeMsg = {
                name: "Sam", 
                message: "Olá! Sou o Bot IFRS. Posso ajudar com informações sobre o campus. Como posso ajudá-lo hoje?"
            };
            this.messages.push(welcomeMsg);
            this.updateChatText(chatBox);
        }
    }

        toggleState(chatBox){
    this.state = !this.state;

    // mostra ou esconde o chat
    if(this.state) {
        chatBox.classList.add('chatbox--active')
    } else {
        chatBox.classList.remove('chatbox--active')
    }
}


    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value.trim();
        
        if (text1 === "" || this.isTyping){
            return;
        }

        // Verificar comandos especiais
        if (text1.toLowerCase() === '/limpar' || text1.toLowerCase() === '/clear') {
            this.clearHistory(chatbox);
            textField.value = '';
            return;
        }

        this.isTyping = true;
        let msg1 = {name: 'User', message: text1 };
        this.messages.push(msg1);
        this.updateChatText(chatbox);

        // Mostrar indicador de digitação
        this.showTypingIndicator(chatbox);

        // Enviar mensagem com contexto melhorado
        fetch($SCRIPT_ROOT + '/predict', {
            method: 'POST',
            body: JSON.stringify({
                message: text1,
                context: this.getRecentContext()
            }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(r => r.json())
        .then(r => {
            this.hideTypingIndicator();
            let msg2 = {name: "Sam", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox);
            textField.value = '';
            this.isTyping = false;
        }).catch((error) => {
            console.error('Error:', error);
            this.hideTypingIndicator();
            let errorMsg = {name: "Sam", message: "Desculpe, ocorreu um erro. Tente novamente." };
            this.messages.push(errorMsg);
            this.updateChatText(chatbox);
            textField.value = '';
            this.isTyping = false;
        });
    }

    getRecentContext() {
        // Retornar as últimas 5 mensagens para contexto
        return this.messages.slice(-5).map(msg => ({
            type: msg.name === 'User' ? 'user' : 'bot',
            message: msg.message
        }));
    }

    showTypingIndicator(chatbox) {
        let typingMsg = {name: "Sam", message: "Bot está digitando..." };
        this.messages.push(typingMsg);
        this.updateChatText(chatbox);
    }

    hideTypingIndicator() {
        // Remove a última mensagem se for o indicador de digitação
        if (this.messages.length > 0 && 
            this.messages[this.messages.length - 1].message === "Bot está digitando...") {
            this.messages.pop();
        }
    }

    clearHistory(chatbox) {
        // Limpar histórico local
        this.messages = [];
        
        // Adicionar mensagem de confirmação
        let clearMsg = {name: "Sam", message: "Histórico da conversa foi limpo. Como posso ajudá-lo?" };
        this.messages.push(clearMsg);
        
        // Limpar histórico no servidor
        fetch($SCRIPT_ROOT + '/clear_history', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
        }).catch((error) => {
            console.error('Erro ao limpar histórico no servidor:', error);
        });
        
        this.updateChatText(chatbox);
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, number) {
            if (item.name === "Sam")
            {
                // Adicionar classe especial para indicador de digitação
                let additionalClass = item.message === "Bot está digitando..." ? " typing-indicator" : "";
                html += '<div class="messages__item messages__item--visitor' + additionalClass + '">' + item.message + '</div>'
            }
            else
            {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
        
        // Scroll para a última mensagem
        chatmessage.scrollTop = chatmessage.scrollHeight;
    }
}

const chatbox = new Chatbox();
chatbox.display();