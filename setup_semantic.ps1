# Script PowerShell para configurar o sistema de chatbot semântico
# Compatível com Windows PowerShell

Write-Host "🚀 Configuração do Sistema de Chatbot Semântico" -ForegroundColor Green
Write-Host "=" * 50

# Verificar Python
Write-Host "🐍 Verificando instalação do Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ $pythonVersion encontrado!" -ForegroundColor Green
} catch {
    Write-Host "❌ Python não encontrado! Instale Python 3.8+ primeiro." -ForegroundColor Red
    exit 1
}

# Verificar se estamos no diretório correto
if (-not (Test-Path "intents.json")) {
    Write-Host "❌ Execute este script no diretório do projeto (onde está o intents.json)" -ForegroundColor Red
    exit 1
}

# Instalar dependências
Write-Host "📦 Instalando dependências..." -ForegroundColor Yellow
try {
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    Write-Host "✅ Dependências instaladas com sucesso!" -ForegroundColor Green
} catch {
    Write-Host "❌ Erro ao instalar dependências: $_" -ForegroundColor Red
    exit 1
}

# Executar script de configuração Python
Write-Host "⚙️ Executando configuração do sistema..." -ForegroundColor Yellow
try {
    python setup_semantic.py
    Write-Host "✅ Sistema configurado!" -ForegroundColor Green
} catch {
    Write-Host "❌ Erro na configuração: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`n🎉 Configuração concluída com sucesso!" -ForegroundColor Green
Write-Host "`n📋 Próximos passos:" -ForegroundColor Cyan
Write-Host "   1. Execute: python app.py" -ForegroundColor White
Write-Host "   2. Acesse: http://localhost:5000" -ForegroundColor White
Write-Host "   3. Para teste interativo: python chat_semantic.py" -ForegroundColor White

# Perguntar se quer iniciar o servidor automaticamente
$response = Read-Host "`n❓ Deseja iniciar o servidor agora? (s/n)"
if ($response -eq "s" -or $response -eq "S" -or $response -eq "sim") {
    Write-Host "🌐 Iniciando servidor Flask..." -ForegroundColor Yellow
    python app.py
}
