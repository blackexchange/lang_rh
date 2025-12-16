# Sistema de Análise e Reformulação de Currículos

Sistema inteligente com dois agentes especializados para análise e reformulação de currículos baseado em vagas de emprego.

## 🚀 Funcionalidades

### Agente Analisador
- Analisa profundamente o currículo e a vaga
- Identifica pontos fortes e fracos
- Detecta habilidades faltantes e subutilizadas
- Gera recomendações específicas
- Calcula score de alinhamento

### Agente Reformulador
- Reformula o currículo baseado na análise
- Mantém todas as informações verdadeiras
- Aplica recomendações da análise
- Destaca habilidades relevantes para a vaga
- Melhora estrutura e clareza

## 📋 Pré-requisitos

- Python 3.8 ou superior
- pip

## 🔧 Instalação

### 1. Clone o repositório (ou navegue até a pasta do projeto)

```bash
cd lang_rh
```

### 2. Crie e ative o ambiente virtual

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

Ou instale manualmente:
```bash
pip install langchain-groq langchain-community langchain-core PyMuPDF docling streamlit python-dotenv pandas
```

### 4. Configure as variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto com:

```env
GROQ_API_KEY=sua_chave_api_groq_aqui
```

## 🎯 Como Usar

### 1. Execute a aplicação

```bash
streamlit run app.py
```

### 2. Acesse no navegador

Abra: `http://localhost:8501`

### 3. Fluxo de trabalho

1. **Upload do Currículo**: Envie um PDF do currículo
2. **Análise Inicial**: O sistema faz a triagem automática
3. **Análise Detalhada**: Clique em "🚀 Executar Análise Detalhada"
4. **Reformulação**: Após a análise, clique em "🔄 Reformular Currículo"
5. **Download**: Baixe o currículo reformulado em formato Markdown

## 📁 Estrutura do Projeto

```
lang_rh/
├── app.py                 # Interface Streamlit principal
├── utils_proj03.py        # Funções utilitárias e agentes
├── requirements.txt       # Dependências do projeto
├── .env                   # Variáveis de ambiente (criar)
├── .gitignore            # Arquivos ignorados pelo git
└── README.md             # Este arquivo
```

## 🛠️ Tecnologias Utilizadas

- **Streamlit**: Interface web
- **LangChain**: Framework para LLMs
- **Groq**: API de linguagem
- **Docling**: Processamento de documentos PDF
- **Pandas**: Manipulação de dados

## 📝 Notas

- O sistema mantém todas as informações verdadeiras do currículo original
- As reformulações são baseadas em recomendações da análise
- O currículo reformulado é salvo em formato Markdown

## 🔒 Segurança

- Não compartilhe seu arquivo `.env` com chaves de API
- Mantenha o `.env` no `.gitignore`

## 📄 Licença

Este projeto é para uso interno/educacional.

