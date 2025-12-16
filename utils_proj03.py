
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
import json
import pandas as pd
import csv
import streamlit as st

# Importa PyMuPDF (mais simples e confiável)
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Tenta importar docling como opção alternativa (pode ter problemas de permissão no Windows)
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except Exception:
    DOCLING_AVAILABLE = False

# Verifica se pelo menos uma biblioteca está disponível
if not PYMUPDF_AVAILABLE and not DOCLING_AVAILABLE:
    import sys
    print("ERRO: Nenhuma biblioteca de PDF disponível. Instale PyMuPDF: pip install PyMuPDF")
    sys.exit(1)

def load_llm(id_model, temperature):
  llm = ChatGroq(
      model=id_model,
      temperature=temperature,
      max_tokens=None,
      timeout=None,
      max_retries=2,
  )
  return llm


def format_res(res, return_thinking=False):
  res = res.strip()

  if return_thinking:
    res = res.replace("<think>", "[pensando...] ")
    res = res.replace("</think>", "\n---\n")

  else:
    if "</think>" in res:
      res = res.split("</think>")[-1].strip()

  return res


def parse_doc(file_path):
  """
  Extrai texto de um arquivo PDF usando PyMuPDF (padrão) ou docling como alternativa.
  
  Args:
    file_path: Caminho para o arquivo PDF
    
  Returns:
    str: Conteúdo do PDF em formato texto
  """
  # Usa PyMuPDF por padrão (mais simples e confiável)
  if PYMUPDF_AVAILABLE:
    try:
      doc = fitz.open(file_path)
      content = ""
      for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text:
          content += f"\n--- Página {page_num + 1} ---\n"
          content += text
      doc.close()
      return content.strip()
    except Exception as e:
      st.error(f"Erro ao processar PDF com PyMuPDF: {e}")
      # Tenta docling como fallback se PyMuPDF falhar
      if DOCLING_AVAILABLE:
        st.info("Tentando docling como alternativa...")
      else:
        raise
  
  # Fallback para docling (se PyMuPDF não estiver disponível ou falhar)
  if DOCLING_AVAILABLE:
    try:
      # Configura variável de ambiente para evitar problemas de cache
      os.environ.setdefault('HF_HOME', os.path.join(os.getcwd(), '.hf_cache'))
      
      converter = DocumentConverter()
      result = converter.convert(file_path)
      content = result.document.export_to_markdown()
      return content
    except (OSError, PermissionError) as e:
      st.error(f"Erro de permissão com docling: {e}")
      st.info("Sugestão: Use PyMuPDF que não requer downloads de modelos.")
      raise
    except Exception as e:
      st.error(f"Erro ao processar PDF com docling: {e}")
      raise
  
  # Se nenhuma biblioteca estiver disponível
  raise ImportError("Nenhuma biblioteca de PDF disponível. Instale PyMuPDF: pip install PyMuPDF")


def parse_res_llm(response_text: str, required_fields: list) -> dict:
    try:
        # Remove a parte do raciocínio (<think>...</think>)
        if "</think>" in response_text:
            response_text = response_text.split("</think>")[-1].strip()

        # Localiza o JSON e faz o parsing
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            raise json.JSONDecodeError("Nenhum JSON encontrado na resposta", response_text, 0)

        json_str = response_text[start_idx:end_idx]
        info_cv = json.loads(json_str)

        for field in required_fields:
            if field not in info_cv:
                info_cv[field] = []

        return info_cv

    except json.JSONDecodeError:
        #Erro ao interpretar a resposta do modelo
        return


def save_json_cv(new_data, path_json, key_name="name"):
    # Carrega o JSON existente, se houver
    if os.path.exists(path_json):
        with open(path_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    if isinstance(data, dict):
        data = [data]

    # Verifica se já existe um currículo com o mesmo nome
    candidates = [entry.get(key_name) for entry in data]
    if new_data.get(key_name) in candidates:
        st.warning(f"Currículo '{new_data.get(key_name)}' já registrado. Ignorando.")
        return

    # Adiciona e salva
    data.append(new_data)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json_cv(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_cv_content_from_json(cv_data):
    """
    Gera conteúdo de currículo em texto a partir dos dados estruturados do JSON.
    
    Args:
        cv_data: Dicionário com os dados do currículo do JSON
    
    Returns:
        str: Conteúdo do currículo em formato texto
    """
    content = f"""# {cv_data.get('name', 'Currículo')}

## Resumo Profissional
{cv_data.get('summary', '')}

## Área de Atuação
{cv_data.get('area', '')}

## Formação Acadêmica
{cv_data.get('education', '')}

## Competências e Habilidades
{', '.join(cv_data.get('skills', []))}

## Pontos Fortes
{chr(10).join(['- ' + s for s in cv_data.get('strengths', [])])}

## Áreas para Desenvolvimento
{chr(10).join(['- ' + a for a in cv_data.get('areas_for_development', [])])}

## Recomendações
{cv_data.get('final_recommendations', '')}
"""
    return content


def generate_analysis_from_json(cv_data):
    """
    Gera uma análise estruturada a partir dos dados do JSON.
    
    Args:
        cv_data: Dicionário com os dados do currículo do JSON
    
    Returns:
        dict: Análise estruturada
    """
    analysis = {
        "analysis_summary": f"Análise do perfil de {cv_data.get('name', 'candidato')}. {cv_data.get('summary', '')}",
        "alignment_score": float(cv_data.get('score', 0.0)),
        "strengths": cv_data.get('strengths', []),
        "weaknesses": cv_data.get('areas_for_development', []),
        "missing_skills": [],
        "underutilized_skills": [],
        "recommendations": [cv_data.get('final_recommendations', '')],
        "key_improvements": cv_data.get('important_considerations', [])
    }
    return analysis


def show_cv_result(result: dict):
    md = f"### 📄 Análise e Resumo do Currículo\n"
    if "name" in result:
        md += f"- **Nome:** {result['name']}\n"
    if "area" in result:
        md += f"- **Área de Atuação:** {result['area']}\n"
    if "skills" in result:
        md += f"- **Competências:** {', '.join(result['skills'])}\n"
    if "summary" in result:
        md += f"- **Resumo do Perfil:** {result['summary']}\n"
    if "interview_questions" in result:
        md += f"- **Perguntas sugeridas:**\n"
        md += "\n".join([f"  - {q}" for q in result["interview_questions"]]) + "\n"
    if "strengths" in result:
        md += f"- **Pontos fortes (ou Alinhamentos):**\n"
        md += "\n".join([f"  - {s}" for s in result["strengths"]]) + "\n"
    if "areas_for_development" in result:
        md += f"- **Pontos a desenvolver (ou Desalinhamentos):**\n"
        md += "\n".join([f"  - {a}" for a in result["areas_for_development"]]) + "\n"
    if "important_considerations" in result:
        md += f"- **Pontos de atenção:**\n"
        md += "\n".join([f"  - {i}" for i in result["important_considerations"]]) + "\n"
    if "final_recommendations" in result:
        md += f"- **Conclusão e recomendações:** {result['final_recommendations']}\n"
    return md


def save_job_to_csv(data, filename):
    headers = ['title', 'description', 'details']
    file_exists = os.path.exists(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=';')
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


def load_job(csv_path):
  try:
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    job = df.iloc[-1]
    prompt_text = f"""
    **Vaga para {job['title']}**

    **Descrição da Vaga:**
    {job['description']}

    **Detalhes Completos:**
    {job['details']}
    """

    return prompt_text.strip()

  except FileNotFoundError:
    return "Erro: Arquivo de vagas não encontrado"

def process_cv(schema, job_details, prompt_template, prompt_score, llm, file_path):

  if file_path:
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

  content = parse_doc(file_path)

  chain = prompt_template | llm
  output = chain.invoke({"schema": schema, "cv": content, "job": job_details, "prompt_score": prompt_score})

  res = format_res(output.content)

  return output, res


def display_json_table(path_json):
  with open(path_json, "r", encoding="utf-8") as f:
    data = json.load(f)

  df = pd.DataFrame(data)
  return df


# ============================================
# AGENTE ANALISADOR - Analisa currículo e vaga
# ============================================

def create_analysis_prompt_template():
    """Cria o template de prompt para o agente analisador"""
    return ChatPromptTemplate.from_template("""
Você é um especialista em Recursos Humanos com vasta experiência em análise de currículos.
Sua tarefa é analisar profundamente o currículo e a vaga, gerando uma análise detalhada e estruturada.

INSTRUÇÕES:
1. Analise o currículo fornecido em detalhes
2. Compare com os requisitos e características da vaga
3. Identifique pontos fortes, fracos, alinhamentos e desalinhamentos
4. Gere recomendações específicas para melhorar o currículo
5. Retorne APENAS um JSON válido com a estrutura abaixo

SCHEMA DE RESPOSTA (JSON):
{{
  "analysis_summary": "Resumo executivo da análise (2-3 parágrafos)",
  "alignment_score": 0.0,
  "strengths": [
    "Lista de pontos fortes e alinhamentos com a vaga"
  ],
  "weaknesses": [
    "Lista de pontos fracos e desalinhamentos com a vaga"
  ],
  "missing_skills": [
    "Habilidades mencionadas na vaga que não estão no currículo"
  ],
  "underutilized_skills": [
    "Habilidades do candidato que poderiam ser melhor destacadas"
  ],
  "recommendations": [
    "Recomendações específicas para melhorar o currículo"
  ],
  "key_improvements": [
    "Melhorias prioritárias que devem ser feitas no currículo"
  ]
}}

CURRÍCULO:
'{cv}'

VAGA:
'{job}'

Retorne APENAS o JSON, sem explicações adicionais.
""")


def analyze_cv_and_job(llm, cv_content, job_details):
    """
    Agente Analisador: Analisa o currículo e a vaga, gerando análise detalhada
    
    Args:
        llm: Modelo de linguagem
        cv_content: Conteúdo do currículo (texto/markdown)
        job_details: Detalhes da vaga (texto)
    
    Returns:
        dict: Análise estruturada em JSON
    """
    prompt_template = create_analysis_prompt_template()
    chain = prompt_template | llm
    
    output = chain.invoke({
        "cv": cv_content,
        "job": job_details
    })
    
    res = format_res(output.content)
    
    # Parse do JSON
    try:
        start_idx = res.find('{')
        end_idx = res.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            raise json.JSONDecodeError("Nenhum JSON encontrado", res, 0)
        
        json_str = res[start_idx:end_idx]
        analysis = json.loads(json_str)
        return analysis
    except json.JSONDecodeError as e:
        st.error(f"Erro ao processar análise: {e}")
        return None


# ============================================
# AGENTE REFORMULADOR - Reformula o currículo
# ============================================

def create_rewrite_prompt_template():
    """Cria o template de prompt para o agente reformulador"""
    return ChatPromptTemplate.from_template("""
Você é um especialista em redação de currículos profissionais.
Sua tarefa é criar um currículo reformulado usando o TEMPLATE fornecido como base estrutural, preenchendo as seções com informações baseadas na análise e no currículo original.

INSTRUÇÕES IMPORTANTES:
1. Use o TEMPLATE fornecido como estrutura base - mantenha a mesma formatação, seções e estilo
2. Preencha as seções do template com informações baseadas na análise e no currículo original
3. Você PODE e DEVE criar/inventar conteúdo relevante para as seções, desde que seja coerente com:
   - As habilidades e experiências do candidato
   - Os pontos fortes identificados na análise
   - Os requisitos da vaga
   - As recomendações da análise
4. Seções a preencher:
   - **Resumo Profissional**: Crie um resumo que destaque os pontos fortes e alinhamento com a vaga
   - **Experiências**: Crie descrições de experiências profissionais relevantes, destacando conquistas e habilidades
   - **Projetos e Consultorias Relevantes**: Crie projetos que demonstrem as habilidades necessárias para a vaga
   - **Hard Skills**: Liste habilidades técnicas relevantes, priorizando as mencionadas na vaga
   - **Soft Skills**: Liste habilidades comportamentais relevantes
5. Use linguagem {style}
6. Mantenha a estrutura e formatação exata do template
7. {focus_instruction}
8. {highlight_instruction}
9. {strengths_instruction}
10. Seja criativo mas realista - crie conteúdo que faça sentido para o perfil do candidato

TEMPLATE DE CV (use esta estrutura):
'{cv_template}'

CURRÍCULO ORIGINAL (referência de informações):
'{original_cv}'

ANÁLISE REALIZADA:
'{analysis}'

VAGA DE REFERÊNCIA:
'{job}'

Crie um currículo completo preenchendo o template com informações relevantes baseadas na análise e no currículo original.
Mantenha a estrutura exata do template, apenas preenchendo as seções com conteúdo novo e relevante.
Retorne o currículo completo no mesmo formato do template.
""")


def rewrite_cv(llm, original_cv_content, analysis, job_details, cv_template=None, rewrite_options=None):
    """
    Agente Reformulador: Reformula o currículo baseado na análise usando um template
    
    Args:
        llm: Modelo de linguagem
        original_cv_content: Conteúdo original do currículo
        analysis: Análise gerada pelo agente analisador (dict)
        job_details: Detalhes da vaga
        cv_template: Template de CV para usar como estrutura base (opcional)
        rewrite_options: Dicionário com opções de reformulação (opcional)
    
    Returns:
        str: Currículo reformulado em markdown
    """
    # Validação de entrada
    if not original_cv_content:
        raise ValueError("Conteúdo do currículo original não pode estar vazio")
    if not analysis:
        raise ValueError("Análise não pode estar vazia")
    if not job_details:
        raise ValueError("Detalhes da vaga não podem estar vazios")
    if not cv_template:
        raise ValueError("Template de CV é obrigatório")
    
    # Opções padrão
    if rewrite_options is None:
        rewrite_options = {
            "focus": "all",
            "style": "professional",
            "highlight_missing": True,
            "emphasize_strengths": True,
            "template": "1"
        }
    
    # Define instruções baseadas nas opções
    focus_map = {
        "all": "Foque em todos os aspectos do currículo",
        "skills": "Dê ênfase especial às habilidades e competências técnicas",
        "experience": "Dê ênfase especial à experiência profissional e histórico de trabalho",
        "summary": "Dê ênfase especial ao resumo profissional e objetivo"
    }
    focus_instruction = focus_map.get(rewrite_options.get("focus", "all"), focus_map["all"])
    
    highlight_instruction = ""
    if rewrite_options.get("highlight_missing", True):
        highlight_instruction = "Destaque claramente as habilidades mencionadas na vaga que estão faltando no currículo, mas NÃO invente que o candidato as possui."
    else:
        highlight_instruction = "Não é necessário destacar habilidades faltantes."
    
    strengths_instruction = ""
    if rewrite_options.get("emphasize_strengths", True):
        strengths_instruction = "Enfatize e destaque os pontos fortes e alinhamentos identificados na análise."
    else:
        strengths_instruction = "Mantenha os pontos fortes sem ênfase especial."
    
    # Converte a análise para texto estruturado
    analysis_text = f"""
RESUMO DA ANÁLISE:
{analysis.get('analysis_summary', 'N/A')}

PONTOS FORTES:
{chr(10).join(['- ' + str(s) for s in analysis.get('strengths', [])])}

PONTOS FRACOS:
{chr(10).join(['- ' + str(w) for w in analysis.get('weaknesses', [])])}

HABILIDADES FALTANTES:
{chr(10).join(['- ' + str(s) for s in analysis.get('missing_skills', [])])}

HABILIDADES SUBUTILIZADAS:
{chr(10).join(['- ' + str(s) for s in analysis.get('underutilized_skills', [])])}

RECOMENDAÇÕES:
{chr(10).join(['- ' + str(r) for r in analysis.get('recommendations', [])])}

MELHORIAS PRIORITÁRIAS:
{chr(10).join(['- ' + str(k) for k in analysis.get('key_improvements', [])])}
"""
    
    try:
        prompt_template = create_rewrite_prompt_template()
        chain = prompt_template | llm
        
        style_map = {
            "professional": "profissional e objetiva",
            "modern": "moderna e dinâmica",
            "concise": "concisa e direta"
        }
        style_text = style_map.get(rewrite_options.get("style", "professional"), "profissional e objetiva")
        
        output = chain.invoke({
            "cv_template": cv_template,
            "original_cv": original_cv_content,
            "analysis": analysis_text,
            "job": job_details,
            "style": style_text,
            "focus_instruction": focus_instruction,
            "highlight_instruction": highlight_instruction,
            "strengths_instruction": strengths_instruction
        })
        
        rewritten_cv = format_res(output.content)
        
        # Validação de saída
        if not rewritten_cv:
            raise ValueError("Resposta do modelo está vazia")
        
        rewritten_cv = rewritten_cv.strip()
        
        if len(rewritten_cv) < 50:
            raise ValueError(f"Currículo reformulado está muito curto ({len(rewritten_cv)} caracteres). Mínimo esperado: 50 caracteres")
        
        return rewritten_cv
    except Exception as e:
        # Não usa st.error aqui pois pode não estar no contexto do Streamlit
        error_msg = f"Erro ao reformular currículo: {str(e)}"
        print(error_msg)  # Log para debug
        raise Exception(error_msg) from e


def save_rewritten_cv(content, filename):
    """Salva o currículo reformulado em um arquivo markdown"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename