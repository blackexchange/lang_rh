import streamlit as st
import uuid
import os
from utils_proj03 import *
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Triagem e Análise de Currículos", page_icon="📄", layout="wide")

id_model = "llama-3.3-70b-versatile"
temperature = 0.7
json_file = 'curriculos.json'
path_job_csv = "vagas.csv"

llm = load_llm(id_model, temperature)

job = {}
job['title'] = "Engenheiro de Dados Pleno (IA)"
job['description'] = "Engenheiro de Dados Pleno (IA):"
job['details'] = """
Experiência em modelagem, arquitetura e integração de dados (DW, Data Lake, Lakehouse).
Domínio de Python e SQL, além de trabalhar bem com frameworks de processamento como Spark, Databricks, Airflow ou equivalentes.
Vivência na construção de pipelines escaláveis e de alto desempenho, com práticas modernas de versionamento, testes e CI/CD.
Conhecimento aplicado em soluções de Machine Learning/IA, incluindo preparação de dados para modelos, feature store, monitoramento e integração com modelos em produção.
Experiência com serviços em nuvem (AWS, Azure ou GCP).
Experiência em SQL Server, incluindo consultas, modelagem de dados e otimização de desempenho.
Conhecimento em ferramentas de automação de processos, como UiPath, Open RPA, N8N ou similares.
 Diferenciais: Participação em projetos de automação IA (Chatgpt, Gemini,Grok, etc).

Capacidade de propor soluções de ponta e trazer visão estratégica para o uso de IA na empresa.

Quais serão os seus desafios?

Estruturar e treinar modelos de IA para automação de tarefas repetitivas.
Garantir integração de IA com sistemas jurídicos.
Sustentação a projetos de inovação e demandas corporativas.
Análise de requisitos e desenvolvimento de soluções técnicas eficientes, atuando diretamente na manutenção de sistemas e aplicações da empresa.

Atuar na análise de necessidades propondo soluções de automações sistêmicas com IA garantindo governança, qualidade, rastreabilidade e disponibilidade dos dados.

Apoiar a evolução da plataforma de dados, definindo boas práticas, padrões e automações.

Identificar oportunidades de melhoria contínua, propondo soluções escaláveis e eficientes para desafios complexos de dados.

Desenvolver scripts ou manipulação de dados para melhoria sistêmica e/ou segurança dos dados.

Sustentação e manutenção de softwares de ERP e HCM, incluindo customizações e suporte a usuários finais.

"""

schema = """
{
  "name": "Nome completo do candidato",
  "area": "Área ou setor principal que o candidato atua. Classifique em apenas uma: Desenvolvimento, Marketing, Vendas, Financeiro, Administrativo, Outros",
  "summary": "Resumo objetivo sobre o perfil profissional do candidato",
  "hard_skills": ["competência 1", "competência 2", "..."],
  "soft_skills": ["competência 1", "competência 2", "..."],
  "academic_info": [{
    "title": "Título do curso",
    "institution": "Instituição",
    "year": "Ano"
  },{...}],
  "training_courses": [{
    "title": "Título do curso",
    "institution": "Instituição"
  }, "..."],
  "experiences": [{"position": "Posição", "company": "Empresa", "start_date": "Data de início", "end_date": "Data de fim", "description": "Descrição da experiência"}, {...}],
  "certifications": ["certificação 1", "certificação 2", "..."],
  "interview_questions": ["Pelo menos 3 perguntas úteis para entrevista com base no currículo, para esclarecer algum ponto ou explorar melhor"],
  "strengths": ["Pontos fortes e aspectos que indicam alinhamento com o perfil ou vaga desejada"],
  "areas_for_development": ["Pontos que indicam possíveis lacunas, fragilidades ou necessidades de desenvolvimento"],
  "important_considerations": ["Observações específicas que merecem verificação ou cuidado adicional"],
  "final_recommendations": "Resumo avaliativo final com sugestões de próximos passos (ex: seguir com entrevista, indicar para outra vaga)",
  "score": 0.0
}
"""

fields = [
    "name",
    "area",
    "summary",
    "hard_skills",
    "soft_skills",
    "academic_info",
    "training_courses",
    "experiences",
    "certifications",
    "interview_questions",
    "strengths",
    "areas_for_development",
    "important_considerations",
    "final_recommendations",
    "score"
]

prompt_score = """
Com base na vaga específica, calcule a pontuação final (de 0.0 a 10.0).
O retorno para esse campo deve conter apenas a pontuação final (x.x) sem mais nenhum texto ou anotação.
Seja justo e rigoroso ao atribuir as notas. A nota 10.0 só deve ser atribuída para candidaturas que superem todas as expectativas da vaga.

Critérios de avaliação:
1. Experiência (Peso: 35% do total): Análise de posições anteriores, tempo de atuação e similaridade com as responsabilidades da vaga.
2. Habilidades Técnicas (Peso: 20% do total): Verifique o alinhamento das habilidades técnicas com os requisitos mencionados na vaga.
3. Soft Skills (Peso: 5% do total): Verifique o alinhamento das soft skills com os requisitos mencionados na vaga.
4. Educação (Peso: 15% do total): Avalie a relevância da graduação/certificações para o cargo, incluindo instituições e anos de estudo.
5. Pontos Fortes (Peso: 15% do total): Avalie a relevância dos pontos fortes (ou alinhamentos) para a vaga.
6. Pontos Fracos (Desconto de até 10%): Avalie a gravidade dos pontos fracos (ou desalinhamentos) para a vaga.
7. Cursos (Peso: 5% do total): Avalie a relevância dos cursos para a vaga.
"""

prompt_template = ChatPromptTemplate.from_template("""
Você é um especialista em Recursos Humanos com vasta experiência em análise de currículos.
Sua tarefa é analisar o conteúdo a seguir e extrair os dados conforme o formato abaixo, para cada um dos campos.
Responda apenas com o JSON estruturado e utilize somente essas chaves. Cuide para que os nomes das chaves sejam exatamente esses.
Não adicione explicações ou anotações fora do JSON.
Schema desejado:
{schema}

---
Para o cálculo do campo score:
{prompt_score}

---

Currículo a ser analisado:
'{cv}'

---

Vaga que o candidato está se candidatando:
'{job}'

""")

if "uploader_key" not in st.session_state:
  st.session_state.uploader_key = str(uuid.uuid4())

if "selected_cv" not in st.session_state:
  st.session_state.selected_cv = None

if "cv_analysis" not in st.session_state:
  st.session_state.cv_analysis = None

if "rewritten_cv" not in st.session_state:
  st.session_state.rewritten_cv = None

if "original_cv_content" not in st.session_state:
  st.session_state.original_cv_content = None

if "rewritten_cvs" not in st.session_state:
  st.session_state.rewritten_cvs = {}  # Dicionário para armazenar CVs reformulados por nome

if "rewrite_options" not in st.session_state:
  st.session_state.rewrite_options = {
    "focus": "all",  # all, skills, experience, summary
    "style": "professional",  # professional, modern, concise
    "highlight_missing": True,
    "emphasize_strengths": True,
    "template": "1"  # 1 ou 2
  }

if "cv_templates" not in st.session_state:
  st.session_state.cv_templates = {
    "1": None,  # cv_base.txt
    "2": None   # cv_base2.txt
  }

# Salva descrição da vaga em um .csv
save_job_to_csv(job, path_job_csv)
job_details = load_job(path_job_csv)

# ============================================
# CARREGAR TEMPLATES DE CV
# ============================================
# Carrega os templates se ainda não foram carregados
if st.session_state.cv_templates["1"] is None:
  try:
    if os.path.exists("cv_base.txt"):
      with open("cv_base.txt", "r", encoding="utf-8") as f:
        st.session_state.cv_templates["1"] = f.read()
  except Exception as e:
    st.sidebar.error(f"Erro ao carregar cv_base.txt: {e}")

if st.session_state.cv_templates["2"] is None:
  try:
    if os.path.exists("cv_base2.txt"):
      with open("cv_base2.txt", "r", encoding="utf-8") as f:
        st.session_state.cv_templates["2"] = f.read()
  except Exception as e:
    st.sidebar.error(f"Erro ao carregar cv_base2.txt: {e}")

# ============================================
# OPÇÕES DE REFORMULAÇÃO
# ============================================
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Opções de Reformulação")

# Seleção do template
st.session_state.rewrite_options["template"] = st.sidebar.selectbox(
  "Template de CV",
  ["1", "2"],
  index=0,
  format_func=lambda x: f"Template {x} (cv_base{x}.txt)"
)

# Mostra status do template
selected_template = st.session_state.rewrite_options["template"]
if st.session_state.cv_templates[selected_template]:
  st.sidebar.success(f"✅ Template {selected_template} carregado")
  with st.sidebar.expander("👁️ Visualizar Template"):
    st.text(st.session_state.cv_templates[selected_template][:300] + "...")
else:
  st.sidebar.warning(f"⚠️ Template {selected_template} não encontrado")

st.session_state.rewrite_options["focus"] = st.sidebar.selectbox(
  "Foco da Reformulação",
  ["all", "skills", "experience", "summary"],
  index=0,
  format_func=lambda x: {
    "all": "Tudo",
    "skills": "Habilidades",
    "experience": "Experiência",
    "summary": "Resumo"
  }[x]
)

st.session_state.rewrite_options["style"] = st.sidebar.selectbox(
  "Estilo",
  ["professional", "modern", "concise"],
  index=0,
  format_func=lambda x: {
    "professional": "Profissional",
    "modern": "Moderno",
    "concise": "Conciso"
  }[x]
)

st.session_state.rewrite_options["highlight_missing"] = st.sidebar.checkbox(
  "Destacar habilidades faltantes",
  value=True
)

st.session_state.rewrite_options["emphasize_strengths"] = st.sidebar.checkbox(
  "Enfatizar pontos fortes",
  value=True
)

col1, col2 = st.columns(2)
with col1:
  st.header("Triagem e Análise de Currículos")
  st.markdown("#### Vaga: {}".format(job["title"]))
with col2:
  uploaded_file = st.file_uploader("Envie um currículo em PDF", type=["pdf"], key=st.session_state.uploader_key)

if uploaded_file is not None:
  path = uploaded_file.name
  with open(path, "wb") as f:
    f.write(uploaded_file.read())
  
  # Extrai o conteúdo do currículo para uso posterior
  st.session_state.original_cv_content = parse_doc(path)
  
  # Análise inicial (triagem)
  with st.spinner("Analisando o currículo (triagem inicial)..."):
    output, res = process_cv(schema, job_details, prompt_template, prompt_score, llm, path)
    structured_data = parse_res_llm(res, fields)
    save_json_cv(structured_data, path_json=json_file, key_name="name")
    st.success("Currículo analisado com sucesso!")
    st.session_state.uploader_key = str(uuid.uuid4())

  st.write(show_cv_result(structured_data))

  with st.expander("Ver dados estruturados (JSON)"):
    st.json(structured_data)
  
  # ============================================
  # SEÇÃO: AGENTE ANALISADOR DETALHADO
  # ============================================
  st.markdown("---")
  st.subheader("🔍 Análise Detalhada - Agente Analisador")
  st.markdown("""
  O **Agente Analisador** realiza uma análise profunda comparando o currículo com a vaga,
  identificando pontos fortes, fracos, habilidades faltantes e gerando recomendações específicas.
  """)
  
  col_analyze1, col_analyze2 = st.columns([1, 4])
  with col_analyze1:
    if st.button("🚀 Executar Análise Detalhada", type="primary", use_container_width=True):
      with st.spinner("Agente Analisador trabalhando..."):
        analysis = analyze_cv_and_job(
          llm, 
          st.session_state.original_cv_content, 
          job_details
        )
        if analysis:
          st.session_state.cv_analysis = analysis
          st.success("Análise concluída!")
          # Limpa o currículo reformulado quando nova análise é feita
          st.session_state.rewritten_cv = None
  
  if st.session_state.cv_analysis:
    analysis = st.session_state.cv_analysis
    
    st.markdown("### 📊 Resultados da Análise")
    
    # Score de alinhamento
    if "alignment_score" in analysis:
      score = analysis["alignment_score"]
      st.metric("Score de Alinhamento", f"{score:.1f}/10.0")
    
    # Resumo da análise
    if "analysis_summary" in analysis:
      st.markdown("#### Resumo Executivo")
      st.info(analysis["analysis_summary"])
    
    # Pontos fortes
    if "strengths" in analysis and analysis["strengths"]:
      st.markdown("#### ✅ Pontos Fortes")
      for strength in analysis["strengths"]:
        st.success(f"• {strength}")
    
    # Pontos fracos
    if "weaknesses" in analysis and analysis["weaknesses"]:
      st.markdown("#### ⚠️ Pontos Fracos")
      for weakness in analysis["weaknesses"]:
        st.warning(f"• {weakness}")
    
    # Habilidades faltantes
    if "missing_skills" in analysis and analysis["missing_skills"]:
      st.markdown("#### 🔴 Habilidades Faltantes")
      for skill in analysis["missing_skills"]:
        st.error(f"• {skill}")
    
    # Habilidades subutilizadas
    if "underutilized_skills" in analysis and analysis["underutilized_skills"]:
      st.markdown("#### 💡 Habilidades Subutilizadas")
      for skill in analysis["underutilized_skills"]:
        st.info(f"• {skill}")
    
    # Recomendações
    if "recommendations" in analysis and analysis["recommendations"]:
      st.markdown("#### 💼 Recomendações")
      for rec in analysis["recommendations"]:
        st.markdown(f"• {rec}")
    
    # Melhorias prioritárias
    if "key_improvements" in analysis and analysis["key_improvements"]:
      st.markdown("#### 🎯 Melhorias Prioritárias")
      for improvement in analysis["key_improvements"]:
        st.markdown(f"**→** {improvement}")
    
    with st.expander("Ver análise completa (JSON)"):
      st.json(analysis)

# ============================================
# SEÇÃO: AGENTE REFORMULADOR (fora do bloco de upload)
# ============================================
# Verifica se temos análise e conteúdo do currículo
has_analysis = st.session_state.cv_analysis is not None
has_cv_content = st.session_state.original_cv_content is not None

if has_analysis and has_cv_content:
  st.markdown("---")
  st.subheader("✏️ Reformulação do Currículo - Agente Reformulador")
  st.markdown("""
  O **Agente Reformulador** utiliza a análise detalhada para reformular o currículo,
  aplicando as recomendações e melhorias identificadas, mantendo todas as informações verdadeiras.
  """)
  
  # Debug info (pode remover depois)
  with st.expander("🔍 Debug Info", expanded=False):
    st.write(f"Análise disponível: {has_analysis}")
    st.write(f"Conteúdo CV disponível: {has_cv_content}")
    st.write(f"CV reformulado salvo: {st.session_state.rewritten_cv is not None}")
    if st.session_state.rewritten_cv:
      st.write(f"Tamanho do CV reformulado: {len(st.session_state.rewritten_cv)} caracteres")
  
  col_rewrite1, col_rewrite2 = st.columns([1, 4])
  with col_rewrite1:
    rewrite_button = st.button("🔄 Reformular Currículo", type="primary", use_container_width=True, key="btn_rewrite")
  
  # Processa a reformulação quando o botão é clicado
  if rewrite_button:
    with st.spinner("Agente Reformulador trabalhando..."):
      try:
        # Verifica se temos todos os dados necessários
        if not st.session_state.original_cv_content:
          st.error("❌ Conteúdo do currículo original não encontrado. Faça upload novamente.")
        elif not st.session_state.cv_analysis:
          st.error("❌ Análise não encontrada. Execute a análise detalhada primeiro.")
        else:
          # Chama a função de reformulação com opções e template
          selected_template = st.session_state.rewrite_options["template"]
          cv_template = st.session_state.cv_templates.get(selected_template)
          
          if not cv_template:
            st.error(f"❌ Template {selected_template} não encontrado. Verifique se o arquivo cv_base{selected_template}.txt existe.")
            rewritten = None
          else:
            rewritten = rewrite_cv(
              llm,
              st.session_state.original_cv_content,
              st.session_state.cv_analysis,
              job_details,
              cv_template=cv_template,
              rewrite_options=st.session_state.rewrite_options
            )
          
          # Valida o resultado
          if rewritten and isinstance(rewritten, str) and len(rewritten.strip()) > 50:
            # Salva no estado da sessão ANTES de qualquer outra coisa
            st.session_state.rewritten_cv = rewritten.strip()
            
            # Confirma que foi salvo
            if st.session_state.rewritten_cv:
              st.success("✅ Currículo reformulado com sucesso e salvo no estado da sessão!")
              
              # Salva também em arquivo
              try:
                filename = "curriculo_reformulado.md"
                save_rewritten_cv(rewritten.strip(), filename)
                st.info(f"💾 Currículo também salvo em arquivo: {filename}")
              except Exception as save_error:
                st.warning(f"⚠️ Não foi possível salvar em arquivo: {save_error}")
            else:
              st.error("❌ Erro: Não foi possível salvar no estado da sessão.")
          else:
            st.error("❌ O currículo reformulado está vazio ou muito curto.")
            if rewritten:
              st.write(f"Tipo: {type(rewritten)}, Tamanho: {len(str(rewritten))} caracteres")
              with st.expander("Ver conteúdo retornado"):
                st.text(str(rewritten)[:500])
      except Exception as e:
        st.error(f"❌ Erro ao reformular currículo: {e}")
        import traceback
        with st.expander("🔍 Detalhes do erro"):
          st.code(traceback.format_exc())
  
  # Exibe o resultado se existir
  if st.session_state.rewritten_cv:
    st.markdown("---")
    st.markdown("### 📝 Currículo Reformulado")
    st.success("✅ Currículo reformulado disponível abaixo!")
    
    # Informações sobre o CV reformulado
    cv_length = len(st.session_state.rewritten_cv)
    st.caption(f"📊 Tamanho: {cv_length} caracteres")
    
    # Comparação lado a lado
    col_original, col_rewritten = st.columns(2)
    
    with col_original:
      st.markdown("#### 📄 Original")
      with st.expander("Ver currículo original", expanded=False):
        if st.session_state.original_cv_content:
          st.markdown(st.session_state.original_cv_content)
        else:
          st.warning("Conteúdo original não disponível")
    
    with col_rewritten:
      st.markdown("#### ✨ Reformulado")
      with st.expander("Ver currículo reformulado", expanded=True):
        st.markdown(st.session_state.rewritten_cv)
    
    # Downloads do currículo reformulado
    col_download_md_main, col_download_pdf_main = st.columns(2)
    
    with col_download_md_main:
      st.download_button(
        label="📄 Baixar Markdown (.md)",
        data=st.session_state.rewritten_cv,
        file_name="curriculo_reformulado.md",
        mime="text/markdown",
        key="download_rewritten_cv_md",
        use_container_width=True
      )
    
    with col_download_pdf_main:
      # Gera PDF
      pdf_bytes = generate_pdf_from_cv(st.session_state.rewritten_cv)
      if pdf_bytes:
        st.download_button(
          label="📕 Baixar PDF (.pdf)",
          data=pdf_bytes,
          file_name="curriculo_reformulado.pdf",
          mime="application/pdf",
          key="download_rewritten_cv_pdf",
          use_container_width=True
        )
elif st.session_state.cv_analysis and not st.session_state.original_cv_content:
  st.info("💡 Faça upload de um currículo e execute a análise para poder reformular.")
elif not st.session_state.cv_analysis and st.session_state.original_cv_content:
  st.info("💡 Execute primeiro a **Análise Detalhada** para poder reformular o currículo.")

if os.path.exists(json_file):
  st.subheader("Lista de currículos analisados", divider="gray")
  df = display_json_table(json_file)
  for i, row in df.iterrows():
    candidate_name = row.get('name', f'Candidato_{i}')
    cv_data = row.to_dict()
    
    # Cria um container para cada currículo
    with st.container():
      cols = st.columns([1, 2, 1, 2, 1, 1])
      
      with cols[0]:
        if st.button("📋 Detalhes", key=f"btn_details_{i}"):
          st.session_state.selected_cv = cv_data
      
      with cols[1]:
        st.write(f"**{candidate_name}**")
      
      with cols[2]:
        score = row.get('score', '-')
        if isinstance(score, (int, float)):
          st.metric("Score", f"{score:.1f}")
        else:
          st.write(f"**Score:** {score}")
      
      with cols[3]:
        summary = row.get('summary', '-')
        if len(summary) > 100:
          summary = summary[:100] + "..."
        st.write(summary)
      
      with cols[4]:
        # Botão de reformulação para este currículo específico
        if st.button("🔄 Reformular CV", key=f"btn_rewrite_{i}", type="primary", use_container_width=True):
          with st.spinner(f"Reformulando currículo de {candidate_name}..."):
            try:
              # Gera conteúdo do CV a partir do JSON
              cv_content = generate_cv_content_from_json(cv_data)
              
              # Gera análise a partir do JSON
              analysis = generate_analysis_from_json(cv_data)
              
              # Executa a reformulação com opções e template
              selected_template = st.session_state.rewrite_options["template"]
              cv_template = st.session_state.cv_templates.get(selected_template)
              
              if not cv_template:
                st.error(f"❌ Template {selected_template} não encontrado. Verifique se o arquivo cv_base{selected_template}.txt existe.")
                rewritten = None
              else:
                rewritten = rewrite_cv(
                  llm,
                  cv_content,
                  analysis,
                  job_details,
                  cv_template=cv_template,
                  rewrite_options=st.session_state.rewrite_options
                )
              
              if rewritten and isinstance(rewritten, str) and len(rewritten.strip()) > 50:
                # Salva no dicionário de CVs reformulados
                st.session_state.rewritten_cvs[candidate_name] = rewritten.strip()
                st.success(f"✅ Currículo de {candidate_name} reformulado com sucesso!")
                
                # Salva em arquivo
                try:
                  filename = f"curriculo_reformulado_{candidate_name.replace(' ', '_')}.md"
                  save_rewritten_cv(rewritten.strip(), filename)
                except Exception as save_error:
                  st.warning(f"⚠️ Não foi possível salvar em arquivo: {save_error}")
              else:
                st.error("❌ O currículo reformulado está vazio ou muito curto.")
            except Exception as e:
              st.error(f"❌ Erro ao reformular currículo: {e}")
      
      with cols[5]:
        # Mostra botões de download se o CV foi reformulado
        if candidate_name in st.session_state.rewritten_cvs:
          col_md, col_pdf = st.columns(2)
          with col_md:
            st.download_button(
              label="📄 MD",
              data=st.session_state.rewritten_cvs[candidate_name],
              file_name=f"curriculo_reformulado_{candidate_name.replace(' ', '_')}.md",
              mime="text/markdown",
              key=f"download_md_{i}",
              use_container_width=True
            )
          with col_pdf:
            # Gera PDF
            pdf_bytes = generate_pdf_from_cv(st.session_state.rewritten_cvs[candidate_name])
            if pdf_bytes:
              st.download_button(
                label="📕 PDF",
                data=pdf_bytes,
                file_name=f"curriculo_reformulado_{candidate_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
                key=f"download_pdf_{i}",
                use_container_width=True
              )
      
      st.divider()

if st.session_state.selected_cv:
  st.markdown("-----")
  selected_name = st.session_state.selected_cv.get('name', 'Candidato')
  
  st.write(show_cv_result(st.session_state.selected_cv))

  with st.expander("Ver dados estruturados (JSON)"):
    st.json(st.session_state.selected_cv)
  
  # Mostra CV reformulado se existir para este candidato
  if selected_name in st.session_state.rewritten_cvs:
    st.markdown("---")
    st.markdown("### ✨ Currículo Reformulado")
    st.success(f"✅ Currículo reformulado de {selected_name} disponível!")
    
    col_original, col_rewritten = st.columns(2)
    
    with col_original:
      st.markdown("#### 📄 Original (do JSON)")
      cv_content = generate_cv_content_from_json(st.session_state.selected_cv)
      with st.expander("Ver currículo original", expanded=False):
        st.markdown(cv_content)
    
    with col_rewritten:
      st.markdown("#### ✨ Reformulado")
      with st.expander("Ver currículo reformulado", expanded=True):
        st.markdown(st.session_state.rewritten_cvs[selected_name])
    
    # Downloads
    col_download_md, col_download_pdf = st.columns(2)
    
    with col_download_md:
      st.download_button(
        label=f"📄 Baixar Markdown (.md)",
        data=st.session_state.rewritten_cvs[selected_name],
        file_name=f"curriculo_reformulado_{selected_name.replace(' ', '_')}.md",
        mime="text/markdown",
        key="download_selected_rewritten_md",
        use_container_width=True
      )
    
    with col_download_pdf:
      # Gera PDF
      pdf_bytes = generate_pdf_from_cv(st.session_state.rewritten_cvs[selected_name])
      if pdf_bytes:
        st.download_button(
          label=f"📕 Baixar PDF (.pdf)",
          data=pdf_bytes,
          file_name=f"curriculo_reformulado_{selected_name.replace(' ', '_')}.pdf",
          mime="application/pdf",
          key="download_selected_rewritten_pdf",
          use_container_width=True
        )

if os.path.exists(json_file):
  with open(json_file, "r", encoding="utf-8") as f:
    json_data = f.read()
  st.download_button(
      label = "📥 Baixar arquivo .json",
      data = json_data,
      file_name = json_file,
      mime="application/json"
  )

  df = display_json_table(json_file)
  st.dataframe(df)