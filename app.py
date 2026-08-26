import os
import pandas as pd
import gradio as gr
import matplotlib
import numpy as np
from fpdf import FPDF
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step
from llama_index.llms.groq import Groq
from llama_index.core import Settings
import ast


# Configurações da Chave e da Fonte:
load_dotenv(dotenv_path='chave.env')
key = os.getenv('minha_chave')
FONTE_DEJAVU = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans.ttf')
FONTE_DEJAVU_BOLD = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans-Bold.ttf')

# Configurações da LLM (Groq)
Settings.llm = Groq(model='openai/gpt-oss-120b', api_key=key)

# Parser que extrai e executa instruções pandas geradas pelo LLM sobre um DataFrame.

class PandasInstructionParser:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def parse(self, output: str):
        return self.default_output_processor(output)

    def default_output_processor(self, output: str):    
        # Remove blocos de código markdown, se existirem
        if "```" in output:
            output = output.split("```")[1]
            output = output.replace("python", "").strip()

        output = output.strip()

        try:
            module = ast.parse(output)
        except SyntaxError as e:
            return f"Erro de sintaxe no código gerado: {e}\nCódigo: {output}"

        local_vars = {"df": self.df, "pd": pd}

        try:
            if len(module.body) > 1:
                exec_ast = ast.Module(body=module.body[:-1], type_ignores=[])
                exec(compile(exec_ast, "<ast>", "exec"), {"__builtins__": {}}, local_vars)

            last_node = module.body[-1]
            if isinstance(last_node, ast.Expr):
                expr_ast = ast.Expression(body=last_node.value)
                result = eval(compile(expr_ast, "<ast>", "eval"), {"__builtins__": {}}, local_vars)
            else:
                exec(compile(ast.Module(body=[last_node], type_ignores=[]), "<ast>", "exec"), {"__builtins__": {}}, local_vars)
                result = "Instrução executada, mas sem valor de retorno explícito."

            return result

        except Exception as e:
            import traceback
            print("ERRO DETALHADO:", traceback.format_exc())
            return f"Erro ao executar instrução pandas: {e}"

# Criação do Prompt
def descricao_colunas(df):
    descricao = '\n'.join([f'{col}: {df[col].dtype}' for col in df.columns])
    return 'Aqui estão os detalhes das colunas do DataFrame:\n' + descricao

# Strings do Prompt
instruction_str = (
    'Dada uma pergunta de entrada, elabore uma resposta natural e clara como um analista de dados a partir dos resultados da consulta. \n'
    'IMPORTANTE: responda apenas com a informação solicitada e apenas com duas casas decimais.  \n'
    'IMPORTANTE: o `print(df.head())` mostrado abaixo é APENAS uma amostra para você entender a estrutura dos dados. '
    'NÃO use `.head()` a menos que a pergunta peça explicitamente por uma amostra ou "primeiros registros". '
    'Se a pergunta pedir um número específico de resultados (ex: "10 maiores"), use exatamente esse número.\n'
    'NÃO mencione o código python ou pandas na sua resposta final, entregue apenas a análise. \n' 
    'Consulta: {query_str}\n\n'
    'Instruções do Pandas (contexto interno, não mencione na resposta):\n{pandas_instructions}\n\n'
    'Saída do pandas: {pandas_output}\n\n'
    'Resposta:'
)

pandas_prompt_str = (
    'Você está trabalhando com um dataframe do pandas em python chamado `df`. \n'
    '{colunas_detalhes}\n\n'
    'Este é o resultado de `print(df.head())`: \n'
    '{df_str}\n'
    'Siga estas instruções: \n'
    '{instruction_str}\n'
    'Consulta: {query_str}\n\n'
    'REGRAS OBRIGATÓRIAS PARA SUA RESPOSTA:\n'
    '1. Responda APENAS com uma linha de código Python válido usando pandas.\n'
    '2. NÃO escreva nenhum texto explicativo, markdown, tabela, comentário ou frase.\n'
    '3. NÃO use blocos de código com ```.\n'
    '4. NÃO responda a pergunta diretamente — gere apenas a expressão pandas que, quando executada, produzirá a resposta.\n'
    '5. Sua resposta deve ser SOMENTE a expressão, começando diretamente com `df`.\n\n'
    'Exemplo de resposta correta para "quais as 5 maiores vendas": df.nlargest(5, "total")\n'
    'Exemplo de resposta INCORRETA: uma tabela markdown ou texto explicando o resultado.\n\n'
    'Expressão:'
)

response_synthesis_prompt_str = (
    'Dada uma pergunta de entrada, elabore uma resposta como um analista de dados a partir dos resultados da consulta. \n'
    'IMPORTANTE: responda apenas com a informação solicitada.  \n'
    'IMPORTANTE: se a saída do pandas contiver uma tabela ou lista de itens, reproduza TODOS os itens/linhas retornados, '
    'sem usar reticências ("...") ou abreviar. Nunca omita linhas.\n'
    'NÃO mencione o código pandas, a consulta técnica ou detalhes de implementação na resposta final. \n'
    'Consulta: {query_str}\n\n'
    'Saída do pandas: {pandas_output}\n\n'
    'Resposta:'
)

# Eventos do Workflow
class PandasCodeEvent(Event):
    query_str: str
    pandas_code: str

class PandasOutputEvent(Event):
    query_str: str
    pandas_output: str
    is_table: bool = False

# Workflow
class PandasQueryWorkflow(Workflow):
    @step
    async def generate_pandas_code(self, ev: StartEvent) -> PandasCodeEvent:
        query_str = ev.query_str
        prompt = self.pandas_prompt.format(query_str=query_str)
        response = await self.llm.acomplete(prompt)
        return PandasCodeEvent(query_str=query_str, pandas_code=str(response))

    @step
    async def execute_pandas_code(self, ev: PandasCodeEvent) -> PandasOutputEvent:
        print("CÓDIGO GERADO:", ev.pandas_code)
        resultado = self.pandas_output_parser.parse(ev.pandas_code)

        if isinstance(resultado, pd.Series):
            resultado = resultado.to_frame()
        elif isinstance(resultado, (list, np.ndarray)):
            resultado = pd.DataFrame(resultado)
        elif isinstance(resultado, dict):
            resultado = pd.DataFrame(list(resultado.items()), columns=["Chave", "Valor"])

        if isinstance(resultado, pd.DataFrame):
            pandas_output_str = resultado.to_markdown(index=False)
            is_table = True
        else:
            pandas_output_str = str(resultado)
            is_table = False

        print("SAÍDA DO PANDAS:", pandas_output_str)
        return PandasOutputEvent(query_str=ev.query_str, pandas_output=pandas_output_str, is_table=is_table)

    @step
    async def synthesize_response(self, ev: PandasOutputEvent) -> StopEvent:
        if ev.is_table:
            # Tabela vai direto, sem passar pelo LLM — evita qualquer alteração nos números
            return StopEvent(result=ev.pandas_output)

        prompt = self.response_synthesis_prompt.format(
            query_str=ev.query_str,
            pandas_output=ev.pandas_output,
        )
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))


# Função do Pipeline
async def pipeline_consulta(pergunta_usuario, df):
    """
    Função assíncrona que recebe a pergunta e o dataframe atual,
    configura o workflow dinamicamente e retorna a resposta do LLM.
    """
    llm = Groq(model='openai/gpt-oss-120b', api_key=key)
    
    # Formata o prompt com os dados específicos do DF enviado
    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str, 
        colunas_detalhes=descricao_colunas(df), 
        df_str=str(df.head(5))
    )
    
    # Inicia e injeta dependências no workflow
    workflow = PandasQueryWorkflow(timeout=60.0)
    workflow.llm = llm
    workflow.pandas_prompt = pandas_prompt
    workflow.response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
    workflow.pandas_output_parser = PandasInstructionParser(df)
    workflow.instruction_str = instruction_str

    # Executa o workflow e retorna o texto final
    resultado = await workflow.run(query_str=pergunta_usuario)
    return str(resultado)

# Interface
# Criando interface


# Fonte DEJAVU do matplot
FONTE_DEJAVU = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans.ttf')
FONTE_DEJAVU_BOLD = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans-Bold.ttf')

# Carregar os dados
def carregar_dados(caminho_arquivo, df_estado):

    if caminho_arquivo is None or caminho_arquivo == "":
        return "Por favor, faça o upload de um arquivo CSV para analiser", pd.DataFrame(), df_estado

    try:
        df = pd.read_csv(caminho_arquivo)
        return 'Arquivo carregado com sucesso.', df.head(), df

    except Exception as e:
        return f'Erro ao carregar arquivo: {str(e)}', pd.DataFrame(), df_estado

# Processar a pergunta
async def processar_pergunta(pergunta, df_estado):
    if df_estado is None:
        return "Erro: Faça o upload do CSV primeiro."
    if not pergunta:
        return "Erro: Digite uma pergunta."
    try:
        resposta = await pipeline_consulta(pergunta, df_estado)
        return resposta
    except Exception as e:
        return f"Ocorreu um erro no processamento: {str(e)}"




# Hitorico

def add_historico(pergunta, respota, historico_estado):
    if pergunta and respota:
        historico_estado.append((pergunta, respota))
        gr.Info('Adicionado ao PDF.', duration=2)
        return historico_estado
    else:
        gr.Warning("Pergunta ou resposta vazias. Nada foi adicionado.")
        return historico_estado
    
# PDF

def gerar_pdf(historico_estado):
    if not historico_estado:
        return 'Nenhum dado para adicionar ao PDF', None

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    caminho_pdf = f'relatorio_perguntas_respostas_{timestamp}.pdf'

    pdf = FPDF(orientation='L', format='A4')  # Paisagem, mais espaço para tabelas largas
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font('DejaVu', '', FONTE_DEJAVU)
    pdf.add_font('DejaVu', 'B', FONTE_DEJAVU_BOLD)

    for pergunta, resposta in historico_estado:
        pdf.add_page()
        pdf.set_font('DejaVu', 'B', 14)
        pdf.multi_cell(0, 8, text=pergunta)
        pdf.ln(2)

        if e_tabela_markdown(resposta):
            desenhar_tabela(pdf, resposta)
        else:
            pdf.set_font('DejaVu', '', 12)
            pdf.multi_cell(0, 8, text=resposta)

        pdf.ln(6)

    pdf.output(caminho_pdf)
    return caminho_pdf


def e_tabela_markdown(texto):
    linhas = texto.strip().split('\n')
    return len(linhas) >= 2 and '|' in linhas[0] and set(linhas[1].replace('|', '').replace(' ', '')) <= set('-:')


def desenhar_tabela(pdf, texto_markdown):
    linhas = [l for l in texto_markdown.strip().split('\n') if l.strip()]
    linhas_dados = [linhas[0]] + linhas[2:]

    tabela = []
    for linha in linhas_dados:
        celulas = [c.strip() for c in linha.strip().strip('|').split('|')]
        tabela.append(celulas)

    num_colunas = len(tabela[0])
    largura_pagina = pdf.w - 2 * pdf.l_margin

    # Calcula largura de cada coluna proporcional ao maior conteúdo dela
    larguras = []
    for col_idx in range(num_colunas):
        maior_texto = max((len(str(linha[col_idx])) for linha in tabela), default=5)
        larguras.append(max(maior_texto, 4))  # mínimo de 4 caracteres

    total_unidades = sum(larguras)
    larguras_px = [(largura / total_unidades) * largura_pagina for largura in larguras]

    # Cabeçalho
    pdf.set_font('DejaVu', 'B', 7)
    for i, celula in enumerate(tabela[0]):
        pdf.cell(larguras_px[i], 7, text=celula, border=1)
    pdf.ln()

    # Linhas de dados
    pdf.set_font('DejaVu', '', 6.5)
    for linha in tabela[1:]:
        y_inicial = pdf.get_y()
        x_inicial = pdf.get_x()

        alturas = []
        for i, celula in enumerate(linha):
            n_linhas = pdf.multi_cell(larguras_px[i], 4, text=celula, border=0, dry_run=True, output="LINES")
            alturas.append(len(n_linhas) * 4)
        altura_max = max(alturas + [5])

        for i, celula in enumerate(linha):
            x = x_inicial + sum(larguras_px[:i])
            pdf.set_xy(x, y_inicial)
            pdf.multi_cell(larguras_px[i], 4, text=celula, border=1)

        pdf.set_xy(x_inicial, y_inicial + altura_max)

        # Quebra de página se necessário
        if pdf.get_y() > pdf.h - 20:
            pdf.add_page()
            pdf.set_font('DejaVu', 'B', 7)
            for i, celula in enumerate(tabela[0]):
                pdf.cell(larguras_px[i], 7, text=celula, border=1)
            pdf.ln()
            pdf.set_font('DejaVu', '', 6.5)

# Limpar pergunta e resposta
def limpar_pergunta_resposta():
    return "",""

# Reset aplicação
def resetar_aplicacao():
    return None, 'Aplicação resetada. Porfavor, faça upload de um novo CSV.', pd.DataFrame(), '', None, [], ''

with gr.Blocks(theme='Soft') as app:

    # Titulo do app e descrição
    gr.Markdown('# Analisando dados com IA:')
    gr.Markdown('''Adicione um arquivo CSV, faça suas perguntas para a IA, armazene a resposta no histórico, gere o PDF com suas análises e finalize com o download de seu PDF  ''')

    # Componentes da Interface
    input_arquivo = gr.File( file_count='single', type='filepath', label='Upload CSV')
    upload_status = gr.Textbox(label='Status do Upload')
    tabela_dados = gr.DataFrame()
    input_pergunta = gr.Textbox( label='Digite sua pergunta sobre os dados' )
    botao_submeter = gr.Button('Enviar')
    output_resposta = gr.Markdown(label='Resposta:')
    with gr.Row():
        bottao_limpeza = gr.Button('Limpar pergunta e resposta.')
        botao_add_pdf = gr.Button('Adicionar ao hitórico do PDF')
        botao_gerar_pdf = gr.Button('Gerar PDF')
    arquivo_pdf = gr.File(label='Download do PDF')
    botao_reset = gr.Button('Analisar outro conjunto de dados.')

    # Gerenciamento de estados
    df_estado = gr.State(value=None)
    historico_estado = gr.State(value=[])


    # Conectando funções aos componentes
    input_arquivo.change(fn=carregar_dados,
                         inputs=[input_arquivo, df_estado],
                         outputs=[upload_status, tabela_dados, df_estado])

    botao_submeter.click(fn=processar_pergunta,
                         inputs=[input_pergunta, df_estado],
                         outputs=output_resposta)

    bottao_limpeza.click(fn=limpar_pergunta_resposta,
                         inputs=[],
                         outputs=[input_pergunta, output_resposta])

    botao_add_pdf.click(fn=add_historico,
                        inputs=[input_pergunta, output_resposta, historico_estado],
                        outputs=[historico_estado])

    botao_gerar_pdf.click(fn=gerar_pdf,
                          inputs=historico_estado,
                          outputs=arquivo_pdf)

    botao_reset.click(fn=resetar_aplicacao,
                      inputs=[],
                      outputs=[input_arquivo, upload_status, tabela_dados, output_resposta, arquivo_pdf, historico_estado, input_pergunta])
    
# Inicialização
if __name__ == '__main__':
    app.launch(debug=True)
