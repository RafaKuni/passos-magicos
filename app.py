import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import plotly.figure_factory as ff
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CARREGAMENTO E TRATAMENTO DE DADOS PARA OS GRÁFICOS
# ==============================================================================

# 1. Carregando a base com o separador correto (vírgula)
df = pd.read_csv("PEDE_Consolidado_2022_2024.csv", sep=",")

# 2. Padronizar todas as colunas para minúsculo para facilitar a leitura
df.columns = df.columns.str.lower()

# 3. Ajustar os nomes das colunas para bater com o que os seus gráficos esperam
if 'ano_pesquisa' in df.columns:
    df.rename(columns={'ano_pesquisa': 'ano_referencia'}, inplace=True)

# 4. Criar a categoria IAN com 4 níveis
if 'ian_cat' not in df.columns and 'ian' in df.columns:
    df['ian_cat'] = df['ian'].apply(
        lambda v: 'Em fase' if v >= 7.5 else (
            'Defasagem leve' if v >= 5.0 else (
                'Defasagem moderada' if v >= 2.5 else 'Defasagem severa'
            )
        )
    )

# 5. Criar a coluna "pedra" genérica (vamos usar a Fase, já que seus gráficos pedem)
if 'pedra' not in df.columns and 'fase' in df.columns:
    df['pedra'] = df['fase']

# 6. Consolidar a coluna INDE (Junta INDE 22, 23, 2024 em uma só chamada 'inde_ano')
if 'inde_ano' not in df.columns:
    # Acha todas as colunas que têm a palavra 'inde'
    colunas_inde = [c for c in df.columns if 'inde' in c]
    
    if len(colunas_inde) > 0:
        # TRATAMENTO NOVO: Força todas as colunas INDE a virarem números puros
        for col in colunas_inde:
            # Troca vírgula por ponto e converte para número (erros viram 'vazio' / NaN)
            df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Pega a nota INDE válida do aluno, ignorando os nulos das outras colunas
        df['inde_ano'] = df[colunas_inde].max(axis=1)
    else:
        # Fallback de segurança para não quebrar os gráficos
        df['inde_ano'] = df['ida']

# ==============================================================================
# CARREGAMENTO DO MOTOR DE MACHINE LEARNING
# ==============================================================================
df_ml = pd.read_csv("base_modelagem.csv", sep=";") # Mantemos a leitura do modelo separada
modelo = joblib.load("modelo_passos_magicos_otimizado.pkl")

# ==============================================================================
# 2. MENU LATERAL - CONFIGURAÇÕES E INFORMAÇÕES
# ==============================================================================

with st.sidebar:
    # Logo Centralizada
    try:
        st.image("Passos-magicos-icon-cor.png", use_container_width=True)
    except:
        st.warning("⚠️ Logo não encontrada.")
    
    st.markdown("---")
    
    # Seção Institucional
    st.subheader("📌 Sobre o Projeto")
    st.markdown("""
    **FIAP - Pós-Tech** *Data Analytics - Fase 5* **Datathon - Passos Mágicos**
    """)
    
    # Seção de Links Úteis com botões ou links formatados
    st.markdown("---")
    st.subheader("🔗 Links Úteis")
    st.markdown("[🌐 Site Passos Mágicos](https://passosmagicos.org.br/)")
    st.markdown("[💻 Repositório GitHub](https://github.com/paulocdvieira/FIAP-DA-F5-TC-GRUPO47)")
    
    # Detalhes Técnicos em um box de destaque
    st.markdown("---")
    with st.expander("🛠️ Detalhes Técnicos", expanded=True):
        st.write("🤖 **Modelo Preditivo:** `Regressão Logística`")
        st.write("🎯 **Estratégia:** `Alta Precisão (Threshold 0.75)`")
        
    # Rodapé do menu
    st.markdown("---")
    st.caption("Desenvolvido pelo Grupo 47")

# ==============================================================================
# 3. INTERFACE E NAVEGAÇÃO (TABS)
# ==============================================================================

st.title("Datathon - Associação Passos Mágicos")

aba1, aba2, aba3 = st.tabs([
    "Visão dos Dados", 
    "Questões Técnicas", 
    "Simulador de Defasagem"
])

# ==============================================================================
# ABA 1: VISÃO DOS DADOS
# ==============================================================================


with aba1:
    st.header("Análise Geral da Base de Modelagem")
    
    if not df.empty:
        # 1. Métricas Principais
        m1, m2, m3, m4 = st.columns(4)
        
        tempo_medio = df['Tempo_Programa'].mean() if 'Tempo_Programa' in df.columns else 0
        inde_medio = df['INDE'].mean() if 'INDE' in df.columns else 0

        with m1: st.metric("Registros Aluno-Ano", len(df))
        with m2: st.metric("Tempo Médio no Programa", f"{tempo_medio:.1f} anos")
        with m3: st.metric("Média Geral INDE", f"{inde_medio:.2f}")
        with m4: st.metric("Taxa de Risco Real", f"{(df['Target_Defasagem_Ano_Seguinte'].mean()*100):.1f}%" if 'Target_Defasagem_Ano_Seguinte' in df.columns else "N/A")

        st.markdown("---")
        
        st.subheader("Visualização dos Microdados (Longitudinais)")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Base de dados 'base_modelagem.csv' não carregada. As métricas estão ocultas.")


# ABA 2: QUESTÕES TÉCNICAS


with aba2:
    st.header("🔍 Resumo das Respostas às Questões Estratégicas")

with st.expander("1. Adequação do nível (IAN) - Qual é o perfil geral de defasagem dos alunos (IAN) e como ele evolui ao longo do ano?"):
        st.markdown("""
        **Análise:** O perfil de defasagem dos alunos demonstra a trajetória de recuperação acadêmica 
        em números absolutos ao longo dos anos, separada em quatro níveis de atenção.
        """)
        
        df_q1 = df.copy()
        
        # 1. Garantir que a coluna 'defasagem' é numérica (e lidar com vírgulas caso existam)
        df_q1['defasagem'] = df_q1['defasagem'].astype(str).str.replace(',', '.')
        df_q1['defasagem'] = pd.to_numeric(df_q1['defasagem'], errors='coerce')
        
        # 2. A sua regra de negócio original do Colab exata!
        def classificar_defasagem(d):
            if pd.isna(d): return None # Ignora os 'Sem Dados' para não poluir o gráfico
            elif d >= 0: return '1. Em Fase'
            elif d == -1: return '2. Defasagem Leve'
            elif d == -2: return '3. Defasagem Moderada'
            else: return '4. Defasagem Severa'
            
        df_q1['ian_cat'] = df_q1['defasagem'].apply(classificar_defasagem)
        
        # Removemos quem ficou sem classificação para espelhar o seu df_valido
        df_q1 = df_q1.dropna(subset=['ian_cat'])
        
        ordem_niveis = ['1. Em Fase', '2. Defasagem Leve', '3. Defasagem Moderada', '4. Defasagem Severa']
        
        # 3. Agrupamento
        df_ian_bruto = df_q1.groupby(['ano_referencia', 'ian_cat']).size().reset_index(name='quantidade')
        
        # Truque para forçar todas as categorias a aparecerem (mesmo zeradas)
        anos = df_ian_bruto['ano_referencia'].unique()
        todas_combinacoes = pd.MultiIndex.from_product([anos, ordem_niveis], names=['ano_referencia', 'ian_cat']).to_frame(index=False)
        df_ian = pd.merge(todas_combinacoes, df_ian_bruto, on=['ano_referencia', 'ian_cat'], how='left').fillna({'quantidade': 0})
        
        # 4. As suas cores originais do Colab
        mapa_cores = {
            '1. Em Fase': '#2ca02c', 
            '2. Defasagem Leve': '#f1c40f', 
            '3. Defasagem Moderada': '#ff7f0e',
            '4. Defasagem Severa': '#d62728'
        }

        # Gráfico
        fig1 = px.bar(
            df_ian, 
            x='ano_referencia', 
            y='quantidade', 
            color='ian_cat', 
            text=df_ian['quantidade'].apply(lambda x: int(x) if x > 0 else ""), 
            barmode='stack',
            title="Evolução do Perfil de Defasagem dos Alunos (2022 - 2024)",
            category_orders={'ian_cat': ordem_niveis}, 
            color_discrete_map=mapa_cores
        )
        
        fig1.update_layout(xaxis=dict(dtick=1), xaxis_title="Ano da Pesquisa", yaxis_title="Número de Alunos", legend_title="Nível de Defasagem (IAN)")
        fig1.update_traces(textposition='inside', textfont=dict(color='white', size=14, weight='bold'))
        
        st.plotly_chart(fig1, use_container_width=True)


with st.expander("2. Desempenho acadêmico (IDA) - O desempenho acadêmico médio (IDA) está melhorando, estagnado ou caindo ao longo das fases e anos?"):
        st.markdown("""
        **Análise:** Acompanhamento do Desempenho Acadêmico Médio (IDA). À esquerda, a evolução global por ano. 
        À direita, o comportamento do IDA ao longo do avanço nas fases do programa.
        """)
        
        # 1. Regra de negócio: Extrair número da fase
        def extrair_numero_fase(valor):
            if pd.isna(valor): return np.nan
            texto = str(valor).strip().upper()
            if 'ALFA' in texto: return 0
            numeros = re.findall(r'\d+', texto)
            return int(numeros[0]) if numeros else np.nan

        df_q2 = df.copy()
        df_q2['fase_numerica'] = df_q2['fase'].apply(extrair_numero_fase)
        
        # Estrutura de colunas do Streamlit
        c1, c2 = st.columns(2)
        
        with c1:
            # Gráfico de Barras: IDA por Ano
            df_ida_ano = df_q2.groupby('ano_referencia')['ida'].mean().reset_index()
            fig2a = px.bar(
                df_ida_ano, x='ano_referencia', y='ida', 
                text=df_ida_ano['ida'].apply(lambda x: f'{x:.2f}'),
                title="Média do IDA por Ano",
                color_discrete_sequence=['#3498db']
            )
            fig2a.update_layout(yaxis=dict(range=[0, 10]), xaxis=dict(dtick=1), xaxis_title="Ano", yaxis_title="Nota Média (IDA)")
            fig2a.update_traces(textposition='outside', textfont=dict(size=12, weight='bold'))
            st.plotly_chart(fig2a, use_container_width=True)
            
        with c2:
            # Gráfico de Linhas: IDA por Fase
            df_ida_fase = df_q2.dropna(subset=['fase_numerica', 'ano_referencia', 'ida'])
            df_ida_fase_grp = df_ida_fase.groupby(['fase_numerica', 'ano_referencia'])['ida'].mean().reset_index()
            df_ida_fase_grp['ano_referencia'] = df_ida_fase_grp['ano_referencia'].astype(str)
            
            fig2b = px.line(
                df_ida_fase_grp, x='fase_numerica', y='ida', color='ano_referencia', 
                markers=True, title="IDA ao longo das Fases",
                color_discrete_sequence=['#2ca02c', '#ff7f0e', '#d62728']
            )
            fig2b.update_traces(marker=dict(size=10), line=dict(width=3))
            fig2b.update_layout(yaxis=dict(range=[0, 10]), xaxis=dict(dtick=1), xaxis_title="Fase do Aluno (0 = Alfa)", yaxis_title="Nota Média (IDA)", legend_title="Ano")
            st.plotly_chart(fig2b, use_container_width=True)

with st.expander("3. Engajamento nas atividades (IEG) - O grau de engajamento dos alunos (IEG) tem relação direta com seus indicadores de desempenho (IDA) e do ponto de virada (IPV)?"):
        st.markdown("""
        **Análise:** A análise de correlação entre o Engajamento (IEG), o Desempenho Acadêmico (IDA) e o Ponto de Virada (IPV) 
        revela que existe uma relação direta e positiva, embora a força dessa conexão varie entre os indicadores.
        """)

        # Garantir que os dados são numéricos (tratando possíveis vírgulas)
        df_corr_data = df[['ieg', 'ida', 'ipv']].copy()
        for col in ['ieg', 'ida', 'ipv']:
            df_corr_data[col] = df_corr_data[col].astype(str).str.replace(',', '.')
            df_corr_data[col] = pd.to_numeric(df_corr_data[col], errors='coerce')
            
        # 1. Preparar os dados (Remover nulos como no Colab)
        df_corr_data = df_corr_data.dropna()
        corr_matrix = df_corr_data.corr()
        
        # 2. Criar Colunas no Streamlit para os dois gráficos
        c1, c2 = st.columns(2)

        with c1:
            # Gráfico 1: IEG vs IDA (Teal com linha Vermelha)
            r_ida = corr_matrix.loc["ieg", "ida"]
            fig_ieg_ida = px.scatter(
                df_corr_data, x='ieg', y='ida', 
                trendline="ols",
                trendline_color_override="red",
                title=f"Correlação Engajamento vs Acadêmico<br><sup>(r = {r_ida:.2f})</sup>",
                opacity=0.3
            )
            fig_ieg_ida.update_traces(marker=dict(color='teal'))
            fig_ieg_ida.update_layout(xaxis_title="Engajamento (IEG)", yaxis_title="Desempenho Acadêmico (IDA)")
            st.plotly_chart(fig_ieg_ida, use_container_width=True)

        with c2:
            # Gráfico 2: IEG vs IPV (Coral com linha Azul)
            r_ipv = corr_matrix.loc["ieg", "ipv"]
            fig_ieg_ipv = px.scatter(
                df_corr_data, x='ieg', y='ipv', 
                trendline="ols",
                trendline_color_override="blue",
                title=f"Correlação Engajamento vs Ponto de Virada<br><sup>(r = {r_ipv:.2f})</sup>",
                opacity=0.3
            )
            fig_ieg_ipv.update_traces(marker=dict(color='coral'))
            fig_ieg_ipv.update_layout(xaxis_title="Engajamento (IEG)", yaxis_title="Ponto de Virada (IPV)")
            st.plotly_chart(fig_ieg_ipv, use_container_width=True)

with st.expander("4. Autoavaliação (IAA) - As percepções dos alunos sobre si mesmos (IAA) são coerentes com seu desempenho real (IDA) e engajamento (IEG)?"):
        st.markdown("""
        **Análise:** Comparamos a percepção do aluno sobre si mesmo (IAA) com seu desempenho real (IDA), 
        engajamento (IEG) e aspecto psicossocial (IPS). O gráfico demonstra a força do alinhamento entre a 
        autoavaliação e os demais pilares da jornada.
        """)
        
        # 1. Preparar os dados garantindo as colunas em minúsculo (padrão do app)
        cols = ['iaa', 'ida', 'ieg', 'ips']
        df_q4 = df[cols].copy()
        
        # Tratamento: trocar vírgula por ponto e garantir numérico
        for col in cols:
            df_q4[col] = df_q4[col].astype(str).str.replace(',', '.')
            df_q4[col] = pd.to_numeric(df_q4[col], errors='coerce')
            
        # 2. Calcular matriz de correlação isolando o IAA
        corr_iaa = df_q4.corr()[['iaa']].drop('iaa')
        
        # Reordenar igual ao seu Colab e preparar para o Plotly
        corr_iaa = corr_iaa.reindex(['ida', 'ieg', 'ips']).reset_index()
        corr_iaa.columns = ['Indicador', 'Correlação']
        corr_iaa['Indicador'] = corr_iaa['Indicador'].str.upper() # Transforma 'ida' em 'IDA' para o gráfico
        
        # 3. Gráfico Plotly (Barra Horizontal)
        fig4 = px.bar(
            corr_iaa, 
            x='Correlação', 
            y='Indicador', 
            orientation='h',
            text=corr_iaa['Correlação'].apply(lambda x: f'{x:.3f}'),
            title='Correlação da Autoavaliação (IAA) com outros indicadores',
            color_discrete_sequence=['#B95246']
        )
        
        fig4.update_layout(
            xaxis_title="Valor da Correlação de Pearson", 
            yaxis_title="",
            xaxis=dict(range=[0, 0.35]) # Um pouco de respiro para o texto caber
        )
        fig4.update_traces(textposition='outside', textfont=dict(size=12, weight='bold'))
        
        st.plotly_chart(fig4, use_container_width=True)

with st.expander("5. Aspectos psicossociais (IPS) - Há padrões psicossociais (IPS) que antecedem quedas de desempenho acadêmico ou de engajamento?"):
        st.markdown("""
        **Análise:** Investigamos se há padrões psicossociais (IPS) que antecedem quedas de desempenho acadêmico (IDA) 
        ou de engajamento (IEG). A análise visual da dispersão e as linhas de tendência evidenciam uma relação 
        praticamente independente entre esses pilares.
        """)
        
        # 1. Preparar os dados garantindo as colunas em minúsculo e tratando as vírgulas
        cols = ['ips', 'ida', 'ieg']
        df_q5 = df[cols].copy()
        
        for col in cols:
            df_q5[col] = df_q5[col].astype(str).str.replace(',', '.')
            df_q5[col] = pd.to_numeric(df_q5[col], errors='coerce')
            
        # Remover nulos para o OLS não falhar
        df_q5 = df_q5.dropna()
        
        # 2. Criar Colunas no Streamlit para os dois gráficos
        c1, c2 = st.columns(2)

        with c1:
            # Gráfico 1: IPS vs IDA
            fig_ips_ida = px.scatter(
                df_q5, x='ips', y='ida', 
                trendline="ols",
                trendline_color_override="#2c3e50",
                title='A "Falsa" Relação:<br>Psicossocial (IPS) vs Desempenho (IDA)',
                opacity=0.4,
                color_discrete_sequence=['#e74c3c']
            )
            fig_ips_ida.update_layout(xaxis_title="Indicador Psicossocial (IPS)", yaxis_title="Desempenho Acadêmico (IDA)")
            st.plotly_chart(fig_ips_ida, use_container_width=True)

        with c2:
            # Gráfico 2: IPS vs IEG
            fig_ips_ieg = px.scatter(
                df_q5, x='ips', y='ieg', 
                trendline="ols",
                trendline_color_override="#2c3e50",
                title='A "Falsa" Relação:<br>Psicossocial (IPS) vs Engajamento (IEG)',
                opacity=0.4,
                color_discrete_sequence=['#3498db']
            )
            fig_ips_ieg.update_layout(xaxis_title="Indicador Psicossocial (IPS)", yaxis_title="Engajamento nas Atividades (IEG)")
            st.plotly_chart(fig_ips_ieg, use_container_width=True)

with st.expander("6. Aspectos psicopedagógicos (IPP) - As avaliações psicopedagógicas (IPP) confirmam ou contradizem a defasagem identificada pelo IAN?"):
        st.markdown("""
        **Análise:** As avaliações psicopedagógicas (IPP) são comparadas com os níveis de defasagem dos alunos 
        para entender se a avaliação profissional de desenvolvimento corrobora com o atraso acadêmico medido.
        """)

        # 1. Limpeza rápida e padronização (tudo minúsculo para o df)
        cols_analise = ['ipp', 'ian', 'defasagem']
        df_q6 = df[cols_analise].copy()
        
        for col in cols_analise:
            df_q6[col] = df_q6[col].astype(str).str.replace(',', '.')
            df_q6[col] = pd.to_numeric(df_q6[col], errors='coerce')

        df_q6 = df_q6.dropna(subset=['ipp', 'defasagem'])

        # 2. Reutilizar a classificação para deixar o gráfico mais legível e bonito
        def classificar_defasagem_q6(d):
            if pd.isna(d): return None
            elif d >= 0: return '1. Em Fase'
            elif d == -1: return '2. Defasagem Leve'
            elif d == -2: return '3. Defasagem Moderada'
            else: return '4. Defasagem Severa'

        df_q6['cat_defasagem'] = df_q6['defasagem'].apply(classificar_defasagem_q6)
        
        ordem_niveis = ['1. Em Fase', '2. Defasagem Leve', '3. Defasagem Moderada', '4. Defasagem Severa']
        
        # Cores padronizadas para manter a consistência com a Pergunta 1
        mapa_cores = {
            '1. Em Fase': '#2ca02c', 
            '2. Defasagem Leve': '#f1c40f', 
            '3. Defasagem Moderada': '#ff7f0e',
            '4. Defasagem Severa': '#d62728'
        }

        # 3. Criar as colunas para o Streamlit
        c1, c2 = st.columns(2)

        with c1:
            # Boxplot - A distribuição real
            fig_box = px.box(
                df_q6, 
                x='cat_defasagem', 
                y='ipp', 
                color='cat_defasagem',
                category_orders={'cat_defasagem': ordem_niveis},
                color_discrete_map=mapa_cores,
                title="Distribuição do IPP por Nível de Defasagem"
            )
            fig_box.update_layout(xaxis_title="", yaxis_title="Nota Psicopedagógica (IPP)", showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        with c2:
            # Gráfico de Barras - A média (Visão Executiva)
            media_ipp = df_q6.groupby('cat_defasagem')['ipp'].mean().reset_index()
            fig_bar = px.bar(
                media_ipp, 
                x='cat_defasagem', 
                y='ipp',
                color='cat_defasagem',
                text=media_ipp['ipp'].apply(lambda x: f'{x:.1f}'),
                category_orders={'cat_defasagem': ordem_niveis},
                color_discrete_map=mapa_cores,
                title="Média do IPP por Nível de Defasagem"
            )
            fig_bar.update_layout(xaxis_title="", yaxis_title="Média do IPP", showlegend=False)
            fig_bar.update_traces(textposition='outside', textfont=dict(size=12, weight='bold'))
            # Garantir que o eixo Y vá até um pouco mais que a nota máxima para caber o texto
            fig_bar.update_yaxes(range=[0, df_q6['ipp'].max() * 1.15]) 
            st.plotly_chart(fig_bar, use_container_width=True)

        # 4. O "Print" da correlação formatado bonitão no Streamlit
        corr_ipp_ian = df_q6[['ipp', 'ian']].corr().iloc[0, 1]
        st.info(f"**A Prova Estatística:** A correlação exata entre o desenvolvimento psicopedagógico (IPP) e a adequação de nível (IAN) é de **{corr_ipp_ian:.3f}**.")

with st.expander("7. Ponto de Virada (IPV) - Quais comportamentos - acadêmicos, emocionais ou de engajamento - mais influenciam o IPV ao longo do tempo?"):
        st.markdown("""
        **Análise:** Avaliamos quais comportamentos (acadêmicos, emocionais ou de engajamento) mais influenciam 
        o Ponto de Virada (IPV) de forma consolidada ao longo dos anos (2022 - 2024).
        """)

        # 1. Preparar os dados (garantindo conversão e minúsculas para o app)
        cols_analise = ['ipv', 'ida', 'ieg', 'iaa', 'ips', 'ipp']
        df_q7 = df[cols_analise].copy()

        for col in cols_analise:
            df_q7[col] = df_q7[col].astype(str).str.replace(',', '.')
            df_q7[col] = pd.to_numeric(df_q7[col], errors='coerce')

        # 2. Calcular matriz de correlação com o IPV
        corr_geral = df_q7.corr()[['ipv']].drop('ipv')

        # Renomear indicadores para ficar legível e amigável no gráfico
        renomeio_indicadores = {
            'ida': 'Desempenho (IDA)',
            'ieg': 'Engajamento (IEG)',
            'iaa': 'Autoavaliação (IAA)',
            'ips': 'Psicossocial (IPS)',
            'ipp': 'Psicopedagógico (IPP)'
        }
        corr_geral = corr_geral.rename(index=renomeio_indicadores)
        corr_geral.columns = ['Correlação']

        # Ordenar do maior para o menor (como no seu Colab)
        corr_geral = corr_geral.sort_values(by='Correlação', ascending=True).reset_index()
        corr_geral.columns = ['Indicador', 'Correlação']

        # 3. Gráfico de Barras Horizontais no Plotly
        fig7 = px.bar(
            corr_geral,
            x='Correlação',
            y='Indicador',
            orientation='h',
            text=corr_geral['Correlação'].apply(lambda x: f'{x:.2f}'),
            title='Influência dos Comportamentos no Ponto de Virada (IPV)<br><sup>Visão Consolidada (2022 - 2024)</sup>',
            color='Correlação',
            color_continuous_scale='Viridis'
        )

        fig7.update_layout(
            xaxis_title="Correlação de Pearson com o IPV",
            yaxis_title="Indicadores Analisados",
            xaxis=dict(range=[-0.1, 1.05]), # Respiro para o texto da barra caber
            coloraxis_showscale=False # Oculta a barra lateral de cores para ficar mais limpo
        )
        
        fig7.update_traces(textposition='outside', textfont=dict(size=12, weight='bold'))

        st.plotly_chart(fig7, use_container_width=True)
        
        # Opcional: Tabela de apoio resumida
        st.dataframe(corr_geral.set_index('Indicador').sort_values(by='Correlação', ascending=False).round(3), use_container_width=True)

with st.expander("8. Multidimensionalidade dos indicadores - Quais combinações de indicadores (IDA + IEG + IPS + IPP) elevam mais a nota global do aluno (INDE)?"):
        st.markdown("""
        **Análise:** Avaliamos como o acúmulo de pilares de destaque (alunos que ficam acima da mediana 
        em IDA, IEG, IPS e IPP simultaneamente) impacta diretamente na sua Nota Global (INDE).
        """)

        # 1. Preparar cópia e limpar colunas (tudo em minúsculo)
        df_multi = df.copy()
        df_multi.columns = [str(col).strip().lower() for col in df_multi.columns]
        
        # Identificar dinamicamente qual coluna de INDE está presente
        col_inde = next((c for c in ['inde', 'inde 2024', 'inde 2023', 'inde_2024', 'inde_2023'] if c in df_multi.columns), None)
        
        pilares = ['ida', 'ieg', 'ips', 'ipp']
        cols_necessarias = pilares + ([col_inde] if col_inde else [])

        # Tratamento de dados (vírgula para ponto e conversão numérica)
        for col in cols_necessarias:
            if col in df_multi.columns:
                df_multi[col] = df_multi[col].astype(str).str.replace(',', '.')
                df_multi[col] = pd.to_numeric(df_multi[col], errors='coerce')

        if col_inde:
            # Remover nulos para a análise correta
            df_multi = df_multi.dropna(subset=cols_necessarias).copy()

            # 2. Definir os pilares e criar colunas booleanas baseadas na mediana
            for pilar in pilares:
                mediana = df_multi[pilar].median()
                df_multi[f'alto_{pilar}'] = (df_multi[pilar] >= mediana).astype(int)

            # 3. Calcular a quantidade de pilares em destaque (0 a 4)
            colunas_booleanas = [f'alto_{pilar}' for pilar in pilares]
            df_multi['combinacao_pilares'] = df_multi[colunas_booleanas].sum(axis=1)

            # 4. Preparar a linha de tendência (Média do INDE por grupo)
            df_tendencia = df_multi.groupby('combinacao_pilares')[col_inde].mean().reset_index()

            # 5. Criar o Gráfico com Plotly (Boxplot + Linha de Tendência)
            fig8 = px.box(
                df_multi, 
                x='combinacao_pilares', 
                y=col_inde, 
                color='combinacao_pilares',
                color_discrete_sequence=px.colors.sequential.Blues,
                title="O Poder da Multidimensionalidade no INDE",
                category_orders={'combinacao_pilares': [0, 1, 2, 3, 4]}
            )

            # Adicionar a linha de tendência das médias
            fig8.add_trace(go.Scatter(
                x=df_tendencia['combinacao_pilares'], 
                y=df_tendencia[col_inde],
                mode='lines+markers',
                name='Tendência da Média',
                line=dict(color='red', dash='dash', width=3),
                marker=dict(color='red', size=10)
            ))

            fig8.update_layout(
                xaxis_title="Número de Indicadores Acima da Mediana (IDA, IEG, IPS, IPP)",
                yaxis_title="Nota Global (INDE)",
                showlegend=False
            )

            st.plotly_chart(fig8, use_container_width=True)

            # 6. Tabela de Apoio (Resumo Técnico)
            st.markdown("##### 📋 Resumo Técnico: INDE Médio por Qtd. de Pilares")
            st.table(df_tendencia.rename(columns={
                'combinacao_pilares': 'Qtd. Pilares em Destaque', 
                col_inde: 'Média INDE'
            }).set_index('Qtd. Pilares em Destaque'))
            
        else:
            st.warning("A coluna correspondente ao INDE não foi identificada automaticamente no conjunto de dados para gerar este gráfico.")

with st.expander("9. Previsão de Risco com Machine Learning (Regressão Logística) - Quais padrões nos indicadores permitem identificar alunos em risco antes de queda no desempenho ou aumento da defasagem?"):
        st.markdown("""
        **Análise:** Construímos um modelo preditivo baseado em **Regressão Logística** para identificar a probabilidade 
        de um aluno entrar em risco de defasagem (`Defasagem < 0`) com base em seus pilares (IDA, IEG, IPS, IAA, IPP).
        """)
                
# 1. Preparar cópia e limpar colunas (tudo em minúsculo)
        df_q9 = df.copy()
        df_q9.columns = [str(col).strip().lower() for col in df_q9.columns]

        cols_features = ['ida', 'ieg', 'ips', 'iaa', 'ipp']
        cols_necessarias = cols_features + ['defasagem']

        # Tratamento de dados (vírgula para ponto e conversão numérica)
        for col in cols_necessarias:
            if col in df_q9.columns:
                df_q9[col] = df_q9[col].astype(str).str.replace(',', '.')
                df_q9[col] = pd.to_numeric(df_q9[col], errors='coerce')

        if all(c in df_q9.columns for c in cols_necessarias):
            df_ml = df_q9.dropna(subset=cols_necessarias).copy()
            df_ml['alvo_risco'] = (df_ml['defasagem'] < 0).astype(int)

            X = df_ml[cols_features]
            y = df_ml['alvo_risco']

            if len(df_ml) > 10 and y.nunique() > 1:
                # 2. Divisão treino e teste
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # 3. Normalização das features (StandardScaler)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # 4. Modelo de Regressão Logística com pesos balanceados
                modelo_lr = LogisticRegression(random_state=42, class_weight='balanced')
                modelo_lr.fit(X_train_scaled, y_train)

                y_pred = modelo_lr.predict(X_test_scaled)
                y_proba = modelo_lr.predict_proba(X_test_scaled)[:, 1]

                # 5. Organizar os coeficientes para visualização no Plotly
                importancias = pd.DataFrame({
                    'Indicador': [c.upper() for c in X.columns],
                    'Coeficiente': modelo_lr.coef_[0]
                }).sort_values(by='Coeficiente', ascending=True)

                fig9 = px.bar(
                    importancias,
                    x='Coeficiente',
                    y='Indicador',
                    orientation='h',
                    text=importancias['Coeficiente'].apply(lambda x: f'{x:.3f}'),
                    title='Padrões de Risco: Impacto dos Indicadores na Probabilidade de Defasagem<br><sup>(Regressão Logística Padronizada)</sup>',
                    color='Coeficiente',
                    color_continuous_scale='RdBu_r'
                )
                
                fig9.update_layout(
                    xaxis_title="Coeficiente Padronizado (Impacto no Risco)",
                    yaxis_title="Indicadores",
                    coloraxis_showscale=False
                )
                fig9.update_traces(textposition='outside', textfont=dict(size=12, weight='bold'))
                
                st.plotly_chart(fig9, use_container_width=True)

                # 6. Métricas de Performance do Modelo
                auc_score = roc_auc_score(y_test, y_proba)
                st.markdown(f"🎯 **Performance do Modelo (AUC-ROC Score):** `{auc_score:.4f}`")

                # 7. Tabela de Alunos em Maior Risco
                df_resultados = X_test.copy()
                df_resultados['Probabilidade_Risco'] = y_proba
                df_resultados['Status_Risco_Real'] = y_test
                
                st.markdown("##### 🚨 Top Alunos com Maior Probabilidade de Risco no Teste")
                st.dataframe(
                    df_resultados.sort_values(by='Probabilidade_Risco', ascending=False).head(10).style.format({
                        'Probabilidade_Risco': '{:.2%}'
                    }), 
                    use_container_width=True
                )
            else:
                st.warning("O conjunto de dados filtrado não possui variação suficiente na variável alvo para treinar o modelo.")
        else:
            st.warning("As colunas necessárias para o modelo preditivo não foram encontradas.")

with st.expander("10. Efetividade do Programa (Evolução por Fase 2022-2024)"):
        st.markdown("""
        **Análise:** Avaliamos o volume e a proporção de alunos em cada fase da jornada 
        (Quartzo, Ágata, Ametista e Topázio) ao longo dos ciclos de 2022 a 2024.
        """)

        # 1. Função robusta para encontrar colunas ignorando maiúsculas/minúsculas e espaços
        def encontrar_coluna(df, nome_alvo):
            for col in df.columns:
                if str(col).strip().lower() == nome_alvo.lower():
                    return col
            return None

        # 2. Mapear as colunas existentes no dataframe principal
        col_22 = encontrar_coluna(df, 'pedra 22')
        col_23 = encontrar_coluna(df, 'pedra 2023')
        col_24 = encontrar_coluna(df, 'pedra 2024')

        dfs_para_concatenar = []

        if col_22:
            df_22 = df[[col_22]].copy().rename(columns={col_22: 'Pedra'})
            df_22['Ano'] = '2022'
            dfs_para_concatenar.append(df_22)

        if col_23:
            df_23 = df[[col_23]].copy().rename(columns={col_23: 'Pedra'})
            df_23['Ano'] = '2023'
            dfs_para_concatenar.append(df_23)

        if col_24:
            df_24 = df[[col_24]].copy().rename(columns={col_24: 'Pedra'})
            df_24['Ano'] = '2024'
            dfs_para_concatenar.append(df_24)

        # 3. Verifica se conseguiu montar a base combinada
        if dfs_para_concatenar:
            df_pedras = pd.concat(dfs_para_concatenar, ignore_index=True)
            
            # Limpeza e padronização dos nomes das pedras
            df_pedras = df_pedras.dropna(subset=['Pedra'])
            df_pedras['Pedra'] = df_pedras['Pedra'].astype(str).str.upper().str.replace('Á', 'A').str.strip()

            pedras_oficiais = ['QUARTZO', 'AGATA', 'AMETISTA', 'TOPAZIO']
            df_pedras = df_pedras[df_pedras['Pedra'].isin(pedras_oficiais)]

            if not df_pedras.empty:
                # 4. Cálculo das contagens absolutas e proporções
                contagem = df_pedras.groupby(['Ano', 'Pedra']).size().unstack(fill_value=0)
                
                # Garantir que todas as pedras oficiais existam na tabela, mesmo zeradas
                for p in pedras_oficiais:
                    if p not in contagem.columns:
                        contagem[p] = 0
                        
                contagem = contagem[pedras_oficiais] # Forçar a ordem correta
                proporcoes = contagem.div(contagem.sum(axis=1), axis=0)

                # 5. Criação do Gráfico 100% Empilhado com Plotly
                fig10 = go.Figure()
                cores = ['#d9d9d9', '#90be6d', '#2d6a4f', '#f9e58f']

                for i, pilar in enumerate(pedras_oficiais):
                    textos_barras = [str(val) if val > 0 else "" for val in contagem[pilar]]
                    
                    fig10.add_trace(go.Bar(
                        name=pilar.title(),
                        x=proporcoes.index,
                        y=proporcoes[pilar],
                        text=textos_barras,
                        textposition='inside',
                        insidetextfont=dict(color='white' if i != 0 else 'black', size=14, family='Arial Black'),
                        marker_color=cores[i],
                        marker_line=dict(color='white', width=1.5)
                    ))

                fig10.update_layout(
                    barmode='stack',
                    title="Evolução do Volume de Alunos por Fase (2022 - 2024)",
                    xaxis_title="Ano",
                    yaxis_title="Proporção",
                    yaxis_tickformat='.0%',
                    legend_title="Fase (Pedra)",
                    legend=dict(traceorder='reversed')
                )

                st.plotly_chart(fig10, use_container_width=True)
                
                # Tabela de Apoio
                st.markdown("##### 📋 Resumo: Volume Absoluto de Alunos")
                st.dataframe(contagem.style.background_gradient(cmap='Greens', axis=None), use_container_width=True)
            else:
                st.warning("As colunas foram encontradas, mas não há dados válidos de Quartzo, Ágata, Ametista ou Topázio registradas nelas.")
        else:
            st.warning("As colunas de pedras ('Pedra 22', 'Pedra 2023', 'Pedra 2024') não foram encontradas no conjunto de dados. Verifique a grafia exata na sua base.")



# ==============================================================================
# ABA 3: SIMULADOR (Com as features do nosso modelo)
# ==============================================================================
with aba3:
    st.header("🔮 Simulador de Risco de Defasagem (Ano Seguinte)")
    
    st.markdown("Insira os dados atuais do aluno e a variação ($\Delta$) em relação ao ano passado para prever a probabilidade de entrar em defasagem severa no próximo ano.")
    
    with st.form("sim"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("**Perfil e Histórico**")
            fase = st.number_input("Fase (Numérico)", 0.0, 10.0, 1.0, step=1.0)
            tempo_prog = st.number_input("Tempo de Programa (Anos)", 1, 10, 2)
            
            st.markdown("**Variações Ano a Ano (Deltas)**")
            delta_ida = st.number_input("Variação IDA (Delta)", -10.0, 10.0, 0.0, step=0.1)
            delta_ieg = st.number_input("Variação IEG (Delta)", -10.0, 10.0, 0.0, step=0.1)
            delta_inde = st.number_input("Variação INDE (Delta)", -10.0, 10.0, 0.0, step=0.1)

        with c2:
            st.markdown("**Indicadores Principais**")
            inde = st.slider("INDE Atual", 0.0, 10.0, 7.0, step=0.1)
            ian = st.slider("IAN Atual", 0.0, 10.0, 7.0, step=0.1)
            ida = st.slider("IDA Atual", 0.0, 10.0, 7.0, step=0.1)
            ieg = st.slider("IEG Atual", 0.0, 10.0, 7.0, step=0.1)

        with c3:
            st.markdown("**Indicadores Complementares**")
            iaa = st.slider("IAA Atual", 0.0, 10.0, 7.0, step=0.1)
            ips = st.slider("IPS Atual", 0.0, 10.0, 7.0, step=0.1)
            ipp = st.slider("IPP Atual", 0.0, 10.0, 7.0, step=0.1)
            ipv = st.slider("IPV Atual", 0.0, 10.0, 7.0, step=0.1)

        if st.form_submit_button("ANALISAR RISCO"):
            # O DataFrame de input deve conter exatamente os mesmos nomes usados no treino
            in_df = pd.DataFrame({
                'INDE': [inde], 'IAN': [ian], 'IDA': [ida], 'IEG': [ieg], 
                'IAA': [iaa], 'IPS': [ips], 'IPP': [ipp], 'IPV': [ipv], 
                'Fase': [fase], 'Tempo_Programa': [tempo_prog], 
                'Delta_IDA': [delta_ida], 'Delta_IEG': [delta_ieg], 'Delta_INDE': [delta_inde]
            })
            
            try:
                # Pegando a probabilidade da classe 1 (Risco)
                prob = modelo.predict_proba(in_df)[0][1]
                st.metric("Probabilidade de Risco", f"{prob*100:.1f}%")
                
                # Aplicando o nosso Threshold super restrito de 0.75 para alta precisão
                if prob >= 0.75: 
                    st.error("🚨 ALTO RISCO (Intervenção Necessária)")
                else: 
                    st.success("✅ ESTÁVEL (Risco Controlado)")
            except Exception as e:
                st.error(f"Erro ao gerar predição. Verifique se os dados estão no formato correto. Detalhe: {e}")

st.caption("Associação Passos Mágicos | Datathon F5 FIAP Data Analytics")
