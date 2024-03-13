import texts2insight as t2i
import streamlit as st
from plotly.subplots import make_subplots


st.config.set_option("theme.primaryColor", '#09A3D5')

# --------- SIDEBAR ---------
st.sidebar.title('Texts2️⃣Insight')
st.sidebar.write('streamlit | pymorphy3 | spacy ')

with st.sidebar.expander('Входные тексты',expanded=True):
    input_size       = st.number_input('Количество собраний',min_value=1,value=2)
    text_collections = [st.file_uploader(f':blue[собрание текстов №{i}]', accept_multiple_files=True) for i in range(input_size)]

with st.sidebar.expander('Настройки'):
    option = st.selectbox(
   "Модель токенизации",
   ("Email", "Home phone", "Mobile phone"),
   index=None,
   placeholder="Select contact method...",)
if [] not in text_collections:
    nlp = t2i.NLP(text_collections, model_name="ru_core_news_lg")
    # st.write(nlp)
    # st.write(nlp.statistics_)
    # st.write(nlp.statistics)

    # --------- Main ---------
    st.header('Предварительный просмотр')
    i = st.selectbox(
            "Выберите коллекцию",
            range(len(text_collections)))

    with st.expander(f'Просмотр коллекции №{i}'):
        st.write('Первые 100 токенов:')
        st.write(nlp.preview(i))
        
    st.header('Общая статистика')
    st.write(nlp.make_stats_table())

    st.header('Лексика')
    st.write(nlp.make_freq_table().T['Несловарные слова_'])
    cols =st.columns(3)
    cols[1].write(nlp.plot_pos_types_freq(type='pos_freq_'),use_container_width=False)
    cols =st.columns(4)
    cols[0].write(nlp.plot_pos_types_freq(type='noun_case_freq_'),use_container_width=False)
    cols[-1].write(nlp.plot_pos_types_freq(type='pron_case_freq_'),use_container_width=False)
    cols =st.columns(4)
    cols[0].write(nlp.plot_pos_types_freq(type='pron_number_freq_'),use_container_width=False)
    cols[-1].write(nlp.plot_pos_types_freq(type='noun_number_freq_'),use_container_width=False)
    cols =st.columns(4)
    cols[0].write(nlp.plot_pos_types_freq(type='aux_tense_freq_'),use_container_width=False)
    cols[-1].write(nlp.plot_pos_types_freq(type='verb_tense_freq_'),use_container_width=False)
    cols =st.columns(4)
    cols[0].write(nlp.plot_pos_types_freq(type='pron_gender_freq_'),use_container_width=False)
    cols[-1].write(nlp.plot_pos_types_freq(type='noun_gender_freq_'),use_container_width=False)
    cols =st.columns(4)
    cols[0].write(nlp.plot_pos_types_freq(type='adj_degree_freq_'),use_container_width=False)
    cols[-1].write(nlp.plot_pos_types_freq(type='noun_animacy_freq_'),use_container_width=False)

    # st.header('Морфология')
    # st.header('Анализ схожести')
    # st.header('Генератор текста')


