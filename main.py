import streamlit as st

from evaluator import Evaluator
from search_engine import SearchEngine

data_sets = {
    'antique': 0, 'wikir': 1
}


def get_search_results(query, data_set_name):
    search_engine = SearchEngine()
    return search_engine.search(query, data_sets[data_set_name])


st.sidebar.title('Navigation')
screen = st.sidebar.radio('', ['Home', 'Statistics'])

if screen == 'Home':
    st.image('assets/google.png', width=300)

    name = st.radio('Select an dataset:', ('antique', 'wikir'))

    embedding_model = st.radio('Select an embedding model:', ('tf-idf', 'sentence transformer'))

    search_query = st.text_input('Enter your search query:')

    if st.button('Submit'):
        if search_query:
            st.markdown(f'<p class="medium-font">Searching for: {search_query} in {name}</p>',
                        unsafe_allow_html=True)
            results = get_search_results(query=search_query, data_set_name=name)
            st.markdown(f'<p class="medium-font"> Search results: </p>', unsafe_allow_html=True)
            for result in results:
                st.write(f'<p class="medium-font">{result}</p>', unsafe_allow_html=True)
        else:
            st.write('Please enter a search query before submitting.')
elif screen == 'Statistics':
    evaluator = Evaluator(search_engine=SearchEngine())
    st.write(f'evaluation of the search engine is {evaluator.calculate_avg_precision_recall()}')
