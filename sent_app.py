import streamlit as st
from sentiment import sentiment
from time import sleep

if __name__ == '__main__':
    # print('Error: \n', file=open('log1.txt', 'w', encoding='utf-8', buffering=1, errors = 'errors'), flush=True)
    st.title('A SENTIMENT CHECKER FOR YORùbá TEXT'.upper())
    col1, col2 = st.columns(2)
    choice = col1.toggle('FILE OR NOT?')
    if choice:
        file = col1.file_uploader('UPLOAD YOUR FILE/S HERE', type = ['txt', 'pdf', 'doc', 'docx'], accept_multiple_files=True)
        tmp = ''
        for i in file:
            tmp+= i.getvalue().decode(encoding='utf-8')
        if 'CONTENT' not in st.session_state:
            st.session_state.CONTENT = tmp
        else:
            del st.session_state.CONTENT
            st.session_state.CONTENT = tmp
            
    response = col2.text_area(label = 'INPUT TEXT HERE')
    button = st.button('REVEAL SENTIMENT')
    if button:
        if not response:
            st.warning('YOU HAVE NOT INPUTED ANYTHING!')
            # with sleep(2):
            st.snow()
        else:
            SENTIMENT = sentiment.app_pred(response)
            st.subheader(f'STATEMENT TENDS TO BE: {SENTIMENT}')
            