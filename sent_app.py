import streamlit as st
from sentiment import sentiment
import re

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
    per_statement = st.toggle('REVEAL SENTIMENT PER STAMENT')
    if not per_statement:
        button = st.button('PREDICT SENTIMENT')
        if button:
            if not response:
                st.warning('PLEASE INPUT TEXT')
                st.snow()
            else:
                st.balloons()
                SENTIMENT = sentiment.app_pred(response)
                if SENTIMENT == "Positive":
                    st.badge(f'STATEMENT TENDS TO BE: {SENTIMENT}', color = 'green')
                elif SENTIMENT == 'Negative':
                    st.badge(f'STATEMENT TENDS TO BE: {SENTIMENT}', color='orange')
                else:
                    st.badge(f'STATEMENT TENDS TO BE: {SENTIMENT}', color='gray')
    else:
        button = st.button('PREDICT SENTIMENT PER STATEMENT')
        if button:
            if not response:
                st.warning('PLEASE INPUT TEXT')
                st.snow()
            else:
                st.balloons()
                statements = re.split(r'[.,?!]\s*', response)
                x = 0
                for i in statements[:-1]:
                    SENTIMENT = sentiment.app_pred(i)
                    if SENTIMENT == "Positive":
                        st.badge(f'STATEMENT {x+1} TENDS TO BE: {SENTIMENT}', color = 'green')
                    elif SENTIMENT == 'Negative':
                        st.badge(f'STATEMENT {x+1} TENDS TO BE: {SENTIMENT}', color='orange')
                    else:
                        st.badge(f'STATEMENT {x+1} TENDS TO BE: {SENTIMENT}', color='gray')
                    x+=1