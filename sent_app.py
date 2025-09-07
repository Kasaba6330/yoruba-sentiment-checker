import streamlit as st
from sentiment import sentiment
import re

if __name__ == '__main__':
    st.set_page_config(page_title='Yorùbá Sentiment Checker', layout='wide')
    st.title('A SENTIMENT CHECKER FOR YORùbá TEXT'.upper())
    col1, col2 = st.columns(2)
    choice = col1.toggle('FILE OR NOT?')
    if choice:
        file = col1.file_uploader('UPLOAD YOUR FILE/S HERE', type = ['txt', 'pdf', 'doc', 'docx'], accept_multiple_files=True)
        tmp = ''
        for i in file:
            try:
                tmp+= i.getvalue().decode(encoding='utf-8')
            except:
                st.toast(f'COULD NOT PROCESS {i.name}')
                pass
        if 'CONTENT' not in st.session_state:
            st.session_state.CONTENT = tmp
        else:
            del st.session_state.CONTENT
            st.session_state.CONTENT = tmp
            
    response = col2.text_area(label = 'INPUT TEXT HERE')
    '___'
    per_statement = st.toggle('REVEAL SENTIMENT PER STATEMENT')
    if not per_statement:
        button = st.button('PREDICT OVERALL SENTIMENT')
        if button:
            if 'CONTENT' in st.session_state:
                st.session_state.CONTENT += response
            else:
                st.session_state.CONTENT = response

            if not st.session_state.CONTENT:
                st.warning('PLEASE INPUT TEXT OR UPLOAD A FILE')
                st.snow()
            else:
                st.balloons()
                SENTIMENT = sentiment.app_pred(st.session_state.CONTENT)
                if SENTIMENT == "Positive":
                    st.badge(f'OVERALL STATEMENT TENDS TO BE: {SENTIMENT}', color = 'green')  
                elif SENTIMENT == 'Negative':
                    st.badge(f'OVERALL STATEMENT TENDS TO BE: {SENTIMENT}', color='orange')
                else:
                    st.badge(f'OVERALL STATEMENT TENDS TO BE: {SENTIMENT}', color='gray')
                    
                del st.session_state.CONTENT
                
    else:
        button = st.button('PREDICT SENTIMENT PER STATEMENT')
        if button:
            if 'CONTENT' in st.session_state:
                st.session_state.CONTENT += response
            else:
                st.session_state.CONTENT = response
              
            if not st.session_state.CONTENT:
                st.warning('PLEASE INPUT TEXT OR UPLOAD A FILE')
                st.snow()
            else:
                st.balloons()
                statements = re.split(r'[.,?!]\s*', st.session_state.CONTENT)
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
                del st.session_state.CONTENT