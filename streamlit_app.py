import streamlit as st
import os
import requests
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from db_manager import Base, Case
import re
import logging
import json
from typing import List
import gdown
import base64

# Streamlit 설정
st.set_page_config(page_title="AI 기반 맞춤형 판례 검색 서비스", layout="wide")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 상수 정의
API_KEY = "D/spYGY15giVS64SLvtShZlNHxAbr9eDi1uU1Ca1wrqCiU+0YMwcnFy53naflVlg5wemikAYwiugNoIepbpexQ=="
API_URL = "https://api.odcloud.kr/api/15069932/v1/uddi:3799441a-4012-4caa-9955-b4d20697b555"
CACHE_FILE = "legal_terms_cache.json"
DB_FILE = "legal_cases.db"
DB_URL = "https://drive.google.com/uc?id=1rBTbbtBE5K5VgiuTvt3JgneuJ8odqCJm"

# 데이터베이스 엔진 정의
engine = create_engine(f'sqlite:///{DB_FILE}')
Session = sessionmaker(bind=engine)

# 배경 이미지 함수
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# CSS 스타일 정의
def local_css():
    st.markdown("""
    <style>
    body {
        font-family: Arial, sans-serif;
        line-height: 1.6;
        color: #333;
    }
    .legal-term {
        color: #007bff;
        cursor: help;
    }
    .main-content {
        text-align: center;
        padding: 2rem;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        margin: 2rem auto;
        max-width: 800px;
    }
    .usage-guide-container, .guide-container {
        background-color: rgba(248, 248, 248, 0.9);
        padding: 2rem;
        margin: 2rem auto;
        border-radius: 10px;
        max-width: 800px;
    }
    .usage-guide-title, .guide-title h2 {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .search-button {
        background-color: #000;
        color: #fff;
        padding: 0.75rem 2rem;
        font-size: 1.2rem;
        font-weight: bold;
        text-decoration: none;
        border-radius: 25px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

# 기존의 함수들 (get_legal_terms, download_db, check_db, load_cases, get_vectorizer_and_matrix)은 그대로 유지

def highlight_legal_terms(text: str) -> str:
    terms = get_legal_terms()
    for term, explanation in terms.items():
        pattern = r'\b' + re.escape(term) + r'\b'
        replacement = f'<span class="legal-term" data-toggle="tooltip" title="{explanation}">{term}</span>'
        text = re.sub(pattern, replacement, text)
    return text

def show_main_page():
    st.markdown("""
    <div class="main-content">
        <h1>AI 기반 맞춤형 판례 검색 서비스</h1>
        <p>당신의 상황에 가장 적합한 판례를 찾아드립니다</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="usage-guide-container">
        <div class="usage-guide-title">이용 방법</div>
        <ul>
            <li><strong>법률 분야 선택:</strong> 검색하고 싶은 법률의 분야를 선택하면 더 정확하게 나와요.</li>
            <li><strong>상황 설명:</strong> 법률 문제를 최대한 자세히 작성해주세요.</li>
            <li><strong>검색 실행:</strong> 날짜, 관련자, 사건 경과를 언급해주세요.</li>
            <li><strong>결과 확인:</strong> 검색 버튼을 눌러 유사 판례를 확인하세요.</li>
            <li><strong>재검색:</strong> 필요시 '재검색' 버튼을 눌러 새로운 검색을 시작하세요.</li>
        </ul>
    </div>
    <div class="guide-container">
        <div class="guide-title"><h2>작성 가이드라인</h2></div>
        <ol>
            <li>사건의 발생 시기와 장소를 명시해주세요.</li>
            <li>관련된 사람들의 관계를 설명해주세요.</li>
            <li>사건의 경과를 시간 순서대로 작성해주세요.</li>
            <li>문제가 되는 행위나 상황을 설명해주세요.</li>
            <li>알고 싶은 법률적 문제를 명확히 해주세요.</li>
        </ol>
        <div class="guide-example">
            "2023년 3월 1일, 서울시 강남구의 한 아파트를 2년 계약으로 월세 100만원에 임대했습니다. 계약 당시 집주인과 구두로 2년 후 재계약 시 월세를 5% 이상 올리지 않기로 약속했습니다. 그러나 계약 만료 3개월 전인 2024년 12월, 집주인이 갑자기 월세를 150만원으로 50% 인상하겠다고 통보했습니다. 이를 거부하면 퇴거해야 한다고 합니다. 구두 약속은 법적 효력이 있는지, 그리고 이런 과도한 월세 인상이 법적으로 가능한지 알고 싶습니다."
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("바로 시작", key="start_button"):
        st.session_state.page = "search"

def show_search_page():
    st.markdown("<h1 style='text-align: center;'>법률 판례 검색</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("<h5 style='font-weight: bold;'>법률 분야 선택</h5>", unsafe_allow_html=True)
        legal_fields = ['민사', '가사', '형사A(생활형)', '형사B(일반형)', '행정', '기업', '근로자', '특허/저작권', '금융조세', '개인정보/ict', '잘모르겠습니다']
        selected_fields = st.multiselect("", legal_fields)

    with col2:
        st.markdown("<h5 style='font-weight: bold;'>상황 설명</h5>", unsafe_allow_html=True)
        st.write("아래 가이드라인을 참고하여 귀하의 법률 상황을 자세히 설명해주세요.")
        
        st.markdown("""
        <div style='background-color: rgba(240, 240, 240, 0.9); padding: 10px; border-radius: 5px;'>
        <h6 style='font-weight: bold;'>작성 가이드라인</h6>
        <ol>
            <li>사건의 발생 시기와 장소를 명시해주세요.</li>
            <li>관련된 사람들의 관계를 설명해주세요. (예: 고용주-직원, 판매자-구매자)</li>
            <li>사건의 경과를 시간 순서대로 설명해주세요.</li>
            <li>문제가 되는 행위나 상황을 구체적으로 설명해주세요.</li>
            <li>현재 상황과 귀하가 알고 싶은 법률적 문제를 명확히 해주세요.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        user_input = st.text_area("상황을 입력하세요:", height=200)

        if st.button("검색", key="search_button"):
            if user_input and len(user_input) > 3:
                st.session_state.user_input = user_input
                st.session_state.selected_fields = selected_fields
                st.session_state.page = "result"
            else:
                st.error("검색어가 없거나 너무 짧습니다")

def show_result_page():
    st.markdown("<h1 style='text-align: center;'>판례 검색 결과</h1>", unsafe_allow_html=True)

    user_input = st.session_state.user_input
    selected_fields = st.session_state.selected_fields

    with st.spinner('판례를 검색 중입니다...'):
        result = get_vectorizer_and_matrix()
        if result is None or len(result) != 3:
            st.error("데이터를 불러오는 데 실패했습니다. 관리자에게 문의해주세요.")
            return
        
        vectorizer, tfidf_matrix, cases = result

        if not selected_fields or '잘모르겠습니다' in selected_fields:
            filtered_cases = cases
            filtered_tfidf_matrix = tfidf_matrix
        else:
            filtered_cases = [case for case in cases if case.class_name in selected_fields]
            filtered_tfidf_matrix = vectorizer.transform([case.summary for case in filtered_cases if case.summary])
        
        if not filtered_cases:
            st.warning("선택한 법률 분야에 해당하는 판례가 없습니다. 다른 분야를 선택해주세요.")
            return

        user_vector = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, filtered_tfidf_matrix)
        most_similar_idx = similarities.argmax()
        case = filtered_cases[most_similar_idx]

    st.markdown("<h3 style='font-weight: bold;'>요약</h3>", unsafe_allow_html=True)
    st.markdown(highlight_legal_terms(case.summary), unsafe_allow_html=True)
    
    if case.jdgmnQuestion:
        st.markdown("<h3 style='font-weight: bold;'>핵심 질문</h3>", unsafe_allow_html=True)
        st.markdown(highlight_legal_terms(case.jdgmnQuestion), unsafe_allow_html=True)
    
    if case.jdgmnAnswer:
        st.markdown("<h3 style='font-weight: bold;'>답변</h3>", unsafe_allow_html=True)
        st.markdown(highlight_legal_terms(case.jdgmnAnswer), unsafe_allow_html=True)

    if st.button("다시 검색하기", key="research_button"):
        st.session_state.page = "search"

def main():
    local_css()
    set_png_as_page_bg('static/photo.png')

    if 'page' not in st.session_state:
        st.session_state.page = "main"

    if st.session_state.page == "main":
        show_main_page()
    elif st.session_state.page == "search":
        show_search_page()
    elif st.session_state.page == "result":
        show_result_page()

if __name__ == '__main__':
    main()