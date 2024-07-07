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
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
    }
    .start-button {
        background-color: #000;
        color: #fff;
        padding: 0.75rem 2rem;
        font-size: 1.2rem;
        font-weight: bold;
        text-decoration: none;
        border-radius: 25px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .start-button:hover {
        background-color: #333;
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
    .guide-example {
        background-color: rgba(240, 240, 240, 0.9);
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def get_legal_terms() -> dict:
    if os.path.exists(CACHE_FILE):
        logging.info("저장된 용어 사전 불러오기")
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            legal_terms_dict = json.load(f)
        logging.info(f"{len(legal_terms_dict)}개의 법률 용어를 캐시에서 불러왔습니다.")
    else:
        logging.info("API에서 법률 용어 데이터 가져오기 시작")
        params = {
            "serviceKey": API_KEY,
            "page": 1,
            "perPage": 1000
        }
        response = requests.get(API_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                legal_terms_dict = {item['용어명']: item['설명'] for item in data['data']}
                logging.info(f"{len(legal_terms_dict)}개의 법률 용어를 가져왔습니다.")
                
                with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(legal_terms_dict, f, ensure_ascii=False, indent=2)
                logging.info("법률 용어 데이터를 캐시 파일에 저장했습니다.")
            else:
                logging.error("API 응답에 'data' 키가 없습니다.")
                legal_terms_dict = {}
        else:
            logging.error(f"API 요청 실패: 상태 코드 {response.status_code}")
            legal_terms_dict = {}
    
    return legal_terms_dict

def download_db():
    try:
        gdown.download(DB_URL, DB_FILE, quiet=False)
        logging.info(f"데이터베이스 다운로드 완료: {DB_FILE}")
        # 데이터베이스 다운로드 후 엔진 재생성
        global engine
        engine = create_engine(f'sqlite:///{DB_FILE}')
        Base.metadata.create_all(engine)
    except Exception as e:
        logging.error(f"데이터베이스 다운로드 실패: {str(e)}")
        st.error("데이터베이스 다운로드에 실패했습니다. 네트워크 연결을 확인하고 다시 시도해주세요.")

def check_db():
    if not os.path.exists(DB_FILE):
        logging.info("데이터베이스 파일이 없습니다. 다운로드를 시작합니다.")
        download_db()
    
    inspector = inspect(engine)
    if not inspector.has_table('cases'):
        logging.warning("cases 테이블이 없습니다. 데이터베이스를 다시 다운로드합니다.")
        download_db()
        return False
    
    return True

@st.cache_resource
def load_cases() -> List[Case]:
    if not check_db():
        return []
    session = Session()

    logging.info("데이터베이스에서 판례 데이터 로딩 시작")
    try:
        total_cases = session.query(Case).count()
        logging.info(f"총 {total_cases}개의 판례가 데이터베이스에 있습니다.")
        
        cases = list(session.query(Case))
        logging.info(f"총 {len(cases)}개의 판례를 로드했습니다.")
        return cases

    except Exception as e:
        logging.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return []

    finally:
        session.close()

@st.cache_resource
def get_vectorizer_and_matrix():
    try:
        cases = load_cases()
        if not cases:
            st.error("케이스 데이터가 비어 있습니다.")
            return None, None, None

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([case.summary for case in cases if case.summary])
        return vectorizer, tfidf_matrix, cases
    except Exception as e:
        logging.error(f"get_vectorizer_and_matrix 함수에서 오류 발생: {str(e)}")
        st.error(f"데이터 처리 중 오류가 발생했습니다: {str(e)}")
        return None, None, None

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
    <div class="button-container">
        <button class="start-button" id="start-button">바로 시작</button>
    </div>
    """, unsafe_allow_html=True)

    # JavaScript를 사용하여 버튼 클릭 이벤트 처리
    st.markdown("""
    <script>
    document.getElementById('start-button').addEventListener('click', function() {
        // Streamlit 컴포넌트에 메시지 전송
        window.parent.postMessage({type: 'streamlit:setComponentValue', value: true}, '*');
    });
    </script>
    """, unsafe_allow_html=True)

    # Streamlit 버튼 (숨겨짐)
    if st.button("Start", key="start_button", style="display:none;"):
        st.session_state.page = "search"
        st.experimental_rerun()

    # JavaScript에서 보낸 메시지 처리
    if st.session_state.get('start_clicked', False):
        st.session_state.page = "search"
        st.session_state.start_clicked = False
        st.experimental_rerun()

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