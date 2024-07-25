import streamlit as st
import pandas as pd
import mammoth
from bs4 import BeautifulSoup
import re
from io import BytesIO
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

import pprint
from typing import Any, Dict
import os
import pandas as pd
from langchain.output_parsers import PandasDataFrameOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import time 

# load_dotenv()
os.environ['OPENAI_API_KEY'] = "sk-proj-Q87yeSCZ8M75Wum8jHAZT3BlbkFJEqtz6O7Wn3Me3pqHipz7"

def dataframe_to_text(df):
    text = ""
    for index, row in df.iterrows():
        row_text = " | ".join(map(str, row.values))  # 각 셀 값을 파이프로 구분
        text += row_text + "\n"
    return text

def dataframe_first_column_as_header2(df):
    # 첫 번째 열을 컬럼명으로 사용
    new_header = df.iloc[0]  # 첫 번째 행을 새로운 헤더로 설정
    df = df[1:]  # 헤더로 사용한 첫 번째 행을 데이터프레임에서 제거
    df.columns = new_header  # 새로운 헤더 설정
    return df

def dataframe_to_markdown2(df):
    # 데이터프레임을 문자열로 변환
    df = dataframe_first_column_as_header2(df)
    df = df.reset_index(drop=True)
    
    # 병합된 셀 처리: NaN 값을 바로 위의 값으로 채우기
    df = df.fillna(method='ffill')

    # 컬럼명이 None일 경우 빈 문자열로 변환
    df.columns = df.columns.fillna('')

    # 컬럼명 변환
    header = '| ' + ' | '.join(map(str, df.columns)) + ' |'
    
    # 구분선 변환
    separator = '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'
    
    # 각 행을 변환
    rows = []
    for _, row in df.iterrows():
        row_values = map(lambda x: str(x).replace('\n', ''), row.values)
        rows.append('| ' + ' | '.join(row_values) + ' |')
    
    # 전체 마크다운 테이블 조합
    markdown_table = header + '\n' + separator + '\n' + '\n'.join(rows)
    return markdown_table



def dataframe_first_column_as_header(df):
    # 첫 번째 열을 컬럼명으로 사용
    new_header = df.iloc[0]  # 첫 번째 행을 새로운 헤더로 설정
    df = df[1:]  # 헤더로 사용한 첫 번째 행을 데이터프레임에서 제거
    df.columns = new_header  # 새로운 헤더 설정
    return df

def dataframe_to_markdown(df):
    # 데이터프레임을 문자열로 변환
    df = dataframe_first_column_as_header(df)
    df = df.reset_index(drop=True)

    # 컬럼명이 None일 경우 빈 문자열로 변환
    df.columns = df.columns.fillna('')

    # 컬럼명 변환
    header = '| ' + ' | '.join(map(str, df.columns)) + ' |'
    
    # 구분선 변환
    separator = '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'
    
    # 각 행을 변환
    rows = []
    for _, row in df.iterrows():
        row_values = map(lambda x: str(x).replace('\n', ''), row.values)
        rows.append('| ' + ' | '.join(row_values) + ' |')
    
    # 전체 마크다운 테이블 조합
    markdown_table = header + '\n' + separator + '\n' + '\n'.join(rows)
    return markdown_table

def llm_invoke(validation_text, split_content):
    # 검증 기준 설정
   
    print(split_content[0])
    text_bf = ''
    detected_text = ''
    text_output = ''
    validation = validation_text

    # split_contents = []
    # split_contents.append(split_content[0])
    # 내용 추가
    # for contents in split_contents :
    for contents in split_content :
        text_output = ''
        for item in contents :

# for contents in split_content :
#                     text_output = ''
#                     for item in contents :
#                         if len(item) == 1 :
#                             text_output += dataframe_to_text(item) + '\n'
#                         else :
#                             text_output += dataframe_to_markdown(item) + '\n'


            if len(item) == 1 :
                text_output += dataframe_to_text(item) + '\n'
            else :
                text_output += dataframe_to_markdown(item) + '\n\n'

            text_output = text_output.replace('   ', '')
        
    
        prompt_t = ChatPromptTemplate.from_messages(
            messages=[
                ('system', """
- [검증기준]은 회차별, 항목별 정확한 값들을 포함하고 있습니다. 이를 기준으로 주어지는 내용의 오류를 찾아내야 합니다.
- 의미가 동일하거나 표현 방식만 다른 경우는 오류로 간주하지 않습니다.
- 표현 형식이 다르지만, 수치나 의미가 동일하면 오류로 처리하지 않습니다.
- 검증 대상 내용 중 수치적 오류나 명백히 다른 내용은 무조건 검출해야 합니다.
- '단위: 원'과 같이 수치에 관련된 정보가 있는 경우, 실제 값과 비교하여 오류를 검출합니다.
- [검증기준]과 상이한 일자, 수치, 내용 등을 정확히 식별하고, 그 결과를 명확히 보고합니다.
- 오류가 발견되면 [출력 예시]와 같이 '항목명:잘못된 내용>>정정된 내용' 형식내용만 출력해야합니다.
- 주어지는 내용을 위에서부터 순차적으로 검증해서 오류만 출력해야합니다.
- 검증 중 잘못된 정보를 발견하지 못하면 계약이 파기될 수 있으므로 주의 깊게 검토해야 합니다.
- 검증은 주어진 내용과 [검증기준]에 제시된 정확한 값들을 비교 분석하는 데 집중해야 합니다.
## 참고사항: 
- 인수금액들 합계가 인수금액 합계랑 같은 내용이니 주의하세요.
- 주어지는 정보는 데이터프레임 리스트를 모두 텍스트로 변환한 데이터야.
- 인수금액 합계는 인수인 들의 인수금액들의 합계입니다.
- 환산 값이 같으면(예: 100,000,000,000 = 1,000억원) 잘 못된 내용이 아니야.
- 오기재된 내용 (ex. 삼천억원(200,000,000,000))도 검출해주세요.
- 같은 의미(ex. 2027년 02월 07일, 2027년 02월 07일(3년))는 잘못된 내용이 아니므로 검출하지 말아야합니다.
- 수치적으로 동일한 값을 다른 형식으로 표현했을 경우(ex. 100,000,000,000, 1,000억원) 오류로 간주하지 않도록 주의하세요.
-----
[출력 예시]:
만기일: 2027년 02월 07일>>2029년 02월 07일
인수금액: 삼천억원 (300,000,000,000 원)>>이백만원 (200,000,000 원)
-----
[검증기준]:
{validation}
"""),
                ('user', '{text_output}')
            ]
        )

        prompt = ChatPromptTemplate.from_messages(
            messages=[
                ('system', """
- [검증기준]은 회차별, 항목별 정확한 값들을 포함하고 있습니다. 이를 기준으로 주어지는 내용의 오류를 찾아내야 합니다.
- 검증 대상 내용 중 수치적 오류나 명백히 다른 내용만을 검출해야 합니다.
- 검증 대상의 회차가 오입력될 경우(ex. 검증기준: 45-1, 45-2, 검증대상: 44-2 포함) 검증기준과 가장 유사한 회차로 변경 인식해야합니다.
- '단위: 원'과 같이 수치에 관련된 정보가 있는 경우, 실제 값과 비교하여 오류를 검출합니다.
- [검증기준]과 상이한 일자, 수치, 내용 등을 정확히 식별하고, 그 결과를 명확히 검출합니다.
- 오류가 발견되면 [출력 예시]와 같이 '순번. 항목명:잘못된 내용>>기준 내용' 형식내용만 출력해야합니다.
- 주어지는 내용을 위에서부터 순차적으로 검증해서 오류만! 출력해야합니다.
- 검증 중 잘못된 정보를 발견하지 못하면 계약이 파기될 수 있으므로 주의 깊게 검토해야 합니다.
- 검증은 주어진 내용과 [검증기준]에 제시된 정확한 값들을 비교 분석하는 데 집중해야 합니다.
- 아래 [참고사항]도 반영해서 오류를 검출해주고 오류가 있을 경우에만 그 내용을 작성하고 없을 경우는 빈칸으로 줘야합니다..
-----
[참고사항]: 
- 오기재된 내용 (ex. 삼천억원(200,000,000,000))도 검출해주세요.
- 의미가 동일하거나 표현 방식만 다른 경우는 오류로 간주하지 않습니다.
- 표현 형식이 다르지만, 수치나 의미가 동일하면 오류로 처리하지 않습니다.
- 인수금액들 합계가 인수금액 합계랑 같은 내용이니 주의하세요.
- 주어지는 정보는 데이터프레임 리스트를 모두 텍스트로 변환한 데이터야.
- 인수금액 합계는 인수인 들의 인수금액들의 합계이므로 참고하세요.
- 환산 값이 같으면(예: 100,000,000,000 = 1,000억원) 오류로 처리하지 않습니다.
- 같은 의미(ex. 2027년 02월 07일, 2027년 02월 07일(3년))는 오류로 처리하지 않습니다.
- 수치적으로 동일한 값을 다른 형식으로 표현했을 경우(ex. 100,000,000,000, 1,000억원) 오류로 간주하지 않습니다.
-----
[출력 예시]:
1. 만기일: 2027년 02월 07일>>2029년 02월 07일
2. 인수금액: 삼천억원(300,000,000,000 원)>>이백만원(200,000,000 원)
-----
[검증기준]:
{validation}
"""),
                ('user', '{text_output}')
            ]
        )

        prompt = ChatPromptTemplate.from_messages(
                messages=[
                    ('system', """
- [검증기준]은 회차별, 항목별 정확한 값들을 포함하고 있습니다. 이를 기준으로 주어지는 내용의 오류를 찾아내야 합니다.
- 검증 대상 내용 중 수치적 오류나 명백히 다른 내용만을 검출해야 합니다.
- '단위: 원'과 같이 수치에 관련된 정보가 있는 경우, 실제 값과 비교하여 오류를 검출합니다.
- [검증기준]과 상이한 일자, 수치, 내용 등을 정확히 식별하고, 그 결과를 명확히 검출합니다.
- 오류가 발견되면 [출력 예시]와 같이 '순번:잘못된 내용>>기준 내용'의 형식으로 무조건 출력해야합니다.
- 숨을 고르고, 주어지는 내용을 위에서부터 순차적으로 검증해서 오류만을 출력해야합니다.
- 검증 중 잘못된 정보를 발견하지 못하면 계약이 파기될 수 있으므로 주의 깊게 검토해야 합니다.
- 검증은 주어진 내용과 [검증기준]에 제시된 정확한 값들을 비교 분석하는 데 집중해야 합니다.
- 아래 [참고사항]도 반영해서 오류를 검출해주고 오류가 있을 경우에만 그 내용을 작성하고 없을 경우는 빈칸으로 줘야합니다.
- 오기재된 내용이 있을 경우는 문맥에 맞게 수정해야 합니다.
-----
[참고사항]:
## 정상
- 의미가 동일하거나 표현 방식만 다른 경우는 오류로 간주하지 않습니다.
- 표현 형식이 다르지만, 수치나 의미가 동일하면 오류로 처리하지 않습니다.
- 인수금액들 합계가 인수금액 합계랑 같은 내용이니 오류로 처리하지 않습니다.
- 수치 환산 값이 같으면 절대 오류로 처리하지 않습니다(ex. 100,000,000,000 / 1,000억원)
- 다른 표현이지만 의미가 같을 경우 절대 오류로 처리하지 않습니다(ex. 2027년 02월 07일 / 2027년 02월 07일(3년))
## 오류
- 오기재된 내용도 검출해야 합니다.(ex. 삼천억원(200,000,000,000)) 
-----
[출력 예시]:
1. 만기일: 2027년 02월 07일>>2029년 02월 07일
2. 인수금액: 삼천억원(300,000,000,000 원)>>이백만원(200,000,000 원)
-----
[검증기준]:
{validation}
"""),
                ('user', '{text_output}')
            ]
            )

           
   
        # Define the maximum number of retries
        max_retries = 5

        # Initialize the number of attempts
        attempt = 0
        attempt = False

        while attempt == False :
            try:
                # Your code block here
                llm = ChatOpenAI(model = 'gpt-4', temperature=0.2, max_tokens=4096, stream=True )# 스트리밍 활성화
                question = llm.invoke(prompt.format(text_output=text_output,validation=validation_text)).content
                attempt = True
                break  # If successful, exit the loop
            except Exception as e:
                print(f"Error occurred: {e}")
                attempt += 1
                # model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
                llm = ChatOpenAI(model = 'gpt-4o', temperature=0.2, max_tokens=4096)
                question = llm.invoke(prompt.format(text_output=text_output,validation=validation_text)).content
                attempt = True
                time.sleep(1)  # Optional: Wait for a second before retrying

        if attempt == max_retries:
            print("Failed after maximum retries.")

        # question = llm.invoke(prompt.format(text_output=text_output,validation_text=validation_text)).content
        detected_text += question + '\n'

    return detected_text




# 스트림릿 앱 제목 설정
st.title("파일 업로드 및 데이터 처리")

# 첫 번째 파일 업로드 위젯: 검증대상 파일
uploaded_file_target = st.file_uploader("검증대상 파일을 첨부해주세요", type=["xlsx", "docx"], key="target")

# 두 번째 파일 업로드 위젯: 검증기준 파일
uploaded_file_standard = st.file_uploader("검증기준 파일을 첨부해주세요", type=["xlsx", "docx"], key="standard")

# 수행 버튼
if st.button("수행"):
    if uploaded_file_target is not None and uploaded_file_standard is not None:
        st.write("두 개의 파일이 업로드되었습니다. 데이터를 처리합니다.")

        def process_excel(file):
            try:
                # 파일을 바이너리 스트림으로 읽음
                data_df = pd.read_excel(file, header=1)
                
                # 데이터프레임 인덱스와 열 정보 출력 (디버그)
                st.write("엑셀 파일 데이터프레임 정보:")
                # st.write(data_df.info())
                
                # 필요 없는 첫 번째 행 제거 (옵션)
                data_df = data_df.reset_index(drop=True)
                
                # 수평으로 병합된 셀 값 복사
                for index, row in data_df.iterrows():
                    for col in range(1, len(data_df.columns)):
                        if pd.isna(row[col]):
                            if col > 0:  # 인덱스가 유효한지 확인
                                data_df.iloc[index, col] = data_df.iloc[index, col - 1]
                return data_df
                print(data_df)
            except Exception as e:
                st.error(f"엑셀 파일 처리 중 오류가 발생했습니다: {e}")
                return None

        def process_docx(file):
            # 바이너리 스트림으로 파일을 읽음
            with BytesIO(file.read()) as docx_file:
                result = mammoth.convert_to_html(docx_file)
            html = result.value  # The generated HTML
            messages = result.messages  # Any messages, such as warnings during conversion

            # HTML 파싱
            soup = BeautifulSoup(html, "html.parser")

            # 텍스트와 테이블을 순서대로 추출
            content = []
            text_buffer = []

            for element in soup.descendants:
                if element.name == 'table':
                    if text_buffer:
                        # 텍스트 버퍼가 비어있지 않다면, 이를 데이터프레임으로 변환 후 content에 추가
                        text_df = pd.DataFrame({'Text': ['\n'.join(text_buffer)]})
                        content.append(text_df)
                        text_buffer = []
                    
                    # 테이블 처리
                    rows = []
                    for row in element.find_all("tr"):
                        cols = [col.get_text(separator="\n") for col in row.find_all(["td", "th"])]
                        rows.append(cols)
                    table_df = pd.DataFrame(rows)
                    content.append(table_df)
                elif element.name is None and element.strip():
                    # 테이블 이외의 텍스트를 버퍼에 추가
                    text_buffer.extend(element.split("\n"))

            # 남은 텍스트 버퍼를 데이터프레임으로 변환 후 content에 추가
            if text_buffer:
                text_df = pd.DataFrame({'Text': ['\n'.join(text_buffer)]})
                content.append(text_df)

            # '회차'를 포함하는 데이터프레임을 기준으로 리스트 분할
            def split_content_by_session(content):
                split_lists = []  # 분할된 데이터프레임 리스트를 저장할 리스트
                current_list = []  # 현재 데이터 프레임을 수집할 리스트

                # 회차 문자열 패턴 정의 (공백을 포함할 수 있음)
                session_pattern = re.compile(r'회\s*차')

                for df in content:
                    # 데이터프레임 내에서 '회차' (공백 포함)가 있는지 검사
                    if df.apply(lambda x: bool(session_pattern.search(x.to_string()))).any():
                        if current_list:
                            split_lists.append(current_list)
                            current_list = []
                    current_list.append(df)

                # 마지막 리스트를 추가
                if current_list:
                    split_lists.append(current_list)

                return split_lists

            # content는 앞서 생성한 데이터프레임 리스트
            split_content = split_content_by_session(content)
            return split_content

        # 검증대상 파일 처리
        if uploaded_file_target.name.endswith('.xlsx'):
            data_df = process_excel(uploaded_file_target)
            # if data_df is not None:
            #     st.write("검증대상 엑셀 데이터:")
                # st.write(data_df)

        elif uploaded_file_target.name.endswith('.docx'):
            target_content = process_docx(uploaded_file_target)
            # if target_content is not None:
            #     st.write("검증대상 워드 파일 내용:")
                # for part in target_content:
                    # st.write(part)

        # 검증기준 파일 처리
        if uploaded_file_standard.name.endswith('.xlsx'):
            df_standard = process_excel(uploaded_file_standard)
            # if df_standard is not None:
            #     st.write("검증기준 엑셀 데이터:")
                # st.write(df_standard)
        elif uploaded_file_standard.name.endswith('.docx'):
            standard_content = process_docx(uploaded_file_standard)
            # if standard_content is not None:
            #     st.write("검증기준 워드 파일 내용:")
                # for part in standard_content:
                #     st.write(part)

        # 두 파일을 활용한 데이터 처리 예제
        if uploaded_file_target.name.endswith('.docx') and uploaded_file_standard.name.endswith('.xlsx'):
            st.write("사업신고서 검증 후 오류 출력:")
            try:
                txt_standard = dataframe_to_markdown2(df_standard)
                output = ''
                output = llm_invoke(txt_standard, target_content)
                # st.write(target_content)
                # st.write(df_standard)
                # st.write(txt_standard)
                st.write(output)
                # target_content
                # df_standard



            except KeyError:
                st.error("공통 열 이름이 일치하지 않습니다. 공통 열 이름을 확인하세요.")

        

    else:
        st.write("두 개의 파일을 모두 업로드해주세요.")
