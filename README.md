# ECG Streamlit Dashboard 실행 방법

이 폴더는 외부 컴퓨터에서 대시보드를 실행하기 위한 공유용 폴더입니다.

## 실행 순서

1. Python 3.10 이상을 설치합니다.
2. 이 폴더에서 터미널을 엽니다.
3. 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

4. 대시보드를 실행합니다.

```bash
streamlit run app.py
```

Windows에서는 `run_dashboard.bat` 파일을 더블클릭해도 됩니다.

## 포함된 주요 파일

- `app.py`: 실행용 Streamlit 앱 파일
- `streamlit_초안9.py`: 원본 앱 파일
- `children-subject-info.csv`, `adults-subject-info.csv`: 환자 메타데이터
- `processed_segments_with_aux.csv`: annotation 보조 데이터
- `ecg_results/`: ECG 요약 및 R-peak 결과
- `leipzig-heart-center-ecg-data/`: WFDB 원본 ECG 데이터

## 선택 설정

Gemini AI 답변 기능을 쓰려면 `.streamlit/secrets.toml`을 참고해 `.streamlit/secrets.toml` 파일을 만들고 `GEMINI_API_KEY`를 직접 입력해야 합니다.
streamlit은 1.53.0을 기반으로 제작되었습니다.
DB 연결 파일과 API 키는 보안 때문에 이 공유 폴더에 포함하지 않았습니다. DB나 Gemini 설정이 없어도 대시보드는 로컬 데이터 기반으로 실행됩니다.
