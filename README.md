# Google Drive 문서 임베딩 시스템

구글 드라이브의 파일 변경을 감지하고, 신규/변경된 파일을 처리하여 Google Gemini 1.5 Pro로 임베딩을 생성한 뒤 Pinecone 벡터 DB에 저장하는 시스템입니다.

## 기능

- 구글 드라이브의 파일 변경 감지
- 특정 폴더만 모니터링 가능
- 다양한 파일 형식 지원 (PDF, Word, Excel, PowerPoint, 텍스트 파일 등)
- Google Gemini 1.5 Pro를 사용한 텍스트 임베딩 생성
- 문서 요약 및 키워드 추출
- Pinecone 벡터 DB에 메타데이터와 함께 저장
- Pinecone 인덱스 자동 생성
- GitHub Actions를 통한 자동 실행

## 설치 및 설정

### 1. 필수 조건

- Python 3.8 이상
- Pinecone 계정
- Google Cloud 계정 및 API 키
- Google Drive API 사용 설정 및 OAuth 인증 정보

### 2. 설치 방법

```bash
# 저장소 클론
git clone https://github.com/your-username/embedding-myself.git
cd embedding-myself

# 의존성 설치
pip install -r requirements.txt

# 데이터 디렉토리 생성
mkdir -p data
```

### 3. 인증 설정

#### Google Drive OAuth 인증 설정

1. [Google Cloud Console](https://console.cloud.google.com/)에서 새 프로젝트 생성
2. Google Drive API 활성화
3. OAuth 동의 화면 구성
4. OAuth 클라이언트 ID 생성 (데스크톱 애플리케이션)
5. 인증 정보(credentials.json) 다운로드
6. `data/credentials.json` 위치에 저장

#### 환경 변수 설정

`.env` 파일을 생성하고 다음 값을 설정:

```
# Pinecone 설정
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX=kb-profile-data  # 선택사항: 설정하지 않으면 "kb-profile-data"로 자동 생성됨

# Google API 설정
GOOGLE_API_KEY=your_google_api_key

# 구글 드라이브 설정
DRIVE_FOLDER_ID=your_drive_folder_id  # 특정 폴더만 모니터링하려면 설정

# 모니터링 설정
CHECK_INTERVAL=300  # 구글 드라이브 확인 간격(초)
```

## 실행 방법

### 로컬에서 실행

기본 실행 (전체 드라이브 모니터링):
```bash
python scripts/create_vectors.py
```

특정 폴더만 모니터링:
```bash
# 환경 변수로 설정된 폴더 사용
python scripts/create_vectors.py

# 또는 명령줄에서 폴더 ID 지정
python scripts/create_vectors.py --folder YOUR_FOLDER_ID
```

> 💡 **폴더 ID 찾는 방법**: 구글 드라이브에서 폴더를 열고 URL을 확인하세요. 주소가 `https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i` 형식이라면, `1a2b3c4d5e6f7g8h9i` 부분이 폴더 ID입니다.

초기 실행 시 Google 인증 절차를 완료해야 합니다. 브라우저가 열리고 구글 계정으로 로그인하여 권한을 부여합니다.

### GitHub Actions를 사용한 자동 실행

GitHub Actions를 사용하여 정기적으로 드라이브 모니터링을 실행할 수 있습니다.

설정 방법:

1. GitHub 저장소에 다음 시크릿 설정:
   - `PINECONE_API_KEY`
   - `PINECONE_ENVIRONMENT`
   - `PINECONE_INDEX`
   - `GOOGLE_API_KEY`
   - `GOOGLE_CREDENTIALS` (credentials.json 파일의 내용을 통째로 복사)

2. GitHub Actions 워크플로우는 이미 `.github/workflows/drive_monitor.yml`에 설정되어 있으며 다음과 같이 실행됩니다:
   - 매일 자동 실행 (UTC 00:00)
   - 수동으로 워크플로우 실행 가능
   - 체크포인트 파일은 GitHub Actions 아티팩트로 저장됨

## 벡터 데이터 구조

Pinecone에 저장되는 벡터 메타데이터 구조:

```json
{
  "text": "문서 내용 (chunk)",
  "doc_type": "resume | publication | profile | ...",
  "summary": "LLM이 생성한 요약",
  "keywords": ["키워드1", "키워드2", "..."],
  "source": "원본 문서 이름 (예: resume.pdf)"
}
```

## 확장하기

파일 형식 지원을 확장하려면 `DriveMonitor` 클래스의 `extract_text` 메서드를 수정하여 추가 파일 형식에 대한 핸들러를 추가하세요.

## 라이선스

MIT License
