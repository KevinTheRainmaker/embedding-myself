#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def load_environment():
    """
    .env 파일에서 환경 변수를 로드합니다.
    
    1. 현재 디렉토리의 .env 파일
    2. 프로젝트 루트의 .env 파일
    3. 환경 변수가 이미 설정되어 있는 경우
    """
    # 1. 현재 디렉토리에서 .env 파일 로드 시도
    if os.path.exists('.env'):
        load_dotenv('.env')
        print("Loaded .env from current directory")
        return True
    
    # 2. 프로젝트 루트 디렉토리 찾기 시도
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    root_dir = current_dir
    
    # 상위 디렉토리로 올라가면서 .env 파일 찾기 (최대 3단계)
    for _ in range(3):
        root_dir = root_dir.parent
        env_path = root_dir / '.env'
        
        if env_path.exists():
            load_dotenv(str(env_path))
            print(f"Loaded .env from {env_path}")
            return True
    
    # 3. 필수 환경 변수 확인
    required_vars = [
        "PINECONE_API_KEY", 
        "PINECONE_ENVIRONMENT", 
        "PINECONE_INDEX", 
        "GOOGLE_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file or set environment variables manually.")
        print("See .env.example for required variables.")
        return False
    
    print("Using environment variables from system")
    return True

if __name__ == "__main__":
    # 스크립트로 직접 실행 시 환경 변수 로드 테스트
    if load_environment():
        print("Environment variables loaded successfully!")
        for key in ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "PINECONE_INDEX", "GOOGLE_API_KEY"]:
            value = os.environ.get(key, "Not set")
            # API 키는 처음 몇 글자만 표시
            if "API_KEY" in key and value != "Not set":
                masked_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
                print(f"{key}: {masked_value}")
            else:
                print(f"{key}: {value}")
    else:
        sys.exit(1) 