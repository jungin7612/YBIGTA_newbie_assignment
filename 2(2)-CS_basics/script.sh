#!/bin/bash
set -e  # 스크립트 실행 중 오류 발생 시 바로 종료

MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_PATH="$HOME/miniconda"
MINICONDA_BIN="$MINICONDA_PATH/bin"
CONDA_EXE="$MINICONDA_BIN/conda"
ENV_NAME="myenv"

# 1) Miniconda 설치 여부 확인
if [ ! -f "$CONDA_EXE" ]; then
    echo "[INFO] Miniconda가 설치되어 있지 않습니다. 설치를 시작합니다."
    wget "https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER" -O "/tmp/$MINICONDA_INSTALLER"
    bash "/tmp/$MINICONDA_INSTALLER" -b -p "$MINICONDA_PATH"
    rm "/tmp/$MINICONDA_INSTALLER"
    echo "[INFO] Miniconda 설치가 완료되었습니다."
fi

# Conda 환경 생성 및 활성화
if [ -f "$CONDA_EXE" ]; then
    echo "[INFO] 현재 쉘에 conda 환경 설정 로드..."
    eval "$("$CONDA_EXE" shell.bash hook)"
else
    echo "[ERROR] conda 실행 파일을 찾을 수 없습니다."
    exit 1
fi

if ! conda env list | grep -q "^$ENV_NAME\s"; then
    echo "[INFO] '$ENV_NAME' 환경이 존재하지 않습니다. 생성을 시작합니다."
    conda create -n "$ENV_NAME" python=3.9 -y
fi

echo "[INFO] '$ENV_NAME' 환경 활성화..."
conda activate "$ENV_NAME"

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "가상환경 활성화: 성공"
else
    echo "가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
REQUIRED_PACKAGES=(mypy)
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! pip show "$package" &>/dev/null; then
        echo "[INFO] $package 패키지를 설치합니다..."
        pip install "$package"
    fi
done

# Submission 폴더 파일 실행
if [ -d "submission" ]; then
    cd "submission"
    for file in *.py; do
        # 파일 이름과 대응하는 input/output 경로
        base_name=$(basename "$file" .py)
        input_file="../input/${base_name}_input"
        output_file="../output/${base_name}_output"

        if [ ! -f "$input_file" ]; then
            echo "[WARN] $input_file 파일이 없어 $file을(를) 건너뜁니다."
            continue
        fi

        echo "[INFO] $file 실행 중..."
        python "$file" < "$input_file" > "$output_file"
        if [ $? -ne 0 ]; then
            echo "[ERROR] $file 실행 중 오류 발생."
            continue
        fi

        # mypy 체크
        echo "[INFO] $file 에 대한 mypy 테스트 실행 중..."
        mypy "$file"
        if [ $? -ne 0 ]; then
            echo "[ERROR] $file mypy 검사 실패."
        fi
    done
else
    echo "[WARN] submission 디렉토리가 존재하지 않아 실행 파일을 찾을 수 없습니다."
fi

# 8) 가상환경 비활성화
echo "[INFO] 가상환경 비활성화..."
conda deactivate

echo "[INFO] 모든 작업이 완료되었습니다."
