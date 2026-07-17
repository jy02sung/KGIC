import os

# 파일 경로
file_path = './dataset/train/_annotations.txt'

# 본인의 데이터셋 경로가 맞는지 검증 (끝에 /가 있으므로 안심하고 결합 가능)
new_base_path = '/workspace/vitis-ai-yolov3-tiny/data/dataset/train/'

# 파일 읽기
with open(file_path, 'r') as file:
    lines = file.readlines()

# 경로 추가
updated_lines = []
for line in lines:
    if line.strip():  # 빈 줄이 아닌 경우에만 처리
        parts = line.split()  # 어노테이션 파일의 형식을 공백 기준으로 분리
        
        # os.path.basename을 사용해 앞의 상대 경로('../.../')를 모두 날리고 파일명만 추출
        image_name = os.path.basename(parts[0])  
        
        # 깔끔해진 파일명과 새 베이스 경로 결합
        new_line = f"{new_base_path}{image_name} {' '.join(parts[1:])}\n"
        updated_lines.append(new_line)
    else:
        updated_lines.append(line)

# 변경된 내용을 파일에 다시 쓰기
with open(file_path, 'w') as file:
    file.writelines(updated_lines)