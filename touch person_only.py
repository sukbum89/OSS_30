import cv2
import numpy as np

def person_only(input_path, output_path, format='mp4'):
    # 사람 인식용 Haar Cascade 로드
    person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
    
    if person_cascade.empty():
        raise ValueError("Cascade Classifier가 제대로 로드되지 않았습니다.")
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError("비디오 파일을 열 수 없습니다.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 형식에 따른 코덱 선택
    if format == 'mp4':
        if not output_path.endswith('.mp4'):
            output_path += '.mp4'
        codec = cv2.VideoWriter_fourcc(*'mp4v')
    elif format == 'avi':
        if not output_path.endswith('.avi'):
            output_path += '.avi'
        codec = cv2.VideoWriter_fourcc(*'XVID')
    else:
        raise ValueError("지원되지 않는 형식입니다. 'mp4' 또는 'avi'를 사용하세요.")

    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        persons = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # 검정 배경 생성
        mask = np.zeros_like(frame)  # frame과 같은 크기의 검정 배경 생성

        for (x, y, w, h) in persons:
            mask[y:y + h, x:x + w] = frame[y:y + h, x:x + w]

        out.write(mask)

    cap.release()
    out.release()
    print(f"처리 완료된 비디오가 '{output_path}'에 저장되었습니다.")
