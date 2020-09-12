import numpy as np
import cv2


# Зчитування відео з вебки та його запис
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:

        # покадровий запис та відображення
        out.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()


# Зчитування відео та його обробка
cap2 = cv2.VideoCapture('output.avi')
out2 = cv2.VideoWriter('output2.avi', fourcc, 20.0, (640, 480))

while cap2.isOpened():
    ret, frame = cap2.read()

    if ret:

        # Конвертування у відтінки сірого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Повернення до простору RGB
        cimage = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Прямокутник
        cv2.rectangle(cimage, (100, 100), (540, 380), (206, 113, 255), 20)
        # Лінія
        cv2.line(cimage, (10, 350), (630, 350), (161, 255, 5), 5, 8, 0)

        # Запис обробленого кадру
        out2.write(cimage)
        cv2.imshow('frame', cimage)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap2.release()
out2.release()
cv2.destroyAllWindows()
