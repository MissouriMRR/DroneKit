import numpy as np
import cv2

pencilCascade = cv2.CascadeClassifier( r'C:\Users\Christopher\Downloads\data\cascade.xml' )
cap = cv2.VideoCapture( 0 )

while(True):
    ret, frame = cap.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pencils = pencilCascade.detectMultiScale( gray, 1.1, 8 )

        for ( x, y, w, h ) in pencils:
            cv2.rectangle( frame, ( x, y ), ( x + w, y + h ), ( 0, 255, 0 ), 3 )

        cv2.imshow( 'Pencil Detection Test', frame )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

