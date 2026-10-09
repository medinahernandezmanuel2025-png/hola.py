import cv2
import numpy as np
import mediapipe as mp

# 1. Inicializar MediaPipe para detección de manos
mp_manos = mp.solutions.hands
manos = mp_manos.Hands(max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.8)
mp_dibujo = mp.solutions.drawing_utils

# 2. Inicializar la cámara web
cap = cv2.VideoCapture(0)

# 3. Crear un lienzo en blanco (negro) para acumular los trazos del dibujo
lienzo = None
x_anterior, y_anterior = 0, 0

print("Programa iniciado. Levanta el dedo ÍNDICE para pintar. Levanta ÍNDICE + MEDIO para pausar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Espejar la imagen para que el movimiento sea natural (tipo espejo)
    frame = cv2.flip(frame, 1)
    alto, ancho, _ = frame.shape
    
    # Inicializar el lienzo negro si aún no se ha creado
    if lienzo is None:
        lienzo = np.zeros((alto, ancho, 3), dtype=np.uint8)
        
    # Convertir la imagen a RGB para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = manos.process(rgb_frame)
    
    # Comprobar si se detectó alguna mano
    if resultado.multi_hand_landmarks:
        for landmarks in resultado.multi_hand_landmarks:
            # Dibujar el esqueleto de la mano en el frame de video (opcional)
            mp_dibujo.draw_landmarks(frame, landmarks, mp_manos.HAND_CONNECTIONS)
            
            # Obtener las coordenadas de las puntas de los dedos clave
            # ID 8 = Punta del Índice, ID 12 = Punta del Medio
            punta_indice = landmarks.landmark[8]
            punta_medio = landmarks.landmark[12]
            
            # Convertir coordenadas normalizadas (0.0 a 1.0) a píxeles de la pantalla
            cx, cy = int(punta_indice.x * ancho), int(punta_indice.y * alto)
            mx, my = int(punta_medio.x * ancho), int(punta_medio.y * alto)
            
            # Detectar qué dedos están arriba basándose en su altura (coordenada Y)
            # En OpenCV, menor Y significa que está más arriba en la pantalla
            indice_arriba = punta_indice.y < landmarks.landmark[6].y
            medio_arriba = punta_medio.y < landmarks.landmark[10].y
            
            # MODO 1: Solo el índice está arriba -> MODO DIBUJAR
            if indice_arriba and not medio_arriba:
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED) # Punto guía verde
                
                if x_anterior == 0 and y_anterior == 0:
                    x_anterior, y_anterior = cx, cy
                
                # Dibujar una línea continua de color azul en el lienzo
                cv2.line(lienzo, (x_anterior, y_anterior), (cx, cy), (255, 0, 0), 5)
                x_anterior, y_anterior = cx, cy
                
            # MODO 2: Ambos dedos arriba -> MODO PAUSA / SELECCIÓN
            elif indice_arriba and medio_arriba:
                # Reiniciar los puntos anteriores para no conectar líneas al volver a pintar
                x_anterior, y_anterior = 0, 0
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), cv2.FILLED) # Punto guía rojo
            else:
                x_anterior, y_anterior = 0, 0

    # 4. Fusionar el lienzo de dibujo con el video de la cámara
    # Convertir el lienzo a escala de grises para crear una máscara binaria
    lienzo_gris = cv2.cvtColor(lienzo, cv2.COLOR_BGR2GRAY)
    _, mascara_inv = cv2.threshold(lienzo_gris, 5, 255, cv2.THRESH_BINARY_INV)
    mascara_inv = cv2.cvtColor(mascara_inv, cv2.COLOR_GRAY2BGR)
    
    # Recortar el área del video donde NO hay dibujo y sumarle el lienzo de color
    frame_combinado = cv2.bitwise_and(frame, mascara_inv)
    frame_final = cv2.add(frame_combinado, lienzo)

    # 5. Mostrar las ventanas
    cv2.imshow('Camara - Air Canvas', frame_final)
    
    # Controles por teclado:
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('c'): # Presiona 'c' para limpiar la pantalla por completo
        lienzo = np.zeros((alto, ancho, 3), dtype=np.uint8)
    elif tecla == ord('q'): # Presiona 'q' para salir del programa
        break

cap.release()
cv2.destroyAllWindows()
