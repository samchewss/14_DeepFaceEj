"""
Aplicación de validación de personas en tiempo real usando DeepFace
Detecta si la persona frente a la cámara está registrada en el Dataset
"""

import cv2
import os
from deepface import DeepFace
import numpy as np
from datetime import datetime
import time


class RealtimeValidator:
    def __init__(self, dataset_path="Dataset"):
        """
        Inicializa el validador en tiempo real

        Args:
            dataset_path: Ruta a la carpeta con imágenes de referencia
        """
        self.dataset_path = dataset_path
        self.db_embeddings = {}
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Configuración
        self.model_name = (
            "Facenet"  # Modelos: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace
        )
        self.distance_metric = "cosine"
        self.detector_backend = "opencv"

        # Control de tiempo para evitar demasiadas verificaciones
        self.last_verification_time = 0
        self.verification_interval = 1.0  # Segundos entre verificaciones

        # Cargar personas del dataset
        self.load_dataset()

    def load_dataset(self):
        """Carga las imágenes del dataset y extrae nombres de personas"""
        self.known_people = set()
        if not os.path.exists(self.dataset_path):
            print(f"❌ No se encontró la carpeta {self.dataset_path}")
            return

        for filename in os.listdir(self.dataset_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                # Extraer nombre de la persona del nombre del archivo
                # Asume formato: "Nombre_Apellido_img0.jpg" o "NombreApellido.jpg"
                name = filename.split("_img")[0].split(".")[0]
                name = name.replace("_", " ")
                self.known_people.add(name)

        print(f"✅ Dataset cargado: {len(self.known_people)} personas encontradas")
        print(f"   Personas: {', '.join(self.known_people)}")

    def verify_face(self, frame):
        """
        Verifica si el rostro en el frame corresponde a alguien del dataset

        Args:
            frame: Frame de video capturado

        Returns:
            tuple: (is_verified, person_name, distance)
        """
        current_time = time.time()

        # Control de frecuencia de verificación
        if current_time - self.last_verification_time < self.verification_interval:
            return None, None, None

        self.last_verification_time = current_time

        try:
            # Guardar frame temporal para DeepFace
            temp_frame_path = "temp_frame.jpg"
            cv2.imwrite(temp_frame_path, frame)

            # Buscar coincidencias en el dataset
            result = DeepFace.find(
                img_path=temp_frame_path,
                db_path=self.dataset_path,
                model_name=self.model_name,
                distance_metric=self.distance_metric,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True,
            )

            # Limpiar archivo temporal
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)

            # Procesar resultados
            if isinstance(result, list) and len(result) > 0 and not result[0].empty:
                df = result[0]
                # Obtener la mejor coincidencia
                best_match = df.iloc[0]
                identity_path = best_match["identity"]
                distance = best_match["distance"]

                # Umbral de distancia (ajustar según necesidad)
                # Para Facenet con cosine: típicamente < 0.4 es buena coincidencia
                threshold = 0.4

                if distance < threshold:
                    # Extraer nombre del archivo
                    filename = os.path.basename(identity_path)
                    person_name = filename.split("_img")[0].split(".")[0]
                    person_name = person_name.replace("_", " ")
                    return True, person_name, distance
                else:
                    return False, "Desconocido", distance
            else:
                return False, "No detectado", None

        except Exception as e:
            print(f"⚠️  Error en verificación: {str(e)}")
            return None, None, None

    def draw_results(self, frame, verified, person_name, distance):
        """
        Dibuja los resultados en el frame

        Args:
            frame: Frame de video
            verified: Si la persona fue verificada
            person_name: Nombre de la persona
            distance: Distancia de similitud
        """
        height, width = frame.shape[:2]

        # Detectar rostros para dibujar rectángulos
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        # Dibujar rectángulos en rostros detectados
        for x, y, w, h in faces:
            if verified is True:
                color = (0, 255, 0)  # Verde para verificado
                label = f"✓ {person_name}"
            elif verified is False:
                color = (0, 0, 255)  # Rojo para no verificado
                label = f"✗ {person_name}"
            else:
                color = (255, 165, 0)  # Naranja para procesando
                label = "Procesando..."

            # Rectángulo alrededor del rostro
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Fondo para el texto
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1
            )

            # Texto con el nombre
            cv2.putText(
                frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Mostrar distancia si está disponible
            if distance is not None:
                dist_text = f"Dist: {distance:.3f}"
                cv2.putText(
                    frame,
                    dist_text,
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        # Panel de información
        info_bg = np.zeros((100, width, 3), dtype=np.uint8)
        info_bg[:] = (40, 40, 40)

        # Información del sistema
        cv2.putText(
            info_bg,
            f"Modelo: {self.model_name}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            info_bg,
            f"Personas en DB: {len(self.known_people)}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            info_bg,
            f"Rostros detectados: {len(faces)}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Instrucciones
        cv2.putText(
            info_bg,
            "Presiona 'q' para salir | 's' para captura",
            (width - 350, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Combinar panel con frame
        frame_with_info = np.vstack([info_bg, frame])

        return frame_with_info

    def run(self):
        """Ejecuta la aplicación de validación en tiempo real"""
        print("\n" + "=" * 60)
        print("🎥 VALIDACIÓN DE PERSONAS EN TIEMPO REAL")
        print("=" * 60)
        print(f"Dataset: {self.dataset_path}")
        print(f"Modelo: {self.model_name}")
        print(f"Métrica: {self.distance_metric}")
        print("=" * 60 + "\n")

        # Iniciar captura de video
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: No se pudo acceder a la cámara")
            return

        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("✅ Cámara iniciada correctamente")
        print("💡 Presiona 'q' para salir")
        print("💡 Presiona 's' para guardar captura\n")

        verified = None
        person_name = None
        distance = None

        while True:
            ret, frame = cap.read()

            if not ret:
                print("❌ Error al capturar frame")
                break

            # Voltear horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)

            # Verificar rostro
            result = self.verify_face(frame)
            if result[0] is not None:
                verified, person_name, distance = result

                # Mostrar en consola
                if verified:
                    print(
                        f"✅ Persona verificada: {person_name} (distancia: {distance:.3f})"
                    )
                else:
                    print(f"❌ Persona no reconocida: {person_name}")

            # Dibujar resultados en el frame
            frame_display = self.draw_results(frame, verified, person_name, distance)

            # Mostrar frame
            cv2.imshow("Validación en Tiempo Real - DeepFace", frame_display)

            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\n👋 Cerrando aplicación...")
                break
            elif key == ord("s"):
                # Guardar captura
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captura_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Captura guardada: {filename}")

        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Aplicación cerrada correctamente")


def main():
    """Función principal"""
    # Crear validador con la ruta al dataset
    validator = RealtimeValidator(dataset_path="Dataset")

    # Ejecutar aplicación
    validator.run()


if __name__ == "__main__":
    main()
