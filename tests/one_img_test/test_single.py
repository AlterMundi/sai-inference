#!/usr/bin/env python3
"""
Test de verificación de configuración en producción SAI-Inference

Este script verifica que los cambios en /etc/sai-inference/production.env
tienen efecto real en el servicio después de reiniciarlo.

Uso:
    python tests/one_img_test/test_single.py

Flujo de verificación:
    1. Cambiar configuración en /etc/sai-inference/production.env
    2. sudo systemctl restart sai-inference
    3. python tests/one_img_test/test_single.py
    4. Ver imagen en tests/one_img_test/result/
"""
import requests
import base64
import sys
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURACIÓN DEL TEST
# =============================================================================
API_URL = "http://localhost:8888"
# =============================================================================


def get_service_config():
    """Obtiene la configuración actual del servicio desde /health"""
    try:
        response = requests.get(f"{API_URL}/api/v1/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print("ERROR: No se puede conectar al servicio SAI-Inference")
        print(f"       Verifica que esté corriendo en {API_URL}")
        print("       sudo systemctl status sai-inference")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def get_loaded_model():
    """Obtiene el nombre real del modelo cargado desde /models"""
    try:
        response = requests.get(f"{API_URL}/api/v1/models", timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("current_model", "unknown")
    except Exception:
        return "unknown"


def find_test_image(img_dir: Path):
    """Busca la primera imagen en el directorio img/"""
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    for ext in extensions:
        images = list(img_dir.glob(f"*{ext}")) + list(img_dir.glob(f"*{ext.upper()}"))
        if images:
            return images[0]
    return None


def run_inference(image_path: Path):
    """Ejecuta inferencia SIN enviar hiperparámetros (usa defaults del servicio)"""
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        # Solo enviamos return_image=true, el resto usa los defaults del servicio
        data = {"return_image": "true"}

        response = requests.post(
            f"{API_URL}/api/v1/infer",
            files=files,
            data=data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()


def save_annotated_image(base64_image: str, result_dir: Path, config: dict, detections: int, model_name: str):
    """Guarda la imagen anotada con nombre descriptivo"""
    # Extraer configuración para el nombre
    conf = config.get("confidence_threshold", "unk")
    iou = config.get("iou_threshold", "unk")
    model = model_name.replace(".pt", "")

    # Crear nombre descriptivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model}_conf_{conf}_iou_{iou}_det_{detections}_{timestamp}.jpg"

    # Decodificar y guardar
    image_data = base64.b64decode(base64_image)
    output_path = result_dir / filename

    with open(output_path, "wb") as f:
        f.write(image_data)

    return output_path


def main():
    # Rutas
    script_dir = Path(__file__).parent
    img_dir = script_dir / "img"
    result_dir = script_dir / "result"

    print("=" * 60)
    print("SAI-INFERENCE - Test de Verificación de Configuración")
    print("=" * 60)

    # 1. Obtener configuración del servicio
    print("\n[1] Consultando configuración del servicio...")
    health = get_service_config()
    runtime_params = health.get("runtime_parameters", {})
    model_info = health.get("loaded_model_info", {})

    # Obtener el nombre real del modelo cargado (no "auto")
    current_model = get_loaded_model()

    print("\n    CONFIGURACIÓN ACTUAL DEL SERVICIO:")
    print("    " + "-" * 40)
    print(f"    Modelo cargado:      {current_model}")
    print(f"    Confidence:          {runtime_params.get('confidence_threshold', 'N/A')}")
    print(f"    IoU:                 {runtime_params.get('iou_threshold', 'N/A')}")
    print(f"    Input size:          {runtime_params.get('input_size', 'N/A')}")
    print(f"    Device:              {runtime_params.get('device', 'N/A')}")
    print(f"    Model dir:           {runtime_params.get('model_dir', 'N/A')}")
    print(f"    Versión servicio:    {health.get('version', 'N/A')}")

    # 2. Buscar imagen de prueba
    print("\n[2] Buscando imagen de prueba...")
    test_image = find_test_image(img_dir)

    if not test_image:
        print(f"    ERROR: No se encontró ninguna imagen en {img_dir}")
        print("    Coloca una imagen (.jpg, .png, etc.) en esa carpeta")
        sys.exit(1)

    print(f"    Imagen encontrada: {test_image.name}")

    # 3. Ejecutar inferencia
    print("\n[3] Ejecutando inferencia (usando configuración del servicio)...")
    result = run_inference(test_image)

    detections = result.get("detections", [])
    detection_count = len(detections)

    print(f"\n    RESULTADO DE INFERENCIA:")
    print("    " + "-" * 40)
    print(f"    Detecciones:         {detection_count}")
    print(f"    Has smoke:           {result.get('has_smoke', False)}")
    print(f"    Has fire:            {result.get('has_fire', False)}")
    print(f"    Alert level:         {result.get('alert_level', 'none')}")
    print(f"    Processing time:     {result.get('processing_time_ms', 0):.1f} ms")

    if detections:
        print("\n    Detecciones individuales:")
        for i, det in enumerate(detections, 1):
            print(f"      {i}. {det.get('class_name', 'unknown')} - conf: {det.get('confidence', 0):.3f}")

    # 4. Guardar imagen anotada
    print("\n[4] Guardando imagen anotada...")

    annotated_image = result.get("annotated_image")
    if not annotated_image:
        print("    ERROR: El servicio no devolvió imagen anotada")
        print("    Verifica que return_image=true esté funcionando")
        sys.exit(1)

    output_path = save_annotated_image(
        annotated_image,
        result_dir,
        runtime_params,
        detection_count,
        current_model
    )

    print(f"    Guardado en: {output_path}")

    # 5. Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Configuración verificada:")
    print(f"  - Modelo:               {current_model}")
    print(f"  - Confidence threshold: {runtime_params.get('confidence_threshold')}")
    print(f"  - IoU threshold:        {runtime_params.get('iou_threshold')}")
    print(f"  - Detecciones:          {detection_count}")
    print(f"\nImagen guardada en:")
    print(f"  {output_path}")
    print("\nPara verificar cambios:")
    print("  1. Editar /etc/sai-inference/production.env")
    print("  2. sudo systemctl restart sai-inference")
    print("  3. Ejecutar este test de nuevo")
    print("  4. Comparar imágenes en result/")
    print("=" * 60)


if __name__ == "__main__":
    main()
