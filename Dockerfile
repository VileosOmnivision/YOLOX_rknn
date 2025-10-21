# Usa la misma imagen base que OVH
FROM ovhcom/ai-training-pytorch:2.4.0

# Instala dependencias del sistema (si faltan)
RUN apt-get update && apt-get install -y git wget unzip && rm -rf /var/lib/apt/lists/*

# Establece directorio de trabajo
WORKDIR /workspace

# Clona tu fork de YOLOX
RUN git clone https://github.com/<tu_usuario>/YOLOX.git && \
    cd YOLOX && \
    pip install -U pip && \
    pip install -v -e . 

# Define variables de entorno
ENV PYTHONUNBUFFERED=1 \
    FORCE_CUDA="1"

# Define comando por defecto (bash o python)
WORKDIR /workspace/YOLOX
CMD ["bash"]
