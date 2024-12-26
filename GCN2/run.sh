# Run GCN, removed resolution requirment. The current model was trained with resolution 320x240 as input. Use other resolution may affect the actual performance. Ideally, GCNv2 should be trained/finetuned in desired resolution.
#GCN_PATH=gcn2_320x240.pt ./rgbd_gcn ../Vocabulary/GCNvoc.bin TUM3_small.yaml ~/home/franz/Datasets/rgbd_dataset_freiburg3_long_office_household ~/home/franz/Datasets/rgbd_dataset_freiburg3_long_office_household/associations.txt

# 640x480 resolution example
#FULL_RESOLUTION=1 GCN_PATH=gcn2_640x480.pt ./rgbd_gcn ../Vocabulary/GCNvoc.bin TUM3.yaml ~/Workspace/Datasets/TUM/freiburg3/rgbd_dataset_freiburg3_long_office_household ~/Workspace/Datasets/TUM/freiburg3/rgbd_dataset_freiburg3_long_office_household/associations.txt

# Reproduce results in comparison with ORB as in our paper, will enable NN_ONLY and use 320x240 resolution.

# GCNv2
# NN_ONLY=1 GCN_PATH=gcn2_320x240.pt ./rgbd_gcn ../Vocabulary/GCNvoc.bin TUM3_small.yaml ~/Workspace/Datasets/TUM/freiburg3/rgbd_dataset_freiburg3_long_office_household ~/Workspace/Datasets/TUM/freiburg3/rgbd_dataset_freiburg3_long_office_household/associations.txt

# Vanilla ORB
# NN_ONLY=1 USE_ORB=1 ./rgbd_gcn ../Vocabulary/ORBvoc.bin TUM3_small.yaml ~/Workspace/Datasets/TUM/freiburg3/rgbd_dataset_freiburg3_long_office_household ~/Workspace/Datasets/TUM/freiburg3/rgbd_dataset_freiburg3_long_office_household/associations.txt

#!/bin/bash
# Run GCN

#!/bin/bash
# Run GCN

# Caminho do modelo GCN
GCN_PATH="/home/franz/GCNv2_SLAM/GCN2/gcn2.pt"

# Caminho para o binário rgbd_gcn
RGBD_GCN="/home/franz/GCNv2_SLAM/GCN2/rgbd_gcn"

# Caminho para o vocabulário
VOCABULARY="/home/franz/GCNv2_SLAM/Vocabulary/GCNvoc.bin"

# Caminho para o arquivo de configuração YAML
CONFIG_FILE="/home/franz/GCNv2_SLAM/GCN2/TUM3_small.yaml"

# Caminho para o dataset
DATASET_PATH="/home/franz/Datasets/rgbd_dataset_freiburg3_long_office_household"

# Caminho para o arquivo de associações
ASSOCIATIONS_FILE="/home/franz/Datasets/rgbd_dataset_freiburg3_long_office_household/associations.txt"

# Verificação de arquivos e diretórios
if [ ! -f "$GCN_PATH" ]; then
    echo "Erro: Arquivo do modelo GCN não encontrado em $GCN_PATH"
    exit 1
fi

if [ ! -f "$RGBD_GCN" ]; then
    echo "Erro: Binário rgbd_gcn não encontrado em $RGBD_GCN"
    exit 1
fi

if [ ! -f "$VOCABULARY" ]; then
    echo "Erro: Arquivo de vocabulário não encontrado em $VOCABULARY"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Erro: Arquivo de configuração YAML não encontrado em $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$DATASET_PATH" ]; then
    echo "Erro: Caminho do dataset não encontrado em $DATASET_PATH"
    exit 1
fi

if [ ! -f "$ASSOCIATIONS_FILE" ]; then
    echo "Erro: Arquivo de associações não encontrado em $ASSOCIATIONS_FILE"
    exit 1
fi

# Comando para executar
$RGBD_GCN $VOCABULARY $CONFIG_FILE $DATASET_PATH $ASSOCIATIONS_FILE

