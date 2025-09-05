# FumoBAT - Small Object YOLO Training

## Setup no Servidor

### 1. Configuração do Ambiente Python
```bash
# Criar ambiente virtual
python -m venv fumobat
source fumobat/bin/activate  # Linux/Mac
# ou
fumobat\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 2. Preparar Dataset
```bash
# Criar diretório para dataset
mkdir -p datasets/my_dataset

# Upload do dataset para datasets/my_dataset/
# Estrutura esperada:
# datasets/my_dataset/
#   ├── images/
#   │   ├── train/
#   │   └── val/
#   └── labels/
#       ├── train/
#       └── val/
```

### 3. Verificar Configurações
```bash
# Verificar configs/data.yaml
# Verificar configs/model.yaml
# Verificar se weights_small_object_yolo.pt está presente
```

### 4. Executar Treinamento
```bash
# Ativar ambiente
source fumobat/bin/activate

# Executar treinamento
python src/train/train.py
```

### 5. Monitorar Treinamento
- Logs serão salvos em `runs/improved_train/`
- Métricas de avaliação em `evaluation_results/`
- Melhor modelo salvo como `improved_best_model.pt`

## Estrutura de Arquivos Essenciais

```
smallObjectYolo/
├── src/
│   ├── data/
│   ├── models/
│   ├── train/
│   └── eval/
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── enhanced_model.yaml
├── weights_small_object_yolo.pt
├── requirements.txt
└── README_SERVER.md
```

## Comandos Úteis

```bash
# Verificar GPU
nvidia-smi

# Monitorar recursos
htop

# Ver logs em tempo real
tail -f runs/improved_train/train.log

# Verificar espaço em disco
df -h
```
