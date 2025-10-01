import os
import numpy as np
from glob import glob
import shutil

def convert_polygon_to_bbox(polygon_coords):
    """
    Converte uma lista de coordenadas de polígono [x1, y1, x2, y2, ...]
    para o formato de bounding box YOLO [x_center, y_center, width, height].
    """
    # Separa as coordenadas x e y
    xs = np.array(polygon_coords[0::2])
    ys = np.array(polygon_coords[1::2])

    # Encontra o min e max para criar a bounding box
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)

    # Converte de (x_min, y_min, x_max, y_max) para YOLO (x_center, y_center, w, h)
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + (width / 2)
    y_center = y_min + (height / 2)

    return [x_center, y_center, width, height]

def process_label_files(input_dir, output_dir):
    """
    Processa todos os arquivos .txt em um diretório, convertendo polígonos
    para bounding boxes e salvando no diretório de saída.
    """
    # Cria o diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Encontra todos os arquivos de label no diretório de entrada
    label_files = glob(os.path.join(input_dir, '*.txt'))
    
    if not label_files:
        print(f"Nenhum arquivo .txt encontrado em: {input_dir}")
        return

    print(f"Encontrados {len(label_files)} arquivos em {input_dir}. Convertendo...")

    for file_path in label_files:
        new_lines = []
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) > 1:
                class_id = parts[0]
                polygon_coords = [float(p) for p in parts[1:]]
                
                # Garante que temos um número par de coordenadas
                if len(polygon_coords) % 2 != 0:
                    print(f"AVISO: Número ímpar de coordenadas no arquivo {os.path.basename(file_path)}. Pulando linha.")
                    continue

                # Converte para bounding box
                bbox_coords = convert_polygon_to_bbox(polygon_coords)

                # Formata a nova linha
                new_line = f"{class_id} {bbox_coords[0]} {bbox_coords[1]} {bbox_coords[2]} {bbox_coords[3]}"
                new_lines.append(new_line)

        # Escreve o novo arquivo de label no diretório de saída
        output_file_path = os.path.join(output_dir, os.path.basename(file_path))
        with open(output_file_path, 'w') as f:
            f.write('\n'.join(new_lines))

    print(f"Conversão concluída! Arquivos salvos em: {output_dir}")


# /src/scripts/convert_polygons.py

if __name__ == '__main__':
    # --- CAMINHOS ADAPTADOS PARA A ESTRUTURA images/train, labels/train ---
    
    base_dir = os.path.join('datasets', 'my_dataset')
    
    # Diretórios de entrada (onde estão os polígonos)
    train_labels_in = os.path.join(base_dir, 'labels', 'train')
    val_labels_in = os.path.join(base_dir, 'labels', 'val')
    
    # Diretórios de saída (onde as novas bounding boxes serão salvas)
    train_labels_out = os.path.join(base_dir, 'labels', 'train_bbox')
    val_labels_out = os.path.join(base_dir, 'labels', 'val_bbox')
    
    # --- EXECUÇÃO DA CONVERSÃO ---
    
    # Processa os arquivos de treino
    process_label_files(train_labels_in, train_labels_out)
    
    # Processa os arquivos de validação
    process_label_files(val_labels_in, val_labels_out)
    
    print("\nLembrete: O próximo passo é renomear as pastas 'train' e 'val' originais")
    print("e depois renomear as novas pastas 'train_bbox' e 'val_bbox'.")
    print("\nExemplo para a pasta de treino:")
    print(f"1. Renomeie '{train_labels_in}' para '{train_labels_in}_polygon_backup'")
    print(f"2. Renomeie '{train_labels_out}' para '{train_labels_in}'")