# /src/scripts/visualize_annotations.py
import cv2
import os
import random

def draw_yolo_bboxes(image_path, label_path):
    """
    Desenha as bounding boxes de um arquivo de label YOLO em uma imagem.
    """
    # Lê a imagem
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao ler a imagem: {image_path}")
        return

    h, w, _ = image.shape

    # Verifica se o arquivo de label existe
    if not os.path.exists(label_path):
        print(f"Arquivo de label não encontrado para {os.path.basename(image_path)}")
        cv2.imshow("Annotation Check", image)
        cv2.waitKey(0)
        return

    # Lê o arquivo de label
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)

                # Converte as coordenadas YOLO (normalizadas) para coordenadas de pixel
                x_center_px = x_center * w
                y_center_px = y_center * h
                width_px = width * w
                height_px = height * h

                # Calcula os cantos da caixa (x_min, y_min)
                x1 = int(x_center_px - (width_px / 2))
                y1 = int(y_center_px - (height_px / 2))
                x2 = int(x_center_px + (width_px / 2))
                y2 = int(y_center_px + (height_px / 2))

                # Desenha o retângulo na imagem
                # Cor (B, G, R) -> Verde. Espessura = 2 pixels.
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Opcional: Desenha o ID da classe
                label_text = f"Class: {int(class_id)}"
                cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Exibe a imagem
    # Redimensiona a imagem se for muito grande para a tela
    max_display_size = 800
    if h > max_display_size or w > max_display_size:
        scale = max_display_size / max(h, w)
        image_resized = cv2.resize(image, (int(w*scale), int(h*scale)))
        cv2.imshow(f"Annotation Check: {os.path.basename(image_path)}", image_resized)
    else:
        cv2.imshow(f"Annotation Check: {os.path.basename(image_path)}", image)

    print(f"Mostrando anotações para: {os.path.basename(image_path)}. Pressione qualquer tecla para continuar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # --- CONFIGURE OS CAMINHOS AQUI ---
    base_dir = os.path.join('datasets', 'my_dataset')
    
    # Vamos verificar o conjunto de treino
    image_dir = os.path.join(base_dir, 'images', 'train')
    label_dir = os.path.join(base_dir, 'labels', 'train') # Use a pasta com os labels convertidos!

    # Pega a lista de todas as imagens
    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not all_images:
        print(f"Nenhuma imagem encontrada em {image_dir}")
    else:
        # Escolhe 5 imagens aleatórias para verificar
        num_images_to_check = 5
        selected_images = random.sample(all_images, min(num_images_to_check, len(all_images)))

        for image_name in selected_images:
            image_path = os.path.join(image_dir, image_name)
            
            # Constrói o caminho do arquivo de label correspondente
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(label_dir, label_name)
            
            draw_yolo_bboxes(image_path, label_path)