import os
from PIL import Image
import matplotlib.pyplot as plt
import math

def show_all_persons_images(dataset_path="dataset", persons_per_row=4):
    """
    Показує по 10 фото кожної людини в одному вікні
    
    Args:
        dataset_path: шлях до папки з датасетом
        persons_per_row: кількість людей в одному рядку (за замовчуванням 4)
    """
    
    # Отримуємо всі папки людей
    folders = sorted([f for f in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, f))])
    
    num_persons = len(folders)
    print(f"Знайдено {num_persons} осіб")
    
    # Розраховуємо розміри сітки
    rows = math.ceil(num_persons / persons_per_row)
    
    # Створюємо велику фігуру
    fig, axes = plt.subplots(rows, persons_per_row, figsize=(persons_per_row * 3, rows * 3))
    
    # Якщо axes одновимірний, робимо його двовимірним
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Для кожної людини
    for idx, folder in enumerate(folders):
        row = idx // persons_per_row
        col = idx % persons_per_row
        
        folder_path = os.path.join(dataset_path, folder)
        
        # Знаходимо всі зображення в папці
        images_files = []
        for ext in ['.pgm', '.png', '.jpg', '.jpeg', '.bmp']:
            images_files.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext)])
        
        images_files.sort()  # сортуємо за назвою
        
        # Беремо перші 10 зображень (або скільки є)
        num_images = min(10, len(images_files))
        
        # Показуємо на зменшеному зображенні
        ax = axes[row, col]
        
        # Створюємо колаж з 10 фото
        if num_images > 0:
            # Беремо перше зображення для розміру
            first_img = Image.open(os.path.join(folder_path, images_files[0]))
            img_width, img_height = first_img.size
            
            # Створюємо колаж 2x5 або 1x10 (залежно від кількості)
            if num_images <= 5:
                collage = Image.new('L', (img_width * num_images, img_height))
                for i, img_file in enumerate(images_files[:num_images]):
                    img = Image.open(os.path.join(folder_path, img_file)).convert('L')
                    collage.paste(img, (i * img_width, 0))
            else:
                # 2 рядки по 5 зображень
                cols = 5
                rows_img = math.ceil(num_images / cols)
                collage = Image.new('L', (img_width * cols, img_height * rows_img))
                for i, img_file in enumerate(images_files[:num_images]):
                    row_img = i // cols
                    col_img = i % cols
                    img = Image.open(os.path.join(folder_path, img_file)).convert('L')
                    collage.paste(img, (col_img * img_width, row_img * img_height))
            
            ax.imshow(collage, cmap='gray')
            ax.set_title(f"{folder}\n({num_images} фото)", fontsize=8)
        else:
            ax.text(0.5, 0.5, f"{folder}\n(немає фото)", 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.axis('off')
    
    # Видаляємо зайві підграфіки
    for idx in range(num_persons, rows * persons_per_row):
        row = idx // persons_per_row
        col = idx % persons_per_row
        axes[row, col].axis('off')
    
    plt.suptitle(f"Всі особи в датасеті (по 10 фото кожна) - всього {num_persons} осіб", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Альтернативний варіант: показати по одній людині у великому форматі
# --------------------------------------------------
def show_person_images(person_folder="s1", dataset_path="dataset"):
    """
    Показує всі 10 фото однієї людини у великому форматі
    """
    person_path = os.path.join(dataset_path, person_folder)
    
    # Знаходимо всі зображення
    images_files = []
    for ext in ['.pgm', '.png', '.jpg', '.jpeg', '.bmp']:
        images_files.extend([f for f in os.listdir(person_path) if f.lower().endswith(ext)])
    
    images_files.sort()
    num_images = len(images_files)
    
    # Визначаємо сітку для показу
    cols = 5
    rows = math.ceil(num_images / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    
    # Якщо axes одновимірний
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_file in enumerate(images_files):
        row = idx // cols
        col = idx % cols
        
        img_path = os.path.join(person_path, img_file)
        img = Image.open(img_path).convert('L')
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f"{img_file}", fontsize=8)
        axes[row, col].axis('off')
    
    # Видаляємо зайві підграфіки
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f"Фото особи {person_folder} (всього {num_images} знімків)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Показати тільки перші 5 фото кожної людини (швидкий огляд)
# --------------------------------------------------
def show_first_5_images_all_persons(dataset_path="dataset", persons_per_row=8):
    """
    Показує перші 5 фото кожної людини для швидкого огляду
    """
    folders = sorted([f for f in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, f))])
    
    num_persons = len(folders)
    rows = math.ceil(num_persons / persons_per_row)
    
    fig, axes = plt.subplots(rows, persons_per_row, figsize=(persons_per_row * 2.5, rows * 2.5))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, folder in enumerate(folders):
        row = idx // persons_per_row
        col = idx % persons_per_row
        
        folder_path = os.path.join(dataset_path, folder)
        
        # Знаходимо перші 5 зображень
        images_files = []
        for ext in ['.pgm', '.png', '.jpg', '.jpeg', '.bmp']:
            images_files.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext)])
        
        images_files.sort()
        first_5 = images_files[:5]
        
        if first_5:
            # Беремо перше зображення для розміру
            first_img = Image.open(os.path.join(folder_path, first_5[0]))
            img_width, img_height = first_img.size
            
            # Створюємо горизонтальний колаж з 5 фото
            collage = Image.new('L', (img_width * len(first_5), img_height))
            for i, img_file in enumerate(first_5):
                img = Image.open(os.path.join(folder_path, img_file)).convert('L')
                collage.paste(img, (i * img_width, 0))
            
            axes[row, col].imshow(collage, cmap='gray')
            axes[row, col].set_title(folder, fontsize=8)
        else:
            axes[row, col].text(0.5, 0.5, f"{folder}\n(немає фото)", 
                               ha='center', va='center', transform=axes[row, col].transAxes)
        
        axes[row, col].axis('off')
    
    # Видаляємо зайві
    for idx in range(num_persons, rows * persons_per_row):
        row = idx // persons_per_row
        col = idx % persons_per_row
        axes[row, col].axis('off')
    
    plt.suptitle(f"Перші 5 фото кожної особи (всього {num_persons} осіб)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Зберегти колаж у файл
# --------------------------------------------------
def save_all_persons_collage(dataset_path="dataset", output_file="all_faces_collage.png"):
    """
    Зберігає колаж з усіма особами у файл
    """
    folders = sorted([f for f in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, f))])
    
    # Розміри для колажу
    persons_per_row = 8
    rows = math.ceil(len(folders) / persons_per_row)
    
    # Отримуємо розмір одного зображення
    first_folder = os.path.join(dataset_path, folders[0])
    first_img_files = [f for f in os.listdir(first_folder) if f.lower().endswith('.pgm')]
    if first_img_files:
        img = Image.open(os.path.join(first_folder, first_img_files[0]))
        img_width, img_height = img.size
    
    # Створюємо велике полотно
    collage_width = img_width * 5 * persons_per_row  # 5 фото на людину
    collage_height = img_height * rows
    collage = Image.new('L', (collage_width, collage_height), color=255)
    
    for idx, folder in enumerate(folders):
        row = idx // persons_per_row
        col = idx % persons_per_row
        
        folder_path = os.path.join(dataset_path, folder)
        images_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.pgm')])
        first_5 = images_files[:5]
        
        for i, img_file in enumerate(first_5):
            img = Image.open(os.path.join(folder_path, img_file)).convert('L')
            x_offset = (col * 5 + i) * img_width
            y_offset = row * img_height
            collage.paste(img, (x_offset, y_offset))
    
    collage.save(output_file)
    print(f"Колаж збережено як {output_file}")
    
    # Показати колаж
    plt.figure(figsize=(20, 20))
    plt.imshow(collage, cmap='gray')
    plt.title(f"Всі особи - перші 5 фото кожної (всього {len(folders)} осіб)")
    plt.axis('off')
    plt.show()

# --------------------------------------------------
# ГОЛОВНА ФУНКЦІЯ
# --------------------------------------------------
if __name__ == "__main__":
    # Варіант 1: Показати всіх людей з по 10 фото (може бути великим)
    print("Варіант 1: Всі люди по 10 фото")
    print("-" * 60)
    show_all_persons_images(dataset_path="dataset", persons_per_row=4)
    
    # Варіант 2: Показати конкретну людину
    print("\nВаріант 2: Конкретна людина (s1)")
    print("-" * 60)
    show_person_images(person_folder="s1", dataset_path="dataset")
    
    # Варіант 3: Швидкий огляд - перші 5 фото кожної
    print("\nВаріант 3: Перші 5 фото кожної людини (швидкий огляд)")
    print("-" * 60)
    show_first_5_images_all_persons(dataset_path="dataset", persons_per_row=8)
    
    # Варіант 4: Зберегти колаж у файл
    print("\nВаріант 4: Збереження колажу у файл")
    print("-" * 60)
    save_all_persons_collage(dataset_path="dataset", output_file="all_faces_collage.png")