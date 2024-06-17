import os

def remplazo(directory, old_number, new_number):
    """
    Replace the first two occurrences of `old_number` with `new_number` in each line of all text files in `directory`.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                new_line = line
                for j in range(2):
                    try:
                        pos = line.index(str(old_number))
                        new_line = line[:pos] + str(new_number) + line[pos+len(str(old_number)):]
                    except ValueError:
                        # old_number not found in this line
                        pass
                lines[i] = new_line
            with open(filepath, 'w') as f:
                f.writelines(lines)
                
    print('PROCESO FINALIZADO CON EXITO')
                
# Example usage: replace the first two numbers with "0" in all the text files in the directory "/path/to/directory"
remplazo("Plagas/detect_plg/Oruga del tomate/train/labels", 6, 2)


# import os
# import shutil

# def separate_files(directory):
#     # Crea las carpetas 'images' y 'labels' si no existen
#     if not os.path.exists(os.path.join(directory, 'images')):
#         os.makedirs(os.path.join(directory, 'images'))
#     if not os.path.exists(os.path.join(directory, 'labels')):
#         os.makedirs(os.path.join(directory, 'labels'))

#     # Lista de extensiones de archivos de imagen
#     image_extensions = ['.jpeg', '.jpg', '.png']

#     # Itera sobre todos los archivos
#     for filename in os.listdir(directory):
#         if filename.endswith(tuple(image_extensions)):
#             # Mueve las imágenes a la carpeta 'images'
#             shutil.move(os.path.join(directory, filename), os.path.join(directory, 'images', filename))
#         elif filename.endswith('.txt'):
#             # Mueve los archivos .txt a la carpeta 'labels'
#             shutil.move(os.path.join(directory, filename), os.path.join(directory, 'labels', filename))

# # Usa la función en tu directorio
# separate_files('data')

