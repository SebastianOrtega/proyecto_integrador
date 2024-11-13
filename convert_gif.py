from moviepy.editor import VideoFileClip

# Carga el archivo de video
clip = VideoFileClip("solucion1.mp4")

# Exporta como GIF, ajustando la duración o el tamaño si es necesario
clip = clip.subclip(0, 5)  # Opcional: recorta los primeros 5 segundos
clip = clip.resize(0.5)     # Opcional: reduce el tamaño al 50%
clip = clip.speedx(2)       # Aumenta la velocidad 2x

# Guarda el clip como GIF
clip.write_gif("solucion1.gif")
