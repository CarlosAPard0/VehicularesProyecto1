from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Cargar el modelo guardado
model = load_model('densidad1.keras')  # Asegúrate que la ruta del archivo sea correcta

# Generar el diagrama
plot_model(
    model,
    to_file="model_plot.png",
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True,  # Para ver detalles de bloques como los residuales
    dpi=96  # Resolución de la imagen
)
