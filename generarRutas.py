import random
import argparse
from lxml import etree

def generar_rutas(semilla, porcentaje_autonomos, total_vehiculos, tiempo_simulacion, output_file="rutas.rou.xml"):
    random.seed(semilla)
    
    # Tipos de vehículos con sus parámetros específicos
    tipos = [
        {
            "id": "conventional",
            "carFollowModel": "Krauss",
            "accel": "2.6",
            "decel": "4.5",
            "tau": "1.2",
            "minGap": "2.5",
            "sigma": "0.5",
            "speedDev": "0.2",
            "color": "200,50,50",
            #"guiShape": "passenger"
        },
        {
            "id": "autonomous",
            "carFollowModel": "CACC",
            "accel": "3.0",
            "decel": "6.0",
            #"tau": "0.5",
            "minGap": "1.0",
            "sigma": "0.0",
            "speedDev": "0.1",
            "color": "50,50,200",
            #"guiShape": "evehicle"
        }
    ]

    
    rutas = [
        ("n1_recto", "n1_to_center center_to_n2"),
        ("n1_izquierda", "n1_to_center center_to_n3"),
        ("n1_derecha", "n1_to_center center_to_n4"),
        ("n2_recto", "n2_to_center center_to_n1"),
        ("n2_izquierda", "n2_to_center center_to_n4"),
        ("n2_derecha", "n2_to_center center_to_n3"),
        ("n3_recto", "n3_to_center center_to_n4"),
        ("n3_izquierda", "n3_to_center center_to_n2"),
        ("n3_derecha", "n3_to_center center_to_n1"),
        ("n4_recto", "n4_to_center center_to_n3"),
        ("n4_izquierda", "n4_to_center center_to_n1"),
        ("n4_derecha", "n4_to_center center_to_n2")
    ]
    
    root = etree.Element("routes")
    
    # Añadir tipos de vehículos (solo con parámetros válidos)
    for tipo in tipos:
        vtype = etree.SubElement(root, "vType")
        for key, value in tipo.items():
            vtype.set(key, value)
    
    # Añadir rutas
    for id, edges in rutas:
        route = etree.SubElement(root, "route", id=id, edges=edges)
    
    # Calcular intervalo entre vehículos
    intervalo = tiempo_simulacion / total_vehiculos if total_vehiculos > 0 else 0
    
    # Calcular el número de vehículos autónomos
    vehiculos_autonomos = int(total_vehiculos * porcentaje_autonomos / 100)
    vehiculos_convencionales = total_vehiculos - vehiculos_autonomos
    
    # Crear una lista con los vehículos (autónomos y convencionales) para mezclarlos
    flujos_vehiculos = ["autonomous"] * vehiculos_autonomos + ["conventional"] * vehiculos_convencionales
    random.shuffle(flujos_vehiculos)  # Mezclar aleatoriamente
    
    # Generar flujos de vehículos
    for i, vtype in enumerate(flujos_vehiculos):
        tiempo_inicio = i * intervalo
        tiempo_fin = tiempo_inicio + 1  # Pequeña ventana de tiempo
        
        route_id = random.choice(rutas)[0]  # Elegir una ruta aleatoria
        
        flow = etree.SubElement(root, "flow",
            id=f"flow_{i}",
            type=vtype,
            route=route_id,
            departLane="best",
            departSpeed="max",
            begin=f"{tiempo_inicio:.2f}",
            end=f"{tiempo_fin:.2f}",
            period="1"
        )
    
    tree = etree.ElementTree(root)
    tree.write(output_file, pretty_print=True, encoding="UTF-8", xml_declaration=True)
    print(f"✔ Archivo generado: {output_file}")
    print(f"  - Total vehículos: {total_vehiculos}")
    print(f"  - Vehículos autónomos: {vehiculos_autonomos} ({porcentaje_autonomos}%)")
    print(f"  - Duración simulación: {tiempo_simulacion} segundos")
    print(f"  - Intervalo entre vehículos: {intervalo:.2f} segundos")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generador de rutas aleatorias para SUMO')
    parser.add_argument('--semilla', type=int, default=42, help='Semilla para reproducibilidad (default: 42)')
    parser.add_argument('--autonomos', type=float, default=30.0, help='Porcentaje de vehículos autónomos (default: 30%%)')
    parser.add_argument('--vehiculos', type=int, default=500, help='Número total de vehículos (default: 500)')
    parser.add_argument('--tiempo', type=float, default=3600, help='Duración total de simulación en segundos (default: 3600)')
    parser.add_argument('--output', type=str, default="rutas.rou.xml", help='Nombre del archivo de salida (default: rutas.rou.xml)')
    
    args = parser.parse_args()
    
    generar_rutas(
        semilla=args.semilla,
        porcentaje_autonomos=args.autonomos,
        total_vehiculos=args.vehiculos,
        tiempo_simulacion=args.tiempo,
        output_file=args.output
    )

