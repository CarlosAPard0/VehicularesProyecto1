import traci
import subprocess
import os
import random
from generarRutas import generar_rutas

# Configuración SUMO
CONFIG = {
    "sumo_cmd": [
        "sumo-gui",
        "-c", "simulacion.sumocfg",
        "--collision.check-junctions",
        "--collision.action=warn",
        "--step-length", "0.5",
        "--collision-output", "collisions.xml"
    ],
    "safety": {
        "max_speed": 30.0
    }
}

# ==================== Percepción ====================
# ==================== Percepción Corregida ====================
class Perception:
    @staticmethod
    def obtener_lider(veh_id, rango=30.0):
        try:
            return traci.vehicle.getLeader(veh_id, rango)
        except:
            return None

    @staticmethod
    def obtener_conflictos(veh_id, radio=30):
        conflictos = []
        pos = traci.vehicle.getPosition(veh_id)
        ruta = traci.vehicle.getRoute(veh_id)
        
        for otro_id in traci.vehicle.getIDList():
            if otro_id == veh_id:
                continue
                
            otro_pos = traci.vehicle.getPosition(otro_id)
            distancia = traci.simulation.getDistance2D(pos[0], pos[1], otro_pos[0], otro_pos[1])
            
            if distancia < radio:
                otro_route = traci.vehicle.getRoute(otro_id)
                interseccion = set(ruta).intersection(otro_route)
                
                if interseccion:
                    conflictos.append({
                        "id": otro_id,
                        "distancia": distancia,
                        "velocidad": traci.vehicle.getSpeed(otro_id)
                    })
        return conflictos

    @staticmethod
    def esta_en_interseccion(veh_id):
        return traci.vehicle.getRoadID(veh_id).startswith(":")


# ==================== Controlador Actualizado ====================
class ControlAutonomo:
    def __init__(self, max_speed):
        self.max_speed = max_speed
        self.umbral_conflicto = 10  # metros

    def decidir_velocidad(self, veh_id):
        speed = traci.vehicle.getSpeed(veh_id)
        en_interseccion = Perception.esta_en_interseccion(veh_id)
        lider = Perception.obtener_lider(veh_id)
        conflictos = Perception.obtener_conflictos(veh_id)
        
        # Lógica mejorada para intersecciones
        if en_interseccion:
            conflicto_activo = any(c["distancia"] < self.umbral_conflicto for c in conflictos)
            
            if conflicto_activo:
                return max(speed - 3.0, 0.0)
            else:
                return min(self.max_speed, speed + 1.0)
        else:
            if lider:
                gap = lider[1]
                return min(self.max_speed, gap/3)  # Control de distancia variable
            else:
                return self.max_speed


    def aplicar_control(self, veh_id):

        nueva_vel = self.decidir_velocidad(veh_id)
        traci.vehicle.setSpeed(veh_id, nueva_vel) 
# ==================== Simulación Principal ====================
def main():
    seed = random.randint(0, 9999)
    total_vehiculos = random.randint(400, 500)
    porcentaje_autonomos = random.randint(30, 40)
    tiempo_sim = random.randint(300, 400)

    print(f"🔧 Generando rutas | Seed: {seed} | Vehículos: {total_vehiculos} | Autónomos: {porcentaje_autonomos}%")
    generar_rutas(seed, porcentaje_autonomos, total_vehiculos, tiempo_sim)

    print("🚦 Iniciando simulación SUMO...")
    traci.start(CONFIG["sumo_cmd"])
    controlador = ControlAutonomo(CONFIG["safety"]["max_speed"])

    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            step += 1
            for veh_id in traci.vehicle.getIDList():
                if traci.vehicle.getTypeID(veh_id) != "autonomous":
                    continue
                controlador.aplicar_control(veh_id)

            traci.simulationStep()

    finally:
        traci.close()
        print("✅ Simulación finalizada.")

if __name__ == "__main__":
    main()

