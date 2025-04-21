import subprocess
import os
import json
from tensorflow.keras.models import load_model
from simulacionTraci import CONFIG, AutonomousController, PerceptionSystem, AdvancedDQNAgent, enhanced_reward
import traci
import numpy as np

def generar_rutas(semilla, porcentaje_autonomos, total_vehiculos, tiempo_simulacion,
                  output_file="rutas.rou.xml"):
    command = [
        "python", "generarRutas.py",
        "--semilla", str(semilla),
        "--autonomos", str(porcentaje_autonomos),
        "--vehiculos", str(total_vehiculos),
        "--tiempo", str(tiempo_simulacion),
        "--output", output_file
    ]
    subprocess.run(command)

def evaluar_modelo(agent):
    try:
        traci.start(CONFIG["sumo_cmd"])
        controller = AutonomousController(agent)

        step = 0
        total_reward = 0
        total_autonomos = 0
        colisiones = 0
        velocidades = []
        detenciones = 0
        tiempo_red = 0

        while traci.simulation.getMinExpectedNumber() > 0:
            step += 1
            vehiculos = traci.vehicle.getIDList()

            for veh_id in vehiculos:
                if traci.vehicle.getTypeID(veh_id) == "autonomous":
                    total_autonomos += 1
                    state = PerceptionSystem.get_state(veh_id)
                    state_seq = agent.get_state_sequence(veh_id, state)
                    action = agent.choose_action(state_seq)
                    controller.apply_safe_action(veh_id, action)

                    reward = enhanced_reward(veh_id, action, controller.last_actions)
                    total_reward += reward
                    controller.last_actions[veh_id] = action

                    velocidad = traci.vehicle.getSpeed(veh_id)
                    velocidades.append(velocidad)

                    if velocidad < 0.1:
                        detenciones += 1

                    tiempo_red += traci.vehicle.getAccumulatedWaitingTime(veh_id)

            traci.simulationStep()

        colisiones = len(traci.simulation.getCollidingVehiclesIDList())
        teletransportes = len(traci.simulation.getEndingTeleportIDList())

    except Exception as e:
        print(f"❌ Error en la evaluación: {e}")
        return None
    finally:
        try:
            traci.close()
        except:
            pass

    promedio_reward = total_reward / max(1, total_autonomos)
    promedio_velocidad = np.mean(velocidades) if velocidades else 0
    tiempo_promedio_red = tiempo_red / max(1, total_autonomos)
    detenciones_promedio = detenciones / max(1, total_autonomos)
    flujo_vehicular = total_autonomos / step if step > 0 else 0

    return {
        "colisiones": colisiones,
        "teletransportes": teletransportes,
        "velocidad_promedio": promedio_velocidad,
        "tiempo_promedio_red": tiempo_promedio_red,
        "detenciones_promedio": detenciones_promedio,
        "flujo_vehicular": flujo_vehicular
    }

def main():
    if not os.path.exists("densidad1.keras"):
        print("❗ No se encontró el modelo entrenado.")
        return

    agent = AdvancedDQNAgent(CONFIG["ia_params"]["state_size"], CONFIG["ia_params"]["action_size"])
    agent.model = load_model("densidad1.keras")
    agent.epsilon = 0.0

    resultados_totales = {}

    porcentajes = [20, 30, 40]
    tiempos = [600, 500, 400]
    #tiempos = [60, 50, 40]
    semilla = 1234
    total_vehiculos = 600

    for porcentaje in porcentajes:
        resultados_totales[f"{porcentaje}%"] = {}

        for tiempo in tiempos:
            print(f"\n🚗 Evaluando escenario: {porcentaje}% autónomos, {tiempo}s")

            generar_rutas(semilla, porcentaje, total_vehiculos, tiempo)

            resultado = evaluar_modelo(agent)

            if resultado:
                resultados_totales[f"{porcentaje}%"][f"{tiempo}s"] = resultado
                print("✅ Evaluación completada.")
            else:
                print("❌ Fallo en la evaluación.")
                resultados_totales[f"{porcentaje}%"][f"{tiempo}s"] = "Error"

    # Guardar resultados en un archivo JSON
    with open("resultados_evaluacion.json", "w") as f:
        json.dump(resultados_totales, f, indent=4)

    print("\n📁 Resultados guardados en 'resultados_evaluacion.json'.")

if __name__ == "__main__":
    main()

