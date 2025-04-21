import traci
import numpy as np
from tensorflow.keras.models import load_model
import os
from simulacionTraci import CONFIG, AutonomousController, PerceptionSystem, AdvancedDQNAgent, enhanced_reward

def evaluar_modelo(agent, mostrar_detalles=False):
    try:
        # Iniciar simulación SUMO
        traci.start(CONFIG["sumo_cmd"])
        print("✅ Simulación iniciada para evaluación.")

        controller = AutonomousController(agent)
        step = 0
        total_reward = 0
        total_autonomos = 0
        colisiones = 0
        velocidades = []
        detenciones = 0  # Contador de detenciones
        tiempo_red = 0  # Tiempo total en la red de los vehículos

        while traci.simulation.getMinExpectedNumber() > 0:
            step += 1
            current_vehicles = traci.vehicle.getIDList()

            for veh_id in current_vehicles:
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

                    # Medir tiempo en la red por vehículo
                    tiempo_red += traci.vehicle.getAccumulatedWaitingTime(veh_id)

                    # Contabilizar detenciones
                    if velocidad < 0.1:
                        detenciones += 1

            traci.simulationStep()

        colisiones = len(traci.simulation.getCollidingVehiclesIDList())

    except Exception as e:
        print(f"❌ Error en la evaluación: {e}")
    finally:
        try:
            traci.close()
        except Exception as e:
            print(f"⚠️ No se pudo cerrar SUMO correctamente: {e}")

    # Estadísticas
    promedio_reward = total_reward / max(1, total_autonomos)
    promedio_velocidad = np.mean(velocidades) if velocidades else 0
    tiempo_promedio_red = tiempo_red / max(1, total_autonomos)  # Tiempo promedio en la red por vehículo
    detenciones_promedio = detenciones / max(1, total_autonomos)  # Detenciones promedio por vehículo
    flujo_vehicular = total_autonomos / step if step > 0 else 0  # Flujo vehicular total

    print("\n📊 Resultados de evaluación:")
    print(f"Total de pasos: {step}")
    print(f"Vehículos autónomos procesados: {total_autonomos}")
    print(f"Colisiones detectadas: {colisiones}")
    print(f"Velocidad promedio: {promedio_velocidad:.2f} m/s")
    print(f"Recompensa promedio por vehículo: {promedio_reward:.2f}")
    print(f"Tiempo promedio en la red por vehículo: {tiempo_promedio_red:.2f} segundos")
    print(f"Cantidad de detenciones por vehículo: {detenciones_promedio:.2f}")
    print(f"Flujo vehicular total: {flujo_vehicular:.2f} vehículos por paso")

    return {
        "pasos": step,
        "autonomos": total_autonomos,
        "colisiones": colisiones,
        "velocidad_promedio": promedio_velocidad,
        "reward_promedio": promedio_reward,
        "tiempo_promedio_red": tiempo_promedio_red,
        "detenciones_promedio": detenciones_promedio,
        "flujo_vehicular": flujo_vehicular
    }

if __name__ == "__main__":
    if not os.path.exists("densidad1.keras"):
        print("❗ No se encontró el modelo entrenado.")
        exit()

    # Cargar agente y modelo
    agent = AdvancedDQNAgent(CONFIG["ia_params"]["state_size"], CONFIG["ia_params"]["action_size"])
    agent.model = load_model("densidad1.keras")
    agent.epsilon = 0.0  # Desactiva la exploración

    resultados = evaluar_modelo(agent)

