import traci
import numpy as np
import random
import subprocess
import os
from collections import deque
from tensorflow.keras.layers import Dense, Dropout, LSTM, Add, Input, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

# Configuración para evitar warnings de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ==================== Definición de CONFIG ====================
CONFIG = {
    "sumo_cmd": [
        "sumo",
        "-c", "simulacion.sumocfg",
        "--collision.check-junctions",
        "--collision.action=warn",
        "--step-length", "0.5",
        "--collision-output=collisions.xml"
    ],
    "safety": { "max_speed": 30.0 },
    "ia_params": {
        "state_size": 10,
        "action_size": 7,
        "batch_size": 64
    }
}

# ==================== Generador de rutas ====================
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

# ==================== Agente DQN Avanzado ====================
class AdvancedDQNAgent:
    def __init__(self, state_size, action_size, sequence_length=25):
        self.state_size      = state_size
        self.action_size     = action_size
        self.sequence_length = sequence_length
        self.memory          = deque(maxlen=4000)
        self.gamma           = 0.95
        self.epsilon         = 0.5
        self.epsilon_min     = 0.01
        self.epsilon_decay   = 0.99    
        self.learning_rate   = 0.0003
        self.model           = self._build_lstm_model()
        self.recent_states   = {}

    def _build_lstm_model(self):
        inp = Input(shape=(self.sequence_length, self.state_size))
        x   = LSTM(128)(inp)

        # Residual block 1
        d1   = Dense(64)(x)
        bn1  = BatchNormalization()(d1)
        a1   = LeakyReLU(0.1)(bn1)
        dr1  = Dropout(0.2)(a1)

        sc1  = Dense(64)(x)
        res1 = Add()([sc1, dr1])

        # Residual block 2
        d2   = Dense(64)(res1)
        bn2  = BatchNormalization()(d2)
        a2   = LeakyReLU(0.2)(bn2)
        dr2  = Dropout(0.2)(a2)

        sc2  = Dense(64)(res1)
        res2 = Add()([sc2, dr2])

        # Suma final
        salida = Add()([res1, res2])

        out = Dense(self.action_size, activation='linear')(salida)

        m = Model(inputs=inp, outputs=out)
        m.compile(loss='mse', optimizer=Adam(self.learning_rate))
        return m

    def remember(self, s_seq, act, rew, ns_seq, done):
        self.memory.append((s_seq, act, rew, ns_seq, done))

    def get_state_sequence(self, vid, new_state):
        arr = np.array(new_state)
        if vid not in self.recent_states:
            self.recent_states[vid] = deque(maxlen=self.sequence_length)
        self.recent_states[vid].append(arr)
        # rellenar con ceros hasta sequence_length
        while len(self.recent_states[vid]) < self.sequence_length:
            self.recent_states[vid].appendleft(np.zeros(self.state_size))
        return np.array(self.recent_states[vid])

    def choose_action(self, state_seq):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        seq = state_seq.reshape(1, self.sequence_length, self.state_size)
        vals = self.model.predict(seq, verbose=0)[0]
        return np.argmax(vals)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for s_seq, act, rew, ns_seq, done in batch:
            target = rew
            if not done:
                ns = ns_seq.reshape(1, self.sequence_length, self.state_size)
                target = rew + self.gamma * np.max(self.model.predict(ns, verbose=0)[0])
            s = s_seq.reshape(1, self.sequence_length, self.state_size)
            t_f = self.model.predict(s, verbose=0)
            t_f[0][act] = target
            self.model.fit(s, t_f, epochs=1, verbose=0)
        # actualizar epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def enhanced_reward(veh_id, action, last_actions):
    speed     = traci.vehicle.getSpeed(veh_id)
    max_speed = CONFIG["safety"]["max_speed"]
    is_junc   = PerceptionSystem.esta_en_interseccion(veh_id)

    leader = traci.vehicle.getLeader(veh_id, 50.0)
    gap    = leader[1] if leader else 100.0

    base_reward = speed / max_speed
    base_reward = min(base_reward, 1.0)

    penalties = 0.0

    # 🚫 Penalización por detenerse cerca de intersección
    if speed < 0.3 and is_junc:
        penalties -= 5.0  # mayor penalización si frena al entrar a cruce

    # 🚫 Penalización por frenar innecesariamente (sin líder ni cruce)
    if action in [0, 1] and not leader and not is_junc:
        penalties -= 2.0

    # 🚫 Penalización por estar completamente detenido sin motivo
    if speed < 0.1 and not is_junc:
        penalties -= 3.0

    # 🚫 Penalización fuerte por ir rápido estando muy cerca de otro
    if gap < 10 and speed > max_speed * 0.5:
        penalties -= 4.0

    # ❌ Colisión
    if veh_id in traci.simulation.getCollidingVehiclesIDList():
        penalties -= 20.0

    # ✅ Bonus por pasar el cruce a buena velocidad
    cross_bonus = 0.0
    if is_junc and speed > max_speed * 0.6:
        cross_bonus += 4.0

    # ✅ Bonus por camino libre y velocidad alta
    reward_bonus = 0.0
    if not leader and not is_junc and speed > max_speed * 0.8:
        reward_bonus = (speed / max_speed) * 2.0

    # ✅ Bonus por mantener buena distancia
    gap_reward = np.tanh(gap / 10.0) * 0.3

    # ✅ Bonus por moverse en tráfico lento
    movement_bonus = 0.3 if speed > 0.3 else 0.0

    reward = base_reward + reward_bonus + gap_reward + movement_bonus + cross_bonus + penalties

    return reward

# ==================== Control y percepción ====================
class AutonomousController:
    def __init__(self, agent):
        self.agent = agent
        self.last_actions = {}

    def apply_safe_action(self, vid, act):
        cur = traci.vehicle.getSpeed(vid)
        mx  = CONFIG["safety"]["max_speed"]
        mapping = {
            0: max(0, cur-0.5),
            1: max(0, cur-1.5),
            2: max(0, cur-2.5),
            3: cur,
            4: min(mx, cur+0.5),
            5: min(mx, cur+1.5),
            6: min(mx, cur+2.5)
        }
        traci.vehicle.setSpeed(vid, mapping[act])

class PerceptionSystem:
    @staticmethod
    def get_state(vid):
        s = []
        spd = traci.vehicle.getSpeed(vid)
        s.append(spd/CONFIG["safety"]["max_speed"])
        leader = traci.vehicle.getLeader(vid)
        s.append((leader[1]/50.0) if leader else 1.0)
        if leader:
            lid = leader[0]
            s.append(traci.vehicle.getSpeed(lid)/CONFIG["safety"]["max_speed"])
        else:
            s.append(0.0)
        is_j = traci.vehicle.getRoadID(vid).startswith(":") or "junction" in traci.vehicle.getRoadID(vid)
        s.append(1.0 if is_j else 0.0)
        lane = traci.vehicle.getLaneID(vid)
        s.append(1.0 if "turn" in lane.lower() else 0.0)
        # nearby vehicles
        cnt=0; x0,y0 = traci.vehicle.getPosition(vid)
        for oid in traci.vehicle.getIDList():
            if oid==vid: continue
            try:
                x1,y1=traci.vehicle.getPosition(oid)
                if abs(x0-x1)<15 and abs(y0-y1)<15: cnt+=1
            except: pass
        s.append(min(cnt/7.0,1.0))
        # líder tipo
        if leader:
            s.append(1.0 if traci.vehicle.getTypeID(leader[0])=="autonomous" else 0.0)
        else:
            s.append(-1.0)
        # padding
        while len(s)<CONFIG["ia_params"]["state_size"]:
            s.append(0.0)
        return np.array(s[:CONFIG["ia_params"]["state_size"]])

    @staticmethod
    def esta_en_interseccion(vid):
        try:
            rid = traci.vehicle.getRoadID(vid)
            return rid.startswith(":") or "junction" in rid
        except:
            return False

# ==================== Ciclo principal ====================
def main(agent):
    step = 0
    total_reward = 0.0
    traci.start(CONFIG["sumo_cmd"])
    ctrl = AutonomousController(agent)
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            step += 1
            for vid in traci.vehicle.getIDList():
                if traci.vehicle.getTypeID(vid) != "autonomous":
                    continue
                st = PerceptionSystem.get_state(vid)
                seq= agent.get_state_sequence(vid, st)
                act= agent.choose_action(seq)
                ctrl.apply_safe_action(vid, act)
                nxt= PerceptionSystem.get_state(vid)
                r  = enhanced_reward(vid, act, ctrl.last_actions)
                total_reward += r
                nsq= agent.get_state_sequence(vid, nxt)
                agent.remember(seq, act, r, nsq, False)
                ctrl.last_actions[vid] = act

            if step % CONFIG["ia_params"]["batch_size"] == 0:
                agent.replay(CONFIG["ia_params"]["batch_size"])
            traci.simulationStep()
    finally:
        traci.close()
        print(f"Recompensa total del episodio: {total_reward:.2f}")
    return total_reward

# ==================== Entrenamiento guardando por recompensa media ====================
if __name__ == '__main__':
    total_escenarios = 10
    repeticiones_por_escenario = 3
    ep_total = 0
    best_avg = -float('inf')

    agent = AdvancedDQNAgent(CONFIG["ia_params"]["state_size"], CONFIG["ia_params"]["action_size"])

    # Cargar modelo previo si existe
    if os.path.exists("densidad3.keras"):
        agent.model = load_model("densidad3.keras")
        print("🔁 Modelo previo cargado.")

    for n in range(total_escenarios):
        # Generamos un solo escenario base
        vehs = random.randint(50, 59)  # entre 30 y 39 vehículos
        porc_autonomos = random.randint(20, 40)
        seed = random.randint(0, 10000)
        tiempo = 60  # fijo

        print(f"\n🌐 Escenario {n+1}/{total_escenarios} | Vehículos: {vehs}, Autónomos: {porc_autonomos}%")

        # Generar rutas solo una vez por escenario
        generar_rutas(seed, porc_autonomos, vehs, tiempo)

        for i in range(repeticiones_por_escenario):
            ep_total += 1
            print(f"\n=== Episodio {ep_total} | Escenario {n+1}, Repetición {i+1}/3 ===")
            ep_reward = main(agent)
            avg_reward = ep_reward / vehs

            if avg_reward > best_avg:
                best_avg = avg_reward
                agent.model.save("densidad1.keras")
                print(f"🏆 Nuevo mejor modelo (avg reward {best_avg:.3f})")
            else:
                print(f"⚠️ Avg reward {avg_reward:.3f} no supera {best_avg:.3f}")

