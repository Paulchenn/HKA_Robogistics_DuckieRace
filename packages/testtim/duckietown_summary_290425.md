# 📅 Zusammenfassung vom 2025-04-29


# 🧠 Zusammenfassung: Duckietown DTProject mit eigenem Script

## 📁 Projektpfad
`~/QuackSquat/HKA_Robogistics_DuckieRace/`

---

## 📦 Aufbau des Repositories (DTProject)

| Ordner / Datei                 | Zweck                                                       |
|-------------------------------|-------------------------------------------------------------|
| `packages/`                   | enthält Python-/ROS-Code (`students/`, `followlane/`, `testtim/`) |
| `launchers/`                  | Startskripte für Docker (z. B. `default.sh`)               |
| `docs/`, `html/`              | Dokumentation (Sphinx)                                     |
| `assets/`                     | Statische Dateien / Konfigurationen                        |
| `Dockerfile`                  | Bauanleitung für das Docker-Image                          |
| `configurations.yaml`         | Build-Konfigurationen für verschiedene Plattformen         |
| `dependencies-*.txt`          | Listen aller benötigten apt/pip-Abhängigkeiten             |
| `README.md`                   | Projektübersicht                                           |

---

## ✅ Dein neues Package: `testtim`

### 📁 Pfad: `packages/testtim/`

### 🗃️ Dateien:
- `__init__.py`
- `my_script.py`

```python
import os
vehicle_name = os.environ['VEHICLE_NAME']
message = f"\nHello from {vehicle_name}!\n"
print(message)
```

---

## 🚀 Launcher-Konfiguration: `launchers/default.sh`

```bash
#!/bin/bash
source /environment.sh
dt-launchfile-init

dt-exec python3 -m "testtim.my_script"

dt-launchfile-join
```

---

## 🔧 Wichtige Befehle (lokal & remote)

### 📁 Lokal im Projektordner:

```bash
cd ~/QuackSquat/HKA_Robogistics_DuckieRace/
```

### 🧱 Image lokal bauen:

```bash
dts devel build -f
```

### ▶️ Image lokal starten:

```bash
dts devel run
```

---

## 🦆 Duckiebot: remote ausführen (Hostname: `daffy.local`)

### 📡 Verbindung testen:

```bash
ping daffy.local
```

### 🏗️ Image auf Duckiebot bauen:

```bash
dts devel build -f -H daffy.local
```

### ▶️ Image auf Duckiebot ausführen:

```bash
dts devel run -H daffy.local
```

---

## 🧪 Extra: `control_lane_node.py` (Zusammenfassung)

- Empfängt Spurinformationen von `detect/lane`
- Veröffentlicht Bewegungsbefehle an den Duckiebot
- Nutzt einen einfachen P-Regler basierend auf Bildmittelpunkt
- Hört nur auf Eingaben, wenn `switch/control` auf `LANE` gesetzt ist

---

## ✅ Fazit:

Du hast heute:
- das Projekt-Repo analysiert,
- ein eigenes Package (`testtim`) aufgebaut,
- ein Script erstellt und in Docker eingebunden,
- es erfolgreich lokal und auf dem Duckiebot ausgeführt 🎉
