import os
import socket
import random
import time
from pathlib import Path
import atexit

# Safe port range for shared cluster use
PORT_RANGE = (10000, 49000)
MIN_SAFE_PORT = 1024

# Location of shared port tracking files
PORT_FILE = Path(os.environ.get("PORT_FILE", "results/used_ports.txt"))
LOCK_FILE = PORT_FILE.with_suffix(".lock")

PORT_FILE.parent.mkdir(parents=True, exist_ok=True)
PORT_FILE.touch(exist_ok=True)

def pick_unique_port():
    for _ in range(1000):
        # Lock file handling (simple mutex)
        while LOCK_FILE.exists():
            time.sleep(0.01)
        LOCK_FILE.touch(exist_ok=True)

        try:
            # Read current used ports
            with open(PORT_FILE, "r") as f:
                content = f.read().strip()
                used = set()
                for line in content.splitlines():
                    try:
                        p = int(line.strip())
                        if MIN_SAFE_PORT <= p <= PORT_RANGE[1]:
                            used.add(p)
                    except ValueError:
                        continue  # Ignore invalid lines

            # Pick a port that is safe and unused
            port = random.randint(*PORT_RANGE)
            if port in used or port < MIN_SAFE_PORT:
                continue

            # Check if port is actually free on this machine
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', port)) != 0:
                    used.add(port)
                    with open(PORT_FILE, "w") as f:
                        f.write("\n".join(map(str, sorted(used))))
                    return port

        finally:
            try:
                LOCK_FILE.unlink()
            except FileNotFoundError:
                pass

    raise RuntimeError("Failed to find a unique port after 1000 attempts.")

# Cleanup files on exit (optional)
@atexit.register
def cleanup():
    try:
        LOCK_FILE.unlink()
    except FileNotFoundError:
        pass
