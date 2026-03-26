import socket

# --- CONFIGURATION ---
LISTEN_IP = "0.0.0.0"  # Listen on all available network interfaces
LISTEN_PORT = 5005

# --- NETWORK SETUP ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LISTEN_IP, LISTEN_PORT))

print(f"Central Hub listening for UDP packets on port {LISTEN_PORT}...")

try:
    while True:
        # Wait for an incoming packet
        data, addr = sock.recvfrom(1024)
        message = data.decode('utf-8')

        # Parse the incoming string
        parts = message.split(',')
        if len(parts) == 4:
            cam_id, cx, cy, area = parts
            print(f"[{addr[0]}] {cam_id} | Target at X:{cx} Y:{cy} (Size: {area})")
        else:
            print(f"Received malformed packet: {message}")

except KeyboardInterrupt:
    print("\nShutting down receiver.")
    sock.close()