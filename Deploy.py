import shutil
import subprocess
import os

# --- CONFIGURATION ---
FIRMWARE_DIR = "firmware"  # The folder containing your node code
ZIP_FILENAME = "firmware"  # The script will automatically append '.zip'
AP_IP = "VisTrack-AP.local"
AP_USER = "vistrack-ap"  # Change this to your Pi 3B username
AP_DEST_DIR = "/home/vistrack-ap/tracker_config"  # The folder where the web server runs


def deploy():
    print("🚀 Starting Swarm Deployment Pipeline...")

    # 1. Verification
    if not os.path.exists(FIRMWARE_DIR):
        print(f"❌ Error: Could not find the '{FIRMWARE_DIR}' directory.")
        return

    # 2. Zipping the Firmware
    print(f"📦 Packaging '{FIRMWARE_DIR}' into {ZIP_FILENAME}.zip...")
    # shutil.make_archive zips the *contents* of the folder, which is exactly what the nodes expect
    shutil.make_archive(ZIP_FILENAME, 'zip', FIRMWARE_DIR)

    zip_path = f"{ZIP_FILENAME}.zip"
    zip_size = os.path.getsize(zip_path) / 1024
    print(f"✅ Packaged successfully! Size: {zip_size:.1f} KB")

    # 3. Uploading via SCP
    print(f"📡 Uploading to AP ({AP_USER}@{AP_IP}:{AP_DEST_DIR})...")
    scp_command = [
        "scp",
        zip_path,
        f"{AP_USER}@{AP_IP}:{AP_DEST_DIR}/firmware.zip"
    ]

    # Run the SCP command. (This will prompt for your Pi password in the console)
    scp_result = subprocess.run(scp_command)

    if scp_result.returncode != 0:
        print("❌ Upload failed. Check your network connection and password.")
        return

    # 4. Verifying the Upload via SSH
    print("🔍 Verifying deployment on the AP...")
    ssh_command = [
        "ssh",
        f"{AP_USER}@{AP_IP}",
        f"ls -lh {AP_DEST_DIR}/firmware.zip"
    ]

    ssh_result = subprocess.run(ssh_command, capture_output=True, text=True)

    if ssh_result.returncode == 0:
        print("✅ Deployment Verified! The AP is now hosting the new firmware.")
        print(f"📄 AP File Info: {ssh_result.stdout.strip()}")
        print("🎉 The nodes will download this on their next reboot.")
    else:
        print("⚠️ Upload seemed to work, but verification failed.")


if __name__ == "__main__":
    deploy()