#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="$(id -un)"
VENV="$PROJECT_DIR/.venv"
ENV_FILE="$PROJECT_DIR/deploy/.env.production"
SERVICE_FILE="/etc/systemd/system/yiwentong.service"

echo "==> Installing system packages"
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git curl

echo "==> Creating Python environment"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
"$VENV/bin/pip" install -i https://mirrors.aliyun.com/pypi/simple/ -r "$PROJECT_DIR/requirements.txt"

if [ ! -f "$ENV_FILE" ]; then
  cp "$PROJECT_DIR/deploy/.env.production.example" "$ENV_FILE"
fi

read -r -p "Demo password [WYU-demo-2026]: " DEMO_PASSWORD
DEMO_PASSWORD="${DEMO_PASSWORD:-WYU-demo-2026}"
cat > "$ENV_FILE" <<EOF
WYU_USE_DENSE=0
WYU_OFFLINE=1
WYU_DEMO_PASSWORD=$DEMO_PASSWORD
WYU_COVERAGE_MIN=0.5
EOF
chmod 600 "$ENV_FILE"

echo "==> Creating systemd service"
sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Yiwentong Streamlit
After=network.target

[Service]
User=$USER_NAME
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$VENV/bin/streamlit run $PROJECT_DIR/ui/wyu_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now yiwentong

echo
echo "Service started."
echo "Open TCP 8501 in the cloud security group."
echo "URL: http://$(curl -fsS --max-time 5 ifconfig.me 2>/dev/null || echo 'SERVER_PUBLIC_IP'):8501"
