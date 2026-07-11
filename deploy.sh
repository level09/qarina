#!/usr/bin/env bash
# Ignite-style direct deploy (no Docker): rsync the app to a fresh Ubuntu box,
# provision uv + systemd + Caddy (auto-SSL + basic auth), start the service.
#
# First deploy:   DOMAIN=qarina.example.org SERVER=root@1.2.3.4 ./deploy.sh
# Re-deploy:      SERVER=root@1.2.3.4 ./deploy.sh        (sync + restart only)
# No domain yet:  DOMAIN=1.2.3.4 SERVER=root@1.2.3.4 ./deploy.sh   (plain HTTP)
set -euo pipefail

SERVER="${SERVER:?Usage: DOMAIN=qarina.example.org SERVER=root@host ./deploy.sh}"
DOMAIN="${DOMAIN:-}"
APP_USER=qarina
APP_DIR=/home/${APP_USER}/app
SERVICE=qarina

echo "==> Syncing project to ${SERVER}"
rsync -az --delete \
  --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
  --exclude .ruff_cache --exclude history.db --exclude knowledge_store \
  ./ "${SERVER}:/tmp/qarina-app/"

ssh "$SERVER" bash -s -- "$DOMAIN" << 'REMOTE'
set -euo pipefail
DOMAIN="$1"
APP_USER=qarina
APP_DIR=/home/${APP_USER}/app
SERVICE=qarina

# ---- one-time provisioning ----
if [ ! -f /etc/systemd/system/${SERVICE}.service ]; then
    echo "==> Installing packages (caddy, uv, ufw, fail2ban)"
    apt-get update -qq
    apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl rsync ufw fail2ban >/dev/null
    if ! command -v caddy >/dev/null; then
        curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
        curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
        apt-get update -qq && apt-get install -y -qq caddy >/dev/null
    fi
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null

    id "$APP_USER" &>/dev/null || useradd -m -s /bin/bash "$APP_USER"

    echo "==> Firewall"
    ufw allow 22/tcp >/dev/null; ufw allow 80/tcp >/dev/null; ufw allow 443/tcp >/dev/null
    ufw --force enable >/dev/null
    systemctl enable --now fail2ban >/dev/null 2>&1

    echo "==> systemd service"
    cat > /etc/systemd/system/${SERVICE}.service << EOF
[Unit]
Description=Qarina
After=network.target

[Service]
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
ExecStart=${APP_DIR}/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8018
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable ${SERVICE} >/dev/null

    echo "==> Caddy (basic auth for the team)"
    TEAM_PASS=$(openssl rand -base64 12)
    HASH=$(caddy hash-password --plaintext "$TEAM_PASS")
    if [[ "$DOMAIN" =~ ^([0-9]+\.){3}[0-9]+$ || "$DOMAIN" == localhost || -z "$DOMAIN" ]]; then
        SITE=":80"
    else
        SITE="$DOMAIN"
    fi
    cat > /etc/caddy/Caddyfile << EOF
${SITE} {
	encode zstd gzip
	header {
		X-Content-Type-Options "nosniff"
		X-Frame-Options "SAMEORIGIN"
		Referrer-Policy "strict-origin-when-cross-origin"
		-Server
	}
	basic_auth {
		team ${HASH}
	}
	reverse_proxy 127.0.0.1:8018
}
EOF
    systemctl restart caddy
    echo "team password: ${TEAM_PASS}" > /home/${APP_USER}/.credentials
    chmod 600 /home/${APP_USER}/.credentials
    echo "==> Team login -> user: team  password: ${TEAM_PASS}"
fi

# ---- every deploy: sync code, install deps, restart ----
echo "==> Installing app"
mkdir -p "$APP_DIR"
rsync -a --delete --exclude .venv --exclude .env --exclude history.db --exclude knowledge_store /tmp/qarina-app/ "$APP_DIR/"
[ -f "$APP_DIR/.env" ] || { cp /tmp/qarina-app/.env "$APP_DIR/.env" 2>/dev/null || echo "!! No .env - create ${APP_DIR}/.env"; }
chown -R "${APP_USER}:${APP_USER}" "/home/${APP_USER}"
sudo -u "$APP_USER" bash -c "cd $APP_DIR && uv sync --frozen --no-dev" >/dev/null
systemctl restart ${SERVICE}
sleep 2
systemctl --no-pager --quiet is-active ${SERVICE} && echo "==> ${SERVICE} running" || { journalctl -u ${SERVICE} -n 20 --no-pager; exit 1; }
echo "==> Done: https://${DOMAIN:-<server-ip>}"
REMOTE
