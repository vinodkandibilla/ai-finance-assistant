#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

sudo apt-get update
sudo apt-get install -y nginx

sudo install -D -m 0644 "$SCRIPT_DIR/finpilot.nginx.conf" \
	/etc/nginx/sites-available/finpilot

sudo ln -sfn /etc/nginx/sites-available/finpilot /etc/nginx/sites-enabled/finpilot
sudo systemctl enable --now nginx
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl status nginx --no-pager