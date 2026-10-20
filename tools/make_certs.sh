#!/usr/bin/env bash
#
# make_certs.sh — a throwaway certificate authority and a server certificate
# it signs, for running the cache with TLS locally.
#
# Clients verify the server against certs/ca.pem, so this is real chain
# verification, not "accept any certificate". The CA key never leaves
# certs/, which is git-ignored.
#
#   ./tools/make_certs.sh
#   python -m cache.server --tls-cert certs/server.pem --tls-key certs/server.key
#   redis-cli --tls --cacert certs/ca.pem -p 6390 PING
#
set -euo pipefail
DIR="${1:-certs}"
mkdir -p "$DIR"
cd "$DIR"

openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout ca.key -out ca.pem -subj "/CN=Evict local CA" 2>/dev/null

openssl req -newkey rsa:2048 -nodes \
    -keyout server.key -out server.csr -subj "/CN=localhost" 2>/dev/null

cat > server.ext <<EXT
subjectAltName = DNS:localhost, IP:127.0.0.1
extendedKeyUsage = serverAuth
EXT
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key -CAcreateserial \
    -days 365 -out server.pem -extfile server.ext 2>/dev/null
rm -f server.csr server.ext ca.srl
chmod 600 ca.key server.key
echo "wrote $DIR/ca.pem, $DIR/server.pem, $DIR/server.key"
