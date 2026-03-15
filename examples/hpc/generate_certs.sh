#!/bin/bash
set -euo pipefail

# Generate self-signed TLS certificates for SFL federation.
#
# Usage:
#   ./generate_certs.sh /scratch/$USER/sfl-certs
#
# Creates: ca.pem, server.pem, server.key, client.pem, client.key
#
# For production, replace with your organization's PKI certificates.

CERTS_DIR="${1:?Usage: $0 <output-directory>}"

mkdir -p "$CERTS_DIR"
cd "$CERTS_DIR"

echo "Generating self-signed certificates in $CERTS_DIR"

# ── CA ────────────────────────────────────────────────────────────
openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout ca.key -out ca.pem \
    -days 365 -subj "/CN=sfl-ca" \
    2>/dev/null
echo "  CA certificate: ca.pem"

# ── Server ────────────────────────────────────────────────────────
openssl req -newkey rsa:2048 -nodes \
    -keyout server.key -out server.csr \
    -subj "/CN=sfl-server" \
    2>/dev/null

# Add SAN for localhost and wildcard (HPC nodes often have internal DNS)
cat > server_ext.cnf <<EOF
[v3_req]
subjectAltName = @alt_names
[alt_names]
DNS.1 = localhost
DNS.2 = *.local
DNS.3 = *.internal
IP.1 = 127.0.0.1
EOF

openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key \
    -CAcreateserial -out server.pem -days 365 \
    -extfile server_ext.cnf -extensions v3_req \
    2>/dev/null
echo "  Server certificate: server.pem"

# ── Client ────────────────────────────────────────────────────────
openssl req -newkey rsa:2048 -nodes \
    -keyout client.key -out client.csr \
    -subj "/CN=sfl-client" \
    2>/dev/null

openssl x509 -req -in client.csr -CA ca.pem -CAkey ca.key \
    -CAcreateserial -out client.pem -days 365 \
    2>/dev/null
echo "  Client certificate: client.pem"

# ── Cleanup CSRs ──────────────────────────────────────────────────
rm -f *.csr *.srl server_ext.cnf

# ── Set permissions ───────────────────────────────────────────────
chmod 600 *.key
chmod 644 *.pem

echo ""
echo "Done. Certificate files:"
ls -la "$CERTS_DIR"/*.pem "$CERTS_DIR"/*.key
echo ""
echo "WARNING: These are self-signed certificates for development."
echo "For production, use your organization's PKI infrastructure."
