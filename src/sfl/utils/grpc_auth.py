"""
gRPC channel security helpers for production Flower deployments.

Provides mTLS configuration and token-based authentication for
securing the Flower gRPC channel between clients and server.

In simulation mode (run_simulation), these are not needed — the channel
is in-process. For multi-node HPC deployments, use mTLS or token auth
to prevent unauthorized clients from joining the federation.

Usage (mTLS)::

    from sfl.utils.grpc_auth import load_tls_certificates, TLSConfig

    tls = TLSConfig(
        ca_cert="certs/ca.pem",
        server_cert="certs/server.pem",
        server_key="certs/server.key",
    )
    certs = load_tls_certificates(tls)
    # Pass to Flower's start_server / start_client

Usage (token auth)::

    from sfl.utils.grpc_auth import TokenAuthInterceptor

    interceptor = TokenAuthInterceptor(token="my-secret-token")
    # Use with grpc.intercept_channel() or grpc.server interceptors
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from sfl.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TLSConfig:
    """Configuration for mutual TLS (mTLS) on the Flower gRPC channel.

    For HPC deployments, certificates should be provisioned by the
    cluster's PKI infrastructure or generated per-job via a shared CA.

    Attributes:
        ca_cert: Path to Certificate Authority PEM file.
        server_cert: Path to server certificate PEM file (server-side only).
        server_key: Path to server private key PEM file (server-side only).
        client_cert: Path to client certificate PEM file (client-side only).
        client_key: Path to client private key PEM file (client-side only).
    """

    ca_cert: str = ""
    server_cert: str = ""
    server_key: str = ""
    client_cert: str = ""
    client_key: str = ""


def load_tls_certificates(
    config: TLSConfig,
    *,
    role: str = "server",
) -> Tuple[bytes, bytes, bytes]:
    """Load TLS certificates from disk for Flower's SSL configuration.

    Args:
        config: TLS certificate paths.
        role: Either "server" or "client".

    Returns:
        Tuple of (ca_cert_bytes, cert_bytes, key_bytes) suitable for
        Flower's ``certificates`` parameter.

    Raises:
        FileNotFoundError: If any certificate file is missing.
        ValueError: If role is not "server" or "client".
    """
    if role not in ("server", "client"):
        raise ValueError(f"role must be 'server' or 'client', got {role!r}")

    ca_path = Path(config.ca_cert)
    if not ca_path.exists():
        raise FileNotFoundError(f"CA certificate not found: {ca_path}")
    ca_bytes = ca_path.read_bytes()

    if role == "server":
        cert_path = Path(config.server_cert)
        key_path = Path(config.server_key)
    else:
        cert_path = Path(config.client_cert)
        key_path = Path(config.client_key)

    if not cert_path.exists():
        raise FileNotFoundError(f"{role} certificate not found: {cert_path}")
    if not key_path.exists():
        raise FileNotFoundError(f"{role} key not found: {key_path}")

    cert_bytes = cert_path.read_bytes()
    key_bytes = key_path.read_bytes()

    logger.info(f"Loaded {role} TLS certificates from {cert_path.parent}")
    return ca_bytes, cert_bytes, key_bytes


def tls_config_from_env() -> Optional[TLSConfig]:
    """Build TLSConfig from environment variables, or None if not set.

    Environment variables:
        SFL_TLS_CA_CERT: Path to CA certificate
        SFL_TLS_SERVER_CERT: Path to server certificate
        SFL_TLS_SERVER_KEY: Path to server key
        SFL_TLS_CLIENT_CERT: Path to client certificate
        SFL_TLS_CLIENT_KEY: Path to client key
    """
    ca = os.environ.get("SFL_TLS_CA_CERT", "")
    if not ca:
        return None
    return TLSConfig(
        ca_cert=ca,
        server_cert=os.environ.get("SFL_TLS_SERVER_CERT", ""),
        server_key=os.environ.get("SFL_TLS_SERVER_KEY", ""),
        client_cert=os.environ.get("SFL_TLS_CLIENT_CERT", ""),
        client_key=os.environ.get("SFL_TLS_CLIENT_KEY", ""),
    )


# ── Token-based authentication ──────────────────────────────────────


@dataclass
class TokenAuthConfig:
    """Configuration for token-based authentication.

    Simpler than mTLS — each client sends a bearer token in the gRPC
    metadata. The server validates it. Suitable for environments where
    PKI is not available but a shared secret can be distributed.

    Attributes:
        token: Shared secret token. In production, read from a file
            or environment variable, never hardcode.
        header_key: gRPC metadata key for the token.
    """

    token: str = ""
    header_key: str = "x-sfl-auth-token"


def token_config_from_env() -> Optional[TokenAuthConfig]:
    """Build TokenAuthConfig from environment, or None if not set.

    Environment variables:
        SFL_AUTH_TOKEN: Bearer token for gRPC authentication
        SFL_AUTH_HEADER: Custom header key (default: x-sfl-auth-token)
    """
    token = os.environ.get("SFL_AUTH_TOKEN", "")
    if not token:
        return None
    return TokenAuthConfig(
        token=token,
        header_key=os.environ.get("SFL_AUTH_HEADER", "x-sfl-auth-token"),
    )


def make_client_auth_interceptor(config: TokenAuthConfig):
    """Create a gRPC client interceptor that adds the auth token to requests.

    Returns:
        A grpc.UnaryUnaryClientInterceptor that injects the token.

    Raises:
        ImportError: If grpc is not installed.
    """
    import grpc

    class _ClientAuthInterceptor(
        grpc.UnaryUnaryClientInterceptor,
        grpc.UnaryStreamClientInterceptor,
        grpc.StreamUnaryClientInterceptor,
        grpc.StreamStreamClientInterceptor,
    ):
        def _add_token(self, client_call_details):
            metadata = list(client_call_details.metadata or [])
            metadata.append((config.header_key, config.token))
            return grpc.aio.ClientCallDetails(
                method=client_call_details.method,
                timeout=client_call_details.timeout,
                metadata=metadata,
                credentials=client_call_details.credentials,
                wait_for_ready=client_call_details.wait_for_ready,
            )

        def intercept_unary_unary(self, continuation, client_call_details, request):
            return continuation(self._add_token(client_call_details), request)

        def intercept_unary_stream(self, continuation, client_call_details, request):
            return continuation(self._add_token(client_call_details), request)

        def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
            return continuation(self._add_token(client_call_details), request_iterator)

        def intercept_stream_stream(self, continuation, client_call_details, request_iterator):
            return continuation(self._add_token(client_call_details), request_iterator)

    return _ClientAuthInterceptor()


def make_server_auth_interceptor(config: TokenAuthConfig):
    """Create a gRPC server interceptor that validates the auth token.

    Returns:
        A grpc.ServerInterceptor that rejects unauthorized requests.

    Raises:
        ImportError: If grpc is not installed.
    """
    import grpc

    class _ServerAuthInterceptor(grpc.ServerInterceptor):
        def intercept_service(self, continuation, handler_call_details):
            metadata = dict(handler_call_details.invocation_metadata or [])
            received = metadata.get(config.header_key, "")
            if received != config.token:
                logger.warning(
                    "Rejected unauthenticated gRPC call to %s",
                    handler_call_details.method,
                )
                return grpc.unary_unary_rpc_method_handler(
                    lambda req, ctx: ctx.abort(
                        grpc.StatusCode.UNAUTHENTICATED,
                        "Invalid or missing authentication token",
                    )
                )
            return continuation(handler_call_details)

    return _ServerAuthInterceptor()


# ── Certificate generation helper (development only) ────────────────


def generate_self_signed_certs(
    output_dir: str,
    *,
    cn: str = "sfl-dev",
    days: int = 365,
) -> TLSConfig:
    """Generate self-signed CA + server + client certificates for development.

    NOT for production use — these are self-signed and not rotated.
    For production, use your organization's PKI or a tool like cert-manager.

    Requires the `cryptography` package.

    Args:
        output_dir: Directory to write certificate files.
        cn: Common Name for the certificates.
        days: Certificate validity in days.

    Returns:
        TLSConfig with paths to the generated files.
    """
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _make_key():
        return rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def _write_key(key, path):
        path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )

    def _write_cert(cert, path):
        path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    now = datetime.datetime.now(datetime.timezone.utc)
    validity = datetime.timedelta(days=days)

    # CA
    ca_key = _make_key()
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"{cn}-ca")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + validity)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    # Server
    srv_key = _make_key()
    srv_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"{cn}-server")])
    srv_cert = (
        x509.CertificateBuilder()
        .subject_name(srv_name)
        .issuer_name(ca_name)
        .public_key(srv_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + validity)
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("*.local"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    # Client
    cli_key = _make_key()
    cli_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"{cn}-client")])
    cli_cert = (
        x509.CertificateBuilder()
        .subject_name(cli_name)
        .issuer_name(ca_name)
        .public_key(cli_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + validity)
        .sign(ca_key, hashes.SHA256())
    )

    _write_cert(ca_cert, out / "ca.pem")
    _write_key(srv_key, out / "server.key")
    _write_cert(srv_cert, out / "server.pem")
    _write_key(cli_key, out / "client.key")
    _write_cert(cli_cert, out / "client.pem")

    logger.info(f"Generated self-signed certificates in {out}")

    return TLSConfig(
        ca_cert=str(out / "ca.pem"),
        server_cert=str(out / "server.pem"),
        server_key=str(out / "server.key"),
        client_cert=str(out / "client.pem"),
        client_key=str(out / "client.key"),
    )
