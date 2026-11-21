"""Reuse authenticated project PEM files with the Hopsworks 4.8 Python client."""

from pathlib import Path


def configure_project_certificates() -> None:
    """Avoid redundant JKS conversion and its hardcoded /tmp output directory.

    Hopsworks already downloads project PEM files at login. The Kafka connector
    otherwise converts the same credentials again through pyjks, whose released
    version cannot read some Java 21 keystores. Only the authenticated project's
    own store paths are intercepted; unrelated connector stores use the SDK.
    """
    from hopsworks_common import client

    instance = client.get_instance()
    original = instance._write_pem
    project_keystore = Path(instance._get_jks_key_store_path()).resolve()
    project_truststore = Path(instance._get_jks_trust_store_path()).resolve()

    def project_pem(keystore_path, keystore_pw, truststore_path, truststore_pw, prefix):
        if (
            Path(keystore_path).resolve() == project_keystore
            and Path(truststore_path).resolve() == project_truststore
        ):
            paths = (
                instance._get_ca_chain_path(),
                instance._get_client_cert_path(),
                instance._get_client_key_path(),
            )
            if not all(Path(path).is_file() for path in paths):
                raise FileNotFoundError("Authenticated project PEM files are missing; reconnect")
            return paths
        return original(keystore_path, keystore_pw, truststore_path, truststore_pw, prefix)

    instance._write_pem = project_pem
