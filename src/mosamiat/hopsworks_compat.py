"""Reuse authenticated project PEM files with the Hopsworks 4.8 Python client."""

import os
from functools import partial
from pathlib import Path
from posixpath import normpath


def list_project_files(api, folder, *, offset, limit):
    """Keep HopsFS paths POSIX on Windows with the pinned Hopsworks SDK."""
    if os.name == "nt":
        from hopsworks_common.core.dataset_api import DatasetApi, client, inode, util

        if isinstance(api, DatasetApi):
            _, items = api._list_dataset_path(normpath(folder), inode.Inode, offset, limit)
            project = client.get_instance()._project_name
            return [util.convert_to_project_rel_path(item.path, project) for item in items]
    return api.list(folder, offset=offset, limit=limit)


def save_registry_model(entry, folder):
    """The SDK model upload config omits per-file chunk concurrency."""
    from hsml.model import Model

    config = {"chunk_size": 1024 * 1024, "simultaneous_uploads": 1}
    if not isinstance(entry, Model):
        return entry.save(folder, keep_original_files=True, upload_configuration=config)
    api = entry._model_engine._engine._dataset_api
    original = api.upload
    # The service can try to assemble the file before parallel chunks arrive.
    api.upload = partial(original, simultaneous_chunks=1)
    try:
        return entry.save(folder, keep_original_files=True, upload_configuration=config)
    finally:
        api.upload = original


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
