"""Stage the four pinned factory IRs and their matching private amp profile."""
import hashlib
import json
import os
from pathlib import Path

from dessamp_release_licenses import copy_open_regular_source, open_regular_source_no_follow

PROJECT = Path(__file__).resolve().parents[1]
BANK = PROJECT / "resources" / "FactoryBank"
CABINET_HASHES = {
    "Desfire.wav": "a4b4ceb38783f181b7d8615ad8d302b10b34608fb0ce3d1fc2ea8b2415d8dc05",
    "SHFLD-Mix.wav": "a905249c3c17d6a7a103fe130149d2a4ab81fde0f760056a2e121b368c0a88cd",
    "Gleichrichter.wav": "ca7f0aeb657bde3afc4f1f1e4a6792255dc6cb722007ceceaeb4162bb8d3c8d7",
    "Green.wav": "f84295fc00c3172179d0cb6505cb21421907db052992c96a3423a092a34d4697"
}


def profile_matches_bank(profile):
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        return False
    amps = profile.get("amplifiers")
    return (profile.get("ir_sha256") == CABINET_HASHES["Desfire.wav"]
            and isinstance(amps, list) and 1 <= len(amps) <= 32
            and all(isinstance(amp, dict) and isinstance(amp.get("cabinet", "Desfire.wav"), str)
                    and amp.get("cabinet", "Desfire.wav") in CABINET_HASHES for amp in amps))


def stage_factory_bank(resource_root, environment=None):
    """The caller resets the generated IR directory before staging.
    Public source-only exports omit the entire private bank.
    """
    environment = os.environ if environment is None else environment
    root = Path(resource_root)
    profile_target = root / "PrivateValidation" / "FactoryDefaults.json"
    profile_target.unlink(missing_ok=True)
    if not os.path.lexists(BANK):
        return
    for name, expected in CABINET_HASHES.items():
        source, metadata = open_regular_source_no_follow(BANK / name, BANK, PROJECT)
        with source:
            if hashlib.sha256(source.read()).hexdigest() != expected:
                raise RuntimeError("Version C cabinet hash mismatch: " + name)
            copy_open_regular_source(source, metadata, root / "IRs" / name)

    if environment.get("DESSAMP_PRIVATE_AMP_VALIDATION") != "YES":
        return
    source, metadata = open_regular_source_no_follow(BANK / "FactoryDefaults.json", BANK, PROJECT)
    with source:
        if metadata.st_size > 65536:
            raise RuntimeError("factory defaults exceed size limit")
        profile = json.load(source)
        if not profile_matches_bank(profile):
            raise RuntimeError("factory profile does not match the Version C bank")
        copy_open_regular_source(source, metadata, profile_target)
