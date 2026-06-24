from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ac_zero.certificates.certificate import Certificate
from ac_zero.environment.goals import exact_standard_goal, signed_permuted_basis_goal
from ac_zero.moves.catalog import ActionCatalog


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Structured outcome of independent certificate replay."""

    ok: bool
    reason: str
    final_hash: str | None = None


class CertificateVerifier:
    """Replay certificates without loading search code or neural checkpoints."""

    def verify(self, certificate: Certificate) -> VerificationResult:
        """Validate schema, primitive moves, intermediate hashes, and final goal."""
        if certificate.schema_version != "aczero-certificate-v1":
            return VerificationResult(False, "unsupported schema version")
        catalog = ActionCatalog(certificate.initial_presentation.rank)
        if certificate.action_catalog_version != catalog.version:
            return VerificationResult(False, "catalog version mismatch")
        pres = certificate.initial_presentation
        for idx, move in enumerate(certificate.moves):
            if move not in catalog.moves:
                return VerificationResult(
                    False, f"move {idx} is not in the strict primitive catalog"
                )
            pres = move.apply(pres)
            if idx >= len(certificate.intermediate_hashes):
                return VerificationResult(False, "missing intermediate hash")
            if pres.content_hash != certificate.intermediate_hashes[idx]:
                return VerificationResult(False, f"intermediate hash mismatch at step {idx + 1}")
        if pres.content_hash != certificate.final_presentation.content_hash:
            return VerificationResult(False, "final presentation hash mismatch")
        if certificate.goal_mode == "exact_standard":
            goal = exact_standard_goal(pres)
        elif certificate.goal_mode == "signed_permuted_basis":
            goal = signed_permuted_basis_goal(pres)
        else:
            return VerificationResult(False, "unknown goal mode")
        if not goal:
            return VerificationResult(
                False, "final presentation is not a goal state", pres.content_hash
            )
        return VerificationResult(True, "verified", pres.content_hash)

    def verify_path(self, path: str | Path) -> VerificationResult:
        """Parse and verify a certificate file, returning failures as data."""
        try:
            certificate = Certificate.from_path(path)
        except Exception as exc:
            return VerificationResult(False, f"certificate parse error: {exc}")
        return self.verify(certificate)
